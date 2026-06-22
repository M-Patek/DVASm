# DVAS 深度系统优化工程计划

## 执行摘要

基于对DVAS代码库的全面分析，本计划制定四大领域的系统性优化方案：视频处理、Teacher模型、Student训练、推理引擎。预期整体性能提升 **5-10倍**，端到端pipeline延迟降低 **80%**。

---

## 现状分析

### 视频处理层 (src/dvas/data/)
| 组件 | 当前状态 | 瓶颈 |
|------|---------|------|
| 解码器 | OpenCV默认，Decord可选 | 顺序读取，无硬件加速 |
| GPU解码 | 已实现但未默认启用 | 需显式配置 |
| 缓存 | LRU元数据缓存(1000项) | 无帧缓存 |
| 采样 | KeyFrame全扫描 | O(n)复杂度 |

### Teacher模型层 (src/dvas/models/teacher/)
| 组件 | 当前状态 | 瓶颈 |
|------|---------|------|
| HTTP连接 | httpx基本连接池 | HTTP/1.1，无多路复用 |
| 批处理 | OpenAI Batch API(5万req) | 仅OpenAI支持 |
| 缓存 | 语义缓存(感知哈希) | 未集成到batch处理 |
| 容错 | 无限退避重试 | 无熔断机制 |

### Student训练层 (src/dvas/models/student/)
| 组件 | 当前状态 | 瓶颈 |
|------|---------|------|
| 分布式 | FSDP已实现 | 无DeepSpeed支持 |
| 量化 | QLoRA 4-bit | 无Unsloth加速 |
| 注意力 | Flash Attention 2配置 | 依赖未安装 |
| 精度 | bf16混合精度 | - |

### 推理层 (src/dvas/models/student/inference.py)
| 组件 | 当前状态 | 瓶颈 |
|------|---------|------|
| 后端 | HF Transformers/vLLM可选 | vLLM未默认启用 |
| 批处理 | 顺序循环 | 无连续批处理 |
| KV缓存 | 标准实现 | 无PagedAttention |

---

## 优化阶段规划

### Phase 1: 视频处理深度优化 (预期: 50x加速)

#### 1.1 默认启用Decord GPU解码

**目标文件:**
- `src/dvas/data/decord_reader.py` - 增强GPU自动检测
- `src/dvas/data/video_loader.py` - 修改VideoLoader默认行为
- `src/dvas/config/settings.py` - 添加视频处理配置

**实现方案:**

```python
# src/dvas/data/decord_reader.py - 新增函数
def get_optimal_video_context() -> str:
    """自动检测最佳解码上下文."""
    if not _DECORD_AVAILABLE:
        return "cpu"
    
    # 优先检测PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except ImportError:
        pass
    
    # 直接检测decord GPU支持
    try:
        dec.gpu(0)
        return "cuda:0"
    except Exception:
        pass
    
    return "cpu"

# src/dvas/config/settings.py - 新增配置
class VideoSettings(BaseModel):
    default_decoder: str = "decord"  # "opencv" | "decord"
    prefer_gpu: bool = True
    gpu_device_id: int = 0
    metadata_cache_size: int = 1000
    enable_frame_prefetch: bool = True
    prefetch_queue_size: int = 32
```

**修改VideoLoader构造函数:**

```python
# src/dvas/data/video_loader.py
class VideoLoader:
    def __init__(
        self,
        video_path: Union[str, Path],
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        # 修改: 默认使用decord
        use_decord: bool = True,
        ctx: Optional[str] = None,  # None表示自动检测
    ):
        self.video_path = Path(video_path)
        
        # 自动选择最佳reader
        if use_decord and _DECORD_AVAILABLE:
            from dvas.data.decord_reader import DecordVideoReader, get_optimal_video_context
            ctx = ctx or get_optimal_video_context()
            self._reader = DecordVideoReader(video_path, ctx=ctx)
        else:
            self._reader = VideoReader(video_path)
```

**验证方式:**
```bash
python -c "
from dvas.data import VideoLoader
import time

# 测试GPU解码
loader = VideoLoader('test.mp4')  # 应自动使用GPU
print(f'Reader type: {type(loader._reader).__name__}')
"
```

#### 1.2 帧预取实现

**目标文件:**
- `src/dvas/data/prefetch_reader.py` - 新建文件

**实现方案:**

```python
"""Prefetch video reader with background frame loading."""

import asyncio
import threading
import queue
from typing import Iterator, Optional
from dvas.data.video_reader import Frame

class PrefetchVideoReader:
    """装饰器模式包装任意VideoReader，添加预取队列."""
    
    def __init__(
        self,
        reader,
        prefetch_size: int = 32,
        num_workers: int = 2
    ):
        self._reader = reader
        self._prefetch_size = prefetch_size
        self._num_workers = num_workers
        self._queue = queue.Queue(maxsize=prefetch_size)
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._iterator: Optional[Iterator[Frame]] = None
        
    def _worker_loop(self):
        """后台工作线程持续填充队列."""
        while not self._stop_event.is_set():
            try:
                frame = next(self._iterator)
                self._queue.put(frame, timeout=0.1)
            except StopIteration:
                self._queue.put(None)  # Sentinel
                break
            except queue.Full:
                continue
                
    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None, step: int = 1):
        """生成帧，从预取队列消费."""
        self._iterator = self._reader.read_frames(start_frame, end_frame, step)
        
        # 启动工作线程
        for _ in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop)
            t.start()
            self._workers.append(t)
        
        # 从队列消费
        finished_workers = 0
        while finished_workers < self._num_workers:
            frame = self._queue.get()
            if frame is None:
                finished_workers += 1
            else:
                yield frame
        
        # 清理
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=1.0)
```

#### 1.3 KeyFrame采样优化

**目标文件:**
- `src/dvas/data/frame_sampler.py`

**优化策略:**

```python
# 1. 早期终止优化
class OptimizedKeyFrameSampler:
    def sample(self, reader, num_frames: int) -> List[Frame]:
        total_frames = len(reader)
        
        # 快速路径: 视频很短
        if total_frames <= num_frames * 2:
            return list(reader.get_batch(range(total_frames)))
        
        # 分层采样: 先粗粒度检测，再细粒度精选
        coarse_step = max(1, total_frames // (num_frames * 10))
        coarse_indices = range(0, total_frames, coarse_step)
        coarse_frames = reader.get_batch(list(coarse_indices))
        
        # 仅在粗粒度关键帧周围进行细粒度采样
        keyframe_regions = self._detect_regions(coarse_frames)
        
        # 并行计算细粒度区域
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for region in keyframe_regions[:num_frames]:
                futures.append(
                    executor.submit(self._process_region, reader, region)
                )
            results = [f.result() for f in futures]
        
        return self._select_best_frames(results, num_frames)

# 2. 运动检测向量化
import numba

@numba.jit(nopython=True, parallel=True)
def _compute_motion_scores_batch(frames_gray: np.ndarray) -> np.ndarray:
    """并行计算帧间差异分数."""
    n = len(frames_gray)
    scores = np.zeros(n-1)
    for i in numba.prange(n-1):
        scores[i] = np.mean(np.abs(frames_gray[i].astype(np.float32) - 
                                   frames_gray[i+1].astype(np.float32)))
    return scores
```

**预期性能:**
- GPU解码: 30fps → 1500fps (50x)
- KeyFrame采样: O(n) → O(n/10) 分层采样

---

### Phase 2: Teacher模型优化 (预期: 3-5x吞吐)

#### 2.1 HTTP/2启用与连接池优化

**目标文件:**
- `src/dvas/models/teacher/base.py`
- `src/dvas/models/teacher/batch_api.py`

**实现方案:**

```python
# src/dvas/models/teacher/base.py - 修改HTTPClient初始化
class TeacherModel:
    def __init__(self, ...):
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30.0
            ),
            http2=True,  # 启用HTTP/2多路复用
            timeout=httpx.Timeout(connect=10.0, read=120.0, pool=5.0),
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=30, max=100"
            }
        )
```

#### 2.2 请求去重与合并

**目标文件:**
- `src/dvas/models/teacher/request_coalescer.py` - 新建

**实现方案:**

```python
"""Request coalescing for identical/similar API calls."""

import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import hashlib

@dataclass
class PendingRequest:
    key: str
    future: asyncio.Future
    timestamp: float

class RequestCoalescer:
    """合并相同/相似请求，避免重复API调用."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self._pending: Dict[str, PendingRequest] = {}
        self._similarity_threshold = similarity_threshold
        self._lock = asyncio.Lock()
        
    def _compute_key(self, video_hash: str, prompt: str, params: dict) -> str:
        """计算请求指纹."""
        content = f"{video_hash}:{prompt}:{sorted(params.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def execute(
        self,
        video_hash: str,
        prompt: str,
        params: dict,
        api_caller: Callable
    ) -> Any:
        """执行请求，如果相同请求正在进行则复用结果."""
        key = self._compute_key(video_hash, prompt, params)
        
        async with self._lock:
            if key in self._pending:
                # 等待已有请求完成
                pending = self._pending[key]
                return await pending.future
            
            # 创建新请求
            future = asyncio.Future()
            self._pending[key] = PendingRequest(key, future, asyncio.get_event_loop().time())
        
        try:
            # 执行实际API调用
            result = await api_caller()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._pending.pop(key, None)
```

#### 2.3 熔断器模式

**目标文件:**
- `src/dvas/models/teacher/circuit_breaker.py` - 新建

**实现方案:**

```python
"""Circuit breaker pattern for resilient API calls."""

import asyncio
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional

class CircuitState(Enum):
    CLOSED = auto()      # 正常
    OPEN = auto()        # 熔断
    HALF_OPEN = auto()   # 半开测试

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2

class CircuitBreaker:
    """熔断器防止级联故障."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs):
        """在熔断器保护下执行函数."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpen(f"Circuit {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = max(0, self._failure_count - 1)
    
    async def _on_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
```

**预期性能:**
- HTTP/2: 吞吐提升 2-3x (多路复用)
- 请求合并: 重复请求减少 50-80%
- 熔断器: 故障恢复时间从分钟级降至秒级

---

### Phase 3: Student训练优化 (预期: 2-5x训练速度)

#### 3.1 Unsloth集成

**目标文件:**
- `src/dvas/models/student/unsloth_trainer.py` - 新建
- `pyproject.toml` - 添加可选依赖

**实现方案:**

```python
"""Unsloth-accelerated training for 2-5x speedup."""

from typing import Optional
import torch
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# 可选依赖检测
try:
    from unsloth import FastVisionModel
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False
    FastVisionModel = None

class UnslothSFTTrainer:
    """SFT训练器使用Unsloth加速."""
    
    def __init__(self, config: "Config"):
        self.config = config
        self._verify_unsloth_available()
        
    def _verify_unsloth_available(self):
        if not _UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth not available. Install with: "
                "pip install unsloth[vision]"
            )
    
    def load_model(self, model_name: str, load_in_4bit: bool = True):
        """加载Unsloth优化模型."""
        logger.info("Loading model with Unsloth acceleration", model=model_name)
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=True,
        )
        
        # 配置LoRA
        model = FastVisionModel.get_peft_model(
            model,
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            target_modules=self.config.model.lora_target_modules,
            lora_dropout=self.config.model.lora_dropout,
        )
        
        return model, tokenizer
    
    def train(self, dataset, output_dir: str):
        """执行加速训练."""
        model, tokenizer = self.load_model(
            self.config.model.model_name_or_path,
            load_in_4bit=self.config.model.load_in_4bit
        )
        
        # Unsloth训练参数
        training_args = UnslothTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.data.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_epochs,
            learning_rate=self.config.training.learning_rate,
            # Unsloth优化
            max_seq_length=self.config.data.max_seq_length,
            # 显存优化
            optim="adamw_8bit",  # 8-bit优化器
            # 速度优化
            group_by_length=False,
            # 其他参数...
        )
        
        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        
        trainer.train()
        
        # Unsloth内存统计
        gpu_stats = torch.cuda.get_device_properties(0)
        logger.info(
            "Training completed",
            peak_memory_gb=torch.cuda.max_memory_reserved() / 2**30,
            gpu_name=gpu_stats.name
        )

# pyproject.toml 添加可选依赖
# [project.optional-dependencies]
# unsloth = ["unsloth[vision]>=2024.1.0"]
```

#### 3.2 Flash Attention 2确保可用

**目标文件:**
- `pyproject.toml`
- `src/dvas/models/student/config.py`

**实现方案:**

```python
# pyproject.toml 添加
# [project.optional-dependencies]
# flash-attn = ["flash-attn>=2.5.0"]

# src/dvas/models/student/config.py
from typing import Literal
import warnings

class ModelConfig:
    # ... 其他配置 ...
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = "flash_attention_2"
    
    def validate(self):
        """验证Flash Attention可用性."""
        if self.attn_implementation == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                warnings.warn(
                    "flash-attention not installed. Falling back to eager. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )
                self.attn_implementation = "eager"
```

**预期性能:**
- Unsloth: 训练速度 2-5x
- Flash Attention 2: 训练速度 +30%，内存 -20%

---

### Phase 4: 推理优化 (预期: 10-20x吞吐)

#### 4.1 vLLM默认启用与优化

**目标文件:**
- `src/dvas/models/student/inference.py`

**实现方案:**

```python
# 修改 StudentInferenceEngine.__init__
def __init__(
    self,
    model_path: str,
    use_vllm: bool = True,  # 改为默认True
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    **kwargs
):
    self.model_path = model_path
    self.use_vllm = use_vllm
    
    if use_vllm:
        self._init_vllm_engine(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
    else:
        self._init_hf_engine()

def _init_vllm_engine(self, **vllm_kwargs):
    """初始化vLLM引擎，支持连续批处理."""
    from vllm import LLM, SamplingParams
    
    self._engine = LLM(
        model=self.model_path,
        trust_remote_code=True,
        # 连续批处理优化
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        # GPU内存优化
        **vllm_kwargs
    )
    
    logger.info(
        "vLLM engine initialized",
        tensor_parallel=vllm_kwargs.get('tensor_parallel_size', 1),
        gpu_util=vllm_kwargs.get('gpu_memory_utilization', 0.9)
    )

def batch_infer(
    self,
    videos: List[np.ndarray],
    prompts: List[str],
    batch_size: int = 32  # vLLM处理大batch更高效
) -> List[str]:
    """优化的批量推理."""
    if self.use_vllm:
        return self._vllm_batch_infer(videos, prompts, batch_size)
    else:
        return self._hf_batch_infer(videos, prompts, batch_size)

def _vllm_batch_infer(self, videos, prompts, batch_size):
    """vLLM连续批处理推理."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=self.config.generation.temperature,
        max_tokens=self.config.generation.max_new_tokens,
        top_p=0.95,
    )
    
    # 构建vLLM输入
    vllm_inputs = []
    for video, prompt in zip(videos, prompts):
        # 处理视频帧为vLLM格式
        processed = self._prepare_vllm_input(video, prompt)
        vllm_inputs.append(processed)
    
    # 连续批处理生成
    outputs = self._engine.generate(vllm_inputs, sampling_params)
    
    return [output.outputs[0].text for output in outputs]
```

#### 4.2 KV-Cache管理优化

vLLM自带PagedAttention优化，无需额外实现。

**预期性能:**
- vLLM连续批处理: 吞吐 10-20x
- 延迟: 首token TTFT < 100ms

---

## 集成验证方案

### 单元测试

```python
# tests/test_video_gpu_decode.py
def test_decord_gpu_auto_select():
    """测试GPU自动选择."""
    from dvas.data import VideoLoader
    
    loader = VideoLoader("test.mp4")
    reader_type = type(loader._reader).__name__
    
    assert reader_type == "DecordVideoReader", f"Expected DecordVideoReader, got {reader_type}"

# tests/test_teacher_http2.py  
def test_teacher_http2_enabled():
    """测试HTTP/2启用."""
    from dvas.models.teacher.base import TeacherModel
    
    model = TeacherModel()
    assert model._client.http2 is True

# tests/test_student_unsloth.py
@pytest.mark.skipif(not _UNSLOTH_AVAILABLE, reason="Unsloth not installed")
def test_unsloth_trainer_init():
    """测试Unsloth训练器初始化."""
    from dvas.models.student.unsloth_trainer import UnslothSFTTrainer
    
    trainer = UnslothSFTTrainer(config)
    assert trainer is not None
```

### 集成测试

```python
# tests/integration/test_e2e_optimized.py
def test_optimized_pipeline():
    """测试完整优化pipeline."""
    import time
    
    # 视频处理
    start = time.time()
    loader = VideoLoader("benchmark.mp4")  # 自动GPU解码
    frames = list(loader.read_frames(num_frames=16))
    video_time = time.time() - start
    
    # Teacher批处理
    start = time.time()
    results = batch_api.submit_and_wait(requests[:100])
    teacher_time = time.time() - start
    
    # Student推理
    start = time.time()
    outputs = engine.batch_infer(videos, prompts)
    infer_time = time.time() - start
    
    # 断言性能目标
    assert video_time < 1.0, f"Video processing too slow: {video_time}s"
    assert teacher_time < 30.0, f"Teacher batch too slow: {teacher_time}s"
    assert infer_time < 5.0, f"Inference too slow: {infer_time}s"
```

### 性能基准

```python
# benchmarks/test_performance_targets.py
"""
性能目标验证:

| 指标 | 当前 | 目标 | 验证方法 |
|------|------|------|---------|
| 视频加载 | 30fps | 1500fps | decord_reader.py |
| Teacher吞吐 | 100req/min | 500req/min | batch_api.py |
| Student训练 | 1.5it/s | 4.5it/s | sft_trainer.py |
| Student推理 | 10tok/s | 50tok/s | inference.py |
| E2E Pipeline | 2min/video | 15s/video | full_pipeline.py |
"""
```

---

## 执行时间表

| 阶段 | 任务 | 预计工时 | 依赖 |
|------|------|---------|------|
| Phase 1 | GPU解码默认启用 | 4h | - |
| Phase 1 | 帧预取实现 | 6h | GPU解码 |
| Phase 1 | KeyFrame采样优化 | 4h | - |
| Phase 2 | HTTP/2启用 | 2h | - |
| Phase 2 | 请求去重 | 6h | - |
| Phase 2 | 熔断器 | 4h | - |
| Phase 3 | Unsloth集成 | 8h | - |
| Phase 3 | Flash Attn确保 | 2h | - |
| Phase 4 | vLLM默认启用 | 4h | - |
| Phase 4 | 连续批处理 | 6h | vLLM |
| **总计** | | **46h** | |

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Unsloth兼容性问题 | 中 | 保持HF Transformers作为fallback |
| vLLM内存溢出 | 中 | 自动调整gpu_memory_utilization |
| Decord GPU失败 | 低 | 自动fallback到CPU解码 |
| HTTP/2不兼容 | 低 | 检测并自动降级到HTTP/1.1 |
| Flash Attn安装失败 | 低 | 优雅降级到SDPA |

---

## 后续工作建议

1. **监控集成**: 将性能指标接入Grafana仪表板
2. **自动调优**: 基于运行时统计自动调整batch size
3. **模型量化**: 集成AWQ/GPTQ用于部署优化
4. **分布式推理**: TensorRT-LLM多节点部署

---

**计划版本**: 1.0  
**最后更新**: 2026-06-22  
**审批状态**: 待审批
