"""Inference engine for student model with vLLM support."""

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dvas.models.base import GenerationResult, GenerationStatus, ModelType, UnifiedModel
from dvas.models.teacher.base import TeacherModel
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def load_frames_from_video(
    video_path: Path,
    num_frames: int = 16,
) -> List[Image.Image]:
    """Load frames from video for inference."""
    from dvas.data.video_loader import VideoLoader

    frames = []
    with VideoLoader(video_path) as loader:
        for frame in loader.read_frames(num_frames=num_frames):
            # Convert BGR to RGB
            rgb = frame.data[:, :, ::-1]
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

    return frames


class StudentInferenceEngine(UnifiedModel):
    """Inference engine for the fine-tuned student model.

    支持两种后端:
    - vLLM (默认): 使用连续批处理和PagedAttention，吞吐量高
    - HF Transformers: 兼容性好，无需额外依赖
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        use_vllm: bool = True,  # 改为默认启用vLLM
        device: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        quantization: Optional[str] = None,
        **vllm_kwargs: Any,
    ):
        """初始化推理引擎.

        Args:
            model_path: 模型路径或HuggingFace模型ID
            use_vllm: 是否使用vLLM后端(默认True)
            device: 设备("auto", "cuda", "cpu")
            tensor_parallel_size: 张量并行大小(多GPU)
            gpu_memory_utilization: GPU显存利用率(0-1)
            max_model_len: 最大序列长度
            quantization: 量化方式("awq", "gptq", 或None)
            **vllm_kwargs: 额外的vLLM参数
        """
        self.model_path = Path(model_path)
        self.use_vllm = use_vllm
        self.device = device
        self.vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "quantization": quantization,
            **vllm_kwargs,
        }

        self.model = None
        self.processor = None
        self._sampling_params = None
        self._load_model()

    @property
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        return ModelType.STUDENT_LOCAL

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        return str(self.model_path.name)

    def _load_model(self) -> None:
        """Load model and processor."""
        logger.info("Loading student model", path=str(self.model_path))

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

    def _load_hf(self) -> None:
        """Load with HuggingFace transformers."""
        import torch

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        logger.info("Model loaded with HF transformers")

    def _load_vllm(self) -> None:
        """Load with vLLM for optimized inference."""
        try:
            from vllm import LLM

            # 配置vLLM优化参数
            vllm_kwargs = {
                "model": str(self.model_path),
                "trust_remote_code": True,
                # 连续批处理优化
                "max_num_seqs": 256,  # 最大并发序列数
                "max_num_batched_tokens": self.vllm_config.get("max_model_len", 4096),
                # GPU内存优化
                "tensor_parallel_size": self.vllm_config.get("tensor_parallel_size", 1),
                "gpu_memory_utilization": self.vllm_config.get("gpu_memory_utilization", 0.9),
                "max_model_len": self.vllm_config.get("max_model_len", 8192),
            }

            # 添加量化配置
            if self.vllm_config.get("quantization"):
                vllm_kwargs["quantization"] = self.vllm_config["quantization"]

            # 合并额外参数
            vllm_kwargs.update(self.vllm_config)

            self.model = LLM(**vllm_kwargs)

            # vLLM uses its own tokenizer
            self.processor = self.model.get_tokenizer()

            logger.info(
                "Model loaded with vLLM",
                tensor_parallel=self.vllm_config.get("tensor_parallel_size", 1),
                gpu_util=self.vllm_config.get("gpu_memory_utilization", 0.9),
                max_seqs=256,
            )

        except ImportError:
            logger.warning("vLLM not available, falling back to HF")
            self.use_vllm = False
            self._load_hf()

    def _capabilities(self) -> List[str]:
        """Return list of supported capabilities."""
        return ["video", "frames", "text", "multimodal"]

    def estimate_cost(
        self,
        num_frames: int = 16,
        prompt_length: int = 500,
    ) -> float:
        """Estimate cost for a generation request."""
        # Local inference: essentially free (compute cost only)
        return 0.0

    async def generate(
        self,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """Generate description for a video."""
        start_time = time.perf_counter()

        try:
            if video_path is None and frames is None:
                return GenerationResult.failure(
                    error_message="Must provide either video_path or frames",
                    model_type=self.model_type,
                    model_version=self.model_version,
                )

            # Load frames if video_path provided
            if frames is None and video_path is not None:
                pil_frames = load_frames_from_video(video_path)
            else:
                pil_frames = []
                for frame in frames or []:
                    if isinstance(frame, np.ndarray):
                        rgb = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
                        pil_frames.append(Image.fromarray(rgb))
                    elif isinstance(frame, Image.Image):
                        pil_frames.append(frame)

            if not pil_frames:
                return GenerationResult.failure(
                    error_message="No frames could be loaded",
                    model_type=self.model_type,
                    model_version=self.model_version,
                )

            default_prompt = (
                "Describe the video in detail, including hand actions and object interactions."
            )
            system_prompt = prompt or default_prompt

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]

            if self.use_vllm:
                text = self._generate_vllm(messages, pil_frames, **kwargs)
            else:
                text = self._generate_hf(messages, pil_frames, **kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return GenerationResult(
                text=text,
                model_type=self.model_type,
                model_version=self.model_version,
                status=GenerationStatus.SUCCESS,
                latency_ms=latency_ms,
                cost_usd=0.0,
                metadata={"device": self.device, "backend": "vllm" if self.use_vllm else "hf"},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                text="",
                model_type=self.model_type,
                model_version=self.model_version,
                status=GenerationStatus.FAILURE,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def generate_batch(
        self, items: List[Dict[str, Any]], batch_size: int = 32, **kwargs
    ) -> List[GenerationResult]:
        """批量推理，支持连续批处理(vLLM).

        Args:
            items: 推理任务列表
            batch_size: vLLM批处理大小(默认32)
            **kwargs: 额外参数

        Returns:
            GenerationResult列表
        """
        if self.use_vllm and len(items) > 1:
            return await self._generate_batch_vllm(items, batch_size, **kwargs)
        else:
            return await self._generate_batch_sequential(items, **kwargs)

    async def _generate_batch_sequential(
        self, items: List[Dict[str, Any]], **kwargs
    ) -> List[GenerationResult]:
        """顺序批量推理(HF Transformers)."""
        results = []
        for item in items:
            result = await self.generate(
                frames=item.get("frames"),
                video_path=item.get("video_path"),
                prompt=item.get("prompt"),
                task=item.get("task", "fine_grained"),
                **kwargs,
            )
            results.append(result)
        return results

    async def _generate_batch_vllm(
        self, items: List[Dict[str, Any]], batch_size: int = 32, **kwargs
    ) -> List[GenerationResult]:
        """vLLM连续批处理推理.

        vLLM的连续批处理允许在单个批次处理过程中
        动态添加新请求，最大化GPU利用率。
        """
        results: List[GenerationResult] = []

        # 批量加载所有视频的帧
        all_frames = []
        all_prompts = []

        for item in items:
            frames = item.get("frames")
            video_path = item.get("video_path")
            prompt = item.get("prompt", "Describe the video.")

            if frames is None and video_path is not None:
                frames = load_frames_from_video(Path(video_path))

            if frames:
                all_frames.append(frames)
                all_prompts.append(prompt)

        if not all_frames:
            return [
                GenerationResult.failure(
                    error_message="No frames loaded for any item",
                    model_type=self.model_type,
                    model_version=self.model_version,
                )
            ] * len(items)

        # 使用vLLM进行批处理生成
        try:
            from vllm import SamplingParams

            max_new_tokens = kwargs.get("max_new_tokens", 512)
            temperature = kwargs.get("temperature", 0.2)

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.95,
            )

            # 准备输入
            vllm_inputs = []
            for prompt, frames in zip(all_prompts, all_frames):
                # 构建多模态输入 (根据vLLM版本可能有所不同)
                vllm_input = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": frames},
                }
                vllm_inputs.append(vllm_input)

            # 分批处理，每批batch_size个
            start_time = time.perf_counter()
            all_outputs = []

            for i in range(0, len(vllm_inputs), batch_size):
                batch = vllm_inputs[i : i + batch_size]
                outputs = self.model.generate(batch, sampling_params)
                all_outputs.extend(outputs)

            total_latency_ms = (time.perf_counter() - start_time) * 1000
            avg_latency_ms = total_latency_ms / len(items)

            # 构建结果
            for i, output in enumerate(all_outputs):
                results.append(
                    GenerationResult(
                        text=output.outputs[0].text,
                        model_type=self.model_type,
                        model_version=self.model_version,
                        status=GenerationStatus.SUCCESS,
                        latency_ms=avg_latency_ms,  # 连续批处理中近似平均延迟
                        cost_usd=0.0,
                        metadata={
                            "device": self.device,
                            "backend": "vllm",
                            "batch_size": len(vllm_inputs),
                            "prompt_tokens": len(output.prompt_token_ids),
                            "output_tokens": len(output.outputs[0].token_ids),
                        },
                    )
                )

        except Exception as e:
            logger.error("vLLM batch generation failed", error=str(e))
            # 回退到顺序处理
            return await self._generate_batch_sequential(items, **kwargs)

        return results

    def _generate_hf(
        self,
        messages: List[Dict],
        frames: List[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        num_beams: int = 1,
    ) -> str:
        """Generate with HF transformers."""
        import torch

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[frames],
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=temperature > 0,
            )

        # Decode
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract only the assistant response
        if "assistant" in output_text:
            output_text = output_text.split("assistant")[-1].strip()

        return output_text

    def _generate_vllm(
        self,
        messages: List[Dict],
        frames: List[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Generate with vLLM."""
        # vLLM handles prompt formatting differently
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        # Note: vLLM multi-modal support varies by version
        # This is a placeholder for the actual implementation
        output = self.model.generate(
            prompts=[messages[0]["content"][1]["text"]],
            sampling_params=sampling_params,
        )

        return output[0].outputs[0].text

    async def generate_stream(
        self,
        video_path: Path,
        prompt: str = "Describe the video in detail.",
        max_new_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Stream generation tokens."""
        # For streaming, we need to use text-generation-inference or similar
        # This is a placeholder implementation
        full_text = self.generate(
            video_path=video_path,
            prompt=prompt,
            task="fine_grained",
        )

        # Yield word by word for demonstration
        words = full_text.split()
        for word in words:
            yield word + " "


class StudentTeacherBridge(TeacherModel):
    """Adapter to use student model as teacher (for cost reduction).

    Supports training data version binding for reproducibility tracking.
    """

    def __init__(
        self,
        student_engine: StudentInferenceEngine,
        fallback_to_teacher: bool = True,
        confidence_threshold: float = 0.7,
        training_data_version: Optional[str] = None,
        model_registry_id: Optional[str] = None,
    ):
        super().__init__(model_name="student-bridge")
        self.student = student_engine
        self.fallback_to_teacher = fallback_to_teacher
        self.confidence_threshold = confidence_threshold
        self._teacher_fallback = None
        self.training_data_version = training_data_version
        self.model_registry_id = model_registry_id

    @property
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        return ModelType.STUDENT_EDGE

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        version_parts = [self.model_name]
        if self.model_registry_id:
            version_parts.append(self.model_registry_id)
        if self.training_data_version:
            version_parts.append(f"data-{self.training_data_version}")
        return "-".join(version_parts)

    def get_version_info(self) -> Dict[str, Optional[str]]:
        """Get version binding information.

        Returns:
            Dictionary with training data version and model registry ID
        """
        return {
            "model_name": self.model_name,
            "model_registry_id": self.model_registry_id,
            "training_data_version": self.training_data_version,
            "confidence_threshold": self.confidence_threshold,
        }

    def _load_fallback_teacher(self):
        """Load teacher model for fallback."""
        if self._teacher_fallback is None:
            from dvas.models.teacher import TeacherModel

            self._teacher_fallback = TeacherModel()

    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """Annotate using student model with fallback option.

        Includes version binding metadata in the result.
        """
        # Try student model
        try:
            result = await self.student.generate(
                frames=frames,
                video_path=video_path,
                prompt=prompt or f"Task: {task}",
                task=task,
            )

            if result.is_success() and result.confidence >= self.confidence_threshold:
                # Add version binding metadata
                result.metadata = {
                    **(result.metadata or {}),
                    "model_registry_id": self.model_registry_id,
                    "training_data_version": self.training_data_version,
                }
                return result

        except Exception as e:
            logger.error("Student inference failed", error=str(e))

        # Fallback to teacher if enabled
        if self.fallback_to_teacher:
            logger.info("Falling back to teacher model")
            self._load_fallback_teacher()
            teacher_result = await self._teacher_fallback.annotate(
                video_path=video_path, frames=frames, prompt=prompt, task=task, **kwargs
            )
            if teacher_result.is_success():
                return GenerationResult.fallback(
                    text=teacher_result.text,
                    fallback_from=self.model_type,
                    model_type=self._teacher_fallback.model_type,
                )
            return teacher_result

        return GenerationResult.failure(
            error_message="Student failed, fallback disabled",
            model_type=self.model_type,
            model_version=self.model_version,
        )

    async def annotate_batch(self, items: List[Dict[str, Any]], **kwargs) -> List[GenerationResult]:
        """Batch annotation with concurrency control."""
        tasks = [
            self.annotate(
                video_path=item.get("video_path"),
                frames=item.get("frames"),
                prompt=item.get("prompt"),
                task=item.get("task", "fine_grained"),
                **kwargs,
            )
            for item in items
        ]
        return await asyncio.gather(*tasks)

    async def generate(
        self,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """UnifiedModel.generate implementation - delegates to annotate."""
        return await self.annotate(
            video_path=video_path, frames=frames, prompt=prompt, task=task, **kwargs
        )

    async def generate_batch(self, items: List[Dict[str, Any]], **kwargs) -> List[GenerationResult]:
        """UnifiedModel.generate_batch implementation - delegates to annotate_batch."""
        return await self.annotate_batch(items, **kwargs)

    def _capabilities(self) -> List[str]:
        """Return list of supported capabilities."""
        return ["video", "frames", "text", "multimodal"]


def batch_inference(
    model_path: Path,
    video_paths: List[Path],
    prompts: Optional[List[str]] = None,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> List[GenerationResult]:
    """Run batch inference on multiple videos."""
    engine = StudentInferenceEngine(model_path)

    results: List[GenerationResult] = []
    prompts = prompts or ["Describe this video in detail."] * len(video_paths)

    for i in range(0, len(video_paths), batch_size):
        batch_paths = video_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        for video_path, prompt in zip(batch_paths, batch_prompts):
            try:
                result = engine.generate(
                    video_path=video_path,
                    prompt=prompt,
                    task="fine_grained",
                )
                # Note: generate is async, but batch_inference is sync
                # In practice, you'd use asyncio.run() or similar
                results.append(result)
            except Exception as e:
                results.append(
                    GenerationResult.failure(
                        error_message=str(e),
                        model_type=engine.model_type,
                        model_version=engine.model_version,
                    )
                )

    return results
