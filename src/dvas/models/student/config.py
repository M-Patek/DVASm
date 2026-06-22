"""Training configuration for student model (Qwen2-VL)."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_data_path: Path
    eval_data_path: Optional[Path] = None
    max_seq_length: int = 2048
    image_resolution: int = 448
    max_num_frames: int = 16
    batch_size: int = 1
    num_workers: int = 4

    # Video sampling
    frame_sampling_strategy: str = "uniform"  # uniform, random, keyframe
    num_frames: int = 16


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = "flash_attention_2"

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def validate(self) -> "ModelConfig":
        """验证配置并检查Flash Attention可用性."""
        if self.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                warnings.warn(
                    "flash-attention not installed. Falling back to sdpa. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )
                self.attn_implementation = "sdpa"
        return self


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Basic settings
    output_dir: Path = Path("outputs/student_model")
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Gradient settings
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Gradient scaling for mixed precision stability
    grad_scaler_enabled: bool = True
    grad_scaler_init_scale: float = 2**16
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000

    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True

    # DPO settings (for preference optimization)
    beta: float = 0.1  # KL divergence coefficient
    label_smoothing: float = 0.0

    # Generation settings for evaluation
    max_new_tokens: int = 512
    temperature: float = 0.2


@dataclass
class HardwareConfig:
    """Hardware/acceleration configuration."""

    device: str = "auto"  # auto, cuda, cpu
    device_map: str = "auto"
    num_gpus: int = 1

    # vLLM settings for inference (Phase 4优化: 默认启用vLLM)
    use_vllm: bool = True  # 默认启用vLLM
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 8192
    vllm_max_num_seqs: int = 256  # 连续批处理最大序列数


@dataclass
class SFTConfig:
    """Complete SFT training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            train_data_path=Path("data/exports/train_llava.jsonl"),
        )
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Experiment tracking
    experiment_name: str = "dvas_student_sft"
    wandb_project: Optional[str] = "dvas"
    wandb_entity: Optional[str] = None
    report_to: str = "wandb"  # none, wandb, tensorboard


@dataclass
class DPOConfig:
    """Complete DPO training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            train_data_path=Path("data/exports/dpo_pairs.jsonl"),
        )
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Reference model (starting from SFT checkpoint)
    ref_model_path: Optional[Path] = None

    experiment_name: str = "dvas_student_dpo"
    wandb_project: Optional[str] = "dvas"
    wandb_entity: Optional[str] = None
    report_to: str = "wandb"
