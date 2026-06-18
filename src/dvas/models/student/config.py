"""Training configuration for student model (Qwen2-VL)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    attn_implementation: str = "flash_attention_2"

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

    # vLLM settings for inference
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9


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
    wandb_project: Optional[str] = None
    report_to: str = "none"  # none, wandb, tensorboard


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
    wandb_project: Optional[str] = None
    report_to: str = "none"
