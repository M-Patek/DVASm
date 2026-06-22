"""Unsloth-accelerated training for 2-5x speedup.

This module provides an alternative training implementation using Unsloth,
which optimizes the training loop for significantly faster iteration.

Usage::

    from dvas.models.student.unsloth_trainer import UnslothSFTTrainer
    from dvas.models.student.config import SFTConfig

    config = SFTConfig()
    trainer = UnslothSFTTrainer(config)
    trainer.train(dataset, output_dir="outputs/unsloth")

Requirements:
    pip install unsloth[vision]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from dvas.models.student.config import SFTConfig

# 可选依赖检测
try:
    import torch
    from unsloth import FastVisionModel
    from unsloth import UnslothTrainer as Trainer
    from unsloth import UnslothTrainingArguments

    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False
    FastVisionModel = None  # type: ignore
    Trainer = None  # type: ignore
    UnslothTrainingArguments = None  # type: ignore


class UnslothSFTTrainer:
    """SFT训练器使用Unsloth加速.

    Unsloth通过以下优化实现2-5x加速:
    - 手动推导的反向传播(减少浮点运算)
    - 优化的LoRA实现
    - 减少显存转置
    - RoPE嵌入优化

    Attributes:
        config: 训练配置
        model: Unsloth优化模型
        tokenizer: 模型tokenizer
    """

    def __init__(self, config: "SFTConfig"):
        """初始化Unsloth训练器.

        Args:
            config: SFT训练配置

        Raises:
            ImportError: 如果Unsloth未安装
        """
        self.config = config
        self._verify_unsloth_available()

        self.model = None
        self.tokenizer = None
        self._trainer = None

    def _verify_unsloth_available(self) -> None:
        """验证Unsloth是否可用."""
        if not _UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth not available. Install with: "
                "pip install 'unsloth[vision]>=2024.1.0'"
            )

    def load_model(
        self,
        model_name: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
    ) -> tuple[Any, Any]:
        """加载Unsloth优化模型.

        Args:
            model_name: 模型名称，默认使用配置中的路径
            load_in_4bit: 是否使用4-bit量化

        Returns:
            (model, tokenizer) 元组
        """
        model_name = model_name or self.config.model.model_name_or_path
        load_in_4bit = (
            load_in_4bit
            if load_in_4bit is not None
            else self.config.model.load_in_4bit
        )

        logger.info(
            "Loading model with Unsloth acceleration",
            model=model_name,
            load_in_4bit=load_in_4bit,
        )

        # 加载模型和tokenizer
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=True,
        )

        # 配置LoRA
        if self.config.model.use_lora:
            model = FastVisionModel.get_peft_model(
                model,
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=self.config.model.lora_target_modules,
                lora_dropout=self.config.model.lora_dropout,
                use_rslora=False,  # 可选: 使用Rank-Stabilized LoRA
            )

            # 打印可训练参数
            model.print_trainable_parameters()

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def train(
        self,
        dataset: Any,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """执行加速训练.

        Args:
            dataset: 训练数据集
            output_dir: 输出目录
            **kwargs: 额外参数覆盖配置

        Returns:
            训练结果对象
        """
        if self.model is None:
            self.load_model()

        output_dir = output_dir or str(self.config.training.output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Unsloth训练参数
        training_args = UnslothTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.data.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps if self.config.training.max_steps > 0 else None,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            # 序列长度优化
            max_seq_length=self.config.data.max_seq_length,
            # 优化器设置
            optim="adamw_8bit",  # 8-bit优化器减少显存
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            # 日志和保存
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            report_to=self.config.report_to if self.config.report_to != "none" else [],
            run_name=self.config.experiment_name,
            # 精度设置
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            # 其他参数
            group_by_length=False,  # Unsloth优化建议
            seed=42,
            **kwargs,
        )

        # 创建trainer
        self._trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        logger.info("Starting Unsloth training", output_dir=output_dir)

        # 训练
        result = self._trainer.train()

        # 内存统计
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_reserved() / 2**30
            gpu_stats = torch.cuda.get_device_properties(0)
            logger.info(
                "Training completed",
                peak_memory_gb=f"{peak_memory_gb:.2f}",
                gpu_name=gpu_stats.name,
            )

        return result

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """保存模型和tokenizer.

        Args:
            output_dir: 输出目录，默认使用配置中的路径
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_dir = output_dir or str(self.config.training.output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Saving model", output_dir=output_dir)

        # Unsloth优化保存
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

    def get_stats(self) -> dict[str, Any]:
        """获取训练统计信息.

        Returns:
            包含训练统计的字典
        """
        stats = {
            "unsloth_available": _UNSLOTH_AVAILABLE,
            "model_loaded": self.model is not None,
        }

        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 2**30

        return stats


def is_unsloth_available() -> bool:
    """检查Unsloth是否可用.

    Returns:
        True if Unsloth is installed and usable
    """
    return _UNSLOTH_AVAILABLE


def create_trainer(
    config: "SFTConfig",
    use_unsloth: bool = True,
) -> Any:
    """工厂函数创建训练器.

    自动选择Unsloth或标准Transformers训练器.

    Args:
        config: 训练配置
        use_unsloth: 是否尝试使用Unsloth

    Returns:
        训练器实例
    """
    if use_unsloth and _UNSLOTH_AVAILABLE:
        logger.info("Using Unsloth accelerated trainer")
        return UnslothSFTTrainer(config)

    # Fallback到标准训练器
    logger.info("Using standard SFT trainer")
    from dvas.models.student.sft_trainer import SFTTrainer

    return SFTTrainer(config)
