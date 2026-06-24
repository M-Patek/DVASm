"""FSDP training launcher for multi-GPU distributed training.

Provides utilities to launch training across multiple GPUs using
torchrun / torch.distributed.launch. Handles process group setup,
model wrapping, and checkpoint saving/loading for FSDP.

Usage::

    # Single node, 8 GPUs
    python -m torchrun --nproc_per_node=8 \
        -m dvas.models.student.fsdp_launcher \
        --config config.yaml

    # Multi-node (2 nodes x 8 GPUs = 16 GPUs)
    # Node 0:
    python -m torchrun --nproc_per_node=8 \
        --nnodes=2 --node_rank=0 \
        --master_addr=192.168.1.1 --master_port=29500 \
        -m dvas.models.student.fsdp_launcher \
        --config config.yaml

    # Node 1:
    python -m torchrun --nproc_per_node=8 \
        --nnodes=2 --node_rank=1 \
        --master_addr=192.168.1.1 --master_port=29500 \
        -m dvas.models.student.fsdp_launcher \
        --config config.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


from dvas.models.student.config import SFTConfig
from dvas.models.student.fsdp_utils import (
    DistributedConfig,
    FSDPConfig,
    cleanup_distributed,
    save_fsdp_checkpoint,
    setup_distributed,
    wrap_model_with_fsdp,
)
from dvas.models.student.sft_trainer import (
    _init_wandb,
    _log_checkpoint_to_wandb,
    load_model_and_processor,
    setup_lora,
)
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def _get_rank() -> int:
    """Get current process rank."""
    return int(os.environ.get("RANK", 0))


def _is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return _get_rank() == 0


def _log_on_main(msg: str, **kwargs) -> None:
    """Log only on main process to avoid duplicate messages."""
    if _is_main_process():
        logger.info(msg, **kwargs)


def _setup_for_distributed_training(config: SFTConfig) -> tuple[SFTConfig, DistributedConfig]:
    """Configure SFTConfig for distributed training.

    Args:
        config: Base SFT configuration

    Returns:
        Tuple of (modified config, distributed config)
    """
    dist_config = DistributedConfig()

    if not dist_config.is_distributed:
        _log_on_main("Running in single-process mode")
        return config, dist_config

    # Adjust batch size for distributed training
    # per_device_batch_size stays the same, total batch = per_device * world_size
    world_size = dist_config.world_size
    _log_on_main(
        "Distributed training configured",
        world_size=world_size,
        per_device_batch_size=config.data.batch_size,
        effective_batch_size=config.data.batch_size
        * world_size
        * config.training.gradient_accumulation_steps,
    )

    # Adjust device map for FSDP (FSDP manages device placement)
    if config.hardware.device_map == "auto":
        config.hardware.device_map = None  # FSDP will handle this

    return config, dist_config


def train_fsdp(config: SFTConfig, fsdp_config: Optional[FSDPConfig] = None) -> Path:
    """Run SFT training with FSDP distributed support.

    Args:
        config: SFT training configuration
        fsdp_config: FSDP configuration (uses defaults if None)

    Returns:
        Path to the trained model checkpoint
    """
    if fsdp_config is None:
        fsdp_config = FSDPConfig()

    # Setup distributed training
    config, dist_config = _setup_for_distributed_training(config)
    setup_distributed(dist_config)

    try:
        _log_on_main("Starting FSDP training", experiment=config.experiment_name)

        # Load model and processor
        model, processor = load_model_and_processor(config)

        # Setup LoRA
        if config.model.use_lora:
            model = setup_lora(model, config)

        # Wrap model with FSDP
        if fsdp_config.enabled and dist_config.is_distributed:
            model = wrap_model_with_fsdp(model, fsdp_config, dist_config)
            _log_on_main(
                "Model wrapped with FSDP",
                sharding=fsdp_config.sharding_strategy,
                world_size=dist_config.world_size,
            )

        # Build training arguments
        from transformers import TrainingArguments

        train_cfg = config.training
        output_dir = train_cfg.output_dir / config.experiment_name

        # Import AMP utilities
        from dvas.models.student.amp_utils import configure_amp_for_trainer, log_amp_status

        log_amp_status(config)
        amp_kwargs = configure_amp_for_trainer(config)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=train_cfg.num_train_epochs,
            max_steps=train_cfg.max_steps if train_cfg.max_steps > 0 else None,
            per_device_train_batch_size=config.data.batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.learning_rate,
            warmup_ratio=train_cfg.warmup_ratio,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            weight_decay=train_cfg.weight_decay,
            max_grad_norm=train_cfg.max_grad_norm,
            logging_steps=train_cfg.logging_steps,
            save_steps=train_cfg.save_steps,
            eval_steps=train_cfg.eval_steps,
            save_total_limit=train_cfg.save_total_limit,
            remove_unused_columns=False,
            dataloader_num_workers=config.data.num_workers,
            report_to=config.report_to if _is_main_process() else "none",
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=25,
            **amp_kwargs,
        )

        # Load dataset
        from datasets import load_dataset

        _log_on_main("Loading dataset", path=str(config.data.train_data_path))

        if config.data.train_data_path.exists():
            dataset = load_dataset(
                "json",
                data_files=str(config.data.train_data_path),
                split="train",
            )
        else:
            logger.error("Dataset not found", path=str(config.data.train_data_path))
            raise FileNotFoundError(f"Dataset not found: {config.data.train_data_path}")

        # Setup trainer
        from trl import SFTTrainer

        _log_on_main("Setting up SFTTrainer with FSDP")
        trainer = SFTTrainer(
            model=model,
            tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text" if "text" in dataset.column_names else None,
            max_seq_length=config.data.max_seq_length,
        )

        # Initialize W&B (only on main process)
        if _is_main_process():
            _init_wandb(config)

        # Check for checkpoint to resume from
        from dvas.models.student.checkpoint_resume import get_resume_kwargs

        resume_kwargs = get_resume_kwargs(config, output_dir)
        if resume_kwargs and _is_main_process():
            logger.info(
                "Resuming from checkpoint",
                checkpoint=resume_kwargs.get("resume_from_checkpoint"),
            )

        # Train
        _log_on_main("Starting training")
        if resume_kwargs:
            trainer.train(resume_from_checkpoint=resume_kwargs["resume_from_checkpoint"])
        else:
            trainer.train()

        # Save final model
        final_path = output_dir / "final"

        if fsdp_config.enabled and dist_config.is_distributed:
            # Save FSDP checkpoint
            save_fsdp_checkpoint(
                model=trainer.model,
                output_dir=str(final_path),
                fsdp_config=fsdp_config,
                optimizer=trainer.optimizer,
            )
        else:
            # Standard save
            if _is_main_process():
                trainer.save_model(str(final_path))
                processor.save_pretrained(str(final_path))

        # Log checkpoint to W&B (only on main process)
        if _is_main_process():
            _log_checkpoint_to_wandb(final_path, config)

            # Finish W&B run
            if config.report_to == "wandb":
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.finish()
                except Exception:
                    pass

        _log_on_main("Training completed", model_path=str(final_path))

        return final_path

    finally:
        cleanup_distributed()


def main():
    """CLI entry point for FSDP training launcher."""
    parser = argparse.ArgumentParser(description="Launch FSDP distributed training")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/student_fsdp"),
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP wrapping",
    )
    parser.add_argument(
        "--sharding-strategy",
        type=str,
        default="full",
        choices=["full", "shard_grad_op", "no_shard"],
        help="FSDP sharding strategy",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        default=False,
        help="Offload parameters to CPU (saves GPU memory)",
    )
    parser.add_argument(
        "--state-dict-type",
        type=str,
        default="full",
        choices=["full", "sharded", "local"],
        help="Checkpoint state dict type",
    )

    args = parser.parse_args()

    # Build config
    config = SFTConfig()
    config.data.train_data_path = args.train_data
    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.model.use_lora = args.use_lora

    # Build FSDP config
    fsdp_config = FSDPConfig(
        enabled=args.fsdp,
        sharding_strategy=args.sharding_strategy,
        cpu_offload=args.cpu_offload,
        state_dict_type=args.state_dict_type,
    )

    # Run training
    final_path = train_fsdp(config, fsdp_config)

    if _is_main_process():
        print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
