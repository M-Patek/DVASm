"""SFT (Supervised Fine-Tuning) training for Qwen2-VL."""

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from dvas.models.student.amp_utils import configure_amp_for_trainer, log_amp_status
from dvas.models.student.config import SFTConfig
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def _init_wandb(config: SFTConfig) -> None:
    """Initialize Weights & Biases logging if configured."""
    if config.report_to != "wandb":
        return

    try:
        import wandb

        wandb.init(
            project=config.wandb_project or "dvas",
            entity=config.wandb_entity,
            name=config.experiment_name,
            config={
                "model": {
                    "name": config.model.model_name_or_path,
                    "lora_r": config.model.lora_r,
                    "lora_alpha": config.model.lora_alpha,
                    "torch_dtype": config.model.torch_dtype,
                    "load_in_4bit": config.model.load_in_4bit,
                },
                "data": {
                    "train_data_path": str(config.data.train_data_path),
                    "batch_size": config.data.batch_size,
                    "max_seq_length": config.data.max_seq_length,
                    "num_frames": config.data.num_frames,
                },
                "training": {
                    "num_train_epochs": config.training.num_train_epochs,
                    "learning_rate": config.training.learning_rate,
                    "warmup_ratio": config.training.warmup_ratio,
                    "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                    "weight_decay": config.training.weight_decay,
                },
            },
        )
        logger.info("W&B initialized", project=config.wandb_project, name=config.experiment_name)
    except ImportError:
        logger.warning("wandb not installed, skipping experiment tracking")
    except Exception as e:
        logger.warning("W&B initialization failed", error=str(e))


def _log_checkpoint_to_wandb(checkpoint_path: Path, config: SFTConfig) -> None:
    """Log model checkpoint as W&B artifact."""
    if config.report_to != "wandb":
        return

    try:
        import wandb

        if wandb.run is None:
            return

        artifact = wandb.Artifact(
            name=f"{config.experiment_name}-checkpoint",
            type="model",
            description=f"SFT checkpoint for {config.experiment_name}",
        )
        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)
        logger.info("Checkpoint logged to W&B", path=str(checkpoint_path))
    except Exception as e:
        logger.warning("Failed to log checkpoint to W&B", error=str(e))


def load_model_and_processor(config: SFTConfig):
    """Load model and processor with appropriate settings."""
    model_cfg = config.model
    hardware_cfg = config.hardware

    # Setup quantization if enabled
    quantization_config = None
    if model_cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        )

    # Load processor (handles both text and images)
    logger.info("Loading processor", model_name=model_cfg.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # Load model
    logger.info("Loading model", model_name=model_cfg.model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        quantization_config=quantization_config,
        device_map=hardware_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=model_cfg.attn_implementation,
    )

    # Enable gradient checkpointing for memory efficiency
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare for LoRA if enabled
    if model_cfg.use_lora and model_cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def setup_lora(model, config: SFTConfig):
    """Setup LoRA adapters."""
    model_cfg = config.model

    logger.info(
        "Setting up LoRA",
        r=model_cfg.lora_r,
        alpha=model_cfg.lora_alpha,
    )

    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        target_modules=model_cfg.lora_target_modules,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train_sft(config: SFTConfig) -> Path:
    """Run SFT training.

    Returns:
        Path to the trained model checkpoint
    """
    logger.info("Starting SFT training", config=config.experiment_name)

    # Load model and processor
    model, processor = load_model_and_processor(config)

    # Setup LoRA
    if config.model.use_lora:
        model = setup_lora(model, config)

    # Training arguments
    train_cfg = config.training
    output_dir = train_cfg.output_dir / config.experiment_name

    # Configure AMP settings
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
        report_to=config.report_to,
        **amp_kwargs,
    )

    # Load dataset
    from datasets import load_dataset

    logger.info("Loading dataset", path=str(config.data.train_data_path))

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
    logger.info("Setting up SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text" if "text" in dataset.column_names else None,
        max_seq_length=config.data.max_seq_length,
    )

    # Initialize W&B before training
    _init_wandb(config)

    # Check for checkpoint to resume from
    from dvas.models.student.checkpoint_resume import (
        get_resume_kwargs,
    )

    resume_kwargs = get_resume_kwargs(config, output_dir)
    if resume_kwargs:
        logger.info(
            "Resuming from checkpoint", checkpoint=resume_kwargs.get("resume_from_checkpoint")
        )

    # Train
    logger.info("Starting training")
    if resume_kwargs:
        trainer.train(resume_from_checkpoint=resume_kwargs["resume_from_checkpoint"])
    else:
        trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    # Log checkpoint to W&B
    _log_checkpoint_to_wandb(final_path, config)

    # Finish W&B run
    if config.report_to == "wandb":
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    logger.info("Training completed", model_path=str(final_path))

    return final_path


def main():
    """CLI entry point for SFT training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train student model with SFT")
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
        default=Path("outputs/student_sft"),
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

    args = parser.parse_args()

    # Build config
    config = SFTConfig()
    config.data.train_data_path = args.train_data
    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.model.use_lora = args.use_lora

    # Run training
    final_path = train_sft(config)
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
