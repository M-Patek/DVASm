"""DPO (Direct Preference Optimization) training."""

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import DPOTrainer

from dvas.models.student.amp_utils import configure_amp_for_trainer, log_amp_status
from dvas.models.student.config import DPOConfig
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def _init_wandb(config: DPOConfig) -> None:
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
                },
                "data": {
                    "train_data_path": str(config.data.train_data_path),
                    "batch_size": config.data.batch_size,
                },
                "training": {
                    "num_train_epochs": config.training.num_train_epochs,
                    "learning_rate": config.training.learning_rate,
                    "beta": config.training.beta,
                },
            },
        )
        logger.info("W&B initialized", project=config.wandb_project, name=config.experiment_name)
    except ImportError:
        logger.warning("wandb not installed, skipping experiment tracking")
    except Exception as e:
        logger.warning("W&B initialization failed", error=str(e))


def _log_checkpoint_to_wandb(checkpoint_path: Path, config: DPOConfig) -> None:
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
            description=f"DPO checkpoint for {config.experiment_name}",
        )
        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)
        logger.info("Checkpoint logged to W&B", path=str(checkpoint_path))
    except Exception as e:
        logger.warning("Failed to log checkpoint to W&B", error=str(e))


def load_model_for_dpo(config: DPOConfig, is_ref: bool = False):
    """Load model for DPO training."""
    model_cfg = config.model

    # Setup quantization
    quantization_config = None
    if model_cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_cfg.bnb_4bit_compute_dtype,
        )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=model_cfg.torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # If reference model, don't apply LoRA
    if is_ref:
        return model, processor

    # Apply LoRA
    if model_cfg.use_lora:
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            target_modules=model_cfg.lora_target_modules,
            lora_dropout=model_cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, processor


def train_dpo(config: DPOConfig) -> Path:
    """Run DPO training."""
    logger.info("Starting DPO training", config=config.experiment_name)

    # Load policy model
    policy_model, processor = load_model_for_dpo(config, is_ref=False)

    # Load reference model (frozen)
    if config.ref_model_path:
        logger.info("Loading reference model", path=str(config.ref_model_path))
        ref_model, _ = load_model_for_dpo(config, is_ref=True)
        # Load SFT checkpoint
        ref_model = PeftModel.from_pretrained(ref_model, str(config.ref_model_path))
    else:
        logger.info("Using base model as reference")
        ref_model, _ = load_model_for_dpo(config, is_ref=True)

    # Load dataset
    logger.info("Loading DPO dataset", path=str(config.data.train_data_path))

    dataset = load_dataset(
        "json",
        data_files=str(config.data.train_data_path),
        split="train",
    )

    # Training arguments
    from transformers import TrainingArguments

    train_cfg = config.training
    output_dir = train_cfg.output_dir / config.experiment_name

    # Configure AMP settings
    log_amp_status(config)
    amp_kwargs = configure_amp_for_trainer(config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=config.data.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        report_to=train_cfg.report_to,
        **amp_kwargs,
    )

    # Setup DPO trainer
    logger.info("Setting up DPOTrainer")
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        beta=train_cfg.beta,
        max_length=config.data.max_seq_length,
        max_prompt_length=config.data.max_seq_length // 2,
    )

    # Initialize W&B before training
    _init_wandb(config)

    # Check for checkpoint to resume from
    from dvas.models.student.checkpoint_resume import get_resume_kwargs

    resume_kwargs = get_resume_kwargs(config, output_dir)
    if resume_kwargs:
        logger.info(
            "Resuming from checkpoint", checkpoint=resume_kwargs.get("resume_from_checkpoint")
        )

    # Train
    logger.info("Starting DPO training")
    if resume_kwargs:
        trainer.train(resume_from_checkpoint=resume_kwargs["resume_from_checkpoint"])
    else:
        trainer.train()

    # Save
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))

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

    logger.info("DPO training completed", model_path=str(final_path))

    return final_path


def main():
    """CLI entry point for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train student model with DPO")
    parser.add_argument(
        "--sft-model",
        type=Path,
        required=True,
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--dpo-data",
        type=Path,
        required=True,
        help="Path to DPO preference pairs JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/student_dpo"),
        help="Output directory",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL divergence coefficient",
    )

    args = parser.parse_args()

    # Build config
    config = DPOConfig()
    config.ref_model_path = args.sft_model
    config.data.train_data_path = args.dpo_data
    config.training.output_dir = args.output_dir
    config.training.beta = args.beta

    # Run training
    final_path = train_dpo(config)
    print(f"DPO model saved to: {final_path}")


if __name__ == "__main__":
    main()
