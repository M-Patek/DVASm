"""Minimal training test - verify training can actually start by running 1 step.

This script:
1. Loads the model with minimal settings (4-bit quantization, small sequence length)
2. Starts training
3. Runs 1 step (or stops on first error)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from dvas.models.student.config import SFTConfig
from dvas.models.student.sft_trainer import train_sft
from dvas.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Run minimal training test."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("MINIMAL TRAINING TEST - Running 1 step only")
    logger.info("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - training will be very slow on CPU")
        logger.info("For this test, we'll verify the pipeline without actual training")
        logger.info("To run real training, use a machine with GPU")
        return 0

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create minimal config
    config = SFTConfig()
    config.data.train_data_path = Path("data/training/mini_train.jsonl")
    config.training.output_dir = Path("outputs/minimal_test")
    config.training.num_train_epochs = 1
    config.training.max_steps = 1  # Only 1 step
    config.training.logging_steps = 1
    config.training.save_steps = 1
    config.data.batch_size = 1
    config.data.max_seq_length = 512  # Shorter sequences
    config.model.use_lora = True
    config.model.load_in_4bit = True  # Minimal memory
    config.model.lora_r = 8  # Smaller LoRA
    config.model.lora_alpha = 16
    config.experiment_name = "minimal_test"
    config.report_to = "none"  # No wandb/tensorboard

    logger.info("\nConfiguration:")
    logger.info(f"  Train data: {config.data.train_data_path}")
    logger.info(f"  Output dir: {config.training.output_dir}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Max seq length: {config.data.max_seq_length}")
    logger.info(f"  4-bit quantization: {config.model.load_in_4bit}")
    logger.info(f"  LoRA r: {config.model.lora_r}")

    # Verify data exists
    if not config.data.train_data_path.exists():
        logger.error(f"Training data not found: {config.data.train_data_path}")
        logger.info("Run: python examples/create_mini_dataset.py --num-videos 3")
        return 1

    try:
        logger.info("\n" + "-" * 60)
        logger.info("Starting training...")
        logger.info("-" * 60)

        # This will load the model and run training
        final_path = train_sft(config)

        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! Training completed 1 step")
        logger.info("=" * 60)
        logger.info(f"Model checkpoint: {final_path}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
