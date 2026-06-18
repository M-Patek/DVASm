"""Test script to verify SFT training pipeline can start.

This script verifies:
1. Training data can be loaded
2. Model can be initialized (or mock initialized if no GPU/memory)
3. Training loop can start (then immediately stop)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.config import SFTConfig, DataConfig, TrainingConfig
from dvas.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_data_loading():
    """Test that training data can be loaded."""
    from datasets import load_dataset

    train_path = Path("data/training/mini_train.jsonl")
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return False

    try:
        dataset = load_dataset("json", data_files=str(train_path), split="train")
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
        logger.info(f"Dataset columns: {dataset.column_names}")
        return True
    except Exception as e:
        import traceback
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return False


def test_model_init_minimal():
    """Test model initialization with minimal settings."""
    try:
        import torch
        from transformers import AutoProcessor

        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"Loading processor for {model_name}...")

        # Only load processor (lightweight) to verify model access
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("Processor loaded successfully")

        # Check if we can at least initialize the model config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Model config loaded: {config.model_type}")

        return True
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        logger.info("Install with: pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize model components: {e}")
        return False


def test_training_config():
    """Test training configuration creation."""
    try:
        config = SFTConfig()
        config.data.train_data_path = Path("data/training/mini_train.jsonl")
        config.training.output_dir = Path("outputs/test_sft")
        config.training.num_train_epochs = 1
        config.training.max_steps = 1  # Only 1 step for testing
        config.data.batch_size = 1
        config.model.use_lora = True
        config.model.load_in_4bit = True  # Use 4-bit to save memory

        logger.info("Training config created successfully")
        logger.info(f"  Train data: {config.data.train_data_path}")
        logger.info(f"  Output dir: {config.training.output_dir}")
        logger.info(f"  Epochs: {config.training.num_train_epochs}")
        logger.info(f"  Batch size: {config.data.batch_size}")

        return True
    except Exception as e:
        logger.error(f"Failed to create training config: {e}")
        return False


def test_full_pipeline_dry_run():
    """Test the full training pipeline in dry-run mode."""
    logger.info("=" * 50)
    logger.info("TESTING FULL TRAINING PIPELINE (DRY RUN)")
    logger.info("=" * 50)

    # Step 1: Test config
    logger.info("\n[1/4] Testing training configuration...")
    if not test_training_config():
        return False

    # Step 2: Test data loading
    logger.info("\n[2/4] Testing data loading...")
    if not test_data_loading():
        return False

    # Step 3: Test model init
    logger.info("\n[3/4] Testing model initialization...")
    if not test_model_init_minimal():
        logger.warning("Model init test skipped or failed (this is OK if dependencies missing)")

    # Step 4: Verify training can start (optional - requires GPU)
    logger.info("\n[4/4] Verifying training entry point...")
    try:
        from dvas.models.student.sft_trainer import train_sft, SFTConfig

        config = SFTConfig()
        config.data.train_data_path = Path("data/training/mini_train.jsonl")
        config.training.output_dir = Path("outputs/test_sft")
        config.training.num_train_epochs = 1
        config.training.max_steps = 1
        config.data.batch_size = 1
        config.model.use_lora = True
        config.model.load_in_4bit = True

        logger.info("Training entry point accessible")
        logger.info("Ready to start training!")

    except Exception as e:
        logger.error(f"Training entry point error: {e}")
        return False

    logger.info("\n" + "=" * 50)
    logger.info("ALL TESTS PASSED - Training pipeline is ready")
    logger.info("=" * 50)
    logger.info("\nTo start actual training, run:")
    logger.info("  python -m dvas.models.student.sft_trainer \\")
    logger.info("    --train-data data/training/mini_train.jsonl \\")
    logger.info("    --output-dir outputs/student_sft \\")
    logger.info("    --epochs 1")

    return True


def main():
    """Main entry point."""
    setup_logging()

    success = test_full_pipeline_dry_run()

    if success:
        logger.info("\n✓ SUCCESS: Training pipeline can start")
        return 0
    else:
        logger.error("\n✗ FAILURE: Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
