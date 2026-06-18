"""Dry-run training test without actual model loading.

This verifies the training code structure is correct by mocking the heavy components.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.config import SFTConfig
from dvas.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_training_code_structure():
    """Test training code structure without loading heavy models."""
    logger.info("Testing training code structure...")

    # Mock all heavy imports
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.bfloat16 = "bfloat16"

    _mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()

    with patch.dict('sys.modules', {
        'torch': mock_torch,
        'transformers': MagicMock(),
        'peft': MagicMock(),
        'trl': MagicMock(),
        'datasets': MagicMock(),
        'bitsandbytes': MagicMock(),
    }):
        # Test 1: Config creation
        logger.info("[1/3] Testing config creation...")
        config = SFTConfig()
        config.data.train_data_path = Path("data/training/mini_train.jsonl")
        config.training.output_dir = Path("outputs/dry_run")
        config.training.num_train_epochs = 1
        config.training.max_steps = 1
        logger.info("  ✓ Config created")

        # Test 2: Import training function
        logger.info("[2/3] Testing module imports...")
        try:
            from dvas.models.student.sft_trainer import (
                load_model_and_processor,
                train_sft,
            )
            logger.info("  ✓ Training module imports successfully")
        except ImportError as e:
            logger.error(f"  ✗ Import failed: {e}")
            return False

        # Test 3: Verify function signatures
        logger.info("[3/3] Verifying function signatures...")
        import inspect

        # Check train_sft signature
        sig = inspect.signature(train_sft)
        params = list(sig.parameters.keys())
        assert 'config' in params, "train_sft must accept 'config' parameter"
        logger.info("  ✓ train_sft signature correct")

        # Check load_model_and_processor signature
        sig = inspect.signature(load_model_and_processor)
        params = list(sig.parameters.keys())
        assert 'config' in params, "load_model_and_processor must accept 'config' parameter"
        logger.info("  ✓ load_model_and_processor signature correct")

    logger.info("\n" + "=" * 50)
    logger.info("CODE STRUCTURE TEST PASSED")
    logger.info("=" * 50)
    logger.info("\nTraining code structure is correct.")
    logger.info("Ready for actual training on GPU machine.")

    return True


def test_data_pipeline():
    """Test that data pipeline is ready."""
    logger.info("\nTesting data pipeline...")

    # Check files exist
    train_path = Path("data/training/mini_train.jsonl")
    if not train_path.exists():
        logger.error(f"  ✗ Training data not found: {train_path}")
        return False
    logger.info(f"  ✓ Training data exists: {train_path}")

    # Check data format
    try:
        import json
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line = json.loads(f.readline())

        required_keys = ['id', 'video', 'conversations']
        for key in required_keys:
            if key not in first_line:
                logger.error(f"  ✗ Missing key: {key}")
                return False
        logger.info(f"  ✓ Data format correct (keys: {list(first_line.keys())})")

        # Count lines
        with open(train_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        logger.info(f"  ✓ Training samples: {count}")

    except Exception as e:
        logger.error(f"  ✗ Data validation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    setup_logging()

    logger.info("=" * 50)
    logger.info("DRY-RUN TRAINING TEST")
    logger.info("=" * 50)

    success = True

    if not test_training_code_structure():
        success = False

    if not test_data_pipeline():
        success = False

    if success:
        logger.info("\n" + "=" * 50)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 50)
        logger.info("\nTraining pipeline is ready!")
        logger.info("\nTo run actual training:")
        logger.info("  1. Ensure you have a GPU with CUDA")
        logger.info("  2. Install dependencies: pip install transformers peft trl bitsandbytes")
        logger.info("  3. Run: python -m dvas.models.student.sft_trainer \\")
        logger.info("       --train-data data/training/mini_train.jsonl \\")
        logger.info("       --output-dir outputs/student_sft \\")
        logger.info("       --epochs 1")
        return 0
    else:
        logger.error("\n" + "=" * 50)
        logger.error("SOME TESTS FAILED")
        logger.error("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
