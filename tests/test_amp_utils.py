"""Tests for mixed precision (AMP) utilities."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock datasets before importing student modules to avoid pyarrow segfault
sys.modules["datasets"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.amp"] = MagicMock()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.amp_utils import (
    configure_amp_for_trainer,
    get_amp_dtype,
    get_grad_scaler,
    get_training_dtype,
)
from dvas.models.student.config import SFTConfig


class TestGetAmpDtype:
    """Test get_amp_dtype function."""

    @patch("dvas.models.student.amp_utils.torch")
    def test_bf16_returns_bfloat16(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        config = SFTConfig()
        config.training.bf16 = True
        config.training.fp16 = False

        dtype = get_amp_dtype(config)

        assert dtype == "bfloat16"

    @patch("dvas.models.student.amp_utils.torch")
    def test_bf16_unsupported_falls_back(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.bf16 = True
        config.training.fp16 = False

        dtype = get_amp_dtype(config)

        assert dtype == "float32"

    @patch("dvas.models.student.amp_utils.torch")
    def test_fp16_returns_float16(self, mock_torch):
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.bf16 = False
        config.training.fp16 = True

        dtype = get_amp_dtype(config)

        assert dtype == "float16"

    @patch("dvas.models.student.amp_utils.torch")
    def test_neither_returns_float32(self, mock_torch):
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.bf16 = False
        config.training.fp16 = False

        dtype = get_amp_dtype(config)

        assert dtype == "float32"


class TestGetGradScaler:
    """Test get_grad_scaler function."""

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_none_when_disabled(self, mock_torch):
        config = SFTConfig()
        config.training.grad_scaler_enabled = False
        config.training.fp16 = True

        scaler = get_grad_scaler(config)

        assert scaler is None

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_none_for_bf16(self, mock_torch):
        config = SFTConfig()
        config.training.bf16 = True
        config.training.fp16 = False
        config.training.grad_scaler_enabled = True

        scaler = get_grad_scaler(config)

        assert scaler is None

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_none_for_fp32(self, mock_torch):
        config = SFTConfig()
        config.training.bf16 = False
        config.training.fp16 = False
        config.training.grad_scaler_enabled = True

        scaler = get_grad_scaler(config)

        assert scaler is None

    @patch("dvas.models.student.amp_utils.GradScaler")
    def test_returns_scaler_for_fp16(self, mock_grad_scaler_class):
        mock_scaler = MagicMock()
        mock_grad_scaler_class.return_value = mock_scaler

        config = SFTConfig()
        config.training.bf16 = False
        config.training.fp16 = True
        config.training.grad_scaler_enabled = True
        config.training.grad_scaler_init_scale = 2**16
        config.training.grad_scaler_growth_factor = 2.0
        config.training.grad_scaler_backoff_factor = 0.5
        config.training.grad_scaler_growth_interval = 2000

        scaler = get_grad_scaler(config)

        assert scaler is mock_scaler
        mock_grad_scaler_class.assert_called_once_with(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True,
        )


class TestGetTrainingDtype:
    """Test get_training_dtype function."""

    @patch("dvas.models.student.amp_utils.torch")
    def test_bfloat16_config(self, mock_torch):
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        config = SFTConfig()
        config.model.torch_dtype = "bfloat16"

        dtype = get_training_dtype(config)

        assert dtype == "bfloat16"

    @patch("dvas.models.student.amp_utils.torch")
    def test_float16_config(self, mock_torch):
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.model.torch_dtype = "float16"

        dtype = get_training_dtype(config)

        assert dtype == "float16"

    @patch("dvas.models.student.amp_utils.torch")
    def test_float32_config(self, mock_torch):
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.model.torch_dtype = "float32"

        dtype = get_training_dtype(config)

        assert dtype == "float32"

    @patch("dvas.models.student.amp_utils.torch")
    def test_unknown_dtype_defaults_to_float32(self, mock_torch):
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.model.torch_dtype = "unknown_dtype"

        dtype = get_training_dtype(config)

        assert dtype == "float32"


class TestConfigureAmpForTrainer:
    """Test configure_amp_for_trainer function."""

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_empty_when_amp_disabled(self, mock_torch):
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.fp16 = False
        config.training.bf16 = False

        result = configure_amp_for_trainer(config)

        assert result == {}

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_fp16_config(self, mock_torch):
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.fp16 = True
        config.training.bf16 = False

        result = configure_amp_for_trainer(config)

        assert result["fp16"] is True
        assert result["bf16"] is False

    @patch("dvas.models.student.amp_utils.torch")
    def test_returns_bf16_config(self, mock_torch):
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        config = SFTConfig()
        config.training.fp16 = False
        config.training.bf16 = True

        result = configure_amp_for_trainer(config)

        assert result["fp16"] is False
        assert result["bf16"] is True
