"""Tests for W&B integration."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dvas.models.student.config import SFTConfig
from dvas.models.student.wandb_tracker import WandBTracker, init_wandb_for_transformers


class TestWandBTracker:
    """Test WandBTracker functionality."""

    def test_enabled_when_report_to_wandb(self):
        config = MagicMock()
        config.report_to = "wandb"
        tracker = WandBTracker(config)
        assert tracker.enabled is True

    def test_disabled_when_report_to_none(self):
        config = MagicMock()
        config.report_to = "none"
        tracker = WandBTracker(config)
        assert tracker.enabled is False

    def test_disabled_when_report_to_tensorboard(self):
        config = MagicMock()
        config.report_to = "tensorboard"
        tracker = WandBTracker(config)
        assert tracker.enabled is False

    @patch("dvas.models.student.wandb_tracker.logger")
    def test_init_logs_warning_when_wandb_not_installed(self, mock_logger):
        config = MagicMock()
        config.report_to = "wandb"
        config.wandb_project = "test_project"
        config.wandb_entity = None
        config.experiment_name = "test_run"

        tracker = WandBTracker(config)

        with patch.dict("sys.modules", {"wandb": None}):
            tracker.init()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "wandb not installed" in str(call_args)

    def test_log_metrics_when_disabled(self):
        config = MagicMock()
        config.report_to = "none"
        tracker = WandBTracker(config)
        # Should not raise
        tracker.log_metrics({"loss": 0.5})

    def test_log_checkpoint_when_disabled(self):
        config = MagicMock()
        config.report_to = "none"
        tracker = WandBTracker(config)
        # Should not raise
        tracker.log_checkpoint(Path("/tmp/checkpoint"))

    def test_finish_when_disabled(self):
        config = MagicMock()
        config.report_to = "none"
        tracker = WandBTracker(config)
        # Should not raise
        tracker.finish()

    @patch("dvas.models.student.wandb_tracker.logger")
    def test_init_with_real_config(self, mock_logger):
        """Test init with SFTConfig - wandb is not installed, should log warning."""
        config = SFTConfig()
        config.report_to = "wandb"
        config.wandb_project = "dvas-test"
        config.experiment_name = "test-sft"

        tracker = WandBTracker(config)
        tracker.init()

        # wandb is not installed, should log warning
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "wandb not installed" in call_args or "W&B initialization failed" in call_args

    @patch("dvas.models.student.wandb_tracker.logger")
    def test_log_metrics(self, mock_logger):
        """Test log_metrics when wandb is not installed."""
        config = MagicMock()
        config.report_to = "wandb"
        config.experiment_name = "test"

        tracker = WandBTracker(config)
        tracker._run = MagicMock()
        tracker.log_metrics({"loss": 0.5, "lr": 2e-4}, step=100)

        # wandb is not installed, should log warning
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "Failed to log metrics" in call_args

    @patch("dvas.models.student.wandb_tracker.logger")
    def test_log_checkpoint(self, mock_logger):
        """Test log_checkpoint when wandb is not installed."""
        config = MagicMock()
        config.report_to = "wandb"
        config.experiment_name = "test"

        tracker = WandBTracker(config)
        tracker._run = MagicMock()
        tracker.log_checkpoint(Path("/tmp/checkpoint"))

        # wandb is not installed, should log warning
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "Failed to log checkpoint" in call_args

    def test_finish(self):
        """Test finish when wandb is not installed - should not raise."""
        config = MagicMock()
        config.report_to = "wandb"

        tracker = WandBTracker(config)
        tracker.finish()  # Should not raise


class TestInitWandbForTransformers:
    """Test init_wandb_for_transformers utility."""

    def test_sets_environment_variables(self):
        config = SFTConfig()
        config.report_to = "wandb"
        config.wandb_project = "my-project"
        config.wandb_entity = "my-team"
        config.experiment_name = "run-123"

        # Clear any existing env vars
        for key in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_NAME"]:
            os.environ.pop(key, None)

        init_wandb_for_transformers(config)

        assert os.environ.get("WANDB_PROJECT") == "my-project"
        assert os.environ.get("WANDB_ENTITY") == "my-team"
        assert os.environ.get("WANDB_RUN_NAME") == "run-123"

    def test_does_not_override_existing_vars(self):
        config = SFTConfig()
        config.report_to = "wandb"
        config.wandb_project = "new-project"

        os.environ["WANDB_PROJECT"] = "existing-project"

        init_wandb_for_transformers(config)

        assert os.environ.get("WANDB_PROJECT") == "existing-project"

    def test_skips_when_report_to_none(self):
        config = SFTConfig()
        config.report_to = "none"

        # Clear any existing env vars
        for key in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_NAME"]:
            os.environ.pop(key, None)

        init_wandb_for_transformers(config)

        assert "WANDB_PROJECT" not in os.environ
