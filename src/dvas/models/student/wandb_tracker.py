"""W&B integration utilities for experiment tracking.

Provides a unified interface for initializing, logging, and managing
Weights & Biases runs across SFT and DPO training.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class WandBTracker:
    """Unified W&B experiment tracker for training runs.

    Usage::

        tracker = WandBTracker(config)
        tracker.init()
        tracker.log_metrics({"loss": 0.5, "lr": 2e-4})
        tracker.log_checkpoint(path)
        tracker.finish()
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._run = None

    @property
    def enabled(self) -> bool:
        """Check if W&B tracking is enabled."""
        return getattr(self.config, "report_to", "none") == "wandb"

    def init(self) -> None:
        """Initialize W&B run with config."""
        if not self.enabled:
            return

        try:
            import wandb

            if wandb.run is not None:
                # Already initialized (e.g., by transformers)
                self._run = wandb.run
                logger.info("W&B run already active", run_id=wandb.run.id)
                return

            wandb.init(
                project=getattr(self.config, "wandb_project", "dvas"),
                entity=getattr(self.config, "wandb_entity", None),
                name=getattr(self.config, "experiment_name", "dvas_run"),
                config=self._build_config(),
            )
            self._run = wandb.run
            logger.info(
                "W&B initialized",
                project=self.config.wandb_project,
                name=self.config.experiment_name,
            )
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")
        except Exception as e:
            logger.warning("W&B initialization failed", error=str(e))

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if not self.enabled or self._run is None:
            return

        try:
            import wandb

            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            logger.warning("Failed to log metrics to W&B", error=str(e))

    def log_checkpoint(self, checkpoint_path: Path, artifact_name: Optional[str] = None) -> None:
        """Log a model checkpoint as a W&B artifact.

        Args:
            checkpoint_path: Path to the checkpoint directory
            artifact_name: Optional custom artifact name
        """
        if not self.enabled or self._run is None:
            return

        try:
            import wandb

            name = artifact_name or f"{self.config.experiment_name}-checkpoint"
            artifact = wandb.Artifact(
                name=name,
                type="model",
                description=f"Checkpoint for {self.config.experiment_name}",
            )
            artifact.add_dir(str(checkpoint_path))
            wandb.log_artifact(artifact)
            logger.info("Checkpoint logged to W&B", path=str(checkpoint_path))
        except Exception as e:
            logger.warning("Failed to log checkpoint to W&B", error=str(e))

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled:
            return

        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
                logger.info("W&B run finished")
        except Exception:
            pass

    def _build_config(self) -> Dict[str, Any]:
        """Build a flat config dict from the config object."""
        config_dict: Dict[str, Any] = {}

        # Try to extract nested config attributes
        for section in ["model", "data", "training", "hardware"]:
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                section_dict: Dict[str, Any] = {}
                for key in dir(section_obj):
                    if not key.startswith("_"):
                        try:
                            val = getattr(section_obj, key)
                            if not callable(val):
                                section_dict[key] = val
                        except Exception:
                            pass
                config_dict[section] = section_dict

        # Add top-level config
        for key in ["experiment_name", "report_to"]:
            if hasattr(self.config, key):
                config_dict[key] = getattr(self.config, key)

        return config_dict


def init_wandb_for_transformers(config: Any) -> None:
    """Configure environment so transformers Trainer auto-logs to W&B.

    Sets WANDB_PROJECT, WANDB_ENTITY, and WANDB_RUN_NAME environment
    variables before the Trainer initializes its report_to logic.
    """
    import os

    if getattr(config, "report_to", "none") != "wandb":
        return

    os.environ.setdefault("WANDB_PROJECT", getattr(config, "wandb_project", "dvas"))
    if getattr(config, "wandb_entity", None):
        os.environ.setdefault("WANDB_ENTITY", config.wandb_entity)
    os.environ.setdefault("WANDB_RUN_NAME", getattr(config, "experiment_name", "dvas_run"))
