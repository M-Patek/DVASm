"""Checkpoint resume utilities for training recovery.

Provides functions to scan, validate, and resume from training checkpoints
for both SFT and DPO training.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dvas.models.student.config import SFTConfig
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Checkpoint directory pattern: checkpoint-<step>
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def find_checkpoints(output_dir: Path) -> List[Path]:
    """Find all checkpoint directories in the output directory.

    Args:
        output_dir: Training output directory

    Returns:
        List of checkpoint paths, sorted by step number
    """
    if not output_dir.exists():
        return []

    checkpoints = []
    for item in output_dir.iterdir():
        if item.is_dir() and CHECKPOINT_PATTERN.match(item.name):
            checkpoints.append(item)

    # Sort by step number
    checkpoints.sort(key=lambda p: int(CHECKPOINT_PATTERN.match(p.name).group(1)))
    return checkpoints


def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Get the most recent valid checkpoint.

    Args:
        output_dir: Training output directory

    Returns:
        Path to latest checkpoint, or None if no valid checkpoint found
    """
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None

    # Check from newest to oldest for valid checkpoint
    for checkpoint in reversed(checkpoints):
        if is_checkpoint_valid(checkpoint):
            return checkpoint

    return None


def is_checkpoint_valid(checkpoint_path: Path) -> bool:
    """Check if a checkpoint is valid and complete.

    Validates:
    - Directory exists and is readable
    - Contains required files (model weights or adapter config)
    - checkpoint_state.json exists and is valid JSON

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if checkpoint is valid
    """
    if not checkpoint_path.exists():
        return False

    if not checkpoint_path.is_dir():
        return False

    # Check for model files (LoRA adapter or full model)
    has_adapter = (checkpoint_path / "adapter_config.json").exists()
    has_model = list(checkpoint_path.glob("*.bin")) or list(checkpoint_path.glob("*.safetensors"))

    if not (has_adapter or has_model):
        logger.warning(
            "Checkpoint missing model files",
            path=str(checkpoint_path),
        )
        return False

    # Validate checkpoint_state.json if present
    state_path = checkpoint_path / "checkpoint_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Verify required fields
            if "step" not in state:
                logger.warning(
                    "Checkpoint state missing 'step' field",
                    path=str(checkpoint_path),
                )
                return False
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Invalid checkpoint state JSON",
                path=str(checkpoint_path),
            )
            return False

    return True


def save_checkpoint_state(
    checkpoint_path: Path,
    step: int,
    epoch: float,
    global_step: int,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    random_state: Optional[Dict] = None,
    extra_metadata: Optional[Dict] = None,
) -> None:
    """Save checkpoint state metadata.

    Args:
        checkpoint_path: Path to checkpoint directory
        step: Current training step
        epoch: Current epoch
        global_step: Global step count
        optimizer_state: Optional optimizer state dict
        scheduler_state: Optional scheduler state dict
        random_state: Optional random state dict
        extra_metadata: Optional extra metadata
    """
    state = {
        "step": step,
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state": optimizer_state or {},
        "scheduler_state": scheduler_state or {},
        "random_state": random_state or {},
        "extra_metadata": extra_metadata or {},
    }

    state_path = checkpoint_path / "checkpoint_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    logger.info(
        "Checkpoint state saved",
        path=str(checkpoint_path),
        step=step,
        epoch=epoch,
    )


def load_checkpoint_state(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint state metadata.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Checkpoint state dict, or None if not found/invalid
    """
    state_path = checkpoint_path / "checkpoint_state.json"
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Failed to load checkpoint state",
            path=str(checkpoint_path),
        )
        return None


def resume_from_checkpoint(
    config: SFTConfig,
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Dict]]:
    """Find and validate the latest checkpoint for resuming.

    Args:
        config: Training configuration
        output_dir: Training output directory

    Returns:
        Tuple of (checkpoint_path, checkpoint_state)
        Returns (None, None) if no valid checkpoint found
    """
    latest = get_latest_checkpoint(output_dir)

    if latest is None:
        logger.info("No checkpoint found for resuming", output_dir=str(output_dir))
        return None, None

    # Validate checkpoint
    if not is_checkpoint_valid(latest):
        logger.warning(
            "Latest checkpoint is invalid, cannot resume",
            path=str(latest),
        )
        return None, None

    # Load checkpoint state
    state = load_checkpoint_state(latest)

    logger.info(
        "Found checkpoint for resuming",
        path=str(latest),
        step=state.get("step", "unknown") if state else "unknown",
    )

    return latest, state


def get_resume_kwargs(
    config: SFTConfig,
    output_dir: Path,
) -> Dict:
    """Get keyword arguments for resuming from checkpoint.

    Returns a dict that can be passed to Trainer's resume_from_checkpoint
    or used to configure TrainingArguments.

    Args:
        config: Training configuration
        output_dir: Training output directory

    Returns:
        Dict with resume configuration
    """
    checkpoint_path, state = resume_from_checkpoint(config, output_dir)

    if checkpoint_path is None:
        return {}

    kwargs = {
        "resume_from_checkpoint": str(checkpoint_path),
    }

    if state:
        kwargs["checkpoint_state"] = state

    return kwargs


def cleanup_old_checkpoints(output_dir: Path, keep_total_limit: int = 3) -> int:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Training output directory
        keep_total_limit: Number of checkpoints to keep

    Returns:
        Number of checkpoints removed
    """
    checkpoints = find_checkpoints(output_dir)

    if len(checkpoints) <= keep_total_limit:
        return 0

    # Keep the most recent checkpoints
    to_remove = checkpoints[:-keep_total_limit]
    removed = 0

    import shutil

    for checkpoint in to_remove:
        try:
            shutil.rmtree(checkpoint)
            logger.info("Removed old checkpoint", path=str(checkpoint))
            removed += 1
        except OSError as e:
            logger.warning(
                "Failed to remove checkpoint",
                path=str(checkpoint),
                error=str(e),
            )

    return removed


def estimate_checkpoint_size(checkpoint_path: Path) -> int:
    """Estimate the total size of a checkpoint in bytes.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    for item in checkpoint_path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def list_checkpoint_info(output_dir: Path) -> List[Dict]:
    """List all checkpoints with metadata.

    Args:
        output_dir: Training output directory

    Returns:
        List of checkpoint info dicts
    """
    checkpoints = find_checkpoints(output_dir)
    info = []

    for checkpoint in checkpoints:
        state = load_checkpoint_state(checkpoint)
        size = estimate_checkpoint_size(checkpoint)

        info.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "step": int(CHECKPOINT_PATTERN.match(checkpoint.name).group(1))
                if CHECKPOINT_PATTERN.match(checkpoint.name)
                else None,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "valid": is_checkpoint_valid(checkpoint),
                "state": state,
            }
        )

    return info
