"""Mixed precision training utilities for DVAS student models.

Provides GradScaler management, AMP context helpers, and dtype selection
logic for optimal training performance.
"""

from __future__ import annotations

import torch
from torch.cuda.amp import GradScaler

from dvas.models.student.config import SFTConfig
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def get_amp_dtype(config: SFTConfig) -> torch.dtype:
    """Determine the optimal AMP dtype based on config and hardware.

    Priority:
    1. bfloat16 (if supported and bf16=True)
    2. float16 (if fp16=True)
    3. float32 (fallback)

    Args:
        config: Training configuration

    Returns:
        torch.dtype for autocast
    """
    if config.training.bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            logger.warning(
                "bf16 requested but not supported on this device, falling back to float32"
            )
            return torch.float32
    elif config.training.fp16:
        return torch.float16
    else:
        return torch.float32


def get_grad_scaler(config: SFTConfig) -> GradScaler | None:
    """Create a GradScaler for mixed precision training.

    Returns None if grad scaling is disabled or not applicable
    (e.g., bfloat16 does not need grad scaling).

    Args:
        config: Training configuration

    Returns:
        GradScaler instance or None
    """
    if not config.training.grad_scaler_enabled:
        logger.info("GradScaler disabled by config")
        return None

    # bfloat16 does not need grad scaling (same exponent range as float32)
    if config.training.bf16:
        logger.info("bf16 mode: GradScaler not needed (same range as fp32)")
        return None

    if not config.training.fp16:
        logger.info("fp32 mode: GradScaler not needed")
        return None

    scaler = GradScaler(
        init_scale=config.training.grad_scaler_init_scale,
        growth_factor=config.training.grad_scaler_growth_factor,
        backoff_factor=config.training.grad_scaler_backoff_factor,
        growth_interval=config.training.grad_scaler_growth_interval,
        enabled=True,
    )
    logger.info(
        "GradScaler initialized",
        init_scale=config.training.grad_scaler_init_scale,
        growth_factor=config.training.grad_scaler_growth_factor,
    )
    return scaler


def get_training_dtype(config: SFTConfig) -> torch.dtype:
    """Get the dtype for model loading and training.

    Maps string config values to torch dtypes.

    Args:
        config: Training configuration

    Returns:
        torch.dtype for model weights
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    torch_dtype_str = config.model.torch_dtype.lower()
    dtype = dtype_map.get(torch_dtype_str, torch.float32)

    # Validate: bfloat16 requires Ampere+ or CPU
    if dtype == torch.bfloat16:
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning(
                "bfloat16 not supported on this GPU (requires Ampere+), falling back to float32"
            )
            return torch.float32

    return dtype


def configure_amp_for_trainer(config: SFTConfig) -> dict:
    """Generate AMP configuration dict for transformers Trainer.

    Returns keyword arguments to pass to TrainingArguments for
    automatic mixed precision support.

    Args:
        config: Training configuration

    Returns:
        Dict of AMP-related TrainingArguments kwargs
    """
    amp_config: dict = {}

    # Determine if AMP is enabled
    amp_enabled = config.training.fp16 or config.training.bf16

    if not amp_enabled:
        logger.info("AMP disabled (fp16=False, bf16=False)")
        return amp_config

    # Set fp16/bf16 flags
    amp_config["fp16"] = config.training.fp16
    amp_config["bf16"] = config.training.bf16

    # GradScaler for fp16 (not needed for bf16)
    if config.training.fp16 and config.training.grad_scaler_enabled:
        amp_config["fp16_full_eval"] = False
        logger.info("fp16 AMP enabled with GradScaler")
    elif config.training.bf16:
        logger.info("bf16 AMP enabled (no GradScaler needed)")

    return amp_config


def log_amp_status(config: SFTConfig) -> None:
    """Log AMP configuration status for debugging.

    Args:
        config: Training configuration
    """
    dtype = get_training_dtype(config)
    amp_dtype = get_amp_dtype(config)
    scaler = get_grad_scaler(config)

    logger.info(
        "AMP Configuration",
        model_dtype=str(dtype),
        amp_dtype=str(amp_dtype),
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        grad_scaler_enabled=scaler is not None,
        cuda_available=torch.cuda.is_available(),
        bf16_supported=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
