"""FSDP (Fully Sharded Data Parallel) support for multi-GPU training.

Provides FSDPConfig, FSDPWarpPolicy, and utilities for wrapping models
with FSDP for efficient distributed training across multiple GPUs.

Usage::

    from dvas.models.student.fsdp_trainer import FSDPTrainer
    from dvas.models.student.config import SFTConfig

    config = SFTConfig()
    trainer = FSDPTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional FSDP imports (PyTorch >= 2.0)
try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
    )

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

    # Create dummy classes for type hints
    class MixedPrecision:  # type: ignore[no-redef]
        pass

    class BackwardPrefetch:  # type: ignore[no-redef]
        pass

    class ShardingStrategy:  # type: ignore[no-redef]
        pass

    class CPUOffload:  # type: ignore[no-redef]
        pass


@dataclass
class FSDPConfig:
    """FSDP configuration for distributed training.

    Attributes:
        enabled: Whether to use FSDP (requires multi-GPU)
        sharding_strategy: How to shard model parameters
            - "full": Full sharding (default, lowest memory)
            - "shard_grad_op": Only shard gradients
            - "no_shard": No sharding (for debugging)
        mixed_precision: Whether to use mixed precision with FSDP
        backward_prefetch: Backward prefetch strategy
            - "backward_pre": Prefetch before backward (lower memory, slightly slower)
            - "backward_post": Prefetch after backward (higher memory, slightly faster)
            - "none": No prefetch
        cpu_offload: Whether to offload parameters to CPU (saves GPU memory)
        limit_all_gathers: Whether to limit concurrent all-gathers (prevents OOM)
        sync_module_states: Whether to sync module states from rank 0
        use_orig_params: Whether to use original parameters (needed for some optimizers)
        auto_wrap_policy: Policy for auto-wrapping submodules
            - "size_based": Wrap modules larger than min_num_params
            - "transformer": Wrap transformer blocks
            - "none": Manual wrapping only
        min_num_params: Minimum number of parameters for size-based auto-wrap
        activation_checkpointing: Whether to use activation checkpointing with FSDP
        state_dict_type: How to save state dicts
            - "full": Full state dict (unsharded, for single-GPU loading)
            - "sharded": Sharded state dict (efficient for large models)
            - "local": Local state dict (per-rank)
    """

    enabled: bool = False
    sharding_strategy: str = "full"
    mixed_precision: bool = True
    backward_prefetch: str = "backward_pre"
    cpu_offload: bool = False
    limit_all_gathers: bool = True
    sync_module_states: bool = True
    use_orig_params: bool = True
    auto_wrap_policy: str = "size_based"
    min_num_params: int = 1_000_000  # 1M params
    activation_checkpointing: bool = True
    state_dict_type: str = "full"

    def __post_init__(self):
        if not FSDP_AVAILABLE and self.enabled:
            logger.warning(
                "FSDP requested but PyTorch FSDP is not available. "
                "Please install PyTorch >= 2.0 with distributed support."
            )
            self.enabled = False


@dataclass
class DistributedConfig:
    """Configuration for distributed training setup.

    Attributes:
        world_size: Total number of processes (auto-detected from env)
        rank: Current process rank (auto-detected from env)
        local_rank: Local rank within node (auto-detected from env)
        backend: Distributed backend (nccl for GPU, gloo for CPU)
        init_method: URL for process group initialization
        master_addr: Master node address (from env or default)
        master_port: Master node port (from env or default)
    """

    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))
    rank: int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    init_method: str = "env://"
    master_addr: str = field(default_factory=lambda: os.environ.get("MASTER_ADDR", "localhost"))
    master_port: str = field(default_factory=lambda: os.environ.get("MASTER_PORT", "29500"))

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is active."""
        return self.world_size > 1


def setup_distributed(config: Optional[DistributedConfig] = None) -> DistributedConfig:
    """Initialize distributed process group.

    Args:
        config: Distributed configuration. If None, auto-detect from environment.

    Returns:
        DistributedConfig with validated settings
    """
    if config is None:
        config = DistributedConfig()

    if not config.is_distributed:
        logger.info("Single-process training (no distributed setup needed)")
        return config

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", config.master_addr)
        os.environ.setdefault("MASTER_PORT", config.master_port)

        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank,
        )
        logger.info(
            "Distributed process group initialized",
            backend=config.backend,
            world_size=config.world_size,
            rank=config.rank,
            local_rank=config.local_rank,
        )

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)

    return config


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def get_sharding_strategy(strategy: str) -> Any:
    """Get FSDP sharding strategy from string name.

    Args:
        strategy: One of "full", "shard_grad_op", "no_shard"

    Returns:
        ShardingStrategy enum value
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available. Install PyTorch >= 2.0.")

    strategies = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown sharding strategy: {strategy}. Use: {list(strategies.keys())}")
    return strategies[strategy]


def get_backward_prefetch(prefetch: str) -> Any:
    """Get FSDP backward prefetch strategy from string name.

    Args:
        prefetch: One of "backward_pre", "backward_post", "none"

    Returns:
        BackwardPrefetch enum value or None
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available. Install PyTorch >= 2.0.")

    if prefetch == "backward_pre":
        return BackwardPrefetch.BACKWARD_PRE
    elif prefetch == "backward_post":
        return BackwardPrefetch.BACKWARD_POST
    elif prefetch == "none":
        return None
    else:
        raise ValueError(f"Unknown backward prefetch: {prefetch}")


def get_mixed_precision_policy(config: FSDPConfig) -> Optional[MixedPrecision]:
    """Create FSDP mixed precision policy.

    Args:
        config: FSDP configuration

    Returns:
        MixedPrecision policy or None if disabled
    """
    if not FSDP_AVAILABLE or not config.mixed_precision:
        return None

    # bfloat16 is preferred for FSDP (more stable than fp16)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )


def get_auto_wrap_policy(config: FSDPConfig):
    """Create FSDP auto-wrap policy based on configuration.

    Args:
        config: FSDP configuration

    Returns:
        Auto-wrap policy function
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available. Install PyTorch >= 2.0.")

    if config.auto_wrap_policy == "size_based":
        return size_based_auto_wrap_policy(
            min_num_params=config.min_num_params,
        )
    elif config.auto_wrap_policy == "none":
        return None
    else:
        # Default to size-based
        return size_based_auto_wrap_policy(
            min_num_params=config.min_num_params,
        )


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    fsdp_config: FSDPConfig,
    dist_config: Optional[DistributedConfig] = None,
) -> torch.nn.Module:
    """Wrap a model with FSDP for distributed training.

    Args:
        model: The model to wrap
        fsdp_config: FSDP configuration
        dist_config: Distributed configuration (auto-detected if None)

    Returns:
        FSDP-wrapped model (or original if FSDP disabled)
    """
    if not fsdp_config.enabled:
        logger.info("FSDP disabled, using single-device training")
        return model

    if not FSDP_AVAILABLE:
        logger.warning("FSDP not available, falling back to single-device training")
        return model

    # Ensure distributed is set up
    dist_config = setup_distributed(dist_config)

    # Build FSDP kwargs
    mp_policy = get_mixed_precision_policy(fsdp_config)
    prefetch = get_backward_prefetch(fsdp_config.backward_prefetch)
    sharding = get_sharding_strategy(fsdp_config.sharding_strategy)
    wrap_policy = get_auto_wrap_policy(fsdp_config)

    fsdp_kwargs: Dict[str, Any] = {
        "sharding_strategy": sharding,
        "auto_wrap_policy": wrap_policy,
        "device_id": torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"),
        "limit_all_gathers": fsdp_config.limit_all_gathers,
        "sync_module_states": fsdp_config.sync_module_states,
        "use_orig_params": fsdp_config.use_orig_params,
    }

    if mp_policy is not None:
        fsdp_kwargs["mixed_precision"] = mp_policy

    if prefetch is not None:
        fsdp_kwargs["backward_prefetch"] = prefetch

    if fsdp_config.cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    # Wrap model
    logger.info(
        "Wrapping model with FSDP",
        sharding=fsdp_config.sharding_strategy,
        mixed_precision=fsdp_config.mixed_precision,
        backward_prefetch=fsdp_config.backward_prefetch,
    )

    model = FSDP(model, **fsdp_kwargs)

    # Enable activation checkpointing if requested
    if fsdp_config.activation_checkpointing:
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
            )

            # Apply checkpointing to transformer blocks
            def check_fn(module):
                # Check for common transformer block patterns
                module_name = module.__class__.__name__
                return any(
                    name in module_name
                    for name in ["DecoderLayer", "EncoderLayer", "TransformerBlock"]
                )

            apply_activation_checkpointing(model, check_fn=check_fn)
            logger.info("Activation checkpointing enabled for FSDP")
        except ImportError:
            logger.warning("Activation checkpointing not available for this PyTorch version")

    return model


def save_fsdp_checkpoint(
    model: torch.nn.Module,
    output_dir: str,
    fsdp_config: FSDPConfig,
    optimizer: Optional[Any] = None,
) -> None:
    """Save FSDP model checkpoint.

    Args:
        model: FSDP-wrapped model
        output_dir: Directory to save checkpoint
        fsdp_config: FSDP configuration
        optimizer: Optional optimizer state to save
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available")

    os.makedirs(output_dir, exist_ok=True)

    if fsdp_config.state_dict_type == "full":
        # Save full state dict (unsharded) - can be loaded on single GPU
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import StateDictType

        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        state_dict = model.state_dict()

        # Only rank 0 saves
        if dist.get_rank() == 0:
            torch.save(state_dict, os.path.join(output_dir, "model.pt"))
            logger.info("Saved full FSDP checkpoint", path=output_dir)

    elif fsdp_config.state_dict_type == "sharded":
        # Save sharded state dict (efficient for large models)
        from torch.distributed.fsdp import ShardedStateDictConfig
        from torch.distributed.fsdp import StateDictType

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "model_shard.pt"))
        logger.info("Saved sharded FSDP checkpoint", path=output_dir)

    else:
        # Local state dict
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        logger.info("Saved local FSDP checkpoint", path=output_dir)


def load_fsdp_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    fsdp_config: FSDPConfig,
) -> None:
    """Load FSDP model checkpoint.

    Args:
        model: FSDP-wrapped model
        checkpoint_dir: Directory containing checkpoint
        fsdp_config: FSDP configuration
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available")

    if fsdp_config.state_dict_type == "full":
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import StateDictType

        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded full FSDP checkpoint", path=checkpoint_dir)

    elif fsdp_config.state_dict_type == "sharded":
        from torch.distributed.fsdp import ShardedStateDictConfig
        from torch.distributed.fsdp import StateDictType

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )
        state_dict = torch.load(os.path.join(checkpoint_dir, "model_shard.pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded sharded FSDP checkpoint", path=checkpoint_dir)

    else:
        state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded local FSDP checkpoint", path=checkpoint_dir)
