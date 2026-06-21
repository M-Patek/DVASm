"""Tests for FSDP distributed training utilities."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.fsdp_utils import (
    DistributedConfig,
    FSDPConfig,
    cleanup_distributed,
    get_auto_wrap_policy,
    get_backward_prefetch,
    get_mixed_precision_policy,
    get_sharding_strategy,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    setup_distributed,
    wrap_model_with_fsdp,
)


class TestFSDPConfig:
    """Test FSDPConfig dataclass."""

    def test_default_config(self):
        config = FSDPConfig()
        assert config.enabled is False
        assert config.sharding_strategy == "full"
        assert config.mixed_precision is True
        assert config.backward_prefetch == "backward_pre"
        assert config.cpu_offload is False
        assert config.limit_all_gathers is True
        assert config.sync_module_states is True
        assert config.use_orig_params is True
        assert config.auto_wrap_policy == "size_based"
        assert config.min_num_params == 1_000_000
        assert config.activation_checkpointing is True
        assert config.state_dict_type == "full"

    def test_custom_config(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            config = FSDPConfig(
                enabled=True,
                sharding_strategy="shard_grad_op",
                cpu_offload=True,
                auto_wrap_policy="none",
            )
            assert config.enabled is True
            assert config.sharding_strategy == "shard_grad_op"
            assert config.cpu_offload is True
            assert config.auto_wrap_policy == "none"

    def test_disabled_when_fsdp_unavailable(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", False):
            config = FSDPConfig(enabled=True)
            assert config.enabled is False


class TestDistributedConfig:
    """Test DistributedConfig dataclass."""

    def test_default_config(self):
        config = DistributedConfig()
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.backend in ("nccl", "gloo")
        assert config.init_method == "env://"

    def test_is_main_process(self):
        config = DistributedConfig(rank=0)
        assert config.is_main_process is True

        config = DistributedConfig(rank=1)
        assert config.is_main_process is False

    def test_is_distributed(self):
        config = DistributedConfig(world_size=1)
        assert config.is_distributed is False

        config = DistributedConfig(world_size=4)
        assert config.is_distributed is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("MASTER_ADDR", "192.168.1.1")
        monkeypatch.setenv("MASTER_PORT", "29501")

        config = DistributedConfig()
        assert config.world_size == 8
        assert config.rank == 3
        assert config.local_rank == 1
        assert config.master_addr == "192.168.1.1"
        assert config.master_port == "29501"


class TestSetupDistributed:
    """Test distributed setup."""

    def test_single_process_noop(self):
        config = DistributedConfig(world_size=1)
        result = setup_distributed(config)
        assert result.world_size == 1
        assert result.rank == 0

    def test_returns_config_when_not_distributed(self):
        config = DistributedConfig(world_size=1)
        result = setup_distributed(config)
        assert result is config


class TestGetShardingStrategy:
    """Test sharding strategy resolution."""

    def test_full_strategy(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            # Mock the ShardingStrategy enum
            mock_fsdp = MagicMock()
            mock_fsdp.ShardingStrategy.FULL_SHARD = "FULL_SHARD"
            mock_fsdp.ShardingStrategy.SHARD_GRAD_OP = "SHARD_GRAD_OP"
            mock_fsdp.ShardingStrategy.NO_SHARD = "NO_SHARD"

            with patch("dvas.models.student.fsdp_utils.ShardingStrategy", mock_fsdp.ShardingStrategy):
                strategy = get_sharding_strategy("full")
                assert strategy == "FULL_SHARD"

    def test_invalid_strategy(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            mock_fsdp = MagicMock()
            mock_fsdp.ShardingStrategy.FULL_SHARD = "FULL_SHARD"
            mock_fsdp.ShardingStrategy.SHARD_GRAD_OP = "SHARD_GRAD_OP"
            mock_fsdp.ShardingStrategy.NO_SHARD = "NO_SHARD"

            with patch("dvas.models.student.fsdp_utils.ShardingStrategy", mock_fsdp.ShardingStrategy):
                with pytest.raises(ValueError, match="Unknown sharding strategy"):
                    get_sharding_strategy("invalid")

    def test_fsdp_not_available(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="FSDP is not available"):
                get_sharding_strategy("full")


class TestGetBackwardPrefetch:
    """Test backward prefetch resolution."""

    def test_backward_pre(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            mock_fsdp = MagicMock()
            mock_fsdp.BackwardPrefetch.BACKWARD_PRE = "BACKWARD_PRE"
            mock_fsdp.BackwardPrefetch.BACKWARD_POST = "BACKWARD_POST"

            with patch("dvas.models.student.fsdp_utils.BackwardPrefetch", mock_fsdp.BackwardPrefetch):
                prefetch = get_backward_prefetch("backward_pre")
                assert prefetch == "BACKWARD_PRE"

    def test_backward_post(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            mock_fsdp = MagicMock()
            mock_fsdp.BackwardPrefetch.BACKWARD_PRE = "BACKWARD_PRE"
            mock_fsdp.BackwardPrefetch.BACKWARD_POST = "BACKWARD_POST"

            with patch("dvas.models.student.fsdp_utils.BackwardPrefetch", mock_fsdp.BackwardPrefetch):
                prefetch = get_backward_prefetch("backward_post")
                assert prefetch == "BACKWARD_POST"

    def test_none_prefetch(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            prefetch = get_backward_prefetch("none")
            assert prefetch is None

    def test_invalid_prefetch(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            with pytest.raises(ValueError, match="Unknown backward prefetch"):
                get_backward_prefetch("invalid")


class TestGetMixedPrecisionPolicy:
    """Test mixed precision policy creation."""

    def test_disabled_returns_none(self):
        config = FSDPConfig(mixed_precision=False)
        policy = get_mixed_precision_policy(config)
        assert policy is None

    def test_bfloat16_when_supported(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            mock_mp = MagicMock()
            mock_mp.return_value = "bfloat16_policy"

            with patch("dvas.models.student.fsdp_utils.MixedPrecision", mock_mp):
                config = FSDPConfig(mixed_precision=True)
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.is_bf16_supported", return_value=True):
                        policy = get_mixed_precision_policy(config)
                        assert policy == "bfloat16_policy"

    def test_float16_when_bf16_not_supported(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", True):
            mock_mp = MagicMock()
            mock_mp.return_value = "float16_policy"

            with patch("dvas.models.student.fsdp_utils.MixedPrecision", mock_mp):
                config = FSDPConfig(mixed_precision=True)
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.is_bf16_supported", return_value=False):
                        policy = get_mixed_precision_policy(config)
                        assert policy == "float16_policy"


class TestWrapModelWithFSDP:
    """Test FSDP model wrapping."""

    def test_disabled_returns_original(self):
        import torch.nn as nn

        model = nn.Linear(10, 10)
        config = FSDPConfig(enabled=False)
        result = wrap_model_with_fsdp(model, config)
        assert result is model

    def test_fsdp_not_available_returns_original(self):
        import torch.nn as nn

        model = nn.Linear(10, 10)
        config = FSDPConfig(enabled=True)

        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", False):
            result = wrap_model_with_fsdp(model, config)
            assert result is model


class TestSaveLoadCheckpoint:
    """Test FSDP checkpoint save/load."""

    def test_save_fsdp_checkpoint_not_available(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="FSDP is not available"):
                save_fsdp_checkpoint(None, "/tmp/test", FSDPConfig())

    def test_load_fsdp_checkpoint_not_available(self):
        with patch("dvas.models.student.fsdp_utils.FSDP_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="FSDP is not available"):
                load_fsdp_checkpoint(None, "/tmp/test", FSDPConfig())


class TestCleanupDistributed:
    """Test distributed cleanup."""

    def test_cleanup_when_not_initialized(self):
        with patch("dvas.models.student.fsdp_utils.dist.is_initialized", return_value=False):
            # Should not raise
            cleanup_distributed()
