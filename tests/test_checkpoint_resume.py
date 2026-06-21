"""Tests for checkpoint resume utilities."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies before importing student modules
sys.modules["datasets"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.amp"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.checkpoint_resume import (
    cleanup_old_checkpoints,
    find_checkpoints,
    get_latest_checkpoint,
    get_resume_kwargs,
    is_checkpoint_valid,
    load_checkpoint_state,
    list_checkpoint_info,
    resume_from_checkpoint,
    save_checkpoint_state,
)


class TestFindCheckpoints:
    """Test find_checkpoints function."""

    def test_empty_when_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = find_checkpoints(tmp_path)
            assert result == []

    def test_finds_checkpoint_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create checkpoint directories
            (tmp_path / "checkpoint-100").mkdir()
            (tmp_path / "checkpoint-200").mkdir()
            (tmp_path / "checkpoint-50").mkdir()
            (tmp_path / "not-a-checkpoint").mkdir()

            result = find_checkpoints(tmp_path)

            assert len(result) == 3
            # Should be sorted by step number
            assert result[0].name == "checkpoint-50"
            assert result[1].name == "checkpoint-100"
            assert result[2].name == "checkpoint-200"

    def test_empty_when_dir_does_not_exist(self):
        result = find_checkpoints(Path("/nonexistent/path"))
        assert result == []


class TestIsCheckpointValid:
    """Test is_checkpoint_valid function."""

    def test_invalid_when_path_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = is_checkpoint_valid(tmp_path / "nonexistent")
            assert result is False

    def test_invalid_when_not_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "not-a-dir"
            file_path.write_text("not a directory")
            result = is_checkpoint_valid(file_path)
            assert result is False

    def test_valid_with_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")

            result = is_checkpoint_valid(checkpoint)
            assert result is True

    def test_valid_with_model_bin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_text("fake model")

            result = is_checkpoint_valid(checkpoint)
            assert result is True

    def test_invalid_when_no_model_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            # Only create a random file
            (checkpoint / "random.txt").write_text("random")

            result = is_checkpoint_valid(checkpoint)
            assert result is False

    def test_valid_with_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")
            state = {"step": 100, "epoch": 1.5}
            (checkpoint / "checkpoint_state.json").write_text(json.dumps(state))

            result = is_checkpoint_valid(checkpoint)
            assert result is True

    def test_invalid_with_corrupt_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "checkpoint_state.json").write_text("not valid json{{{")

            result = is_checkpoint_valid(checkpoint)
            assert result is False


class TestGetLatestCheckpoint:
    """Test get_latest_checkpoint function."""

    def test_none_when_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = get_latest_checkpoint(tmp_path)
            assert result is None

    def test_returns_latest_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create checkpoints
            old = tmp_path / "checkpoint-100"
            old.mkdir()
            (old / "adapter_config.json").write_text("{}")

            latest = tmp_path / "checkpoint-200"
            latest.mkdir()
            (latest / "adapter_config.json").write_text("{}")

            result = get_latest_checkpoint(tmp_path)

            assert result is not None
            assert result.name == "checkpoint-200"

    def test_skips_invalid_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create invalid checkpoint (no model files)
            invalid = tmp_path / "checkpoint-100"
            invalid.mkdir()
            (invalid / "random.txt").write_text("random")

            # Create valid checkpoint
            valid = tmp_path / "checkpoint-200"
            valid.mkdir()
            (valid / "adapter_config.json").write_text("{}")

            result = get_latest_checkpoint(tmp_path)

            assert result is not None
            assert result.name == "checkpoint-200"


class TestSaveAndLoadCheckpointState:
    """Test save_checkpoint_state and load_checkpoint_state."""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()

            save_checkpoint_state(
                checkpoint_path=checkpoint,
                step=100,
                epoch=2.5,
                global_step=100,
                optimizer_state={"lr": 0.001},
                scheduler_state={"last_epoch": 10},
                random_state={"seed": 42},
                extra_metadata={"run_id": "abc123"},
            )

            state = load_checkpoint_state(checkpoint)

            assert state is not None
            assert state["step"] == 100
            assert state["epoch"] == 2.5
            assert state["global_step"] == 100
            assert state["optimizer_state"] == {"lr": 0.001}
            assert state["scheduler_state"] == {"last_epoch": 10}
            assert state["random_state"] == {"seed": 42}
            assert state["extra_metadata"] == {"run_id": "abc123"}

    def test_load_returns_none_when_no_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()

            state = load_checkpoint_state(checkpoint)

            assert state is None

    def test_load_returns_none_for_corrupt_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "checkpoint_state.json").write_text("not valid json")

            state = load_checkpoint_state(checkpoint)

            assert state is None


class TestResumeFromCheckpoint:
    """Test resume_from_checkpoint function."""

    def test_none_when_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from dvas.models.student.config import SFTConfig

            config = SFTConfig()
            checkpoint, state = resume_from_checkpoint(config, tmp_path)

            assert checkpoint is None
            assert state is None

    def test_returns_checkpoint_and_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from dvas.models.student.config import SFTConfig

            # Create a valid checkpoint
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")
            save_checkpoint_state(checkpoint, step=100, epoch=2.0, global_step=100)

            config = SFTConfig()
            result_checkpoint, state = resume_from_checkpoint(config, tmp_path)

            assert result_checkpoint is not None
            assert result_checkpoint.name == "checkpoint-100"
            assert state is not None
            assert state["step"] == 100


class TestGetResumeKwargs:
    """Test get_resume_kwargs function."""

    def test_empty_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from dvas.models.student.config import SFTConfig

            config = SFTConfig()
            kwargs = get_resume_kwargs(config, tmp_path)

            assert kwargs == {}

    def test_returns_resume_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from dvas.models.student.config import SFTConfig

            # Create a valid checkpoint
            checkpoint = tmp_path / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}")
            save_checkpoint_state(checkpoint, step=100, epoch=2.0, global_step=100)

            config = SFTConfig()
            kwargs = get_resume_kwargs(config, tmp_path)

            assert "resume_from_checkpoint" in kwargs
            assert kwargs["resume_from_checkpoint"] == str(checkpoint)
            assert "checkpoint_state" in kwargs


class TestCleanupOldCheckpoints:
    """Test cleanup_old_checkpoints function."""

    def test_removes_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create checkpoints
            for step in [100, 200, 300, 400]:
                cp = tmp_path / f"checkpoint-{step}"
                cp.mkdir()
                (cp / "adapter_config.json").write_text("{}")

            removed = cleanup_old_checkpoints(tmp_path, keep_total_limit=2)

            assert removed == 2
            remaining = find_checkpoints(tmp_path)
            assert len(remaining) == 2
            assert remaining[0].name == "checkpoint-300"
            assert remaining[1].name == "checkpoint-400"

    def test_no_removal_when_under_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create only 2 checkpoints
            for step in [100, 200]:
                cp = tmp_path / f"checkpoint-{step}"
                cp.mkdir()
                (cp / "adapter_config.json").write_text("{}")

            removed = cleanup_old_checkpoints(tmp_path, keep_total_limit=3)

            assert removed == 0
            assert len(find_checkpoints(tmp_path)) == 2


class TestListCheckpointInfo:
    """Test list_checkpoint_info function."""

    def test_lists_all_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create checkpoints
            for step in [100, 200]:
                cp = tmp_path / f"checkpoint-{step}"
                cp.mkdir()
                (cp / "adapter_config.json").write_text("{}")
                save_checkpoint_state(cp, step=step, epoch=1.0, global_step=step)

            info = list_checkpoint_info(tmp_path)

            assert len(info) == 2
            assert info[0]["name"] == "checkpoint-100"
            assert info[1]["name"] == "checkpoint-200"
            assert info[0]["step"] == 100
            assert info[0]["valid"] is True
