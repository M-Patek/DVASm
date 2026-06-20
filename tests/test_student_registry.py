"""Tests for student model registry."""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.models.student.registry import (
    AdapterMetadata,
    LoRAAdapterRegistry,
    ModelArtifactRegistry,
)


class TestAdapterMetadata:
    """Test AdapterMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from datetime import datetime

        meta = AdapterMetadata(
            adapter_id="test_001",
            adapter_name="test_adapter",
            base_model="Qwen/Qwen2-VL-7B",
            adapter_type="lora",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            training_data_hash="abc123",
            training_config_hash="def456",
            parent_adapter_id="parent_001",
            metrics={"loss": 0.5},
            tags=["sft", "v1"],
            description="Test adapter",
            data_version="v1.0",
            epic_split="train",
        )

        data = meta.to_dict()
        assert data["adapter_id"] == "test_001"
        assert data["adapter_name"] == "test_adapter"
        assert data["parent_adapter_id"] == "parent_001"
        assert data["data_version"] == "v1.0"
        assert data["epic_split"] == "train"

    def test_from_dict(self):
        """Test creation from dictionary."""
        from datetime import datetime

        data = {
            "adapter_id": "test_002",
            "adapter_name": "test_adapter",
            "base_model": "Qwen/Qwen2-VL-7B",
            "adapter_type": "lora",
            "created_at": "2024-01-01T12:00:00",
            "training_data_hash": "abc123",
            "training_config_hash": "def456",
            "parent_adapter_id": None,
            "metrics": {},
            "tags": [],
            "description": None,
            "data_version": "v2.0",
            "epic_split": None,
        }

        meta = AdapterMetadata.from_dict(data)
        assert meta.adapter_id == "test_002"
        assert meta.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert meta.data_version == "v2.0"


class TestLoRAAdapterRegistry:
    """Test LoRAAdapterRegistry."""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LoRAAdapterRegistry(tmpdir)

    def test_init_creates_directories(self, temp_registry):
        """Test registry creates necessary directories."""
        assert temp_registry.adapters_dir.exists()
        assert temp_registry.metadata_dir.exists()

    def test_register_and_get_adapter(self, temp_registry):
        """Test registering and retrieving an adapter."""
        # Create fake adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text('{"test": true}')

            # Register
            adapter_id = temp_registry.register_adapter(
                adapter_path=adapter_path,
                adapter_name="test_adapter",
                base_model="Qwen/Qwen2-VL-7B",
                adapter_type="lora",
                description="Test adapter",
            )

            # Verify ID format
            assert adapter_id.startswith("test_adapter_")

            # Get path
            retrieved_path = temp_registry.get_adapter(adapter_id)
            assert retrieved_path is not None
            assert (retrieved_path / "adapter_config.json").exists()

    def test_register_with_version_binding(self, temp_registry):
        """Test registering with data version binding."""
        # Create fake adapter and data
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            (adapter_path / "config.json").write_text("{}")

            data_path = Path(tmpdir) / "train.jsonl"
            data_path.write_text('{"test": "data"}\n')

            # Register with version info
            adapter_id = temp_registry.register_adapter(
                adapter_path=adapter_path,
                adapter_name="versioned_adapter",
                base_model="Qwen/Qwen2-VL-7B",
                training_data_path=data_path,
                data_version="epic-train-v1",
                epic_split="train",
            )

            # Verify metadata
            metadata = temp_registry.get_metadata(adapter_id)
            assert metadata is not None
            assert metadata.data_version == "epic-train-v1"
            assert metadata.epic_split == "train"
            assert metadata.training_data_hash != "unknown"

    def test_parent_child_lineage(self, temp_registry):
        """Test adapter lineage tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Parent adapter (SFT)
            sft_path = Path(tmpdir) / "sft"
            sft_path.mkdir()
            (sft_path / "config.json").write_text("{}")

            sft_id = temp_registry.register_adapter(
                adapter_path=sft_path,
                adapter_name="sft_adapter",
                base_model="Qwen/Qwen2-VL-7B",
                adapter_type="lora",
            )

            # Child adapter (DPO)
            dpo_path = Path(tmpdir) / "dpo"
            dpo_path.mkdir()
            (dpo_path / "config.json").write_text("{}")

            dpo_id = temp_registry.register_adapter(
                adapter_path=dpo_path,
                adapter_name="dpo_adapter",
                base_model="Qwen/Qwen2-VL-7B",
                adapter_type="lora",
                parent_adapter_id=sft_id,
            )

            # Verify lineage
            lineage = temp_registry.get_adapter_lineage(dpo_id)
            assert len(lineage) == 2
            assert lineage[0].adapter_id == sft_id
            assert lineage[1].adapter_id == dpo_id

    def test_list_adapters_with_filters(self, temp_registry):
        """Test listing with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple adapters
            for i, (atype, model) in enumerate([
                ("lora", "Qwen/Qwen2-VL-7B"),
                ("lora", "Qwen/Qwen2-VL-7B"),
                ("full", "Qwen/Qwen2-VL-2B"),
            ]):
                path = Path(tmpdir) / f"adapter_{i}"
                path.mkdir()
                (path / "config.json").write_text("{}")

                temp_registry.register_adapter(
                    adapter_path=path,
                    adapter_name=f"adapter_{i}",
                    base_model=model,
                    adapter_type=atype,
                    tags=["test", atype],
                )

            # Filter by type
            lora_adapters = temp_registry.list_adapters(adapter_type="lora")
            assert len(lora_adapters) == 2

            # Filter by base model
            qwen_adapters = temp_registry.list_adapters(
                base_model="Qwen/Qwen2-VL-7B"
            )
            assert len(qwen_adapters) == 2

            # Filter by tags
            full_adapters = temp_registry.list_adapters(tags=["full"])
            assert len(full_adapters) == 1

    def test_verify_data_binding(self, temp_registry):
        """Test data binding verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            (adapter_path / "config.json").write_text("{}")

            data_path = Path(tmpdir) / "train.jsonl"
            data_path.write_text('{"test": "data"}\n')

            # Register with data
            adapter_id = temp_registry.register_adapter(
                adapter_path=adapter_path,
                adapter_name="test_adapter",
                base_model="Qwen/Qwen2-VL-7B",
                training_data_path=data_path,
            )

            # Verify matching data
            assert temp_registry.verify_data_binding(adapter_id, data_path)

            # Verify non-matching data
            wrong_data = Path(tmpdir) / "wrong.jsonl"
            wrong_data.write_text('{"different": "data"}\n')
            assert not temp_registry.verify_data_binding(adapter_id, wrong_data)

    def test_get_latest_adapter(self, temp_registry):
        """Test getting most recent adapter."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = Path(tmpdir) / f"adapter_{i}"
                path.mkdir()
                (path / "config.json").write_text("{}")

                temp_registry.register_adapter(
                    adapter_path=path,
                    adapter_name="test_adapter",
                    base_model="Qwen/Qwen2-VL-7B",
                )
                time.sleep(0.01)  # Ensure different timestamps

            latest = temp_registry.get_latest_adapter()
            assert latest is not None
            assert latest.adapter_name == "test_adapter"

    def test_delete_adapter(self, temp_registry):
        """Test adapter deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapter"
            path.mkdir()
            (path / "config.json").write_text("{}")

            adapter_id = temp_registry.register_adapter(
                adapter_path=path,
                adapter_name="to_delete",
                base_model="Qwen/Qwen2-VL-7B",
            )

            # Verify exists
            assert temp_registry.get_adapter(adapter_id) is not None

            # Delete
            assert temp_registry.delete_adapter(adapter_id)

            # Verify gone
            assert temp_registry.get_adapter(adapter_id) is None
            assert temp_registry.get_metadata(adapter_id) is None

            # Delete non-existent returns False
            assert not temp_registry.delete_adapter("non_existent")


class TestModelArtifactRegistry:
    """Test ModelArtifactRegistry."""

    @pytest.fixture
    def temp_artifact_registry(self):
        """Create temporary artifact registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelArtifactRegistry(tmpdir)

    def test_register_merged_model(self, temp_artifact_registry):
        """Test registering merged model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "merged_model"
            model_path.mkdir()
            (model_path / "model.safetensors").write_text("fake")

            artifact_id = temp_artifact_registry.register_merged_model(
                merged_model_path=model_path,
                adapter_id="lora_001",
                artifact_name="qwen_merged",
                merge_method="linear",
            )

            assert artifact_id.startswith("merged_qwen_merged_")

            # Verify retrieval
            retrieved = temp_artifact_registry.get_artifact(artifact_id)
            assert retrieved is not None

    def test_save_and_list_checkpoints(self, temp_artifact_registry):
        """Test checkpoint management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()
            (checkpoint_path / "pytorch_model.bin").write_text("fake")

            # Save multiple checkpoints
            for step in [100, 200, 300]:
                saved = temp_artifact_registry.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    training_run_id="run_001",
                    step=step,
                    metrics={"loss": 1.0 / step},
                )
                assert saved.exists()

            # List checkpoints
            checkpoints = temp_artifact_registry.list_checkpoints("run_001")
            assert len(checkpoints) == 3

            # Verify sorted by step
            steps = [c["step"] for c in checkpoints]
            assert steps == [100, 200, 300]
