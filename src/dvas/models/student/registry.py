"""LoRA adapter and model artifact registry for student models.

Provides versioned storage and retrieval of:
- LoRA adapter weights
- Full model checkpoints
- Training metadata and data version bindings
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata for a LoRA adapter or model artifact."""

    adapter_id: str
    adapter_name: str
    base_model: str
    adapter_type: str  # "lora", "full", "merged"
    created_at: datetime
    training_data_hash: str  # Hash of training dataset for reproducibility
    training_config_hash: str  # Hash of training configuration
    parent_adapter_id: Optional[str] = None  # For DPO, points to SFT adapter
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    data_version: Optional[str] = None  # Version of training data
    epic_split: Optional[str] = None  # EPIC-KITCHENS split used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "base_model": self.base_model,
            "adapter_type": self.adapter_type,
            "created_at": self.created_at.isoformat(),
            "training_data_hash": self.training_data_hash,
            "training_config_hash": self.training_config_hash,
            "parent_adapter_id": self.parent_adapter_id,
            "metrics": self.metrics,
            "tags": self.tags,
            "description": self.description,
            "data_version": self.data_version,
            "epic_split": self.epic_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterMetadata":
        """Create from dictionary."""
        return cls(
            adapter_id=data["adapter_id"],
            adapter_name=data["adapter_name"],
            base_model=data["base_model"],
            adapter_type=data["adapter_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training_data_hash=data["training_data_hash"],
            training_config_hash=data["training_config_hash"],
            parent_adapter_id=data.get("parent_adapter_id"),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
            description=data.get("description"),
            data_version=data.get("data_version"),
            epic_split=data.get("epic_split"),
        )


class LoRAAdapterRegistry:
    """Registry for managing LoRA adapters and model artifacts.

    Provides:
    - Versioned storage of adapters
    - Data version binding for reproducibility
    - Adapter lineage tracking (SFT -> DPO)
    - Metadata querying and filtering
    """

    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.adapters_dir = self.registry_dir / "adapters"
        self.metadata_dir = self.registry_dir / "metadata"

        # Create directories
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized LoRAAdapterRegistry",
            registry_dir=str(self.registry_dir),
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def _compute_data_hash(self, data_path: Path) -> str:
        """Compute hash of training data (JSONL file or directory)."""
        if data_path.is_file():
            return self._compute_file_hash(data_path)
        elif data_path.is_dir():
            # Hash all JSONL files in directory
            sha256 = hashlib.sha256()
            for jsonl_file in sorted(data_path.glob("*.jsonl")):
                sha256.update(self._compute_file_hash(jsonl_file).encode())
            return sha256.hexdigest()[:16]
        return "unknown"

    def _compute_config_hash(self, config: Any) -> str:
        """Compute hash of training configuration."""
        config_str = json.dumps(config, default=str, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def register_adapter(
        self,
        adapter_path: Path,
        adapter_name: str,
        base_model: str,
        adapter_type: str = "lora",
        training_data_path: Optional[Path] = None,
        training_config: Optional[Any] = None,
        parent_adapter_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        data_version: Optional[str] = None,
        epic_split: Optional[str] = None,
    ) -> str:
        """Register a new adapter in the registry.

        Args:
            adapter_path: Path to adapter files (LoRA weights or full model)
            adapter_name: Human-readable name
            base_model: Base model name (e.g., "Qwen/Qwen2-VL-7B-Instruct")
            adapter_type: Type of adapter ("lora", "full", "merged")
            training_data_path: Path to training data for hash computation
            training_config: Training configuration object
            parent_adapter_id: ID of parent adapter (for DPO lineage)
            metrics: Training/evaluation metrics
            tags: Tags for filtering
            description: Human-readable description
            data_version: Version identifier for training data
            epic_split: EPIC-KITCHENS split used for training

        Returns:
            adapter_id: Unique identifier for the registered adapter
        """
        # Generate adapter ID from name and timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        adapter_id = f"{adapter_name}_{timestamp}"

        # Compute hashes
        training_data_hash = (
            self._compute_data_hash(training_data_path) if training_data_path else "unknown"
        )
        training_config_hash = (
            self._compute_config_hash(training_config) if training_config else "unknown"
        )

        # Create metadata
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            base_model=base_model,
            adapter_type=adapter_type,
            created_at=datetime.utcnow(),
            training_data_hash=training_data_hash,
            training_config_hash=training_config_hash,
            parent_adapter_id=parent_adapter_id,
            metrics=metrics or {},
            tags=tags or [],
            description=description,
            data_version=data_version,
            epic_split=epic_split,
        )

        # Copy adapter files to registry
        target_dir = self.adapters_dir / adapter_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if adapter_path.is_file():
            import shutil

            shutil.copy2(adapter_path, target_dir / adapter_path.name)
        elif adapter_path.is_dir():
            import shutil

            for item in adapter_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)

        # Save metadata
        metadata_path = self.metadata_dir / f"{adapter_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(
            "Registered adapter",
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            adapter_type=adapter_type,
        )

        return adapter_id

    def get_adapter(self, adapter_id: str) -> Optional[Path]:
        """Get path to adapter files by ID.

        Args:
            adapter_id: Unique adapter identifier

        Returns:
            Path to adapter directory, or None if not found
        """
        adapter_dir = self.adapters_dir / adapter_id
        if adapter_dir.exists():
            return adapter_dir
        return None

    def get_metadata(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get metadata for an adapter.

        Args:
            adapter_id: Unique adapter identifier

        Returns:
            AdapterMetadata, or None if not found
        """
        metadata_path = self.metadata_dir / f"{adapter_id}.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return AdapterMetadata.from_dict(data)

    def list_adapters(
        self,
        adapter_type: Optional[str] = None,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        epic_split: Optional[str] = None,
    ) -> List[AdapterMetadata]:
        """List adapters with optional filtering.

        Args:
            adapter_type: Filter by adapter type
            base_model: Filter by base model
            tags: Filter by tags (all must match)
            epic_split: Filter by EPIC split

        Returns:
            List of matching adapter metadata
        """
        results = []

        for metadata_path in self.metadata_dir.glob("*.json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = AdapterMetadata.from_dict(data)

            # Apply filters
            if adapter_type and metadata.adapter_type != adapter_type:
                continue
            if base_model and metadata.base_model != base_model:
                continue
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            if epic_split and metadata.epic_split != epic_split:
                continue

            results.append(metadata)

        # Sort by creation time (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results

    def get_latest_adapter(
        self,
        adapter_type: Optional[str] = None,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[AdapterMetadata]:
        """Get the most recent adapter matching criteria.

        Args:
            adapter_type: Filter by adapter type
            base_model: Filter by base model
            tags: Filter by tags

        Returns:
            Most recent matching adapter metadata, or None
        """
        adapters = self.list_adapters(
            adapter_type=adapter_type,
            base_model=base_model,
            tags=tags,
        )
        return adapters[0] if adapters else None

    def get_adapter_lineage(self, adapter_id: str) -> List[AdapterMetadata]:
        """Get lineage chain for an adapter (parent -> child).

        Args:
            adapter_id: Starting adapter ID

        Returns:
            List of metadata from oldest ancestor to the given adapter
        """
        lineage = []
        current_id = adapter_id

        # First, collect all ancestors
        ancestors = []
        while current_id:
            metadata = self.get_metadata(current_id)
            if not metadata:
                break
            ancestors.append(metadata)
            current_id = metadata.parent_adapter_id

        # Reverse to get chronological order
        lineage = list(reversed(ancestors))
        return lineage

    def verify_data_binding(
        self,
        adapter_id: str,
        data_path: Path,
    ) -> bool:
        """Verify that adapter was trained on specific data.

        Args:
            adapter_id: Adapter to verify
            data_path: Path to data to check against

        Returns:
            True if data hash matches, False otherwise
        """
        metadata = self.get_metadata(adapter_id)
        if not metadata:
            return False

        current_hash = self._compute_data_hash(data_path)
        return current_hash == metadata.training_data_hash

    def delete_adapter(self, adapter_id: str) -> bool:
        """Delete an adapter from the registry.

        Args:
            adapter_id: Adapter to delete

        Returns:
            True if deleted, False if not found
        """
        adapter_dir = self.adapters_dir / adapter_id
        metadata_path = self.metadata_dir / f"{adapter_id}.json"

        deleted = False
        if adapter_dir.exists():
            import shutil

            shutil.rmtree(adapter_dir)
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        if deleted:
            logger.info("Deleted adapter", adapter_id=adapter_id)

        return deleted


class ModelArtifactRegistry:
    """Registry for model artifacts beyond LoRA adapters.

    Manages:
    - Merged models (base + LoRA)
    - ONNX exports
    - Quantized models (GPTQ, AWQ)
    - Training checkpoints
    """

    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.checkpoints_dir = self.registry_dir / "checkpoints"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def register_merged_model(
        self,
        merged_model_path: Path,
        adapter_id: str,
        artifact_name: str,
        merge_method: str = "linear",
    ) -> str:
        """Register a merged model (base + LoRA).

        Args:
            merged_model_path: Path to merged model files
            adapter_id: Source adapter ID
            artifact_name: Human-readable name
            merge_method: Method used for merging

        Returns:
            artifact_id: Unique identifier
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_id = f"merged_{artifact_name}_{timestamp}"

        target_dir = self.artifacts_dir / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        import shutil

        if merged_model_path.is_dir():
            for item in merged_model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)

        # Save metadata
        metadata = {
            "artifact_id": artifact_id,
            "artifact_name": artifact_name,
            "artifact_type": "merged_model",
            "adapter_id": adapter_id,
            "merge_method": merge_method,
            "created_at": datetime.utcnow().isoformat(),
        }

        metadata_path = target_dir / "artifact_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Registered merged model",
            artifact_id=artifact_id,
            adapter_id=adapter_id,
        )

        return artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[Path]:
        """Get path to artifact files."""
        artifact_dir = self.artifacts_dir / artifact_id
        if artifact_dir.exists():
            return artifact_dir
        return None

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        training_run_id: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint files
            training_run_id: Unique training run identifier
            step: Training step number
            metrics: Current metrics

        Returns:
            Path to saved checkpoint
        """
        target_dir = self.checkpoints_dir / training_run_id / f"checkpoint-{step}"
        target_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        if checkpoint_path.is_dir():
            for item in checkpoint_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)

        # Save checkpoint metadata
        metadata = {
            "training_run_id": training_run_id,
            "step": step,
            "metrics": metrics or {},
            "saved_at": datetime.utcnow().isoformat(),
        }

        metadata_path = target_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return target_dir

    def list_checkpoints(self, training_run_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a training run."""
        run_dir = self.checkpoints_dir / training_run_id
        if not run_dir.exists():
            return []

        checkpoints = []
        for checkpoint_dir in sorted(run_dir.glob("checkpoint-*")):
            metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)

        return checkpoints
