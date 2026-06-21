"""Feature store for video annotation features.

Provides FeatureStore for extracting, versioning, and querying
video features for model training and inference.

Usage::

    from dvas.data.feature_store import FeatureStore, FeatureConfig

    store = FeatureStore("data/features")
    store.extract_and_store(video_path, annotation_id)

    # Query features
    features = store.get_features(annotation_id)
    similar = store.find_similar(features, top_k=10)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


@dataclass
class FeatureConfig:
    """Configuration for feature store.

    Attributes:
        root_path: Root directory for feature storage
        embedding_dim: Dimension of feature embeddings
        enable_faiss: Whether to use FAISS for similarity search
        enable_versioning: Whether to version features
        compression: Parquet compression codec
        feature_types: Types of features to extract
    """

    root_path: Path = Path("data/features")
    embedding_dim: int = 768
    enable_faiss: bool = True
    enable_versioning: bool = True
    compression: str = "zstd"
    feature_types: List[str] = field(
        default_factory=lambda: [
            "visual",
            "temporal",
            "textual",
            "metadata",
        ]
    )


@dataclass
class FeatureVector:
    """A feature vector with metadata.

    Attributes:
        annotation_id: Annotation ID
        video_id: Video ID
        feature_type: Type of feature
        version: Feature version
        data: Feature data (numpy array or dict)
        timestamp: Creation timestamp
        metadata: Additional metadata
    """

    annotation_id: str
    video_id: str
    feature_type: str
    version: str = "1.0"
    data: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """Feature store for video annotation features.

    Provides efficient storage and retrieval of extracted features
    with versioning and similarity search support.

    Attributes:
        config: FeatureConfig
        root_path: Root directory for features
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.root_path = Path(self.config.root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookup
        self._index: Dict[str, FeatureVector] = {}
        self._faiss_index: Optional[Any] = None
        self._faiss_ids: List[str] = []

        if self.config.enable_faiss and not FAISS_AVAILABLE:
            logger.warning("faiss not available, similarity search disabled")
            self.config.enable_faiss = False

    def _get_feature_path(self, annotation_id: str, feature_type: str) -> Path:
        """Get storage path for a feature.

        Args:
            annotation_id: Annotation ID
            feature_type: Feature type

        Returns:
            Path to feature file
        """
        subdir = annotation_id[:2]
        return self.root_path / subdir / f"{annotation_id}_{feature_type}.npy"

    def _get_metadata_path(self, annotation_id: str) -> Path:
        """Get metadata path for an annotation.

        Args:
            annotation_id: Annotation ID

        Returns:
            Path to metadata file
        """
        subdir = annotation_id[:2]
        return self.root_path / subdir / f"{annotation_id}_metadata.json"

    def store(
        self,
        annotation_id: str,
        video_id: str,
        feature_type: str,
        data: np.ndarray,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Store a feature vector.

        Args:
            annotation_id: Annotation ID
            video_id: Video ID
            feature_type: Type of feature
            data: Feature data
            version: Feature version
            metadata: Additional metadata

        Returns:
            Path to stored feature
        """
        # Create feature vector
        feature = FeatureVector(
            annotation_id=annotation_id,
            video_id=video_id,
            feature_type=feature_type,
            version=version,
            data=data,
            metadata=metadata or {},
        )

        # Store in memory
        key = f"{annotation_id}:{feature_type}:{version}"
        self._index[key] = feature

        # Store on disk
        feature_path = self._get_feature_path(annotation_id, feature_type)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(feature_path), data)

        # Store metadata
        meta_path = self._get_metadata_path(annotation_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "annotation_id": annotation_id,
                    "video_id": video_id,
                    "feature_type": feature_type,
                    "version": version,
                    "timestamp": feature.timestamp,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "metadata": metadata or {},
                },
                f,
                indent=2,
            )

        logger.debug(
            "feature_stored",
            annotation_id=annotation_id,
            feature_type=feature_type,
            version=version,
        )

        return feature_path

    def get(
        self,
        annotation_id: str,
        feature_type: str = "visual",
        version: str = "1.0",
    ) -> Optional[np.ndarray]:
        """Get a feature vector.

        Args:
            annotation_id: Annotation ID
            feature_type: Type of feature
            version: Feature version

        Returns:
            Feature data or None
        """
        # Check memory cache
        key = f"{annotation_id}:{feature_type}:{version}"
        if key in self._index:
            return self._index[key].data

        # Load from disk
        feature_path = self._get_feature_path(annotation_id, feature_type)
        if feature_path.exists():
            data = np.load(str(feature_path))
            return data

        return None

    def get_metadata(self, annotation_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an annotation.

        Args:
            annotation_id: Annotation ID

        Returns:
            Metadata dict or None
        """
        meta_path = self._get_metadata_path(annotation_id)
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def delete(self, annotation_id: str, feature_type: Optional[str] = None) -> bool:
        """Delete features for an annotation.

        Args:
            annotation_id: Annotation ID
            feature_type: Optional feature type filter

        Returns:
            True if deleted
        """
        deleted = False

        # Remove from memory
        keys_to_remove = [
            k for k in self._index
            if k.startswith(f"{annotation_id}:")
            and (feature_type is None or feature_type in k)
        ]
        for key in keys_to_remove:
            del self._index[key]
            deleted = True

        # Remove from disk
        if feature_type:
            feature_path = self._get_feature_path(annotation_id, feature_type)
            if feature_path.exists():
                feature_path.unlink()
                deleted = True
        else:
            # Remove all features for annotation
            subdir = self.root_path / annotation_id[:2]
            if subdir.exists():
                for file in subdir.glob(f"{annotation_id}_*"):
                    file.unlink()
                    deleted = True

        return deleted

    def list_features(self, annotation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all stored features.

        Args:
            annotation_id: Optional annotation filter

        Returns:
            List of feature metadata
        """
        features = []

        if annotation_id:
            # List features for specific annotation
            subdir = self.root_path / annotation_id[:2]
            if subdir.exists():
                for file in subdir.glob(f"{annotation_id}_*.npy"):
                    features.append({
                        "annotation_id": annotation_id,
                        "path": str(file),
                        "feature_type": file.stem.split("_")[-1],
                    })
        else:
            # List all features
            for subdir in self.root_path.iterdir():
                if subdir.is_dir():
                    for file in subdir.glob("*_*.npy"):
                        parts = file.stem.rsplit("_", 1)
                        if len(parts) == 2:
                            features.append({
                                "annotation_id": parts[0],
                                "path": str(file),
                                "feature_type": parts[1],
                            })

        return features

    def build_faiss_index(self) -> None:
        """Build FAISS index for similarity search."""
        if not self.config.enable_faiss:
            logger.warning("faiss not enabled")
            return

        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is required for similarity search")

        # Collect all features
        features = []
        ids = []

        for key, feature in self._index.items():
            if feature.data is not None:
                # Flatten feature vector
                flat = feature.data.flatten()
                features.append(flat)
                ids.append(key)

        if not features:
            logger.warning("no_features_to_index")
            return

        # Build FAISS index
        vectors = np.array(features).astype("float32")
        dim = vectors.shape[1]

        self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
        self._faiss_index.add(vectors)
        self._faiss_ids = ids

        logger.info("faiss_index_built", vectors=len(ids), dim=dim)

    def find_similar(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar features using FAISS.

        Args:
            query: Query feature vector
            top_k: Number of results

        Returns:
            List of (annotation_id, similarity_score)
        """
        if not self.config.enable_faiss or self._faiss_index is None:
            logger.warning("faiss_index_not_built")
            return []

        # Normalize query
        query_flat = query.flatten().astype("float32")
        query_norm = query_flat / np.linalg.norm(query_flat)
        query_norm = query_norm.reshape(1, -1)

        # Search
        distances, indices = self._faiss_index.search(query_norm, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self._faiss_ids):
                results.append((self._faiss_ids[idx], float(dist)))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics.

        Returns:
            Dict with statistics
        """
        feature_count = len(self._index)
        total_size = 0

        for subdir in self.root_path.iterdir():
            if subdir.is_dir():
                for file in subdir.glob("*.npy"):
                    total_size += file.stat().st_size

        return {
            "memory_entries": feature_count,
            "total_size_mb": total_size / (1024 * 1024),
            "root_path": str(self.root_path),
            "faiss_enabled": self.config.enable_faiss,
            "faiss_built": self._faiss_index is not None,
        }

    def export_to_parquet(self, output_path: Path) -> None:
        """Export features to Parquet format.

        Args:
            output_path: Output Parquet file path
        """
        if not PYARROW_AVAILABLE:
            raise RuntimeError("pyarrow is required for Parquet export")

        data = {
            "annotation_id": [],
            "video_id": [],
            "feature_type": [],
            "version": [],
            "timestamp": [],
        }

        for key, feature in self._index.items():
            data["annotation_id"].append(feature.annotation_id)
            data["video_id"].append(feature.video_id)
            data["feature_type"].append(feature.feature_type)
            data["version"].append(feature.version)
            data["timestamp"].append(feature.timestamp)

        table = pa.Table.from_pydict(data)
        pq.write_table(table, str(output_path), compression=self.config.compression)

        logger.info("features_exported_to_parquet", path=str(output_path))

    def close(self) -> None:
        """Close the feature store and release resources."""
        self._index.clear()
        if self._faiss_index is not None:
            del self._faiss_index
            self._faiss_index = None


class FeatureExtractor:
    """Extract features from video annotations.

    Provides methods to extract different types of features
    from video annotations for model training and inference.
    """

    @staticmethod
    def extract_visual_features(video_path: str) -> Dict[str, np.ndarray]:
        """Extract visual features from a video.

        Args:
            video_path: Path to video file

        Returns:
            Dict of feature name -> feature array
        """
        # Placeholder: In production, use CLIP, ResNet, etc.
        logger.info("extracting_visual_features", path=video_path)

        # Return dummy features for testing
        return {
            "frame_embeddings": np.random.randn(16, 768).astype(np.float32),
            "motion_features": np.random.randn(8, 256).astype(np.float32),
            "scene_features": np.random.randn(4, 512).astype(np.float32),
        }

    @staticmethod
    def extract_textual_features(text: str) -> Dict[str, np.ndarray]:
        """Extract textual features from annotation text.

        Args:
            text: Annotation text

        Returns:
            Dict of feature name -> feature array
        """
        # Placeholder: In production, use BERT, Sentence-BERT, etc.
        logger.info("extracting_textual_features", text_length=len(text))

        return {
            "text_embedding": np.random.randn(768).astype(np.float32),
            "token_features": np.random.randn(128, 768).astype(np.float32),
        }

    @staticmethod
    def extract_temporal_features(
        timestamps: List[float],
        durations: List[float],
    ) -> Dict[str, np.ndarray]:
        """Extract temporal features from video timestamps.

        Args:
            timestamps: List of timestamps
            durations: List of durations

        Returns:
            Dict of feature name -> feature array
        """
        if not timestamps or not durations:
            return {
                "temporal_stats": np.zeros(4, dtype=np.float32),
            }

        timestamps_arr = np.array(timestamps)
        durations_arr = np.array(durations)

        return {
            "temporal_stats": np.array([
                np.mean(timestamps_arr),
                np.std(timestamps_arr),
                np.mean(durations_arr),
                np.std(durations_arr),
            ], dtype=np.float32),
            "temporal_histogram": np.histogram(timestamps_arr, bins=10)[0].astype(np.float32),
        }

    @staticmethod
    def compute_similarity_matrix(
        features1: np.ndarray,
        features2: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """Compute similarity matrix between two sets of features.

        Args:
            features1: First set of features (N, D)
            features2: Second set of features (M, D)
            metric: Similarity metric

        Returns:
            Similarity matrix (N, M)
        """
        if metric == "cosine":
            # Normalize
            f1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            f2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            return np.dot(f1_norm, f2_norm.T)
        elif metric == "euclidean":
            diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
            return -np.linalg.norm(diff, axis=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
