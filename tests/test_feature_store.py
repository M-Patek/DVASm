"""Tests for feature store."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.data.feature_store import (
    FeatureConfig,
    FeatureExtractor,
    FeatureStore,
    FeatureVector,
)


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        config = FeatureConfig()
        assert config.embedding_dim == 768
        assert config.enable_faiss is True
        assert config.enable_versioning is True
        assert config.compression == "zstd"

    def test_custom_config(self):
        config = FeatureConfig(
            embedding_dim=512,
            enable_faiss=False,
            compression="snappy",
        )
        assert config.embedding_dim == 512
        assert config.enable_faiss is False
        assert config.compression == "snappy"


class TestFeatureVector:
    """Test FeatureVector dataclass."""

    def test_creation(self):
        vec = FeatureVector(
            annotation_id="ann-001",
            video_id="vid-001",
            feature_type="visual",
            data=np.array([1.0, 2.0, 3.0]),
        )
        assert vec.annotation_id == "ann-001"
        assert vec.video_id == "vid-001"
        assert vec.feature_type == "visual"
        assert vec.version == "1.0"
        assert vec.timestamp is not None


class TestFeatureStore:
    """Test FeatureStore."""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)
            assert store.root_path.exists()

    def test_store_and_get(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            data = np.array([1.0, 2.0, 3.0, 4.0])
            store.store(
                annotation_id="ann-001",
                video_id="vid-001",
                feature_type="visual",
                data=data,
            )

            result = store.get("ann-001", "visual")
            assert result is not None
            np.testing.assert_array_equal(result, data)

    def test_get_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            result = store.get("nonexistent", "visual")
            assert result is None

    def test_get_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            data = np.array([1.0, 2.0, 3.0])
            store.store(
                annotation_id="ann-001",
                video_id="vid-001",
                feature_type="visual",
                data=data,
                metadata={"fps": 30.0},
            )

            meta = store.get_metadata("ann-001")
            assert meta is not None
            assert meta["annotation_id"] == "ann-001"
            assert meta["metadata"]["fps"] == 30.0

    def test_get_metadata_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            meta = store.get_metadata("nonexistent")
            assert meta is None

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            data = np.array([1.0, 2.0, 3.0])
            store.store("ann-001", "vid-001", "visual", data)

            assert store.delete("ann-001", "visual") is True
            assert store.get("ann-001", "visual") is None

    def test_delete_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            assert store.delete("nonexistent") is False

    def test_list_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            store.store("ann-001", "vid-001", "visual", np.array([1.0, 2.0]))
            store.store("ann-001", "vid-001", "textual", np.array([3.0, 4.0]))

            features = store.list_features("ann-001")
            assert len(features) == 2

    def test_list_all_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            store.store("ann-001", "vid-001", "visual", np.array([1.0, 2.0]))
            store.store("ann-002", "vid-002", "visual", np.array([3.0, 4.0]))

            features = store.list_features()
            assert len(features) == 2

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            store.store("ann-001", "vid-001", "visual", np.array([1.0, 2.0]))

            stats = store.get_statistics()
            assert stats["memory_entries"] == 1
            assert stats["faiss_enabled"] is True

    def test_close(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir))
            store = FeatureStore(config)

            store.store("ann-001", "vid-001", "visual", np.array([1.0, 2.0]))
            store.close()

            assert len(store._index) == 0

    def test_faiss_disabled(self):
        with patch("dvas.data.feature_store.FAISS_AVAILABLE", False):
            config = FeatureConfig(enable_faiss=True)
            store = FeatureStore(config)
            assert store.config.enable_faiss is False

    def test_build_faiss_index_no_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir), enable_faiss=False)
            store = FeatureStore(config)
            store.build_faiss_index()
            assert store._faiss_index is None

    def test_find_similar_no_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatureConfig(root_path=Path(tmp_dir), enable_faiss=False)
            store = FeatureStore(config)

            results = store.find_similar(np.array([1.0, 2.0, 3.0]))
            assert results == []


class TestFeatureExtractor:
    """Test FeatureExtractor."""

    def test_extract_visual_features(self):
        features = FeatureExtractor.extract_visual_features("/tmp/test.mp4")
        assert "frame_embeddings" in features
        assert "motion_features" in features
        assert "scene_features" in features

    def test_extract_textual_features(self):
        features = FeatureExtractor.extract_textual_features("This is a test caption.")
        assert "text_embedding" in features
        assert "token_features" in features

    def test_extract_temporal_features(self):
        timestamps = [0.0, 1.0, 2.0, 3.0]
        durations = [1.0, 1.0, 1.0, 1.0]
        features = FeatureExtractor.extract_temporal_features(timestamps, durations)

        assert "temporal_stats" in features
        assert "temporal_histogram" in features
        assert len(features["temporal_stats"]) == 4

    def test_extract_temporal_features_empty(self):
        features = FeatureExtractor.extract_temporal_features([], [])
        assert "temporal_stats" in features
        assert np.all(features["temporal_stats"] == 0)

    def test_compute_similarity_matrix_cosine(self):
        f1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        f2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = FeatureExtractor.compute_similarity_matrix(f1, f2, metric="cosine")

        assert sim.shape == (2, 2)
        assert sim[0, 0] == pytest.approx(1.0, abs=0.01)
        assert sim[1, 1] == pytest.approx(1.0, abs=0.01)

    def test_compute_similarity_matrix_euclidean(self):
        f1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        f2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = FeatureExtractor.compute_similarity_matrix(f1, f2, metric="euclidean")

        assert sim.shape == (2, 2)
        assert sim[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_compute_similarity_matrix_unknown_metric(self):
        f1 = np.array([[1.0, 0.0]])
        f2 = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown metric"):
            FeatureExtractor.compute_similarity_matrix(f1, f2, metric="unknown")
