"""Tests for Parquet-backed annotation storage."""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.data.schemas import Annotation
from dvas.data.storage_parquet import (
    ParquetAnnotationStore,
    ParquetStoreConfig,
    SemanticSearchResult,
)


class TestParquetStoreConfig:
    """Test ParquetStoreConfig dataclass."""

    def test_default_config(self):
        config = ParquetStoreConfig()
        assert config.partition_by == "source"
        assert config.row_group_size == 10000
        assert config.compression == "zstd"
        assert config.enable_duckdb is True
        assert config.enable_semantic_search is False
        assert config.embedding_dim == 768

    def test_custom_config(self):
        config = ParquetStoreConfig(
            partition_by="date",
            compression="snappy",
            row_group_size=5000,
        )
        assert config.partition_by == "date"
        assert config.compression == "snappy"
        assert config.row_group_size == 5000


class TestParquetAnnotationStore:
    """Test ParquetAnnotationStore."""

    def _create_test_annotation(self, annotation_id="test-001", source="student", video_id="vid-001"):
        """Create a test annotation."""
        from dvas.data.schemas import VideoMetadata

        return Annotation(
            id=annotation_id,
            video_id=video_id,
            video_path="/tmp/test.mp4",
            source=source,
            model_version="v1.0",
            quality_score=0.85,
            created_at=datetime.now(timezone.utc),
            segments=[],
            tags=["test"],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=60.0,
                total_frames=1800,
            ),
        )

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)
            assert store.root_path.exists()
            assert store.config.compression == "zstd"

    def test_annotation_to_dict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)
            annotation = self._create_test_annotation()

            data = store._annotation_to_dict(annotation)

            assert data["id"] == "test-001"
            assert data["video_id"] == "vid-001"
            assert data["source"] == "student"
            assert data["model_version"] == "v1.0"
            assert data["quality_score"] == 0.85
            assert data["num_segments"] == 0
            assert "json_data" in data

    def test_get_partition_path_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir), partition_by="source")
            store = ParquetAnnotationStore(config)
            annotation = self._create_test_annotation(source="teacher")

            path = store._get_partition_path(annotation)
            assert path == Path(tmp_dir) / "teacher"

    def test_get_partition_path_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir), partition_by="date")
            store = ParquetAnnotationStore(config)
            annotation = self._create_test_annotation()

            path = store._get_partition_path(annotation)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            assert path == Path(tmp_dir) / today

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)
            annotation = self._create_test_annotation("save-test-001")

            # Save
            path = store.save(annotation)
            assert path.exists()

            # Load
            loaded = store.load("save-test-001")
            assert loaded is not None
            assert loaded.id == "save-test-001"
            assert loaded.video_id == "vid-001"

    def test_load_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            result = store.load("nonexistent")
            assert result is None

    def test_save_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            annotations = [
                self._create_test_annotation(f"batch-{i}")
                for i in range(5)
            ]

            paths = store.save_batch(annotations)
            assert len(paths) > 0

            # Verify all can be loaded
            for i in range(5):
                loaded = store.load(f"batch-{i}")
                assert loaded is not None
                assert loaded.id == f"batch-{i}"

    def test_save_batch_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            paths = store.save_batch([])
            assert paths == []

    def test_query_by_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            # Save annotations with different sources
            store.save(self._create_test_annotation("q1", source="student"))
            store.save(self._create_test_annotation("q2", source="teacher"))
            store.save(self._create_test_annotation("q3", source="student"))

            # Query by source
            results, total = store.query(source="student", limit=10)
            assert len(results) == 2
            assert total == 2

    def test_query_by_video_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            annotation = self._create_test_annotation("v1", video_id="special-vid")
            store.save(annotation)

            results, total = store.query(video_id="special-vid", limit=10)
            assert len(results) == 1
            assert results[0].video_id == "special-vid"

    def test_query_by_quality(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            low_quality = self._create_test_annotation("low")
            low_quality.quality_score = 0.3
            high_quality = self._create_test_annotation("high")
            high_quality.quality_score = 0.9

            store.save(low_quality)
            store.save(high_quality)

            results, total = store.query(min_quality=0.5, limit=10)
            assert len(results) == 1
            assert results[0].id == "high"

    def test_get_columnar_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            # Save some annotations
            for i in range(3):
                store.save(self._create_test_annotation(f"stat-{i}"))

            stats = store.get_columnar_stats()
            assert stats["total_files"] > 0
            assert stats["total_rows"] == 3
            assert stats["total_size_mb"] > 0

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)

            stats = store.get_statistics()
            assert stats["store_type"] == "parquet"
            assert stats["compression"] == "zstd"
            assert "partition_by" in stats

    def test_close(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(root_path=Path(tmp_dir))
            store = ParquetAnnotationStore(config)
            store.close()
            assert store._duckdb_conn is None

    def test_pyarrow_not_available(self):
        with patch("dvas.data.storage_parquet.PYARROW_AVAILABLE", False):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ParquetStoreConfig(root_path=Path(tmp_dir))
                store = ParquetAnnotationStore(config)
                annotation = self._create_test_annotation()

                with pytest.raises(RuntimeError, match="pyarrow is required"):
                    store.save(annotation)


class TestSemanticSearchResult:
    """Test SemanticSearchResult dataclass."""

    def test_creation(self):
        result = SemanticSearchResult(
            annotation_id="test-001",
            video_id="vid-001",
            score=0.95,
        )
        assert result.annotation_id == "test-001"
        assert result.video_id == "vid-001"
        assert result.score == 0.95
        assert result.embedding is None

    def test_with_embedding(self):
        result = SemanticSearchResult(
            annotation_id="test-002",
            video_id="vid-002",
            score=0.88,
            embedding=[0.1, 0.2, 0.3],
        )
        assert result.embedding == [0.1, 0.2, 0.3]


class TestParquetAnnotationStoreSemanticSearch:
    """Test semantic search methods."""

    def test_semantic_search_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(
                root_path=Path(tmp_dir),
                enable_semantic_search=False,
            )
            store = ParquetAnnotationStore(config)

            results = store.semantic_search([0.1, 0.2, 0.3], top_k=5)
            assert results == []

    def test_semantic_search_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ParquetStoreConfig(
                root_path=Path(tmp_dir),
                enable_semantic_search=True,
            )
            store = ParquetAnnotationStore(config)

            results = store.semantic_search([0.1, 0.2, 0.3], top_k=5)
            assert results == []
