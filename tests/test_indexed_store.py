"""Tests for indexed annotation store.

Uses in-memory SQLite for testing.
"""

import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import Annotation, Segment, VideoMetadata
from dvas.persistence.indexed_store import (
    AnnotationIndex,
    AnnotationQuery,
    IndexStore,
    IndexStoreConfig,
)


class TestIndexedStore:
    """Test IndexStore with in-memory database."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        config = IndexStoreConfig(db_path=db_path)
        store = IndexStore(config)
        store.create_index()  # Initialize tables

        yield store, db_path

        # Cleanup - close store first, then remove file
        try:
            store.close()
        except Exception:
            pass

        # On Windows, SQLite files may still be locked
        # Use retry logic or ignore deletion errors
        import time

        for _ in range(5):
            try:
                db_path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.1)

    @pytest.fixture
    def sample_annotation(self):
        """Create sample annotation."""
        return Annotation(
            id="ann_test_001",
            video_id="vid_001",
            video_path="/v/001.mp4",
            segments=[Segment(start_time=0.0, end_time=5.0, caption="Test action")],
            metadata=VideoMetadata(
                video_id="vid_001",
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
            source="teacher",
            model_version="gpt-5.5",
            quality_score=0.85,
            tags=["cooking", "kitchen"],
        )

    def test_store_initialization(self, temp_db):
        """Test store initialization."""
        store, db_path = temp_db
        assert store is not None
        assert store.config.db_path == db_path

    def test_index_annotation(self, temp_db, sample_annotation):
        """Test indexing annotation into store."""
        store, db_path = temp_db

        store.index_annotation(sample_annotation)

        # Verify by querying
        results, total = store.query(AnnotationQuery(video_id="vid_001"))
        assert len(results) == 1
        assert results[0].video_id == "vid_001"

    def test_update_annotation(self, temp_db, sample_annotation):
        """Test updating existing annotation."""
        store, db_path = temp_db

        # Index original
        store.index_annotation(sample_annotation)

        # Update with new data
        updated = sample_annotation.model_copy()
        updated.quality_score = 0.95

        store.index_annotation(updated)  # Should update existing

        # Verify update
        results, total = store.query(AnnotationQuery(video_id="vid_001"))
        assert len(results) == 1
        assert results[0].quality_score == 0.95

    def test_delete_annotation(self, temp_db, sample_annotation):
        """Test deleting annotation."""
        store, db_path = temp_db

        store.index_annotation(sample_annotation)
        store.delete("ann_test_001")

        # Verify deletion
        results, total = store.query(AnnotationQuery(video_id="vid_001"))
        assert len(results) == 0

    def test_query_by_source(self, temp_db, sample_annotation):
        """Test querying by source filter."""
        store, db_path = temp_db

        # Index teacher annotation
        store.index_annotation(sample_annotation)

        # Index student annotation
        student_ann = sample_annotation.model_copy()
        student_ann.id = "ann_student"
        student_ann.source = "student"
        store.index_annotation(student_ann)

        # Query teacher only - returns tuple (results, total_count)
        results, total = store.query(AnnotationQuery(source="teacher"))
        assert len(results) == 1
        assert results[0].source == "teacher"

    def test_query_by_quality(self, temp_db, sample_annotation):
        """Test querying by quality score range."""
        store, db_path = temp_db

        store.index_annotation(sample_annotation)

        # Query with quality filter - returns tuple (results, total_count)
        results, total = store.query(AnnotationQuery(min_quality=0.8, max_quality=0.9))
        assert len(results) == 1

        # Query outside range
        results, total = store.query(AnnotationQuery(min_quality=0.9))
        assert len(results) == 0

    def test_get_statistics(self, temp_db, sample_annotation):
        """Test getting store statistics."""
        store, db_path = temp_db

        store.index_annotation(sample_annotation)

        stats = store.get_statistics()
        assert stats["total_annotations"] == 1
        assert "by_source" in stats

    def test_get_sync_history(self, temp_db, sample_annotation):
        """Test getting sync history."""
        store, db_path = temp_db

        # Index annotation
        store.index_annotation(sample_annotation)

        # Get sync history
        history = store.get_sync_history(limit=10)
        assert isinstance(history, list)

    def test_create_version(self, temp_db, sample_annotation):
        """Test creating named version."""
        store, db_path = temp_db

        store.index_annotation(sample_annotation)
        version = store.create_version("v1.0", "Initial version")

        assert version is not None
        assert version.name == "v1.0"
        assert version.description == "Initial version"


class TestAnnotationQuery:
    """Test query builder."""

    def test_empty_query(self):
        """Test query with no filters."""
        query = AnnotationQuery()
        assert query.video_id is None
        assert query.source is None

    def test_query_with_filters(self):
        """Test query with multiple filters."""
        query = AnnotationQuery(
            video_id="vid_123",
            source="teacher",
            min_quality=0.8,
            tags=["cooking"],
        )
        assert query.video_id == "vid_123"
        assert query.source == "teacher"
        assert query.min_quality == 0.8


class TestAnnotationIndex:
    """Test index data class."""

    def test_index_creation(self):
        """Test creating annotation index."""
        index = AnnotationIndex(
            id="ann_001",
            video_id="vid_001",
            video_path="/v/1.mp4",
            source="teacher",
            model_version="gpt-5.5",
            quality_score=0.85,
            created_at="2024-01-01T00:00:00",
            updated_at=None,
            num_segments=5,
            total_duration=25.0,
            tags='["cooking"]',
        )
        assert index.id == "ann_001"
        assert index.num_segments == 5
