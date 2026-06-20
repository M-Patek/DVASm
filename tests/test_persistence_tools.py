"""Tests for persistence tools (diff, rollback, backup, compaction)."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.persistence import (
    BackupManager,
    CompactionManager,
    IndexManager,
    RollbackManager,
    compute_annotation_diff,
)
from dvas.persistence.backends import LocalFSBackend, LocalFSConfig, SQLiteBackend, SQLiteConfig


@pytest.fixture
def sample_annotation():
    """Create a sample annotation."""
    return Annotation(
        id="test-001",
        video_id="video-001",
        video_path="/path/to/video.mp4",
        source="teacher",
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.0,
            total_frames=300,
        ),
    )


@pytest.fixture
def modified_annotation():
    """Create a modified version of sample annotation."""
    return Annotation(
        id="test-001",
        video_id="video-001",
        video_path="/path/to/video.mp4",
        source="student",  # Changed source
        quality_score=0.95,  # Added quality score
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.0,
            total_frames=300,
        ),
    )


class TestDiffTool:
    """Tests for diff tool."""

    def test_compute_diff_same_annotation(self, sample_annotation):
        """Test diff of identical annotations."""
        diff = compute_annotation_diff(sample_annotation, sample_annotation)
        assert diff.unchanged is True
        assert len(diff.field_changes) == 0

    def test_compute_diff_modified(self, sample_annotation, modified_annotation):
        """Test diff of modified annotation."""
        diff = compute_annotation_diff(sample_annotation, modified_annotation)
        assert diff.unchanged is False
        assert len(diff.field_changes) > 0

        # Check source change
        source_change = next((c for c in diff.field_changes if c.field == "source"), None)
        assert source_change is not None
        assert source_change.old_value == "teacher"
        assert source_change.new_value == "student"

    def test_diff_different_ids_raises(self):
        """Test diff of annotations with different IDs raises."""
        ann1 = Annotation(
            id="test-001",
            video_id="video-001",
            video_path="/path",
            metadata=VideoMetadata(fps=30, resolution=[1920, 1080], duration=10, total_frames=300),
        )
        ann2 = Annotation(
            id="test-002",
            video_id="video-001",
            video_path="/path",
            metadata=VideoMetadata(fps=30, resolution=[1920, 1080], duration=10, total_frames=300),
        )
        with pytest.raises(ValueError):
            compute_annotation_diff(ann1, ann2)


class TestRollbackManager:
    """Tests for rollback manager."""

    def test_rollback_to_version(self, sample_annotation):
        """Test rollback to version."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Setup backends
            storage_config = LocalFSConfig(root_path=Path(tmpdir) / "storage")
            storage = LocalFSBackend(storage_config)
            storage.open()

            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                # Create version - use teacher source to match sample_annotation
                storage.save(sample_annotation, source="teacher")
                metadata.index_annotation(sample_annotation)
                version = metadata.create_version("v1.0")

                # Modify annotation
                modified = sample_annotation.model_copy(update={"source": "student"})
                storage.save(modified, source="student", overwrite=True)
                metadata.index_annotation(modified)

                # Rollback
                manager = RollbackManager(metadata, storage)
                stats = manager.rollback_to_version(version.id)

                # Verify rollback was performed
                assert stats["annotations_restored"] == 1
                assert len(stats["errors"]) == 0
            finally:
                storage.close()
                metadata.close()

    def test_preview_rollback(self, sample_annotation):
        """Test preview rollback."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                metadata.index_annotation(sample_annotation)
                version = metadata.create_version("v1.0")

                manager = RollbackManager(metadata)
                preview = manager.preview_rollback(version.id)

                assert preview["annotations_restored"] == 1
            finally:
                metadata.close()


class TestBackupManager:
    """Tests for backup manager."""

    def test_create_and_list_backups(self, sample_annotation):
        """Test backup creation and listing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            storage_config = LocalFSConfig(root_path=Path(tmpdir) / "storage")
            storage = LocalFSBackend(storage_config)
            storage.open()

            try:
                storage.save(sample_annotation, source="model")

                backup_root = Path(tmpdir) / "backups"
                manager = BackupManager(backup_root)

                backup_path = manager.create_backup(storage_backend=storage, name="test-backup")
                assert backup_path.exists()

                backups = manager.list_backups()
                assert len(backups) == 1
                # Name might include .tar extension
                assert "test-backup" in backups[0]["name"]
            finally:
                storage.close()

    def test_restore_backup(self, sample_annotation):
        """Test backup restore."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Create original storage
            storage_config = LocalFSConfig(root_path=Path(tmpdir) / "storage")
            storage = LocalFSBackend(storage_config)
            storage.open()

            try:
                storage.save(sample_annotation, source="model")

                # Create backup
                backup_root = Path(tmpdir) / "backups"
                manager = BackupManager(backup_root)
                backup_path = manager.create_backup(storage_backend=storage, name="test-backup")

                # Create new storage and restore
                restore_config = LocalFSConfig(root_path=Path(tmpdir) / "restore")
                restore_storage = LocalFSBackend(restore_config)
                restore_storage.open()

                try:
                    stats = manager.restore_backup(backup_path, storage_backend=restore_storage)
                    assert "storage" in stats["components"]
                finally:
                    restore_storage.close()
            finally:
                storage.close()


class TestCompactionManager:
    """Tests for compaction manager."""

    def test_compact_no_op_on_clean_storage(self, sample_annotation):
        """Test compaction on clean storage."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            storage_config = LocalFSConfig(root_path=Path(tmpdir) / "storage")
            storage = LocalFSBackend(storage_config)
            storage.open()

            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                storage.save(sample_annotation, source="model")
                metadata.index_annotation(sample_annotation)

                manager = CompactionManager(storage, metadata)
                stats = manager.compact()

                assert stats["duplicates_found"] == 0
                assert stats["duplicates_removed"] == 0
            finally:
                storage.close()
                metadata.close()

    def test_rebuild_index(self, sample_annotation):
        """Test index rebuild."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            storage_config = LocalFSConfig(root_path=Path(tmpdir) / "storage")
            storage = LocalFSBackend(storage_config)
            storage.open()

            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                storage.save(sample_annotation, source="model")

                manager = CompactionManager(storage, metadata)
                stats = manager.rebuild_index()

                assert stats["annotations_processed"] == 1
                assert stats["annotations_indexed"] == 1
            finally:
                storage.close()
                metadata.close()


class TestIndexManager:
    """Tests for index manager."""

    def test_index_manager_annotation(self, sample_annotation):
        """Test index manager annotation indexing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                manager = IndexManager(metadata)
                manager.index_annotation(sample_annotation)

                entry = manager.get_annotation(sample_annotation.id)
                assert entry is not None
                assert entry["id"] == sample_annotation.id
            finally:
                metadata.close()

    def test_video_hash_index(self):
        """Test video hash indexing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                manager = IndexManager(metadata)
                manager.index_video_hash(
                    video_id="video-001",
                    video_path="/path/to/video.mp4",
                    content_hash="abc123",
                )

                result = manager.find_video_by_hash("abc123")
                assert result is not None
                assert result["video_id"] == "video-001"
            finally:
                metadata.close()

    def test_is_video_duplicate(self):
        """Test duplicate video detection."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            meta_config = SQLiteConfig(db_path=Path(tmpdir) / "metadata.db")
            metadata = SQLiteBackend(meta_config)
            metadata.open()

            try:
                manager = IndexManager(metadata)

                assert manager.is_video_duplicate("abc123") is False

                manager.index_video_hash(
                    video_id="video-001",
                    video_path="/path/to/video.mp4",
                    content_hash="abc123",
                )

                assert manager.is_video_duplicate("abc123") is True
            finally:
                metadata.close()
