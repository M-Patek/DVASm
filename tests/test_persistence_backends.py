"""Tests for persistence backends."""

import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.persistence.backends import (
    LocalFSBackend,
    LocalFSConfig,
    SQLiteBackend,
    SQLiteConfig,
)


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


class TestLocalFSBackend:
    """Tests for LocalFS storage backend."""

    def test_init(self):
        """Test backend initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            assert backend.root_path == Path(tmpdir)

    def test_open_close(self):
        """Test backend open/close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()
            assert not backend._closed
            backend.close()
            assert backend._closed

    def test_health_check(self):
        """Test health check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()
            healthy, msg = backend.health_check()
            assert healthy is True
            assert msg == "healthy"

    def test_save_and_load(self, sample_annotation):
        """Test save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()

            path = backend.save(sample_annotation, source="model")
            assert path is not None

            loaded = backend.load(sample_annotation.id, source="model")
            assert loaded is not None
            assert loaded.id == sample_annotation.id
            assert loaded.video_id == sample_annotation.video_id

    def test_load_all(self, sample_annotation):
        """Test load_all generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()

            backend.save(sample_annotation, source="model")

            annotations = list(backend.load_all(source="model"))
            assert len(annotations) == 1
            assert annotations[0].id == sample_annotation.id

    def test_delete(self, sample_annotation):
        """Test delete operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()

            backend.save(sample_annotation, source="model")
            assert backend.exists(sample_annotation.id) is True

            backend.delete(sample_annotation.id)
            assert backend.exists(sample_annotation.id) is False

    def test_exists(self, sample_annotation):
        """Test exists check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()

            assert backend.exists(sample_annotation.id) is False
            backend.save(sample_annotation, source="model")
            assert backend.exists(sample_annotation.id) is True

    def test_get_statistics(self, sample_annotation):
        """Test statistics gathering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LocalFSConfig(root_path=Path(tmpdir))
            backend = LocalFSBackend(config)
            backend.open()

            stats = backend.get_statistics()
            assert stats.total_annotations == 0

            backend.save(sample_annotation, source="model")
            stats = backend.get_statistics()
            assert stats.total_annotations == 1
            assert "model" in stats.by_source


class TestSQLiteBackend:
    """Tests for SQLite metadata backend."""

    def test_init(self):
        """Test backend initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            assert backend.config.db_path == Path(tmpdir) / "test.db"

    def test_open_close(self):
        """Test backend open/close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            backend.open()
            assert not backend._closed
            backend.close()
            assert backend._closed

    def test_index_and_get(self, sample_annotation):
        """Test index and get operations."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)

                entry = backend.get(sample_annotation.id)
                assert entry is not None
                assert entry.id == sample_annotation.id
                assert entry.video_id == sample_annotation.video_id
            finally:
                backend.close()

    def test_query(self, sample_annotation):
        """Test query operation."""
        from dvas.persistence.backends.base import QueryFilter

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)

                query = QueryFilter(video_id=sample_annotation.video_id)
                results, total = backend.query(query)
                assert total == 1
                assert len(results) == 1
                assert results[0].video_id == sample_annotation.video_id
            finally:
                backend.close()

    def test_search(self, sample_annotation):
        """Test full-text search."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(
                db_path=Path(tmpdir) / "test.db",
                enable_fts=True,
            )
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)

                results = backend.search(sample_annotation.video_id)
                assert len(results) > 0
            finally:
                backend.close()

    def test_versioning(self, sample_annotation):
        """Test version operations."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(
                db_path=Path(tmpdir) / "test.db",
                enable_versioning=True,
            )
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)

                version = backend.create_version("test-version")
                assert version is not None
                assert version.name == "test-version"

                versions = backend.list_versions()
                assert len(versions) == 1

                restored = backend.restore_version(version.id)
                assert len(restored) == 1
            finally:
                backend.close()

    def test_delete_index(self, sample_annotation):
        """Test delete from index."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)
                assert backend.get(sample_annotation.id) is not None

                backend.delete_index(sample_annotation.id)
                assert backend.get(sample_annotation.id) is None
            finally:
                backend.close()

    def test_hash_index(self, sample_annotation):
        """Test video and frame hash indexing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = SQLiteConfig(db_path=Path(tmpdir) / "test.db")
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_video_hash(
                    video_id="video-001",
                    video_path="/path/to/video.mp4",
                    content_hash="abc123",
                    frame_count=300,
                    duration=10.0,
                )

                result = backend.get_video_by_hash("abc123")
                assert result is not None
                assert result["video_id"] == "video-001"

                backend.index_frame_hash("video-001", 0, "frame_hash_001")
                frames = backend.get_frames_by_hash("frame_hash_001")
                assert len(frames) == 1
                assert frames[0]["frame_index"] == 0
            finally:
                backend.close()

    def test_backup_restore(self, sample_annotation):
        """Test backup and restore."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLiteConfig(db_path=db_path)
            backend = SQLiteBackend(config)
            backend.open()

            try:
                backend.index_annotation(sample_annotation)

                # Backup
                backup_path = Path(tmpdir) / "backups"
                backup_file = backend.backup(backup_path)
                assert backup_file.exists()

                backend.close()

                # Restore
                config2 = SQLiteConfig(db_path=db_path)
                backend2 = SQLiteBackend(config2)
                backend2.open()
                backend2.restore(backup_file)

                entry = backend2.get(sample_annotation.id)
                assert entry is not None
            finally:
                backend2.close()
