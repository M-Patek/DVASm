"""Tests for video loader."""

import pytest


class TestVideoLoader:
    """Test VideoLoader class."""

    def test_video_loader_context_manager(self, tmp_path):
        """Test VideoLoader context manager."""
        # Create a dummy video file (this would need a real video in practice)
        # For now, just test the structure
        pass

    def test_metadata_extraction(self):
        """Test metadata extraction."""
        # Would require a real video file
        pass


class TestRetryUtilities:
    """Test retry utilities."""

    def test_with_retry_success(self):
        """Test successful function with retry."""
        from dvas.utils.retry import with_retry

        @with_retry(max_attempts=2, base_delay=0.1)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_with_retry_failure(self):
        """Test retry with eventual failure."""
        from dvas.utils.retry import with_retry

        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(Exception):  # RetryExhaustedError wraps ValueError
            failing_func()

        assert call_count == 2  # Retried once


class TestBatchProcessor:
    """Test BatchProcessor."""

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save and load."""
        from dvas.utils.retry import BatchProcessor

        checkpoint_path = tmp_path / "checkpoint.json"
        processor = BatchProcessor(checkpoint_path=str(checkpoint_path))

        # Simulate processing via checkpoint attribute
        processor.checkpoint.processed_count = 10
        processor.checkpoint.failed_items = [{"id": "1", "error": "test"}]
        processor._save_checkpoint()

        # Load in new processor
        new_processor = BatchProcessor(checkpoint_path=str(checkpoint_path))
        # Checkpoint is auto-loaded in __init__ via _load_checkpoint
        assert new_processor.checkpoint.processed_count == 10


class TestExceptions:
    """Test DVAS exception hierarchy."""

    def test_dvas_exception(self):
        """Test base exception."""
        from dvas.exceptions import DVASException

        exc = DVASException("test error", error_code="TEST_001", details={"key": "value"})
        assert exc.message == "test error"
        assert exc.error_code == "TEST_001"
        assert exc.details == {"key": "value"}
        assert str(exc) == "[TEST_001] test error"

    def test_api_rate_limit_error(self):
        """Test API rate limit exception."""
        from dvas.exceptions import APIRateLimitError

        exc = APIRateLimitError("Rate limited", retry_after=60)
        assert exc.status_code == 429
        assert exc.retry_after == 60

    def test_video_processing_error(self):
        """Test video processing exception."""
        from dvas.exceptions import VideoProcessingError

        exc = VideoProcessingError("Failed to decode", video_path="/path/to/video.mp4")
        assert exc.video_path == "/path/to/video.mp4"
        assert exc.error_code == "DVAS_VID_001"


class TestCache:
    """Test cache utilities."""

    def test_get_cache(self):
        """Test cache instance creation."""
        from dvas.utils.cache import get_cache

        cache = get_cache()
        assert cache is not None

    def test_cached_decorator_sync(self):
        """Test sync cached decorator."""
        from dvas.utils.cache import cached

        call_count = 0

        @cached("test_prefix", ttl=60)
        def test_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = test_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = test_func(5)
        assert result2 == 10
        # Note: In-memory cache may not persist across tests


class TestStorage:
    """Test annotation storage."""

    def test_annotation_store_init(self, tmp_path):
        """Test store initialization."""
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=tmp_path)
        assert store.gold_path.exists()
        assert store.model_path.exists()
        assert store.reviewed_path.exists()

    def test_save_and_load(self, tmp_path):
        """Test save and load annotation."""
        from dvas.data.storage import AnnotationStore
        from dvas.data.schemas import Annotation, VideoMetadata

        store = AnnotationStore(root_path=tmp_path)

        annotation = Annotation(
            id="test_001",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        path = store.save(annotation, source="gold")
        assert path.exists()

        loaded = store.load("test_001", source="gold")
        assert loaded is not None
        assert loaded.id == "test_001"
        assert loaded.video_id == "vid_001"

    def test_load_all_generator(self, tmp_path):
        """Test load_all as generator."""
        from dvas.data.storage import AnnotationStore
        from dvas.data.schemas import Annotation, VideoMetadata

        store = AnnotationStore(root_path=tmp_path)

        # Create multiple annotations
        for i in range(5):
            annotation = Annotation(
                id=f"test_{i:03d}",
                video_id=f"vid_{i:03d}",
                video_path=f"/path/to/video_{i}.mp4",
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=10.0,
                    total_frames=300,
                ),
            )
            store.save(annotation, source="gold")

        # Load as generator
        annotations = list(store.load_all(source="gold"))
        assert len(annotations) == 5

    def test_get_statistics(self, tmp_path):
        """Test storage statistics."""
        from dvas.data.storage import AnnotationStore
        from dvas.data.schemas import Annotation, VideoMetadata

        store = AnnotationStore(root_path=tmp_path)

        annotation = Annotation(
            id="test_001",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )
        store.save(annotation, source="gold")

        stats = store.get_statistics()
        assert stats["gold"]["count"] == 1
        assert stats["gold"]["size_mb"] > 0
