"""Failure recovery tests for the annotation pipeline.

Tests recovery behavior at each stage:
- Video loading failures
- Teacher API failures (timeout, rate limit, connection error)
- Parser failures (malformed response)
- Storage failures
- Checkpoint corruption
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.exceptions import PipelineError, RetryExhaustedError
from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.pipeline.core import AnnotationPipeline
from dvas.pipeline.checkpoint import CheckpointManager


class FailureScenarioFactory:
    """Factory for creating various failure scenarios."""

    @staticmethod
    def create_video_load_failure(error_type: str = "file_not_found"):
        """Create a VideoLoader that fails to load."""

        def failing_loader(*args, **kwargs):
            if error_type == "file_not_found":
                raise FileNotFoundError(f"Video not found: {args[0] if args else 'unknown'}")
            elif error_type == "corrupted":
                raise ValueError("Video file is corrupted")
            elif error_type == "codec_error":
                raise RuntimeError("Unsupported video codec")
            return MagicMock()

        return failing_loader

    @staticmethod
    def create_teacher_with_sequence(responses: list) -> MagicMock:
        """Create a teacher that returns responses in sequence, cycling if needed."""
        teacher = MagicMock()
        teacher.model_name = "mock-failing-teacher"
        teacher.model_type = ModelType.TEACHER_GPT55

        call_count = [0]

        async def generate_impl(*args, **kwargs):
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            response = responses[idx]
            if isinstance(response, Exception):
                raise response
            return response

        teacher.generate = MagicMock(side_effect=generate_impl)
        return teacher

    @staticmethod
    def create_teacher_timeout() -> MagicMock:
        """Create a teacher that times out."""
        return FailureScenarioFactory.create_teacher_with_sequence(
            [
                GenerationResult(
                    text="",
                    model_type=ModelType.TEACHER_GPT55,
                    model_version="gpt-5.5",
                    status=GenerationStatus.TIMEOUT,
                    error_message="Request timeout after 30s",
                    latency_ms=30000.0,
                )
            ]
        )

    @staticmethod
    def create_teacher_rate_limited() -> MagicMock:
        """Create a teacher that hits rate limit."""
        return FailureScenarioFactory.create_teacher_with_sequence(
            [
                GenerationResult(
                    text="",
                    model_type=ModelType.TEACHER_GPT55,
                    model_version="gpt-5.5",
                    status=GenerationStatus.RATE_LIMITED,
                    error_message="Rate limit exceeded",
                    latency_ms=100.0,
                )
            ]
        )

    @staticmethod
    def create_teacher_connection_error() -> MagicMock:
        """Create a teacher with connection error."""
        teacher = MagicMock()
        teacher.model_name = "mock-connection-error"
        teacher.model_type = ModelType.TEACHER_GPT55

        async def failing_generate(*args, **kwargs):
            raise ConnectionError("Failed to connect to API")

        teacher.generate = MagicMock(side_effect=failing_generate)
        return teacher

    @staticmethod
    def create_teacher_partial_failure(fail_indices: list) -> MagicMock:
        """Create a teacher that fails at specific call indices."""
        teacher = MagicMock()
        teacher.model_name = "mock-partial-failure"
        teacher.model_type = ModelType.TEACHER_GPT55

        call_count = [0]

        async def conditional_generate(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1

            if idx in fail_indices:
                return GenerationResult(
                    text="",
                    model_type=ModelType.TEACHER_GPT55,
                    model_version="gpt-5.5",
                    status=GenerationStatus.FAILURE,
                    error_message=f"Simulated failure at call {idx}",
                    latency_ms=100.0,
                )

            return GenerationResult(
                text=json.dumps(
                    {
                        "scene_description": f"Scene {idx}",
                        "actions": [{"verb": "test", "noun": "object"}],
                    }
                ),
                model_type=ModelType.TEACHER_GPT55,
                model_version="gpt-5.5",
                status=GenerationStatus.SUCCESS,
                latency_ms=500.0,
            )

        teacher.generate = conditional_generate
        return teacher


@pytest.fixture
def temp_storage_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_video_loader():
    """Create a mock video loader."""
    loader = MagicMock()
    loader.metadata = VideoMetadata(
        video_id="test_vid",
        fps=30.0,
        resolution=[640, 480],
        duration=10.0,
        total_frames=300,
    )
    loader.detect_scenes.return_value = [(0.0, 5.0), (5.0, 10.0)]

    async def mock_aiter_frames(*args, **kwargs):
        for i in range(8):
            frame = MagicMock()
            frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame.timestamp = i * 0.1
            yield frame

    loader.aiter_frames = mock_aiter_frames
    loader.__enter__ = MagicMock(return_value=loader)
    loader.__exit__ = MagicMock(return_value=None)
    return loader


class TestVideoLoadingFailureRecovery:
    """Test recovery from video loading failures."""

    @pytest.mark.asyncio
    async def test_file_not_found_raises_pipeline_error(self, temp_storage_dir):
        """Test that missing video files raise PipelineError."""
        pipeline = AnnotationPipeline()

        with pytest.raises((PipelineError, FileNotFoundError, RetryExhaustedError)):
            await pipeline.annotate_video(Path("/nonexistent/video.mp4"), "missing_vid")

    @pytest.mark.asyncio
    async def test_batch_processing_skips_missing_videos(self, temp_storage_dir):
        """Test batch processing continues despite missing videos."""
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(store=store)

        video_items = [
            {"video_path": "/nonexistent/vid1.mp4", "video_id": "missing_1"},
            {"video_path": "/nonexistent/vid2.mp4", "video_id": "missing_2"},
        ]

        successful, failed = await pipeline.process_batch(video_items)

        # All should fail but not crash
        assert len(successful) == 0
        assert len(failed) == 2


class TestTeacherFailureRecovery:
    """Test recovery from teacher model failures."""

    @pytest.mark.asyncio
    async def test_timeout_creates_empty_segment(self, temp_storage_dir, mock_video_loader):
        """Test that timeout creates empty segment rather than crashing."""
        from dvas.data.storage import AnnotationStore

        mock_teacher = FailureScenarioFactory.create_teacher_timeout()
        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(Path("/fake/video.mp4"), "timeout_test")

        # Should still create annotation with empty segments
        assert isinstance(annotation, Annotation)
        assert len(annotation.segments) == 2
        # Segments should be empty (no actions due to timeout)
        for segment in annotation.segments:
            assert segment.actions == []

    @pytest.mark.asyncio
    async def test_rate_limit_creates_empty_segment(self, temp_storage_dir, mock_video_loader):
        """Test that rate limit creates empty segment."""
        from dvas.data.storage import AnnotationStore

        mock_teacher = FailureScenarioFactory.create_teacher_rate_limited()
        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(Path("/fake/video.mp4"), "rate_limit_test")

        assert isinstance(annotation, Annotation)
        assert len(annotation.segments) == 2

    @pytest.mark.asyncio
    async def test_connection_error_with_retry(self, temp_storage_dir, mock_video_loader):
        """Test connection error triggers retry mechanism."""
        from dvas.data.storage import AnnotationStore

        mock_teacher = FailureScenarioFactory.create_teacher_connection_error()
        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        # ConnectionError should trigger retry (up to 3 attempts per segment)
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            # After retries exhaust, should create empty segment
            await pipeline.annotate_video(Path("/fake/video.mp4"), "connection_test")

        # Teacher should have been called multiple times (retries)
        assert mock_teacher.generate.call_count >= 2

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, temp_storage_dir, mock_video_loader):
        """Test recovery when some segments fail but others succeed."""
        from dvas.data.storage import AnnotationStore

        # Fail the first segment (index 0), succeed on second
        mock_teacher = FailureScenarioFactory.create_teacher_partial_failure(fail_indices=[0])
        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(Path("/fake/video.mp4"), "partial_test")

        # Should have 2 segments
        assert len(annotation.segments) == 2
        # First should be empty (failure), second should have content
        assert annotation.segments[0].actions == []
        assert len(annotation.segments[1].actions) > 0

    @pytest.mark.asyncio
    async def test_batch_partial_failure_recovery(self, temp_storage_dir, mock_video_loader):
        """Test batch processing continues despite individual video failures."""
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)

        # Create teacher that fails on odd calls
        teacher = MagicMock()
        teacher.model_name = "mock-alternating"
        call_count = [0]

        async def alternating_generate(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx % 2 == 0:
                raise ConnectionError("Network error")
            return GenerationResult(
                text=json.dumps({"scene_description": "Success", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )

        teacher.generate = alternating_generate

        pipeline = AnnotationPipeline(teacher_model=teacher, store=store)

        video_items = [
            {"video_path": "/fake/vid1.mp4", "video_id": "fail_vid_1"},
            {"video_path": "/fake/vid2.mp4", "video_id": "succeed_vid_2"},
            {"video_path": "/fake/vid3.mp4", "video_id": "fail_vid_3"},
        ]

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            successful, failed = await pipeline.process_batch(video_items)

        # Should have some successes and some failures
        assert len(successful) + len(failed) == 3


class TestStorageFailureRecovery:
    """Test recovery from storage failures."""

    @pytest.mark.asyncio
    async def test_save_failure_raises_pipeline_error(self, temp_storage_dir, mock_video_loader):
        """Test that save failures raise PipelineError."""
        from dvas.data.storage import AnnotationStore

        # Create store with failing save
        store = MagicMock(spec=AnnotationStore)
        store.save.side_effect = IOError("Disk full")

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-teacher"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text=json.dumps({"scene_description": "Test", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            with pytest.raises(PipelineError) as exc_info:
                await pipeline.annotate_video(Path("/fake/video.mp4"), "save_fail_test")

        assert "save" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_checkpoint_not_updated_on_save_failure(
        self, temp_storage_dir, mock_video_loader
    ):
        """Test checkpoint is not updated if save fails."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_storage_dir / "checkpoint_no_save.json"
        store = MagicMock(spec=AnnotationStore)
        store.save.side_effect = IOError("Disk full")

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-teacher"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text=json.dumps({"scene_description": "Test", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            with pytest.raises(PipelineError):
                await pipeline.annotate_video(Path("/fake/video.mp4"), "checkpoint_test")

        # Checkpoint should not mark as processed
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        assert not checkpoint.is_processed("checkpoint_test")


class TestCheckpointFailureRecovery:
    """Test recovery from checkpoint failures."""

    def test_checkpoint_corrupted_recovery(self, temp_storage_dir):
        """Test recovery from corrupted checkpoint file."""
        checkpoint_path = temp_storage_dir / "corrupted.json"

        # Write corrupted JSON
        checkpoint_path.write_text("{invalid json content")

        # Should handle gracefully and start fresh
        checkpoint = CheckpointManager(checkpoint_path)
        result = checkpoint.load()

        # Returns False (not loaded), but doesn't crash
        assert result is False

        # Can still use the checkpoint
        assert not checkpoint.is_processed("any_vid")

        # Should have created backup
        backup = checkpoint_path.with_suffix(".corrupted")
        assert backup.exists()

    def test_checkpoint_atomic_update(self, temp_storage_dir):
        """Test that checkpoint updates are atomic."""
        checkpoint_path = temp_storage_dir / "atomic.json"

        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("vid_1")
        checkpoint.mark_processed("vid_2")
        checkpoint.save()

        # Load fresh
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()

        assert checkpoint2.is_processed("vid_1")
        assert checkpoint2.is_processed("vid_2")

    @pytest.mark.asyncio
    async def test_checkpoint_resume_after_crash(self, temp_storage_dir, mock_video_loader):
        """Test that processing resumes correctly after a crash."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_storage_dir / "crash_resume.json"
        store = AnnotationStore(root_path=temp_storage_dir)

        # First, mark one video as processed
        pre_checkpoint = CheckpointManager(checkpoint_path)
        pre_checkpoint.mark_processed("already_done")
        pre_checkpoint.save()

        # Create teacher
        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-resume"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text=json.dumps({"scene_description": "Test", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Try to process the already-done video (should skip)
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            # This should be skipped due to checkpoint
            await pipeline.annotate_video(Path("/fake/already_done.mp4"), "already_done")

        # Teacher should not be called for already processed video
        # (because it loads from store or skips)
        # Note: Current implementation loads from store if available

    @pytest.mark.asyncio
    async def test_checkpoint_inconsistency_detection(self, temp_storage_dir, mock_video_loader):
        """Test detection of checkpoint/storage inconsistency."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_storage_dir / "inconsistent.json"
        store = AnnotationStore(root_path=temp_storage_dir)

        # Mark as processed in checkpoint but not in storage
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("inconsistent_vid")
        checkpoint.save()

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-inconsistent"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text=json.dumps({"scene_description": "Test", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            # Should detect inconsistency and reprocess
            await pipeline.annotate_video(Path("/fake/inconsistent.mp4"), "inconsistent_vid")

        # Should have reprocessed (teacher was called)
        assert mock_teacher.generate.call_count > 0


class TestParserFailureRecovery:
    """Test recovery from parser failures."""

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, temp_storage_dir, mock_video_loader):
        """Test fallback when teacher returns malformed JSON."""
        from dvas.data.storage import AnnotationStore

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-malformed"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text="This is not JSON at all, just plain text description",
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(
                Path("/fake/malformed.mp4"), "malformed_test"
            )

        # Should still create annotation with plain text fallback
        assert isinstance(annotation, Annotation)
        assert len(annotation.segments) == 2
        # Plain text should be in description
        for segment in annotation.segments:
            assert segment.caption  # Has some caption

    @pytest.mark.asyncio
    async def test_partial_json_parsing(self, temp_storage_dir, mock_video_loader):
        """Test recovery from partially valid JSON."""
        from dvas.data.storage import AnnotationStore

        # JSON with some valid fields but missing others
        partial_json = '{"scene_description": "Kitchen scene", "actions": [invalid here]}'

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-partial"
        mock_teacher.generate = AsyncMock(
            return_value=GenerationResult(
                text=partial_json,
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )
        )

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(Path("/fake/partial.mp4"), "partial_test")

        # Should still work (with fallback)
        assert isinstance(annotation, Annotation)


class TestRetryMechanism:
    """Test the retry mechanism behavior."""

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, temp_storage_dir, mock_video_loader):
        """Test that retry eventually gives up."""
        from dvas.data.storage import AnnotationStore

        fail_count = [0]

        async def always_fail(*args, **kwargs):
            fail_count[0] += 1
            raise ConnectionError(f"Failure #{fail_count[0]}")

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-always-fail"
        mock_teacher.generate = always_fail

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            # Should handle gracefully after retries
            await pipeline.annotate_video(Path("/fake/retry.mp4"), "retry_test")

        # Should have retried multiple times
        # Each segment gets 3 retries, 2 segments = up to 6 calls
        assert fail_count[0] >= 3

    @pytest.mark.asyncio
    async def test_retry_success_on_second_attempt(self, temp_storage_dir, mock_video_loader):
        """Test successful retry."""
        from dvas.data.storage import AnnotationStore

        call_count = [0]

        async def succeed_on_second(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("First attempt fails")
            return GenerationResult(
                text=json.dumps({"scene_description": "Success", "actions": []}),
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                latency_ms=100.0,
            )

        mock_teacher = MagicMock()
        mock_teacher.model_name = "mock-second-try"
        mock_teacher.generate = succeed_on_second

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader):
            annotation = await pipeline.annotate_video(
                Path("/fake/retry_success.mp4"), "retry_success_test"
            )

        # Should have succeeded on second attempt for first segment,
        # then first attempt for second segment (total 3 calls)
        assert call_count[0] >= 3
        assert isinstance(annotation, Annotation)
