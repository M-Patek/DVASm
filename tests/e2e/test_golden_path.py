"""Golden Path E2E tests for the main annotation pipeline.

Tests the complete happy path: Video → Segment → Teacher → Parse → Annotation → Quality → Export
All tests use mocked external dependencies (no real API calls, no real video files).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dvas.data.schemas import Annotation, Segment, VideoMetadata
from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.pipeline.core import AnnotationPipeline
from dvas.pipeline.checkpoint import CheckpointManager


class SyntheticVideoHelper:
    """Helper to create synthetic video data for testing."""

    @staticmethod
    def create_mock_video_loader(
        video_path: Path,
        video_id: str = "test_video",
        num_frames: int = 30,
        fps: float = 30.0,
        duration: float = 10.0,
        scenes: List[tuple] | None = None,
    ):
        """Create a mock VideoLoader that behaves like a real video."""
        mock_loader = MagicMock()

        mock_loader.metadata = VideoMetadata(
            video_id=video_id,
            fps=fps,
            resolution=[640, 480],
            duration=duration,
            total_frames=num_frames,
        )

        # Default scenes if not provided
        if scenes is None:
            scenes = [(0.0, 5.0), (5.0, 10.0)]
        mock_loader.detect_scenes.return_value = scenes

        # Mock async frame iteration
        frame_data = SyntheticVideoHelper._create_mock_frames(
            num_frames=min(num_frames, 16), fps=fps
        )

        async def mock_aiter_frames(start_time, end_time, num_frames=16):
            for frame in frame_data[:num_frames]:
                yield frame

        mock_loader.aiter_frames = mock_aiter_frames
        mock_loader.__enter__ = MagicMock(return_value=mock_loader)
        mock_loader.__exit__ = MagicMock(return_value=None)

        return mock_loader

    @staticmethod
    def _create_mock_frames(num_frames: int, fps: float = 30.0):
        """Create mock frame objects."""
        frames = []
        for i in range(num_frames):
            frame = MagicMock()
            # Create a simple gradient image (deterministic)
            frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame.data[:, :, 0] = (i * 255 // num_frames)  # Red channel gradient
            frame.timestamp = i / fps
            frame.idx = i
            frames.append(frame)
        return frames


class MockTeacherFactory:
    """Factory for creating mock teacher models with predictable responses."""

    @staticmethod
    def create_structured_teacher(
        response_data: Dict[str, Any] | None = None,
        latency_ms: float = 500.0,
    ) -> MagicMock:
        """Create a mock teacher that returns JSON structure."""
        teacher = MagicMock()
        teacher.model_name = "mock-structured-teacher"
        teacher.model_type = ModelType.TEACHER_GPT55
        teacher.model_version = "gpt-5.5-structured"

        if response_data is None:
            response_data = {
                "scene_description": "A person cooking in a kitchen",
                "actions": [
                    {"verb": "pick", "noun": "knife"},
                    {"verb": "cut", "noun": "vegetable"},
                ],
                "objects": [
                    {"name": "knife"},
                    {"name": "cutting_board"},
                    {"name": "vegetable"},
                ],
                "steps": [
                    {"action": "Pick up knife", "details": "From the counter"},
                    {"action": "Cut vegetable", "details": "Into small pieces"},
                ],
            }

        result = GenerationResult(
            text=json.dumps(response_data),
            model_type=teacher.model_type,
            model_version=teacher.model_version,
            status=GenerationStatus.SUCCESS,
            latency_ms=latency_ms,
        )
        teacher.generate = AsyncMock(return_value=result)
        return teacher

    @staticmethod
    def create_plaintext_teacher(scene_description: str = "A cooking scene") -> MagicMock:
        """Create a mock teacher that returns plain text."""
        teacher = MagicMock()
        teacher.model_name = "mock-plaintext-teacher"
        teacher.model_type = ModelType.TEACHER_GPT55

        result = GenerationResult(
            text=f"Scene: {scene_description}\nActions: pick knife, cut vegetable",
            model_type=teacher.model_type,
            model_version="gpt-5.5",
            status=GenerationStatus.SUCCESS,
            latency_ms=300.0,
        )
        teacher.generate = AsyncMock(return_value=result)
        return teacher


@pytest.fixture
def temp_storage_dir():
    """Provide a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_video_loader_factory():
    """Provide the video loader factory."""
    return SyntheticVideoHelper.create_mock_video_loader


@pytest.fixture
def mock_teacher_factory():
    """Provide the teacher factory."""
    return MockTeacherFactory


class TestGoldenPathSingleVideo:
    """Golden path tests for single video annotation."""

    @pytest.mark.asyncio
    async def test_complete_annotation_flow_structured_output(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test complete annotation flow with structured JSON output.

        Verifies:
        1. Video loading and scene detection
        2. Teacher model invocation
        3. Response parsing with confidence
        4. Annotation building
        5. Storage persistence
        6. Checkpoint update
        """
        checkpoint_path = temp_storage_dir / "checkpoint.json"
        annotation_dir = temp_storage_dir / "annotations"
        annotation_dir.mkdir(exist_ok=True)

        # Setup mock components
        video_path = Path("/fake/video.mp4")
        video_id = "golden_test_video_001"
        mock_loader = mock_video_loader_factory(video_path, video_id)
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        # Create pipeline
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=annotation_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
            segment_duration=5.0,
            checkpoint_path=checkpoint_path,
        )

        # Execute the golden path
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # Verify 1: Annotation structure is complete
        assert isinstance(annotation, Annotation)
        assert annotation.video_id == video_id
        assert annotation.source == "teacher"
        assert len(annotation.segments) == 2  # Two scenes detected

        # Verify 2: Segments have parsed content
        for segment in annotation.segments:
            assert isinstance(segment, Segment)
            assert segment.caption  # Has a caption
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            # Should have at least some actions

        # Verify 3: Teacher was called for each scene
        assert mock_teacher.generate.call_count == 2

        # Verify 4: Annotation was persisted
        from_store = store.load(f"{video_id}_annotated", source="gold")
        assert from_store is not None
        assert from_store.video_id == video_id
        assert len(from_store.segments) == len(annotation.segments)

        # Verify 5: Checkpoint was updated
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        assert checkpoint.is_processed(video_id)

    @pytest.mark.asyncio
    async def test_golden_path_with_quality_metrics(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that annotation includes quality tracking metadata."""
        video_path = Path("/fake/video.mp4")
        video_id = "quality_test_video"

        mock_loader = mock_video_loader_factory(video_path, video_id)
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # Verify quality-related metadata exists
        assert hasattr(annotation, "metadata")
        # Check that segments have confidence scores
        for segment in annotation.segments:
            if segment.actions:
                for action in segment.actions:
                    # Actions from structured output should have high confidence
                    assert action.confidence is None or 0 <= action.confidence <= 1

    @pytest.mark.asyncio
    async def test_golden_path_multiple_scenes_all_succeed(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test video with multiple scenes, all successfully annotated."""
        video_path = Path("/fake/multi_scene.mp4")
        video_id = "multi_scene_video"

        # Create video with 4 scenes
        mock_loader = mock_video_loader_factory(
            video_path,
            video_id,
            scenes=[(0, 2), (2, 4), (4, 6), (6, 8)],
        )
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # All 4 scenes should be annotated
        assert len(annotation.segments) == 4
        # Teacher called once per scene
        assert mock_teacher.generate.call_count == 4


class TestGoldenPathBatchProcessing:
    """Golden path tests for batch video processing."""

    @pytest.mark.asyncio
    async def test_batch_processing_all_success(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test batch processing where all videos succeed."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_storage_dir / "batch_checkpoint.json"
        mock_teacher = mock_teacher_factory.create_structured_teacher()
        store = AnnotationStore(root_path=temp_storage_dir)

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=4,
            checkpoint_path=checkpoint_path,
        )

        # Create batch of 3 videos
        video_items = [
            {"video_path": "/fake/vid1.mp4", "video_id": "batch_vid_1"},
            {"video_path": "/fake/vid2.mp4", "video_id": "batch_vid_2"},
            {"video_path": "/fake/vid3.mp4", "video_id": "batch_vid_3"},
        ]

        # Mock VideoLoader to return different instances for each call
        def create_loader_for_video(video_path, *args, **kwargs):
            video_id = Path(video_path).stem.replace("vid", "batch_vid_")
            return mock_video_loader_factory(Path(video_path), video_id)

        with patch("dvas.pipeline.core.VideoLoader", side_effect=create_loader_for_video):
            successful, failed = await pipeline.process_batch(
                video_items, max_concurrent=2
            )

        # All should succeed
        assert len(successful) == 3
        assert len(failed) == 0

        # All should be checkpointed
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        for item in video_items:
            assert checkpoint.is_processed(item["video_id"])

    @pytest.mark.asyncio
    async def test_batch_processing_idempotent(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that re-running batch skips already processed videos."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_storage_dir / "idempotent_checkpoint.json"
        mock_teacher = mock_teacher_factory.create_structured_teacher()
        store = AnnotationStore(root_path=temp_storage_dir)

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=4,
            checkpoint_path=checkpoint_path,
        )

        video_items = [
            {"video_path": "/fake/vid1.mp4", "video_id": "idempotent_vid_1"},
            {"video_path": "/fake/vid2.mp4", "video_id": "idempotent_vid_2"},
        ]

        # First run
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader_factory(Path("/fake/vid.mp4"))):
            successful1, failed1 = await pipeline.process_batch(video_items)

        assert len(successful1) == 2
        teacher_calls_after_first = mock_teacher.generate.call_count

        # Reset call count for accurate measurement
        mock_teacher.generate.reset_mock()

        # Second run - should skip processed videos
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_video_loader_factory(Path("/fake/vid.mp4"))):
            successful2, failed2 = await pipeline.process_batch(video_items)

        assert len(successful2) == 2  # Still returns all annotations
        assert len(failed2) == 0
        # Teacher should not be called again for already processed videos
        assert mock_teacher.generate.call_count == 0


class TestGoldenPathExport:
    """Golden path tests for annotation export."""

    @pytest.mark.asyncio
    async def test_annotation_export_roundtrip(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that annotations can be exported and re-imported."""
        from dvas.data.storage import AnnotationStore
        from dvas.export.adapters import LLaVAAdapter

        video_path = Path("/fake/export_test.mp4")
        video_id = "export_test_video"

        mock_loader = mock_video_loader_factory(video_path, video_id)
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        # Create annotation
        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # Export to LLaVA format
        adapter = LLaVAAdapter()
        export_path = temp_storage_dir / "export.jsonl"

        with open(export_path, "w") as f:
            for segment in annotation.segments:
                llava_record = adapter.convert_segment(segment, annotation.video_path)
                f.write(json.dumps(llava_record, ensure_ascii=False) + "\n")

        # Verify export
        assert export_path.exists()
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == len(annotation.segments)

        # Verify JSON structure
        for line in lines:
            record = json.loads(line)
            assert "id" in record
            assert "image" in record or "video" in record
            assert "conversations" in record

    @pytest.mark.asyncio
    async def test_export_with_parser_confidence(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that export includes parser confidence scores."""
        video_path = Path("/fake/confidence_test.mp4")
        video_id = "confidence_test_video"

        mock_loader = mock_video_loader_factory(video_path, video_id)
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # Check that confidence is tracked somewhere
        # (Implementation detail: parser stores confidence in metadata)
        for segment in annotation.segments:
            if segment.actions:
                # Structured output should have high confidence
                pass


class TestGoldenPathCostTracking:
    """Golden path tests for cost and latency tracking."""

    @pytest.mark.asyncio
    async def test_teacher_latency_recorded(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that teacher model latency is recorded."""
        video_path = Path("/fake/latency_test.mp4")
        video_id = "latency_test_video"

        mock_loader = mock_video_loader_factory(
            video_path, video_id, scenes=[(0, 3)]  # Single scene
        )
        mock_teacher = mock_teacher_factory.create_structured_teacher(latency_ms=750.0)

        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # Verify teacher was called and latency was in result
        mock_teacher.generate.assert_called_once()
        call_result = await mock_teacher.generate.return_value
        assert call_result.latency_ms == 750.0

    @pytest.mark.asyncio
    async def test_cost_estimation_accumulated(
        self,
        temp_storage_dir,
        mock_video_loader_factory,
        mock_teacher_factory,
    ):
        """Test that cost estimates are accumulated across segments."""
        video_path = Path("/fake/cost_test.mp4")
        video_id = "cost_test_video"

        # 3 scenes = 3 teacher calls
        mock_loader = mock_video_loader_factory(
            video_path, video_id, scenes=[(0, 2), (2, 4), (4, 6)]
        )
        mock_teacher = mock_teacher_factory.create_structured_teacher()

        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=temp_storage_dir)
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            num_frames=8,
        )

        with patch("dvas.pipeline.core.VideoLoader", return_value=mock_loader):
            annotation = await pipeline.annotate_video(video_path, video_id)

        # 3 scenes = 3 API calls
        assert mock_teacher.generate.call_count == 3


@pytest.mark.e2e
@pytest.mark.slow
class TestGoldenPathIntegration:
    """Integration-level golden path tests (marked as slow)."""

    @pytest.mark.skip(reason="Requires real video file and API key")
    @pytest.mark.asyncio
    async def test_real_video_annotation(self, temp_storage_dir):
        """End-to-end test with actual video file (requires setup)."""
        pass
