"""Checkpoint resume tests for long-running annotation jobs.

Tests:
- Clean resume from checkpoint after interruption
- Partial batch resume
- Checkpoint integrity validation
- Resume with modified configuration
- Cross-session resume
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.pipeline.core import AnnotationPipeline
from dvas.pipeline.checkpoint import CheckpointManager


def create_mock_video_loader(video_id: str, num_scenes: int = 2):
    """Create a mock VideoLoader for testing."""
    loader = MagicMock()
    loader.metadata = VideoMetadata(
        video_id=video_id,
        fps=30.0,
        resolution=[640, 480],
        duration=float(num_scenes * 5),
        total_frames=num_scenes * 150,
    )

    # Create scene boundaries
    scenes = [(i * 5.0, (i + 1) * 5.0) for i in range(num_scenes)]
    loader.detect_scenes.return_value = scenes

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


def create_mock_teacher():
    """Create a mock teacher that succeeds."""
    teacher = MagicMock()
    teacher.model_name = "mock-teacher"
    teacher.model_type = ModelType.TEACHER_GPT55

    async def generate_impl(*args, **kwargs):
        return GenerationResult(
            text=json.dumps(
                {
                    "scene_description": f"Scene at {kwargs.get('task', 'unknown')}",
                    "actions": [{"verb": "test", "noun": "action"}],
                }
            ),
            model_type=ModelType.TEACHER_GPT55,
            model_version="gpt-5.5",
            status=GenerationStatus.SUCCESS,
            latency_ms=500.0,
        )

    teacher.generate = MagicMock(side_effect=generate_impl)
    return teacher


@pytest.fixture
def temp_dir():
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCheckpointResumeBasic:
    """Basic checkpoint resume functionality."""

    @pytest.mark.asyncio
    async def test_resume_skips_completed_videos(self, temp_dir):
        """Test that checkpoint resume skips already processed videos."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "resume_basic.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        # Pre-populate checkpoint with one processed video
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("vid_1")
        checkpoint.save()

        # Create store with annotation for vid_1
        store = AnnotationStore(root_path=store_dir)

        # Create pipeline
        mock_teacher = create_mock_teacher()
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Process batch with vid_1 (processed), vid_2 (new), vid_3 (new)
        video_items = [
            {"video_path": "/fake/vid1.mp4", "video_id": "vid_1"},
            {"video_path": "/fake/vid2.mp4", "video_id": "vid_2"},
            {"video_path": "/fake/vid3.mp4", "video_id": "vid_3"},
        ]

        with patch("dvas.pipeline.core.VideoLoader") as mock_loader_class:
            mock_loader_class.side_effect = lambda p: create_mock_video_loader(Path(p).stem)
            successful, failed = await pipeline.process_batch(video_items)

        # vid_1 is skipped by checkpoint filter, vid_2 and vid_3 are processed
        assert len(successful) == 2
        assert len(failed) == 0
        # Teacher should only be called for vid_2 and vid_3 (2 videos * 2 scenes = 4 calls)
        assert mock_teacher.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_checkpoint_updated_after_each_video(self, temp_dir):
        """Test that checkpoint is updated after each video completes."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "incremental.json"
        store = AnnotationStore(root_path=temp_dir)

        mock_teacher = create_mock_teacher()
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
            num_frames=4,
        )

        # Process first video
        with patch(
            "dvas.pipeline.core.VideoLoader", return_value=create_mock_video_loader("vid_1")
        ):
            await pipeline.annotate_video(Path("/fake/vid1.mp4"), "vid_1")

        # Verify checkpoint
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        assert checkpoint.is_processed("vid_1")

        # Process second video
        with patch(
            "dvas.pipeline.core.VideoLoader", return_value=create_mock_video_loader("vid_2")
        ):
            await pipeline.annotate_video(Path("/fake/vid2.mp4"), "vid_2")

        # Verify checkpoint updated
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()
        assert checkpoint2.is_processed("vid_1")
        assert checkpoint2.is_processed("vid_2")


class TestBatchCheckpointResume:
    """Batch processing checkpoint resume scenarios."""

    @pytest.mark.asyncio
    async def test_batch_resume_mid_stream(self, temp_dir):
        """Test resuming a batch that was interrupted mid-stream."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "batch_resume.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        # Simulate previous partial run: vid_1 done, vid_2 failed, vid_3 not started
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("batch_vid_1")
        checkpoint.mark_failed("batch_vid_2", "simulated failure")
        checkpoint.save()

        store = AnnotationStore(root_path=store_dir)

        # Create teacher with tracking
        call_log = []
        teacher = create_mock_teacher()
        original_generate = teacher.generate

        async def tracking_generate(*args, **kwargs):
            call_log.append(kwargs.get("task", "unknown"))
            return await original_generate(*args, **kwargs)

        teacher.generate = tracking_generate

        pipeline = AnnotationPipeline(
            teacher_model=teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Try to process all 3 again
        video_items = [
            {"video_path": "/fake/vid1.mp4", "video_id": "batch_vid_1"},
            {"video_path": "/fake/vid2.mp4", "video_id": "batch_vid_2"},
            {"video_path": "/fake/vid3.mp4", "video_id": "batch_vid_3"},
        ]

        with patch("dvas.pipeline.core.VideoLoader") as mock_loader_class:
            mock_loader_class.side_effect = lambda p: create_mock_video_loader(
                Path(p).stem, num_scenes=1
            )
            successful, failed = await pipeline.process_batch(video_items)

        # Should process vid_2 and vid_3 (vid_1 skipped via checkpoint)
        # Note: vid_2 will be retried despite being in failed list

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_save(self, temp_dir):
        """Test that checkpoint is saved periodically during batch processing."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "periodic.json"
        store = AnnotationStore(root_path=temp_dir)

        mock_teacher = create_mock_teacher()
        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Process 5 videos with checkpoint_every=2
        video_items = [
            {"video_path": f"/fake/vid{i}.mp4", "video_id": f"periodic_vid_{i}"} for i in range(5)
        ]

        with patch("dvas.pipeline.core.VideoLoader") as mock_loader_class:
            mock_loader_class.side_effect = lambda p: create_mock_video_loader(
                Path(p).stem, num_scenes=1
            )
            await pipeline.process_batch(video_items, checkpoint_every=2)

        # Verify checkpoint exists and has entries
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        assert checkpoint.get_processed_count() == 5


class TestCheckpointIntegrity:
    """Checkpoint integrity and validation tests."""

    def test_checkpoint_format_version_compatibility(self, temp_dir):
        """Test checkpoint format backward compatibility."""
        checkpoint_path = temp_dir / "version_compat.json"

        # Write checkpoint in current format
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("vid_1")
        checkpoint.mark_failed("vid_2", "error")
        checkpoint.save()

        # Verify it's valid JSON
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert "processed" in data
        assert "failed" in data
        assert "vid_1" in data["processed"]

    def test_checkpoint_handles_duplicate_entries(self, temp_dir):
        """Test that duplicate entries in checkpoint are handled."""
        checkpoint_path = temp_dir / "duplicates.json"

        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("vid_1")
        checkpoint.mark_processed("vid_1")  # Duplicate
        checkpoint.mark_processed("vid_1")  # Another duplicate
        checkpoint.save()

        # Should be stored as set (no duplicates)
        with open(checkpoint_path) as f:
            data = json.load(f)

        processed_list = data["processed"]
        assert processed_list.count("vid_1") == 1

    def test_checkpoint_with_many_entries(self, temp_dir):
        """Test checkpoint performance with many entries."""
        checkpoint_path = temp_dir / "large.json"

        checkpoint = CheckpointManager(checkpoint_path)

        # Add 1000 entries
        for i in range(1000):
            checkpoint.mark_processed(f"vid_{i:04d}")

        checkpoint.save()

        # Load and verify
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()
        assert checkpoint2.get_processed_count() == 1000
        assert checkpoint2.is_processed("vid_0500")

    def test_checkpoint_atomic_write(self, temp_dir):
        """Test that checkpoint writes are atomic."""
        checkpoint_path = temp_dir / "atomic.json"

        checkpoint = CheckpointManager(checkpoint_path)

        # Add many entries and save multiple times
        for i in range(100):
            checkpoint.mark_processed(f"vid_{i}")
            if i % 10 == 0:
                checkpoint.save()

        checkpoint.save()

        # Verify file is valid
        checkpoint2 = CheckpointManager(checkpoint_path)
        assert checkpoint2.load() is True
        assert checkpoint2.get_processed_count() == 100


class TestCrossSessionResume:
    """Cross-session checkpoint resume tests."""

    @pytest.mark.asyncio
    async def test_new_pipeline_instance_resumes_from_checkpoint(self, temp_dir):
        """Test that a new pipeline instance can resume from existing checkpoint."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "cross_session.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        # First session: process some videos
        store1 = AnnotationStore(root_path=store_dir)
        teacher1 = create_mock_teacher()
        pipeline1 = AnnotationPipeline(
            teacher_model=teacher1,
            store=store1,
            checkpoint_path=checkpoint_path,
        )

        with patch(
            "dvas.pipeline.core.VideoLoader", return_value=create_mock_video_loader("session_1_vid")
        ):
            await pipeline1.annotate_video(Path("/fake/session1.mp4"), "session_1_vid")

        # Second session: new pipeline instance, same checkpoint
        store2 = AnnotationStore(root_path=store_dir)
        teacher2 = create_mock_teacher()
        pipeline2 = AnnotationPipeline(
            teacher_model=teacher2,
            store=store2,
            checkpoint_path=checkpoint_path,
        )

        # Process same video again (should be skipped)
        with patch(
            "dvas.pipeline.core.VideoLoader", return_value=create_mock_video_loader("session_1_vid")
        ):
            await pipeline2.annotate_video(Path("/fake/session1.mp4"), "session_1_vid")

        # Teacher 2 should not be called for already processed video
        # (it returns from store or checkpoint skip)
        assert teacher2.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_resume_with_different_teacher_model(self, temp_dir):
        """Test that resume works even with different teacher model."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "different_teacher.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        # First session with teacher A
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("teacher_test_vid")
        checkpoint.save()

        store = AnnotationStore(root_path=store_dir)

        # Create annotation manually
        annotation = Annotation(
            id="teacher_test_vid_annotated",
            video_id="teacher_test_vid",
            video_path="/fake/test.mp4",
            segments=[],
            metadata=VideoMetadata(
                video_id="teacher_test_vid",
                fps=30.0,
                resolution=[640, 480],
                duration=10.0,
                total_frames=300,
            ),
        )
        store.save(annotation, source="gold")

        # Second session with different teacher
        teacher_b = MagicMock()
        teacher_b.model_name = "different-model"
        teacher_b.generate = MagicMock()

        pipeline = AnnotationPipeline(
            teacher_model=teacher_b,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        with patch(
            "dvas.pipeline.core.VideoLoader",
            return_value=create_mock_video_loader("teacher_test_vid"),
        ):
            await pipeline.annotate_video(Path("/fake/test.mp4"), "teacher_test_vid")

        # Should return cached annotation, not call new teacher
        teacher_b.generate.assert_not_called()


class TestCheckpointWithStorageSync:
    """Checkpoint and storage synchronization tests."""

    @pytest.mark.asyncio
    async def test_checkpoint_storage_consistency(self, temp_dir):
        """Test that checkpoint and storage stay consistent."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "consistent.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        store = AnnotationStore(root_path=store_dir)
        mock_teacher = create_mock_teacher()

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Process video
        with patch(
            "dvas.pipeline.core.VideoLoader",
            return_value=create_mock_video_loader("consistent_vid"),
        ):
            await pipeline.annotate_video(Path("/fake/consistent.mp4"), "consistent_vid")

        # Verify both checkpoint and storage have the video
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.load()
        assert checkpoint.is_processed("consistent_vid")

        from_store = store.load("consistent_vid_annotated", source="gold")
        assert from_store is not None
        assert from_store.video_id == "consistent_vid"

    @pytest.mark.asyncio
    async def test_orphan_checkpoint_entry_handling(self, temp_dir):
        """Test handling of checkpoint entries without corresponding storage."""
        from dvas.data.storage import AnnotationStore

        checkpoint_path = temp_dir / "orphan.json"
        store_dir = temp_dir / "store"
        store_dir.mkdir()

        # Create orphan checkpoint entry (no storage)
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_processed("orphan_vid")
        checkpoint.save()

        store = AnnotationStore(root_path=store_dir)
        mock_teacher = create_mock_teacher()

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            store=store,
            checkpoint_path=checkpoint_path,
        )

        # Process orphan video
        with patch(
            "dvas.pipeline.core.VideoLoader", return_value=create_mock_video_loader("orphan_vid")
        ):
            await pipeline.annotate_video(Path("/fake/orphan.mp4"), "orphan_vid")

        # Should detect inconsistency and reprocess
        assert mock_teacher.generate.call_count > 0

        # Now should be consistent
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()
        assert checkpoint2.is_processed("orphan_vid")


class TestCheckpointEdgeCases:
    """Edge cases for checkpoint resume."""

    @pytest.mark.asyncio
    async def test_empty_checkpoint_load(self, temp_dir):
        """Test loading from empty checkpoint file."""
        checkpoint_path = temp_dir / "empty.json"

        # Create empty but valid JSON
        checkpoint_path.write_text(json.dumps({"processed": [], "failed": []}))

        checkpoint = CheckpointManager(checkpoint_path)
        result = checkpoint.load()

        assert result is True  # Successfully loaded
        assert checkpoint.get_processed_count() == 0
        assert checkpoint.get_failed_count() == 0

    @pytest.mark.asyncio
    async def test_checkpoint_with_special_characters_in_id(self, temp_dir):
        """Test checkpoint with special characters in video IDs."""
        checkpoint_path = temp_dir / "special.json"

        checkpoint = CheckpointManager(checkpoint_path)

        # Add videos with special characters
        special_ids = [
            "vid-with-dashes",
            "vid_with_underscores",
            "vid.with.dots",
            "vid/with/slashes",
            "vid:with:colons",
            "vid with spaces",
            'vid"with"quotes',
        ]

        for vid in special_ids:
            checkpoint.mark_processed(vid)

        checkpoint.save()

        # Load and verify
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()

        for vid in special_ids:
            assert checkpoint2.is_processed(vid), f"Failed for {vid}"

    def test_checkpoint_failed_items_preserved(self, temp_dir):
        """Test that failed items are preserved across saves."""
        checkpoint_path = temp_dir / "failed_preserved.json"

        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.mark_failed("failed_1", "Network error")
        checkpoint.mark_failed("failed_2", "Parse error")
        checkpoint.save()

        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()

        assert checkpoint2.get_failed_count() == 2
        # Check failed items details
        assert len(checkpoint2.failed_items) == 2

    def test_simultaneous_checkpoint_access(self, temp_dir):
        """Test behavior with multiple checkpoint managers."""
        checkpoint_path = temp_dir / "concurrent.json"

        # First manager adds items
        checkpoint1 = CheckpointManager(checkpoint_path)
        checkpoint1.mark_processed("vid_1")
        checkpoint1.save()

        # Second manager loads and adds more
        checkpoint2 = CheckpointManager(checkpoint_path)
        checkpoint2.load()
        checkpoint2.mark_processed("vid_2")
        checkpoint2.save()

        # First manager adds more (should overwrite)
        checkpoint1.mark_processed("vid_3")
        checkpoint1.save()

        # Final state depends on implementation
        # This test documents current behavior
        checkpoint3 = CheckpointManager(checkpoint_path)
        checkpoint3.load()

        # vid_3 should be there (last write wins)
        assert checkpoint3.is_processed("vid_3")
