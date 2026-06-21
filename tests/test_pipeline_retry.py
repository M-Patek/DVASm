"""E2E tests for 04-pipeline retry/checkpoint/resume behavior.

Closes known_gap: "No batch retry logic for API failures" by exercising
@with_retry on annotate_video + CheckpointManager.mark_failed + resume
across real (mocked) teacher failures.
"""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.models.base import GenerationResult, GenerationStatus, ModelType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generation_result(text: str = "ok") -> GenerationResult:
    """Build a successful GenerationResult the way real teachers return it."""
    return GenerationResult(
        text=text,
        model_type=ModelType.TEACHER_GPT55,
        model_version="gpt-5.5",
        status=GenerationStatus.SUCCESS,
        confidence=1.0,
    )


def _make_failing_result(error: str) -> GenerationResult:
    return GenerationResult(
        text="",
        model_type=ModelType.TEACHER_GPT55,
        model_version="gpt-5.5",
        status=GenerationStatus.FAILURE,
        error_message=error,
        confidence=0.0,
    )


def _patch_video_loader(scene_count: int = 1) -> MagicMock:
    """Patch VideoLoader in pipeline.core to return a fake loader with N scenes."""
    mock_loader = MagicMock()
    mock_loader.__enter__ = MagicMock(return_value=mock_loader)
    mock_loader.__exit__ = MagicMock(return_value=None)
    mock_loader.metadata = VideoMetadata(
        fps=30.0,
        resolution=[1920, 1080],
        duration=10.0,
        total_frames=300,
    )
    scenes = [(i * 5.0, (i + 1) * 5.0) for i in range(scene_count)] or [(0.0, 5.0)]
    mock_loader.detect_scenes.return_value = scenes

    async def _aiter(start_time, end_time, num_frames):  # type: ignore[no-untyped-def]
        for i in range(num_frames):
            yield MagicMock(
                data=np.zeros((64, 64, 3), dtype=np.uint8),
                idx=i,
                timestamp=float(i),
            )

    mock_loader.aiter_frames = _aiter
    return mock_loader


def _build_teacher(result_sequence: List[GenerationResult]) -> MagicMock:
    """Build a mock teacher whose .generate() yields each result in order, then
    repeats the last one forever."""
    teacher = MagicMock()
    teacher.model_name = "gpt-5.5"
    teacher.model_type = ModelType.TEACHER_GPT55
    teacher.model_version = "gpt-5.5"

    call_index = {"i": 0}

    async def _gen(*args: Any, **kwargs: Any) -> GenerationResult:
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(result_sequence):
            return result_sequence[idx]
        return result_sequence[-1]

    teacher.generate = _gen
    teacher.annotate = _gen
    return teacher


# ---------------------------------------------------------------------------
# 1. Retry recovers from transient failures
# ---------------------------------------------------------------------------


class TestRetryRecovery:
    """@with_retry on _call_teacher_with_retry recovers from transient failures.

    We test the retry decorator directly on the pipeline's internal method
    to avoid coupling to VideoLoader / scene detection / save. The point of
    THIS test is the retry loop's behavior under transient vs persistent failure.
    """

    @pytest.mark.asyncio
    async def test_call_teacher_with_retry_succeeds_after_two_transient_failures(self):
        """Teacher raises ConnectionError twice, then succeeds on the 3rd attempt."""
        from dvas.utils.retry import with_retry as real_with_retry
        from dvas.pipeline.core import AnnotationPipeline

        def fast_retry(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            kwargs["base_delay"] = 0.0
            kwargs["max_delay"] = 0.0
            return real_with_retry(*args, **kwargs)

        call_count = {"i": 0}

        async def _gen(*args: Any, **kwargs: Any) -> GenerationResult:
            call_count["i"] += 1
            if call_count["i"] < 3:
                raise ConnectionError(f"transient outage #{call_count['i']}")
            return _make_generation_result("recovered")

        teacher = MagicMock()
        teacher.model_name = "gpt-5.5"
        teacher.generate = _gen

        pipeline = AnnotationPipeline(teacher_model=teacher, num_frames=4)

        with patch("dvas.pipeline.core.with_retry", side_effect=fast_retry):
            result = await pipeline._call_teacher_with_retry([])

        assert result.is_success()
        assert result.text == "recovered"
        # 2 failures + 1 success = 3 attempts
        assert call_count["i"] == 3

    @pytest.mark.asyncio
    async def test_call_teacher_with_retry_raises_retry_exhausted_after_max_attempts(self):
        """When teacher keeps failing, _call_teacher_with_retry raises
        RetryExhaustedError after max_attempts (3)."""
        from dvas.exceptions import RetryExhaustedError
        from dvas.utils.retry import with_retry as real_with_retry
        from dvas.pipeline.core import AnnotationPipeline

        def fast_retry(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            kwargs["base_delay"] = 0.0
            kwargs["max_delay"] = 0.0
            return real_with_retry(*args, **kwargs)

        call_count = {"i": 0}

        async def _gen(*args: Any, **kwargs: Any) -> GenerationResult:
            call_count["i"] += 1
            raise ConnectionError(f"persistent outage #{call_count['i']}")

        teacher = MagicMock()
        teacher.model_name = "gpt-5.5"
        teacher.generate = _gen

        pipeline = AnnotationPipeline(teacher_model=teacher, num_frames=4)

        with patch("dvas.pipeline.core.with_retry", side_effect=fast_retry):
            with pytest.raises(RetryExhaustedError) as exc_info:
                await pipeline._call_teacher_with_retry([])

        # Decorator attempted exactly max_attempts (3) times
        assert call_count["i"] == 3
        # Underlying error is preserved on RetryExhaustedError.last_error
        assert "persistent outage" in str(exc_info.value.last_error)


# ---------------------------------------------------------------------------
# 2. Batch processing: partial failure isolation
# ---------------------------------------------------------------------------


class TestBatchPartialFailure:
    """process_batch should isolate per-item failures and not crash the batch.

    Implementation note: we monkey-patch ``annotate_video`` so the test
    controls per-video outcomes (success / fail) without going through the
    real VideoLoader + teacher plumbing. This isolates the orchestration
    behavior under test (batch concurrency + checkpoint bookkeeping) from
    the inner annotate_video logic (covered by the TestRetryRecovery class).
    """

    @pytest.mark.asyncio
    async def test_process_batch_isolates_failures_and_books_keeps(self, tmp_path):
        """A mix of succeeding and failing videos:
        - successes end up in `successful`
        - failures end up in `failed`
        - checkpoint reflects both states
        """
        from dvas.data.schemas import Annotation, VideoMetadata

        def make_annotation(video_id: str) -> Annotation:
            return Annotation(
                id=video_id,
                video_id=video_id,
                video_path=f"/fake/{video_id}.mp4",
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=10.0,
                    total_frames=300,
                ),
            )

        # Map of video_id -> outcome ("ok" or "fail")
        outcomes = {
            "vid_ok_1": "ok",
            "vid_fail_1": "fail",
            "vid_ok_2": "ok",
            "vid_fail_2": "fail",
            "vid_ok_3": "ok",
        }

        async def fake_annotate_video(self, video_path, video_id):  # type: ignore[no-untyped-def]
            outcome = outcomes.get(video_id, "fail")
            if outcome == "fail":
                raise ConnectionError(f"simulated outage for {video_id}")
            return make_annotation(video_id)

        teacher = MagicMock()
        teacher.model_name = "gpt-5.5"

        from dvas.pipeline.core import AnnotationPipeline

        pipeline = AnnotationPipeline(
            teacher_model=teacher,
            checkpoint_path=tmp_path / "cp.json",
            num_frames=2,
        )

        # Patch annotate_video to control per-item outcomes
        with patch.object(AnnotationPipeline, "annotate_video", new=fake_annotate_video):
            items = [{"video_id": vid, "video_path": f"/fake/{vid}.mp4"} for vid in outcomes]
            successful, failed = await pipeline.process_batch(items, max_concurrent=2)

        # 3 successes, 2 failures
        assert len(successful) == 3
        assert {a.video_id for a in successful} == {"vid_ok_1", "vid_ok_2", "vid_ok_3"}
        assert len(failed) == 2
        failed_ids = {f["item"]["video_id"] for f in failed}
        assert failed_ids == {"vid_fail_1", "vid_fail_2"}

        # Checkpoint state matches outcomes
        assert pipeline.checkpoint is not None
        assert pipeline.checkpoint.get_processed_count() == 3
        assert pipeline.checkpoint.get_failed_count() == 2


# ---------------------------------------------------------------------------
# 3. Checkpoint resume skips already-processed videos
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """A second run with the same checkpoint should skip processed video_ids."""

    @pytest.mark.asyncio
    async def test_second_run_skips_already_processed_videos(self, tmp_path):
        """Run 1 processes 2 videos and saves them to the store + checkpoint.
        Run 2 reuses the same checkpoint and adds 1 new video. Only the new
        video triggers an annotation call; the resumed ones are loaded from
        the store and returned to the caller."""
        from dvas.data.schemas import Annotation, VideoMetadata
        from dvas.data.storage import AnnotationStore

        def make_annotation(video_id: str) -> Annotation:
            return Annotation(
                id=video_id,
                video_id=video_id,
                video_path=f"/fake/{video_id}.mp4",
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=10.0,
                    total_frames=300,
                ),
            )

        # Track calls to the fake annotate_video
        annotated_ids: List[str] = []

        async def fake_annotate_video(self, video_path, video_id):  # type: ignore[no-untyped-def]
            annotated_ids.append(video_id)
            # Real annotate_video stores with id = f"{video_id}_annotated"
            # (see pipeline/core.py:106, the resume path's lookup key).
            ann = make_annotation(video_id)
            ann.id = f"{video_id}_annotated"
            self.store.save(ann, source="gold")
            return ann

        teacher = MagicMock()
        teacher.model_name = "gpt-5.5"

        from dvas.pipeline.core import AnnotationPipeline

        cp_path = tmp_path / "cp.json"
        store_root = tmp_path / "store"

        with patch.object(AnnotationPipeline, "annotate_video", new=fake_annotate_video):
            # Run 1: process 2 videos
            pipeline1 = AnnotationPipeline(
                teacher_model=teacher,
                checkpoint_path=cp_path,
                store=AnnotationStore(root_path=store_root),
                num_frames=2,
            )
            items1 = [
                {"video_id": "vid_a", "video_path": "/fake/a.mp4"},
                {"video_id": "vid_b", "video_path": "/fake/b.mp4"},
            ]
            ok1, _ = await pipeline1.process_batch(items1, max_concurrent=2)
            assert len(ok1) == 2
            assert sorted(annotated_ids) == ["vid_a", "vid_b"]
            # Checkpoint has the 2 processed ids
            assert "vid_a" in pipeline1.checkpoint.processed_ids
            assert "vid_b" in pipeline1.checkpoint.processed_ids
            # Store has the 2 annotations
            assert len(list(pipeline1.store.load_all(source="gold"))) == 2

            # Run 2: same checkpoint, same items + 1 new
            pipeline2 = AnnotationPipeline(
                teacher_model=teacher,
                checkpoint_path=cp_path,
                store=AnnotationStore(root_path=store_root),
                num_frames=2,
            )
            items2 = items1 + [{"video_id": "vid_c", "video_path": "/fake/c.mp4"}]
            ok2, _ = await pipeline2.process_batch(items2, max_concurrent=2)

        # All 3 annotations returned (2 hydrated from store + 1 fresh)
        assert len(ok2) == 3
        returned_ids = {a.video_id for a in ok2}
        assert returned_ids == {"vid_a", "vid_b", "vid_c"}
        # Only vid_c was newly annotated in run 2 (vid_a/vid_b were skipped
        # via checkpoint + hydrated from store)
        annotated_in_run_2 = annotated_ids[len(items1) :]
        assert annotated_in_run_2 == ["vid_c"], (
            f"Expected only vid_c to be annotated in run 2, got {annotated_in_run_2}"
        )

    def test_checkpoint_persists_across_reload(self, tmp_path):
        """Verify CheckpointManager saves/loads state across instances."""
        from dvas.pipeline.checkpoint import CheckpointManager

        cp_path = tmp_path / "cp.json"
        cp1 = CheckpointManager(cp_path)
        cp1.mark_processed("vid_1")
        cp1.mark_processed("vid_2")
        cp1.mark_failed("vid_3", "api timeout")
        cp1.save()

        assert cp_path.exists()
        raw = json.loads(cp_path.read_text())
        assert set(raw["processed"]) == {"vid_1", "vid_2"}
        assert len(raw["failed"]) == 1

        # Fresh instance loads from disk
        cp2 = CheckpointManager(cp_path)
        assert cp2.load() is True
        assert "vid_1" in cp2.processed_ids
        assert cp2.get_failed_count() == 1
        assert cp2.is_processed("vid_2")
        assert not cp2.is_processed("vid_unknown")
