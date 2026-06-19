"""Tests for lineage tracking — pytest function style.

Covers LineageTracker provenance recording, schema compatibility
checks, and statistics.
"""

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    AnnotationStandard,
    Hand,
    Segment,
    VideoMetadata,
)
from dvas.lineage import LineageStep, LineageTracker, SchemaCompatibility, SchemaVersion


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tracker() -> LineageTracker:
    """Fresh LineageTracker instance."""
    return LineageTracker()


@pytest.fixture
def sample_annotation() -> Annotation:
    """A v2.0 annotation for compatibility tests."""
    return Annotation(
        id="ann_v2",
        video_id="vid_001",
        video_path="/tmp/test.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=1.0,
                caption="test",
                actions=[
                    Action(
                        verb="pick",
                        noun="cup",
                        hand=Hand.RIGHT,
                        instrument="hand",
                    )
                ],
            )
        ],
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=1.0,
            total_frames=30,
        ),
        annotation_standard=AnnotationStandard.EPIC_KITCHENS,
    )


# ── Basic lineage tests ───────────────────────────────────────────────────


def test_record_step(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "pipeline_annotation", {"model": "gpt-4v"})
    provenance = tracker.get_provenance("ann_001")
    assert len(provenance) == 1
    assert provenance[0].step == "pipeline_annotation"
    assert provenance[0].metadata == {"model": "gpt-4v"}


def test_record_multiple_steps(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "pipeline_annotation")
    tracker.record_step("ann_001", "human_review", {"reviewer": "alice"})
    tracker.record_step("ann_001", "export_llava")

    provenance = tracker.get_provenance("ann_001")
    assert len(provenance) == 3
    assert [s.step for s in provenance] == [
        "pipeline_annotation",
        "human_review",
        "export_llava",
    ]


def test_get_last_step(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "step_a")
    tracker.record_step("ann_001", "step_b")
    last = tracker.get_last_step("ann_001")
    assert last is not None
    assert last.step == "step_b"


def test_get_last_step_unknown(tracker: LineageTracker) -> None:
    assert tracker.get_last_step("unknown") is None


def test_get_provenance_unknown(tracker: LineageTracker) -> None:
    assert tracker.get_provenance("unknown") == []


def test_record_step_with_agent(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "pipeline_annotation", agent="teacher")
    step = tracker.get_last_step("ann_001")
    assert step is not None
    assert step.agent == "teacher"


# ── Compatibility tests ─────────────────────────────────────────────────


def test_compat_same_version(tracker: LineageTracker, sample_annotation: Annotation) -> None:
    result = tracker.check_compatibility(sample_annotation, "2.0")
    assert result.compatible is True
    assert result.source_version == "2.0"
    assert result.target_version == "2.0"


def test_compat_v1_to_v2(tracker: LineageTracker) -> None:
    """v1.0 -> v2.0 is backward compatible but may warn."""
    ann = Annotation(
        id="ann_v1",
        video_id="vid_001",
        video_path="/tmp/test.mp4",
        segments=[],
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=0.0,
            total_frames=0,
        ),
        annotation_standard=AnnotationStandard.EPIC_KITCHENS,
    )
    result = tracker.check_compatibility(ann, "2.0")
    assert result.compatible is True


def test_compat_v2_to_v1_blocked(tracker: LineageTracker, sample_annotation: Annotation) -> None:
    """v2.0 -> v1.0 should fail if enhanced fields are present."""
    result = tracker.check_compatibility(sample_annotation, "1.0")
    assert result.compatible is False
    assert len(result.errors) > 0
    assert "not supported" in result.errors[0]


def test_compat_unknown_version(tracker: LineageTracker, sample_annotation: Annotation) -> None:
    result = tracker.check_compatibility(sample_annotation, "9.9")
    assert result.compatible is False
    assert len(result.errors) == 1


# ── Statistics tests ────────────────────────────────────────────────────


def test_statistics_empty(tracker: LineageTracker) -> None:
    stats = tracker.get_statistics()
    assert stats["total_annotations"] == 0
    assert stats["total_steps"] == 0
    assert stats["step_breakdown"] == {}


def test_statistics(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "pipeline_annotation")
    tracker.record_step("ann_001", "human_review")
    tracker.record_step("ann_002", "pipeline_annotation")

    stats = tracker.get_statistics()
    assert stats["total_annotations"] == 2
    assert stats["total_steps"] == 3
    assert stats["step_breakdown"]["pipeline_annotation"] == 2
    assert stats["step_breakdown"]["human_review"] == 1


# ── Clear tests ─────────────────────────────────────────────────────────


def test_clear_single(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "step_a")
    tracker.record_step("ann_002", "step_b")
    tracker.clear("ann_001")
    assert tracker.get_provenance("ann_001") == []
    assert len(tracker.get_provenance("ann_002")) == 1


def test_clear_all(tracker: LineageTracker) -> None:
    tracker.record_step("ann_001", "step_a")
    tracker.record_step("ann_002", "step_b")
    tracker.clear()
    assert tracker.get_statistics()["total_annotations"] == 0


# ── SchemaVersion enum tests ────────────────────────────────────────────


def test_schema_version_values() -> None:
    assert SchemaVersion.V1_0.value == "1.0"
    assert SchemaVersion.V2_0.value == "2.0"
    assert SchemaVersion.V3_0.value == "3.0"
