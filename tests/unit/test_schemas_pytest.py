"""Tests for data schemas — pytest function style.

Converted from unittest.TestCase class style to pytest function style
with fixtures. Covers BoundingBox, Segment, Annotation, VideoMetadata,
and Hand enums.
"""

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    BoundingBox,
    Hand,
    Segment,
    VideoMetadata,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_metadata() -> VideoMetadata:
    return VideoMetadata(
        fps=30.0,
        resolution=[1920, 1080],
        duration=10.0,
        total_frames=300,
    )


@pytest.fixture
def sample_segment() -> Segment:
    return Segment(
        start_time=0.0,
        end_time=5.0,
        caption="Test action",
        actions=[
            Action(verb="pick", noun="spoon", hand=Hand.RIGHT),
        ],
    )


# ── BoundingBox tests ───────────────────────────────────────────────────


def test_bounding_box_valid_coordinates() -> None:
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    assert bbox.to_list() == [0.1, 0.2, 0.5, 0.6]


def test_bounding_box_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        BoundingBox(x1=1.5, y1=0.0, x2=0.5, y2=1.0)


# ── Segment tests ───────────────────────────────────────────────────────


def test_segment_creation() -> None:
    segment = Segment(
        start_time=0.0,
        end_time=5.0,
        caption="Test caption",
    )
    assert segment.duration == 5.0


def test_segment_end_before_start() -> None:
    with pytest.raises(ValueError):
        Segment(
            start_time=5.0,
            end_time=2.0,
            caption="Invalid",
        )


def test_segment_with_actions(sample_segment: Segment) -> None:
    assert len(sample_segment.actions) == 1
    assert sample_segment.actions[0].verb == "pick"
    assert sample_segment.actions[0].hand == Hand.RIGHT


# ── Annotation tests ────────────────────────────────────────────────────


def test_annotation_creation(sample_metadata: VideoMetadata) -> None:
    annotation = Annotation(
        id="test_001",
        video_id="vid_001",
        video_path="/path/to/video.mp4",
        metadata=sample_metadata,
    )
    assert annotation.id == "test_001"
    assert annotation.quality_score is None


def test_annotation_to_llava_format(sample_metadata: VideoMetadata) -> None:
    annotation = Annotation(
        id="test_001",
        video_id="vid_001",
        video_path="/path/to/video.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Test action",
                actions=[Action(verb="pick", noun="spoon", hand=Hand.RIGHT)],
            ),
        ],
        metadata=sample_metadata,
    )

    llava = annotation.to_llava_format()
    assert "id" in llava
    assert "conversations" in llava
    assert len(llava["conversations"]) == 2  # human + gpt


def test_annotation_get_unique_verbs(sample_metadata: VideoMetadata) -> None:
    annotation = Annotation(
        id="test_001",
        video_id="vid_001",
        video_path="/path/to/video.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Segment 1",
                actions=[
                    Action(verb="cut", noun="onion"),
                    Action(verb="pick", noun="knife"),
                ],
            ),
            Segment(
                start_time=5.0,
                end_time=10.0,
                caption="Segment 2",
                actions=[
                    Action(verb="cut", noun="tomato"),
                ],
            ),
        ],
        metadata=sample_metadata,
    )

    verbs = annotation.get_action_verbs()
    assert verbs == ["cut", "pick"]
