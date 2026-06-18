"""Tests for data schemas."""

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    BoundingBox,
    Hand,
    Segment,
    VideoMetadata,
)


class TestBoundingBox:
    """Test BoundingBox model."""

    def test_valid_coordinates(self):
        """Test valid bounding box creation."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        assert bbox.to_list() == [0.1, 0.2, 0.5, 0.6]

    def test_invalid_coordinates(self):
        """Test validation of out-of-bounds coordinates."""
        with pytest.raises(ValueError):
            BoundingBox(x1=1.5, y1=0.0, x2=0.5, y2=1.0)


class TestSegment:
    """Test Segment model."""

    def test_segment_creation(self):
        """Test creating a segment."""
        segment = Segment(
            start_time=0.0,
            end_time=5.0,
            caption="Test caption",
        )
        assert segment.duration == 5.0

    def test_end_before_start(self):
        """Test validation of end_time before start_time."""
        with pytest.raises(ValueError):
            Segment(
                start_time=5.0,
                end_time=2.0,
                caption="Invalid",
            )

    def test_segment_with_actions(self):
        """Test segment with actions."""
        segment = Segment(
            start_time=0.0,
            end_time=3.0,
            caption="Cooking",
            actions=[
                Action(verb="cut", noun="onion", hand=Hand.RIGHT),
            ],
        )
        assert len(segment.actions) == 1
        assert segment.actions[0].verb == "cut"


class TestAnnotation:
    """Test Annotation model."""

    def test_annotation_creation(self):
        """Test creating an annotation."""
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
        assert annotation.id == "test_001"
        assert annotation.quality_score is None

    def test_to_llava_format(self):
        """Test conversion to LLaVA format."""
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
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        llava = annotation.to_llava_format()
        assert "id" in llava
        assert "conversations" in llava
        assert len(llava["conversations"]) == 2  # human + gpt

    def test_get_unique_verbs(self):
        """Test extracting unique action verbs."""
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
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        verbs = annotation.get_action_verbs()
        assert verbs == ["cut", "pick"]
