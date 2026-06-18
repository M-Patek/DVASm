"""Snapshot tests for DVAS.

Tests that verify output matches stored snapshots for regression detection.
"""

import pytest
from pathlib import Path

from dvas.testing import SnapshotStore

from dvas.data.schemas import (
    Action,
    Annotation,
    Hand,
    Object,
    QAPair,
    Segment,
    VideoMetadata,
)


SNAPSHOT_DIR = Path(__file__).parent / "__snapshots__"


class TestAnnotationSnapshots:
    """Snapshot tests for annotation serialization."""

    @pytest.fixture
    def store(self):
        """Create snapshot store."""
        return SnapshotStore(SNAPSHOT_DIR)

    def test_llava_format_snapshot(self, store):
        """Test LLaVA format matches snapshot."""
        annotation = Annotation(
            id="snapshot_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=5.0,
                    caption="A person is cutting vegetables in the kitchen.",
                    actions=[
                        Action(verb="cut", noun="vegetables", hand=Hand.RIGHT),
                    ],
                    objects=[
                        Object(name="knife", confidence=0.95),
                        Object(name="cutting_board", confidence=0.88),
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

        llava = annotation.to_llava_format()
        store.assert_match("test_llava_format", llava)

    def test_openai_format_snapshot(self, store):
        """Test OpenAI format matches snapshot."""
        annotation = Annotation(
            id="snapshot_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=5.0,
                    caption="A person is cutting vegetables.",
                ),
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        openai = annotation.to_openai_format()
        store.assert_match("test_openai_format", openai)

    def test_annotation_dict_snapshot(self, store):
        """Test annotation dict matches snapshot."""
        annotation = Annotation(
            id="snapshot_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=3.0,
                    caption="First segment",
                    actions=[
                        Action(verb="pick", noun="spoon", hand=Hand.RIGHT),
                    ],
                ),
                Segment(
                    start_time=3.0,
                    end_time=6.0,
                    caption="Second segment",
                    actions=[
                        Action(verb="stir", noun="soup", hand=Hand.LEFT),
                    ],
                ),
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
            tags=["cooking", "kitchen"],
        )

        store.assert_match("test_annotation_dict", annotation.model_dump())

    def test_empty_annotation_snapshot(self, store):
        """Test empty annotation matches snapshot."""
        annotation = Annotation(
            id="empty_test",
            video_id="vid_empty",
            video_path="/path/to/empty.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        store.assert_match("test_empty_annotation", annotation.model_dump())


class TestSegmentSnapshots:
    """Snapshot tests for segment serialization."""

    @pytest.fixture
    def store(self):
        return SnapshotStore(SNAPSHOT_DIR)

    def test_segment_with_qa_pairs(self, store):
        """Test segment with Q&A pairs matches snapshot."""
        segment = Segment(
            start_time=0.0,
            end_time=5.0,
            caption="A cooking scene",
            qa_pairs=[
                QAPair(question="What is happening?", answer="Cooking"),
                QAPair(question="What tool is used?", answer="A knife"),
            ],
        )

        store.assert_match("test_segment_with_qa", segment.model_dump())


class TestVideoMetadataSnapshots:
    """Snapshot tests for video metadata."""

    @pytest.fixture
    def store(self):
        return SnapshotStore(SNAPSHOT_DIR)

    def test_video_metadata_snapshot(self, store):
        """Test video metadata matches snapshot."""
        metadata = VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=120.0,
            total_frames=3600,
            codec="h264",
            bitrate=5000000,
            has_audio=True,
        )

        store.assert_match("test_video_metadata", metadata.model_dump())


class TestActionSnapshots:
    """Snapshot tests for action serialization."""

    @pytest.fixture
    def store(self):
        return SnapshotStore(SNAPSHOT_DIR)

    def test_action_with_all_fields(self, store):
        """Test action with all fields matches snapshot."""
        action = Action(
            verb="cut",
            noun="vegetables",
            hand=Hand.RIGHT,
            start_time=1.0,
            end_time=3.0,
            confidence=0.95,
        )

        store.assert_match("test_action_full", action.model_dump())

    def test_action_minimal(self, store):
        """Test minimal action matches snapshot."""
        action = Action(verb="pick", noun="spoon")

        store.assert_match("test_action_minimal", action.model_dump())
