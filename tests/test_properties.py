"""Property-based tests for DVAS data schemas.

Tests invariants that should hold for ALL valid inputs, not just specific examples.
"""

from dvas.testing import ArbitraryValue

from dvas.data.schemas import (
    Action,
    Annotation,
    BoundingBox,
    Segment,
    VideoMetadata,
)


class TestBoundingBoxProperties:
    """Property-based tests for BoundingBox."""

    def test_bounding_box_coordinates_always_in_range(self):
        """Bounding box coordinates must always be within [0, 1]."""
        gen = ArbitraryValue(seed=42)
        for x1 in gen.floats(0.0, 0.4).take(20):
            for y1 in gen.floats(0.0, 0.4).take(5):
                for x2 in gen.floats(0.5, 1.0).take(5):
                    for y2 in gen.floats(0.5, 1.0).take(5):
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        assert 0.0 <= bbox.x1 <= 1.0
                        assert 0.0 <= bbox.y1 <= 1.0
                        assert 0.0 <= bbox.x2 <= 1.0
                        assert 0.0 <= bbox.y2 <= 1.0

    def test_bounding_box_to_list_roundtrip(self):
        """to_list() preserves all coordinates."""
        gen = ArbitraryValue(seed=42)
        for x1 in gen.floats(0.0, 0.4).take(20):
            for y1 in gen.floats(0.0, 0.4).take(5):
                for x2 in gen.floats(0.5, 1.0).take(5):
                    for y2 in gen.floats(0.5, 1.0).take(5):
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        coords = bbox.to_list()
                        assert len(coords) == 4
                        assert coords == [x1, y1, x2, y2]


class TestSegmentProperties:
    """Property-based tests for Segment."""

    def test_segment_duration_is_positive(self):
        """Segment duration must always be positive."""
        gen = ArbitraryValue(seed=42)
        for start in gen.floats(0.0, 100.0).take(50):
            for duration in gen.floats(0.1, 50.0).take(5):
                segment = Segment(
                    start_time=start,
                    end_time=start + duration,
                    caption="Test",
                )
                assert abs(segment.duration - duration) < 1e-10
                assert segment.duration > 0

    def test_segment_with_actions_preserves_count(self):
        """Adding actions preserves the count."""
        gen = ArbitraryValue(seed=42)
        for start in gen.floats(0.0, 100.0).take(20):
            for duration in gen.floats(0.1, 50.0).take(5):
                actions = [
                    Action(verb="cut", noun="onion"),
                    Action(verb="pick", noun="spoon"),
                ]
                segment = Segment(
                    start_time=start,
                    end_time=start + duration,
                    caption="Test",
                    actions=actions,
                )
                assert len(segment.actions) == 2


class TestAnnotationProperties:
    """Property-based tests for Annotation."""

    def test_annotation_total_duration_is_sum(self):
        """Total duration equals sum of segment durations."""
        gen = ArbitraryValue(seed=42)
        for num_segments in gen.integers(1, 10).take(10):
            segments = []
            expected_total = 0.0
            for i in range(num_segments):
                duration = 5.0
                segments.append(
                    Segment(
                        start_time=i * duration,
                        end_time=(i + 1) * duration,
                        caption=f"Segment {i}",
                    )
                )
                expected_total += duration

            annotation = Annotation(
                id="test",
                video_id="vid_001",
                video_path="/path/to/video.mp4",
                segments=segments,
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=expected_total,
                    total_frames=int(30.0 * expected_total),
                ),
            )

            assert annotation.get_total_duration() == expected_total

    def test_unique_verbs_are_actually_unique(self):
        """get_action_verbs returns unique verbs."""
        gen = ArbitraryValue(seed=42)
        for num_actions in gen.integers(1, 20).take(10):
            actions = []
            for i in range(num_actions):
                actions.append(Action(verb=f"verb_{i % 5}", noun=f"noun_{i}"))

            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Test",
                actions=actions,
            )

            annotation = Annotation(
                id="test",
                video_id="vid_001",
                video_path="/path/to/video.mp4",
                segments=[segment],
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=5.0,
                    total_frames=150,
                ),
            )

            verbs = annotation.get_action_verbs()
            assert len(verbs) == len(set(verbs))  # No duplicates
            assert len(verbs) <= 5  # At most 5 unique verbs


class TestVideoMetadataProperties:
    """Property-based tests for VideoMetadata."""

    def test_total_frames_consistent_with_fps_and_duration(self):
        """total_frames should be approximately fps * duration."""
        gen = ArbitraryValue(seed=42)
        for fps in gen.floats(1.0, 120.0).take(50):
            for duration in gen.floats(1.0, 3600.0).take(5):
                total_frames = int(fps * duration)
                metadata = VideoMetadata(
                    fps=fps,
                    resolution=[1920, 1080],
                    duration=duration,
                    total_frames=total_frames,
                )
                assert metadata.total_frames == total_frames
                assert abs(metadata.total_frames - fps * duration) < 1.0


class TestLLaVAFormatProperties:
    """Property-based tests for LLaVA format conversion."""

    def test_llava_format_always_has_id_and_conversations(self):
        """LLaVA format always has required fields."""
        annotation = Annotation(
            id="test_001",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(start_time=0.0, end_time=5.0, caption="Test"),
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        llava = annotation.to_llava_format()
        assert "id" in llava
        assert "video" in llava
        assert "conversations" in llava
        assert len(llava["conversations"]) > 0
        assert llava["conversations"][0]["from"] == "human"
        assert llava["conversations"][1]["from"] == "gpt"

    def test_llava_conversations_count_matches_segments(self):
        """Each segment produces 2 conversation entries."""
        gen = ArbitraryValue(seed=42)
        for num_segments in gen.integers(1, 10).take(10):
            segments = [
                Segment(start_time=i * 5.0, end_time=(i + 1) * 5.0, caption=f"Seg {i}")
                for i in range(num_segments)
            ]

            annotation = Annotation(
                id="test",
                video_id="vid_001",
                video_path="/path/to/video.mp4",
                segments=segments,
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=num_segments * 5.0,
                    total_frames=int(num_segments * 5.0 * 30.0),
                ),
            )

            llava = annotation.to_llava_format()
            assert len(llava["conversations"]) == num_segments * 2


class TestOpenAIFormatProperties:
    """Property-based tests for OpenAI format conversion."""

    def test_openai_format_has_messages(self):
        """OpenAI format always has messages array."""
        annotation = Annotation(
            id="test_001",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(start_time=0.0, end_time=5.0, caption="Test"),
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        openai = annotation.to_openai_format()
        assert "id" in openai
        assert "messages" in openai
        assert len(openai["messages"]) == 2
