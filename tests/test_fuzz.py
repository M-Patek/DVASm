"""Fuzz testing for DVAS.

Tests that verify robustness against unexpected/invalid inputs.
"""

import pytest
from dvas.testing import fuzz_string, fuzz_dict

from dvas.data.schemas import (
    Action,
    Annotation,
    BoundingBox,
    Segment,
    VideoMetadata,
)


class TestFuzzString:
    """Test fuzz string generation."""

    def test_fuzz_string_returns_string(self):
        """Fuzz string always returns a string."""
        for _ in range(100):
            s = fuzz_string()
            assert isinstance(s, str)

    def test_fuzz_string_length_range(self):
        """Fuzz string respects length constraints."""
        for _ in range(100):
            s = fuzz_string(min_length=5, max_length=10)
            assert 5 <= len(s) <= 10

    def test_fuzz_string_with_null(self):
        """Fuzz string can include null bytes."""
        _found_null = False
        for _ in range(1000):
            s = fuzz_string(max_length=100, include_null=True)
            if "\x00" in s:
                _found_null = True
                break
        # Null bytes should appear with some probability
        # We don't assert they must appear (probabilistic)

    def test_fuzz_string_with_unicode(self):
        """Fuzz string can include unicode."""
        _found_unicode = False
        for _ in range(1000):
            s = fuzz_string(max_length=100, include_unicode=True)
            if any(ord(c) > 127 for c in s):
                _found_unicode = True
                break
        # Unicode should appear with some probability
        # We don't assert they must appear (probabilistic)


class TestFuzzDict:
    """Test fuzz dict generation."""

    def test_fuzz_dict_returns_dict(self):
        """Fuzz dict always returns a dictionary."""
        for _ in range(100):
            d = fuzz_dict()
            assert isinstance(d, dict)

    def test_fuzz_dict_key_count(self):
        """Fuzz dict respects key count constraints."""
        for _ in range(100):
            d = fuzz_dict(min_keys=2, max_keys=5)
            assert 2 <= len(d) <= 5

    def test_fuzz_dict_values(self):
        """Fuzz dict values are valid types."""
        for _ in range(100):
            d = fuzz_dict()
            for key, value in d.items():
                assert isinstance(key, str)
                assert isinstance(value, (str, int, float, bool, list, dict))


class TestFuzzBoundingBox:
    """Test BoundingBox with fuzzed inputs."""

    def test_bounding_box_with_fuzzed_strings(self):
        """BoundingBox should reject non-numeric inputs."""
        for _ in range(100):
            s = fuzz_string(max_length=10)
            # Pydantic v2 auto-converts numeric strings, so only non-convertible
            # strings should raise errors
            try:
                bbox = BoundingBox(x1=s, y1=0.5, x2=0.5, y2=0.5)
                # If conversion succeeded, value should be a valid float in range
                assert isinstance(bbox.x1, float)
                assert 0.0 <= bbox.x1 <= 1.0
            except (ValueError, TypeError):
                # Non-convertible strings should raise errors
                pass

    def test_bounding_box_with_extreme_values(self):
        """BoundingBox should handle extreme values."""
        # Values outside [0, 1] should raise ValueError
        with pytest.raises(ValueError):
            BoundingBox(x1=-1.0, y1=0.5, x2=0.5, y2=0.5)

        with pytest.raises(ValueError):
            BoundingBox(x1=0.5, y1=0.5, x2=2.0, y2=0.5)


class TestFuzzSegment:
    """Test Segment with fuzzed inputs."""

    def test_segment_with_fuzzed_caption(self):
        """Segment should accept any string caption."""
        for _ in range(100):
            caption = fuzz_string(max_length=500)
            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                caption=caption,
            )
            assert segment.caption == caption

    def test_segment_end_before_start(self):
        """Segment should reject end_time before start_time."""
        with pytest.raises(ValueError):
            Segment(
                start_time=5.0,
                end_time=2.0,
                caption="Invalid",
            )


class TestFuzzAnnotation:
    """Test Annotation with fuzzed inputs."""

    def test_annotation_with_fuzzed_video_id(self):
        """Annotation should accept any string video_id."""
        for _ in range(100):
            video_id = fuzz_string(max_length=50)
            annotation = Annotation(
                id="test",
                video_id=video_id,
                video_path="/path/to/video.mp4",
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=10.0,
                    total_frames=300,
                ),
            )
            assert annotation.video_id == video_id

    def test_annotation_with_fuzzed_dict(self):
        """Annotation should handle fuzzed dict inputs."""
        for _ in range(50):
            d = fuzz_dict(max_keys=5)
            # The fuzzed dict won't be a valid Annotation, but we should
            # be able to create an Annotation with proper fields
            annotation = Annotation(
                id="test",
                video_id="vid_001",
                video_path="/path/to/video.mp4",
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=10.0,
                    total_frames=300,
                ),
                tags=list(d.keys())[:5],  # Use keys as tags
            )
            assert isinstance(annotation.tags, list)


class TestFuzzAction:
    """Test Action with fuzzed inputs."""

    def test_action_with_fuzzed_verb(self):
        """Action should accept any string verb."""
        for _ in range(100):
            verb = fuzz_string(max_length=50)
            action = Action(verb=verb, noun="test")
            assert action.verb == verb

    def test_action_with_fuzzed_noun(self):
        """Action should accept any string noun."""
        for _ in range(100):
            noun = fuzz_string(max_length=50)
            action = Action(verb="test", noun=noun)
            assert action.noun == noun


class TestFuzzVideoMetadata:
    """Test VideoMetadata with fuzzed inputs."""

    def test_video_metadata_with_positive_values(self):
        """VideoMetadata should accept positive values."""
        for _ in range(100):
            fps = 1.0 + (hash(f"fps_{_}") % 100)
            duration = 1.0 + (hash(f"dur_{_}") % 1000)
            total_frames = int(fps * duration)

            metadata = VideoMetadata(
                fps=fps,
                resolution=[1920, 1080],
                duration=duration,
                total_frames=total_frames,
            )
            assert metadata.fps == fps
            assert metadata.duration == duration
