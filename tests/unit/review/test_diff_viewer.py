"""Tests for annotation diff computation."""

import pytest

from dvas.data.schemas import Action, Annotation, Hand, Object, Segment, VideoMetadata
from dvas.review.diff_viewer import AnnotationDiff, DiffType


class TestAnnotationDiff:
    """Test suite for AnnotationDiff."""

    def _make_annotation(self, ann_id: str = "ann1") -> Annotation:
        """Create a test annotation with two segments."""
        metadata = VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.0,
            total_frames=300,
        )
        segment1 = Segment(
            start_time=0.0,
            end_time=3.0,
            caption="First segment",
            actions=[
                Action(verb="take", noun="cup", hand=Hand.RIGHT),
                Action(verb="pour", noun="water", hand=Hand.LEFT),
            ],
            objects=[Object(name="cup"), Object(name="water")],
        )
        segment2 = Segment(
            start_time=3.0,
            end_time=6.0,
            caption="Second segment",
            actions=[Action(verb="place", noun="cup", hand=Hand.RIGHT)],
            objects=[Object(name="cup")],
        )
        return Annotation(
            id=ann_id,
            video_id="vid1",
            video_path="test.mp4",
            metadata=metadata,
            segments=[segment1, segment2],
        )

    def test_identical_annotations(self):
        """Test diff between identical annotations."""
        ann = self._make_annotation("ann1")
        diff = AnnotationDiff(ann, ann)
        result = diff.compute_diff()

        assert result.total_changes == 0
        assert len(result.unchanged_segments) == 2
        assert len(result.added_segments) == 0
        assert len(result.removed_segments) == 0

    def test_caption_difference(self):
        """Test diff with caption change."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments[0].caption = "Modified first segment"

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert result.total_changes == 1
        assert result.modifications == 1
        assert len(result.unchanged_segments) == 1

        seg_diff = result.segment_diffs[0]
        assert seg_diff.caption_diff is not None
        assert seg_diff.caption_diff.old_value == "First segment"
        assert seg_diff.caption_diff.new_value == "Modified first segment"

    def test_action_added(self):
        """Test diff with added action."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments[0].actions.append(Action(verb="stir", noun="water", hand=Hand.RIGHT))

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert result.total_changes == 1
        assert len(result.segment_diffs[0].added_actions) == 1
        assert result.segment_diffs[0].added_actions[0]["verb"] == "stir"

    def test_action_removed(self):
        """Test diff with removed action."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments[0].actions.pop(0)  # Remove first action

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert result.total_changes == 1
        assert len(result.segment_diffs[0].removed_actions) == 1
        assert result.segment_diffs[0].removed_actions[0]["verb"] == "take"

    def test_object_added(self):
        """Test diff with added object."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments[0].objects.append(Object(name="spoon"))

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert len(result.segment_diffs[0].added_objects) == 1
        assert result.segment_diffs[0].added_objects[0]["name"] == "spoon"

    def test_segment_added(self):
        """Test diff with added segment."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        new_segment = Segment(
            start_time=6.0,
            end_time=9.0,
            caption="Third segment",
        )
        ann_b.segments.append(new_segment)

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert len(result.added_segments) == 1
        assert result.additions == 1

    def test_segment_removed(self):
        """Test diff with removed segment."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments.pop(0)

        diff = AnnotationDiff(ann_a, ann_b)
        result = diff.compute_diff()

        assert len(result.removed_segments) == 1
        assert result.removals == 1

    def test_diff_statistics(self):
        """Test diff statistics computation."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")
        ann_b.segments[0].caption = "Changed"
        ann_b.segments[0].objects.append(Object(name="spoon"))

        diff = AnnotationDiff(ann_a, ann_b)
        stats = diff.get_diff_statistics()

        assert stats["total_segments"] == 2
        assert stats["unchanged_segments"] == 1
        assert stats["total_changes"] > 0
        assert "change_rate" in stats

    def test_has_changes(self):
        """Test has_changes method."""
        ann_a = self._make_annotation("ann_a")
        ann_b = self._make_annotation("ann_b")

        diff = AnnotationDiff(ann_a, ann_b)
        assert not diff.has_changes()

        ann_b.segments[0].caption = "Changed"
        diff = AnnotationDiff(ann_a, ann_b)
        assert diff.has_changes()
