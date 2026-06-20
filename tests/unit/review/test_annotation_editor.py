"""Tests for annotation editor change tracking."""

import pytest

from dvas.data.schemas import Action, Annotation, Hand, Object, Segment, VideoMetadata
from dvas.review.annotation_editor import AnnotationEditor, ChangeType


class TestAnnotationEditor:
    """Test suite for AnnotationEditor."""

    def _make_annotation(self) -> Annotation:
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
            ],
            objects=[Object(name="cup")],
        )
        segment2 = Segment(
            start_time=3.0,
            end_time=6.0,
            caption="Second segment",
            actions=[],
            objects=[],
        )
        return Annotation(
            id="ann1",
            video_id="vid1",
            video_path="test.mp4",
            metadata=metadata,
            segments=[segment1, segment2],
        )

    def test_add_action(self):
        """Test adding an action to a segment."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        new_action = Action(verb="pour", noun="water", hand=Hand.LEFT)
        editor.add_action(0, new_action)

        assert len(editor.annotation.segments[0].actions) == 2
        assert editor.has_changes is True
        assert editor.edit_history.change_count == 1

    def test_remove_action(self):
        """Test removing an action from a segment."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        removed = editor.remove_action(0, 0)

        assert removed is not None
        assert removed.verb == "take"
        assert len(editor.annotation.segments[0].actions) == 0
        assert editor.has_changes is True

    def test_edit_action(self):
        """Test editing an action."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        new_action = Action(verb="grab", noun="cup", hand=Hand.LEFT)
        editor.edit_action(0, 0, new_action)

        assert editor.annotation.segments[0].actions[0].verb == "grab"
        assert editor.annotation.segments[0].actions[0].hand == Hand.LEFT

    def test_add_object(self):
        """Test adding an object to a segment."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        new_obj = Object(name="spoon")
        editor.add_object(0, new_obj)

        assert len(editor.annotation.segments[0].objects) == 2
        assert editor.annotation.segments[0].objects[1].name == "spoon"

    def test_remove_object(self):
        """Test removing an object from a segment."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        removed = editor.remove_object(0, 0)

        assert removed is not None
        assert removed.name == "cup"
        assert len(editor.annotation.segments[0].objects) == 0

    def test_edit_caption(self):
        """Test editing a segment caption."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        editor.edit_caption(0, "Updated caption")

        assert editor.annotation.segments[0].caption == "Updated caption"

    def test_undo_add_action(self):
        """Test undoing an add action."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        original_count = len(ann.segments[0].actions)
        editor.add_action(0, Action(verb="pour", noun="water"))
        assert len(editor.annotation.segments[0].actions) == original_count + 1

        undone = editor.undo()
        assert undone is not None
        assert len(editor.annotation.segments[0].actions) == original_count

    def test_redo_add_action(self):
        """Test redoing an undone action."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        editor.add_action(0, Action(verb="pour", noun="water"))
        original_count = len(editor.annotation.segments[0].actions)

        editor.undo()
        assert len(editor.annotation.segments[0].actions) == original_count - 1

        editor.redo()
        assert len(editor.annotation.segments[0].actions) == original_count

    def test_undo_empty_stack(self):
        """Test undo with empty stack."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        result = editor.undo()
        assert result is None

    def test_redo_empty_stack(self):
        """Test redo with empty stack."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        result = editor.redo()
        assert result is None

    def test_change_summary(self):
        """Test change summary."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        editor.add_action(0, Action(verb="pour", noun="water"))
        editor.add_object(0, Object(name="spoon"))
        editor.edit_caption(0, "Updated")

        summary = editor.get_change_summary()
        assert summary[ChangeType.ADD_ACTION.value] == 1
        assert summary[ChangeType.ADD_OBJECT.value] == 1
        assert summary[ChangeType.EDIT_CAPTION.value] == 1

    def test_finalize(self):
        """Test finalizing edits."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann, reviewer_id="rev1")

        editor.add_action(0, Action(verb="pour", noun="water"))
        result = editor.finalize()

        assert result is editor.annotation
        assert editor.edit_history.is_complete is True
        assert editor.edit_history.reviewer_id == "rev1"

    def test_out_of_range_segment(self):
        """Test operations with invalid segment index."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        with pytest.raises(IndexError):
            editor.add_action(10, Action(verb="pour", noun="water"))

    def test_out_of_range_action(self):
        """Test operations with invalid action index."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        with pytest.raises(IndexError):
            editor.remove_action(0, 10)

    def test_original_preserved(self):
        """Test that original annotation is preserved."""
        ann = self._make_annotation()
        editor = AnnotationEditor(ann)

        editor.add_action(0, Action(verb="pour", noun="water"))

        assert len(editor.original.segments[0].actions) == 1
        assert len(editor.annotation.segments[0].actions) == 2
