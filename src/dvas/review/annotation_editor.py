"""Annotation editor with change tracking for review workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Action, Annotation, Object
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class ChangeType(str, Enum):
    """Type of change made to an annotation."""

    ADD_ACTION = "add_action"
    REMOVE_ACTION = "remove_action"
    EDIT_ACTION = "edit_action"
    ADD_OBJECT = "add_object"
    REMOVE_OBJECT = "remove_object"
    EDIT_OBJECT = "edit_object"
    EDIT_SEGMENT = "edit_segment"
    ADD_SEGMENT = "add_segment"
    REMOVE_SEGMENT = "remove_segment"
    EDIT_CAPTION = "edit_caption"
    EDIT_METADATA = "edit_metadata"


@dataclass
class ChangeRecord:
    """Record of a single change to an annotation."""

    change_type: ChangeType
    segment_index: Optional[int] = None
    action_index: Optional[int] = None
    object_index: Optional[int] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reviewer_id: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "segment_index": self.segment_index,
            "action_index": self.action_index,
            "object_index": self.object_index,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "reviewer_id": self.reviewer_id,
            "reason": self.reason,
        }


@dataclass
class AnnotationEdit:
    """Complete edit session for an annotation."""

    annotation_id: str
    changes: List[ChangeRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def change_count(self) -> int:
        return len(self.changes)

    def add_change(self, change: ChangeRecord) -> None:
        self.changes.append(change)

    def get_changes_by_type(self, change_type: ChangeType) -> List[ChangeRecord]:
        return [c for c in self.changes if c.change_type == change_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "changes": [c.to_dict() for c in self.changes],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reviewer_id": self.reviewer_id,
            "change_count": self.change_count,
        }


class AnnotationEditor:
    """Editor for annotations with change tracking."""

    def __init__(self, annotation: Annotation, reviewer_id: Optional[str] = None):
        self._original = annotation
        self._annotation = annotation.model_copy(deep=True)
        self._reviewer_id = reviewer_id
        self._edit = AnnotationEdit(annotation_id=annotation.id, reviewer_id=reviewer_id)
        self._undo_stack: List[ChangeRecord] = []
        self._redo_stack: List[ChangeRecord] = []

    @property
    def annotation(self) -> Annotation:
        return self._annotation

    @property
    def original(self) -> Annotation:
        return self._original

    @property
    def edit_history(self) -> AnnotationEdit:
        return self._edit

    @property
    def has_changes(self) -> bool:
        return len(self._edit.changes) > 0

    def add_action(self, segment_index: int, action: Action, reason: Optional[str] = None) -> None:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        segment.actions.append(action)
        change = ChangeRecord(
            change_type=ChangeType.ADD_ACTION,
            segment_index=segment_index,
            action_index=len(segment.actions) - 1,
            old_value=None,
            new_value=action.model_dump(),
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info("action_added", annotation_id=self._annotation.id, segment_index=segment_index)

    def remove_action(
        self, segment_index: int, action_index: int, reason: Optional[str] = None
    ) -> Optional[Action]:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        if action_index < 0 or action_index >= len(segment.actions):
            raise IndexError(f"Action index {action_index} out of range")
        removed = segment.actions.pop(action_index)
        change = ChangeRecord(
            change_type=ChangeType.REMOVE_ACTION,
            segment_index=segment_index,
            action_index=action_index,
            old_value=removed.model_dump(),
            new_value=None,
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info(
            "action_removed",
            annotation_id=self._annotation.id,
            segment_index=segment_index,
            action_index=action_index,
        )
        return removed

    def edit_action(
        self,
        segment_index: int,
        action_index: int,
        new_action: Action,
        reason: Optional[str] = None,
    ) -> None:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        if action_index < 0 or action_index >= len(segment.actions):
            raise IndexError(f"Action index {action_index} out of range")
        old_action = segment.actions[action_index]
        segment.actions[action_index] = new_action
        change = ChangeRecord(
            change_type=ChangeType.EDIT_ACTION,
            segment_index=segment_index,
            action_index=action_index,
            old_value=old_action.model_dump(),
            new_value=new_action.model_dump(),
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info(
            "action_edited",
            annotation_id=self._annotation.id,
            segment_index=segment_index,
            action_index=action_index,
        )

    def add_object(self, segment_index: int, obj: Object, reason: Optional[str] = None) -> None:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        segment.objects.append(obj)
        change = ChangeRecord(
            change_type=ChangeType.ADD_OBJECT,
            segment_index=segment_index,
            object_index=len(segment.objects) - 1,
            old_value=None,
            new_value=obj.model_dump(),
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info(
            "object_added",
            annotation_id=self._annotation.id,
            segment_index=segment_index,
            object_name=obj.name,
        )

    def remove_object(
        self, segment_index: int, object_index: int, reason: Optional[str] = None
    ) -> Optional[Object]:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        if object_index < 0 or object_index >= len(segment.objects):
            raise IndexError(f"Object index {object_index} out of range")
        removed = segment.objects.pop(object_index)
        change = ChangeRecord(
            change_type=ChangeType.REMOVE_OBJECT,
            segment_index=segment_index,
            object_index=object_index,
            old_value=removed.model_dump(),
            new_value=None,
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info(
            "object_removed",
            annotation_id=self._annotation.id,
            segment_index=segment_index,
            object_index=object_index,
        )
        return removed

    def edit_caption(
        self, segment_index: int, new_caption: str, reason: Optional[str] = None
    ) -> None:
        if segment_index < 0 or segment_index >= len(self._annotation.segments):
            raise IndexError(f"Segment index {segment_index} out of range")
        segment = self._annotation.segments[segment_index]
        old_caption = segment.caption
        segment.caption = new_caption
        change = ChangeRecord(
            change_type=ChangeType.EDIT_CAPTION,
            segment_index=segment_index,
            old_value=old_caption,
            new_value=new_caption,
            reviewer_id=self._reviewer_id,
            reason=reason,
        )
        self._edit.add_change(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()
        logger.info(
            "caption_edited", annotation_id=self._annotation.id, segment_index=segment_index
        )

    def undo(self) -> Optional[ChangeRecord]:
        if not self._undo_stack:
            return None
        change = self._undo_stack.pop()
        self._redo_stack.append(change)
        self._revert_change(change)
        logger.info(
            "change_undone", annotation_id=self._annotation.id, change_type=change.change_type.value
        )
        return change

    def redo(self) -> Optional[ChangeRecord]:
        if not self._redo_stack:
            return None
        change = self._redo_stack.pop()
        self._undo_stack.append(change)
        self._apply_change(change)
        logger.info(
            "change_redone", annotation_id=self._annotation.id, change_type=change.change_type.value
        )
        return change

    def _revert_change(self, change: ChangeRecord) -> None:
        if change.segment_index is None:
            return
        segment = self._annotation.segments[change.segment_index]
        if change.change_type == ChangeType.ADD_ACTION:
            if change.action_index is not None and change.action_index < len(segment.actions):
                segment.actions.pop(change.action_index)
        elif change.change_type == ChangeType.REMOVE_ACTION:
            if change.old_value and change.action_index is not None:
                action = Action.model_validate(change.old_value)
                segment.actions.insert(change.action_index, action)
        elif change.change_type == ChangeType.EDIT_ACTION:
            if change.old_value and change.action_index is not None:
                segment.actions[change.action_index] = Action.model_validate(change.old_value)
        elif change.change_type == ChangeType.ADD_OBJECT:
            if change.object_index is not None and change.object_index < len(segment.objects):
                segment.objects.pop(change.object_index)
        elif change.change_type == ChangeType.REMOVE_OBJECT:
            if change.old_value and change.object_index is not None:
                obj = Object.model_validate(change.old_value)
                segment.objects.insert(change.object_index, obj)
        elif change.change_type == ChangeType.EDIT_CAPTION:
            if change.old_value is not None:
                segment.caption = change.old_value

    def _apply_change(self, change: ChangeRecord) -> None:
        if change.segment_index is None:
            return
        segment = self._annotation.segments[change.segment_index]
        if change.change_type == ChangeType.ADD_ACTION:
            if change.new_value:
                segment.actions.append(Action.model_validate(change.new_value))
        elif change.change_type == ChangeType.REMOVE_ACTION:
            if change.action_index is not None and change.action_index < len(segment.actions):
                segment.actions.pop(change.action_index)
        elif change.change_type == ChangeType.EDIT_ACTION:
            if change.new_value and change.action_index is not None:
                segment.actions[change.action_index] = Action.model_validate(change.new_value)
        elif change.change_type == ChangeType.ADD_OBJECT:
            if change.new_value:
                segment.objects.append(Object.model_validate(change.new_value))
        elif change.change_type == ChangeType.REMOVE_OBJECT:
            if change.object_index is not None and change.object_index < len(segment.objects):
                segment.objects.pop(change.object_index)
        elif change.change_type == ChangeType.EDIT_CAPTION:
            if change.new_value is not None:
                segment.caption = change.new_value

    def get_change_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for change in self._edit.changes:
            key = change.change_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    def finalize(self) -> Annotation:
        self._annotation.updated_at = datetime.now(timezone.utc)
        self._edit.completed_at = datetime.utcnow()
        logger.info(
            "annotation_edit_finalized",
            annotation_id=self._annotation.id,
            change_count=self._edit.change_count,
        )
        return self._annotation
