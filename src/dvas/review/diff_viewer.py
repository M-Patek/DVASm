"""Annotation diff viewer for comparing annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from dvas.data.schemas import Action, Annotation, Object, Segment
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DiffType(str, Enum):
    """Types of differences between annotations."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class FieldDiff:
    """Difference for a single field."""
    field_name: str
    diff_type: DiffType
    old_value: Any
    new_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "diff_type": self.diff_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass
class SegmentDiff:
    """Difference between two segments."""
    segment_idx: int
    diff_type: DiffType
    caption_diff: Optional[FieldDiff] = None
    action_diffs: List[FieldDiff] = field(default_factory=list)
    object_diffs: List[FieldDiff] = field(default_factory=list)
    added_actions: List[Dict[str, Any]] = field(default_factory=list)
    removed_actions: List[Dict[str, Any]] = field(default_factory=list)
    added_objects: List[Dict[str, Any]] = field(default_factory=list)
    removed_objects: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_idx": self.segment_idx,
            "diff_type": self.diff_type.value,
            "caption_diff": self.caption_diff.to_dict() if self.caption_diff else None,
            "action_diffs": [d.to_dict() for d in self.action_diffs],
            "object_diffs": [d.to_dict() for d in self.object_diffs],
            "added_actions": self.added_actions,
            "removed_actions": self.removed_actions,
            "added_objects": self.added_objects,
            "removed_objects": self.removed_objects,
        }


@dataclass
class AnnotationDiffResult:
    """Complete diff result between two annotations."""
    annotation_a_id: str
    annotation_b_id: str
    segment_diffs: List[SegmentDiff] = field(default_factory=list)
    added_segments: List[int] = field(default_factory=list)
    removed_segments: List[int] = field(default_factory=list)
    unchanged_segments: List[int] = field(default_factory=list)
    total_changes: int = 0
    additions: int = 0
    removals: int = 0
    modifications: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_a_id": self.annotation_a_id,
            "annotation_b_id": self.annotation_b_id,
            "segment_diffs": [sd.to_dict() for sd in self.segment_diffs],
            "added_segments": self.added_segments,
            "removed_segments": self.removed_segments,
            "unchanged_segments": self.unchanged_segments,
            "total_changes": self.total_changes,
            "additions": self.additions,
            "removals": self.removals,
            "modifications": self.modifications,
        }


class AnnotationDiff:
    """Diff viewer for comparing two annotations."""

    def __init__(self, annotation_a: Annotation, annotation_b: Annotation):
        self.annotation_a = annotation_a
        self.annotation_b = annotation_b

    def compute_diff(self) -> AnnotationDiffResult:
        """Compute the full diff between the two annotations."""
        result = AnnotationDiffResult(
            annotation_a_id=self.annotation_a.id,
            annotation_b_id=self.annotation_b.id,
        )

        segments_a = self.annotation_a.segments
        segments_b = self.annotation_b.segments
        max_segments = max(len(segments_a), len(segments_b))

        for i in range(max_segments):
            if i >= len(segments_a):
                result.added_segments.append(i)
                result.additions += 1
                result.total_changes += 1
            elif i >= len(segments_b):
                result.removed_segments.append(i)
                result.removals += 1
                result.total_changes += 1
            else:
                seg_diff = self._compare_segments(i, segments_a[i], segments_b[i])
                result.segment_diffs.append(seg_diff)
                if seg_diff.diff_type == DiffType.UNCHANGED:
                    result.unchanged_segments.append(i)
                elif seg_diff.diff_type == DiffType.MODIFIED:
                    result.modifications += 1
                    result.total_changes += 1

        return result

    def _compare_segments(self, segment_idx: int, segment_a: Segment, segment_b: Segment) -> SegmentDiff:
        """Compare two segments and return their diff."""
        seg_diff = SegmentDiff(
            segment_idx=segment_idx,
            diff_type=DiffType.UNCHANGED,
        )

        # Compare captions
        if segment_a.caption != segment_b.caption:
            seg_diff.diff_type = DiffType.MODIFIED
            seg_diff.caption_diff = FieldDiff(
                field_name="caption",
                diff_type=DiffType.MODIFIED,
                old_value=segment_a.caption,
                new_value=segment_b.caption,
            )

        # Compare actions
        actions_a = [a.model_dump() for a in segment_a.actions]
        actions_b = [a.model_dump() for a in segment_b.actions]
        action_diffs = self._compare_lists(actions_a, actions_b, ["verb", "noun"])

        seg_diff.added_actions = action_diffs["added"]
        seg_diff.removed_actions = action_diffs["removed"]
        seg_diff.action_diffs = action_diffs["modified"]

        if seg_diff.added_actions:
            seg_diff.diff_type = DiffType.MODIFIED
        if seg_diff.removed_actions:
            seg_diff.diff_type = DiffType.MODIFIED

        # Compare objects
        objects_a = [o.model_dump() for o in segment_a.objects]
        objects_b = [o.model_dump() for o in segment_b.objects]
        object_diffs = self._compare_lists(objects_a, objects_b, ["name"])

        seg_diff.added_objects = object_diffs["added"]
        seg_diff.removed_objects = object_diffs["removed"]
        seg_diff.object_diffs = object_diffs["modified"]

        if seg_diff.added_objects:
            seg_diff.diff_type = DiffType.MODIFIED
        if seg_diff.removed_objects:
            seg_diff.diff_type = DiffType.MODIFIED

        return seg_diff

    def _compare_lists(self, list_a: List[Dict[str, Any]], list_b: List[Dict[str, Any]], key_fields: List[str]) -> Dict[str, Any]:
        """Compare two lists of dictionaries by key fields."""
        def make_key(item: Dict[str, Any]) -> str:
            return "|".join(str(item.get(f, "")) for f in key_fields)

        keys_a = {make_key(item): item for item in list_a}
        keys_b = {make_key(item): item for item in list_b}

        added = []
        removed = []
        modified = []

        # Find added (in B but not in A)
        for key, item in keys_b.items():
            if key not in keys_a:
                added.append(item)

        # Find removed (in A but not in B)
        for key, item in keys_a.items():
            if key not in keys_b:
                removed.append(item)

        # Find modified (same key but different fields)
        for key in keys_a:
            if key in keys_b:
                item_a = keys_a[key]
                item_b = keys_b[key]
                if item_a != item_b:
                    for field_name in item_a:
                        if field_name in item_b and item_a[field_name] != item_b[field_name]:
                            modified.append(
                                FieldDiff(
                                    field_name=field_name,
                                    diff_type=DiffType.MODIFIED,
                                    old_value=item_a[field_name],
                                    new_value=item_b[field_name],
                                )
                            )

        return {"added": added, "removed": removed, "modified": modified}

    def get_diff_statistics(self) -> Dict[str, Any]:
        """Get statistics about the diff."""
        result = self.compute_diff()
        total_segments = max(len(self.annotation_a.segments), len(self.annotation_b.segments))

        return {
            "total_segments": total_segments,
            "unchanged_segments": len(result.unchanged_segments),
            "added_segments": len(result.added_segments),
            "removed_segments": len(result.removed_segments),
            "modified_segments": result.modifications,
            "total_changes": result.total_changes,
            "additions": result.additions,
            "removals": result.removals,
            "change_rate": result.total_changes / total_segments if total_segments > 0 else 0.0,
        }

    def has_changes(self) -> bool:
        """Check if there are any differences between the annotations."""
        result = self.compute_diff()
        return result.total_changes > 0
