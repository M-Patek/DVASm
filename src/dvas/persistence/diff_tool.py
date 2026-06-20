"""Annotation diff tool for comparing versions.

Provides utilities for:
- Comparing two annotations (diff)
- Comparing annotation versions
- Generating human-readable diffs
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import orjson

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import DiffResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FieldChange:
    """A single field change."""

    field: str
    old_value: Any
    new_value: Any
    change_type: str  # "added", "removed", "modified"


@dataclass
class SegmentChange:
    """Segment-level change."""

    index: int
    change_type: str  # "added", "removed", "modified"
    fields_changed: List[FieldChange] = None

    def __post_init__(self):
        if self.fields_changed is None:
            self.fields_changed = []


@dataclass
class AnnotationDiff:
    """Complete diff between two annotations."""

    annotation_id: str
    field_changes: List[FieldChange]
    segment_changes: List[SegmentChange]
    unchanged: bool


def compute_annotation_diff(
    annotation1: Annotation,
    annotation2: Annotation,
) -> AnnotationDiff:
    """Compute detailed diff between two annotations.

    Args:
        annotation1: First annotation (old)
        annotation2: Second annotation (new)

    Returns:
        AnnotationDiff with detailed changes
    """
    if annotation1.id != annotation2.id:
        raise ValueError(
            f"Cannot diff annotations with different IDs: {annotation1.id} vs {annotation2.id}"
        )

    data1 = annotation1.model_dump()
    data2 = annotation2.model_dump()

    field_changes = _compute_field_changes(data1, data2)
    segment_changes = _compute_segment_changes(
        data1.get("segments", []),
        data2.get("segments", []),
    )

    unchanged = len(field_changes) == 0 and len(segment_changes) == 0

    return AnnotationDiff(
        annotation_id=annotation1.id,
        field_changes=field_changes,
        segment_changes=segment_changes,
        unchanged=unchanged,
    )


def _compute_field_changes(data1: Dict, data2: Dict) -> List[FieldChange]:
    """Compute field-level changes."""
    changes = []
    all_fields = set(data1.keys()) | set(data2.keys())

    # Exclude segments (handled separately)
    all_fields.discard("segments")

    for field in all_fields:
        val1 = data1.get(field)
        val2 = data2.get(field)

        if field not in data1:
            changes.append(FieldChange(field, None, val2, "added"))
        elif field not in data2:
            changes.append(FieldChange(field, val1, None, "removed"))
        elif val1 != val2:
            changes.append(FieldChange(field, val1, val2, "modified"))

    return changes


def _compute_segment_changes(segments1: List[Dict], segments2: List[Dict]) -> List[SegmentChange]:
    """Compute segment-level changes."""
    changes = []

    # Find added/removed/modified segments
    max_len = max(len(segments1), len(segments2))

    for i in range(max_len):
        if i >= len(segments1):
            # Segment added
            changes.append(SegmentChange(i, "added", []))
        elif i >= len(segments2):
            # Segment removed
            changes.append(SegmentChange(i, "removed", []))
        elif segments1[i] != segments2[i]:
            # Segment modified
            field_changes = _compute_field_changes(segments1[i], segments2[i])
            changes.append(SegmentChange(i, "modified", field_changes))

    return changes


def format_diff(diff: AnnotationDiff, verbose: bool = False) -> str:
    """Format diff as human-readable string.

    Args:
        diff: AnnotationDiff to format
        verbose: Include full values or just summaries

    Returns:
        Formatted diff string
    """
    if diff.unchanged:
        return f"No changes for annotation {diff.annotation_id}"

    lines = [f"Changes for annotation {diff.annotation_id}:", ""]

    # Field changes
    if diff.field_changes:
        lines.append("Field Changes:")
        for change in diff.field_changes:
            if change.change_type == "added":
                lines.append(f"  + {change.field}: {change.new_value}")
            elif change.change_type == "removed":
                lines.append(f"  - {change.field}: (was {change.old_value})")
            else:
                if verbose:
                    lines.append(f"  ~ {change.field}:")
                    lines.append(f"    - {change.old_value}")
                    lines.append(f"    + {change.new_value}")
                else:
                    lines.append(f"  ~ {change.field}: changed")
        lines.append("")

    # Segment changes
    if diff.segment_changes:
        lines.append("Segment Changes:")
        for change in diff.segment_changes:
            prefix = "+" if change.change_type == "added" else "-" if change.change_type == "removed" else "~"
            lines.append(f"  [{prefix}] Segment {change.index}")

            if change.fields_changed and verbose:
                for field_change in change.fields_changed:
                    lines.append(f"      {field_change.field}: {field_change.change_type}")
        lines.append("")

    return "\n".join(lines)


def diff_annotations(
    annotation1: Annotation,
    annotation2: Annotation,
    format_output: bool = False,
) -> AnnotationDiff:
    """Diff two annotations.

    Args:
        annotation1: First annotation
        annotation2: Second annotation
        format_output: Also return formatted string

    Returns:
        AnnotationDiff, optionally with formatted string
    """
    diff = compute_annotation_diff(annotation1, annotation2)

    if format_output:
        return diff, format_diff(diff)

    return diff


def diff_json(json1: str, json2: str) -> AnnotationDiff:
    """Diff two annotation JSON strings.

    Args:
        json1: First annotation JSON
        json2: Second annotation JSON

    Returns:
        AnnotationDiff
    """
    ann1 = Annotation.model_validate(orjson.loads(json1))
    ann2 = Annotation.model_validate(orjson.loads(json2))
    return compute_annotation_diff(ann1, ann2)


def diff_from_backend_result(result: DiffResult) -> AnnotationDiff:
    """Convert backend DiffResult to AnnotationDiff.

    Args:
        result: Backend DiffResult

    Returns:
        AnnotationDiff
    """
    field_changes = []
    for field, (old, new) in result.field_changes.items():
        change_type = "added" if old is None else "removed" if new is None else "modified"
        field_changes.append(FieldChange(field, old, new, change_type))

    segment_changes = []
    for idx in result.segments_added:
        segment_changes.append(SegmentChange(idx, "added"))
    for idx in result.segments_removed:
        segment_changes.append(SegmentChange(idx, "removed"))
    for idx in result.segments_modified:
        segment_changes.append(SegmentChange(idx, "modified"))

    return AnnotationDiff(
        annotation_id=result.annotation_id,
        field_changes=field_changes,
        segment_changes=segment_changes,
        unchanged=result.unchanged,
    )
