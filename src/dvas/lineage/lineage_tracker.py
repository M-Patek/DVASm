"""Lineage — Schema version management and data provenance tracking.

Tracks the full lifecycle of annotations from raw video to training data,
including schema version compatibility checks.

Usage:
    from dvas.lineage import LineageTracker, SchemaVersion

    tracker = LineageTracker()
    tracker.record_step("ann_001", "pipeline_annotation", {"model": "gpt-5.5"})
    provenance = tracker.get_provenance("ann_001")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation, AnnotationStandard


class SchemaVersion(str, Enum):
    """Schema version identifiers."""

    V1_0 = "1.0"  # EPIC-KITCHENS baseline
    V2_0 = "2.0"  # VLA enhanced + World Model compatible
    V3_0 = "3.0"  # Future: full World Model support


@dataclass
class LineageStep:
    """A single step in an annotation's lifecycle."""

    step: str  # e.g. "pipeline_annotation", "human_review", "export_llava"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent: Optional[str] = None  # "teacher", "human_reviewer", "export_adapter"


@dataclass
class SchemaCompatibility:
    """Result of a compatibility check."""

    compatible: bool
    source_version: str
    target_version: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class LineageTracker:
    """Track data lineage and schema compatibility."""

    # Schema compatibility matrix
    # Maps (source_version, target_version) -> compatible?
    _COMPATIBILITY = {
        ("1.0", "1.0"): True,
        ("1.0", "2.0"): True,  # v2.0 is backward compatible
        ("1.0", "3.0"): False,  # v3.0 may drop support
        ("2.0", "1.0"): False,  # v2.0 data cannot go back to v1.0
        ("2.0", "2.0"): True,
        ("2.0", "3.0"): True,  # v3.0 will be backward compatible
    }

    def __init__(self):
        self._lineage: Dict[str, List[LineageStep]] = {}

    def record_step(
        self,
        annotation_id: str,
        step: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Record a processing step for an annotation.

        Args:
            annotation_id: Unique annotation ID
            step: Step name (e.g. "pipeline_annotation")
            metadata: Additional metadata about this step
            agent: Agent that performed this step
        """
        if annotation_id not in self._lineage:
            self._lineage[annotation_id] = []

        self._lineage[annotation_id].append(
            LineageStep(
                step=step,
                metadata=metadata or {},
                agent=agent,
            )
        )

    def get_provenance(self, annotation_id: str) -> List[LineageStep]:
        """Get full provenance for an annotation.

        Returns:
            List of lineage steps in chronological order.
        """
        return self._lineage.get(annotation_id, [])

    def get_last_step(self, annotation_id: str) -> Optional[LineageStep]:
        """Get the most recent step for an annotation."""
        steps = self._lineage.get(annotation_id)
        return steps[-1] if steps else None

    def check_compatibility(
        self, annotation: Annotation, target_version: str
    ) -> SchemaCompatibility:
        """Check if an annotation is compatible with a target schema version.

        Args:
            annotation: The annotation to check
            target_version: Target schema version string

        Returns:
            SchemaCompatibility result with warnings/errors.
        """
        source_version = annotation.schema_version
        key = (source_version, target_version)

        compatible = self._COMPATIBILITY.get(key, False)
        warnings = []
        errors = []

        if not compatible:
            errors.append(
                f"Schema {source_version} -> {target_version} is not supported"
            )
            return SchemaCompatibility(
                compatible=False,
                source_version=source_version,
                target_version=target_version,
                errors=errors,
            )

        # v1.0 -> v2.0: warn about missing enhanced fields
        if source_version == "1.0" and target_version == "2.0":
            if not annotation.is_v2_enhanced():
                warnings.append(
                    "v1.0 annotation loaded as v2.0 — enhanced fields will be empty"
                )

        # v2.0 -> v1.0: error if enhanced fields are populated
        if source_version == "2.0" and target_version == "1.0":
            if annotation.is_v2_enhanced():
                errors.append(
                    "v2.0 enhanced fields (instrument, physical, temporal_relations) "
                    "will be lost when downgrading to v1.0"
                )
                compatible = False

        return SchemaCompatibility(
            compatible=compatible,
            source_version=source_version,
            target_version=target_version,
            warnings=warnings,
            errors=errors,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get lineage statistics."""
        total = len(self._lineage)
        step_counts: Dict[str, int] = {}
        for steps in self._lineage.values():
            for step in steps:
                step_counts[step.step] = step_counts.get(step.step, 0) + 1

        return {
            "total_annotations": total,
            "total_steps": sum(len(s) for s in self._lineage.values()),
            "step_breakdown": step_counts,
        }

    def clear(self, annotation_id: Optional[str] = None) -> None:
        """Clear lineage data.

        Args:
            annotation_id: If provided, clear only this annotation's lineage.
                          If None, clear all lineage data.
        """
        if annotation_id:
            self._lineage.pop(annotation_id, None)
        else:
            self._lineage.clear()
