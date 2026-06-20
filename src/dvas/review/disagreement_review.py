"""Disagreement review for handling teacher/student disagreements.

Provides disagreement visualization and resolution workflow
for cases where teacher and student models disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DisagreementType(str, Enum):
    """Types of disagreements between models."""

    ACTION = "action"
    OBJECT = "object"
    CAPTION = "caption"
    TEMPORAL = "temporal"
    OVERALL = "overall"


class DisagreementStatus(str, Enum):
    """Status of a disagreement review."""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"


class Resolution(str, Enum):
    """Possible resolutions for a disagreement."""

    TEACHER = "teacher"
    STUDENT = "student"
    BOTH = "both"
    NEITHER = "neither"
    HYBRID = "hybrid"


@dataclass
class DisagreementVisualization:
    """Visualization data for a disagreement."""

    annotation_id: str
    video_id: str
    disagreement_type: DisagreementType
    teacher_output: str
    student_output: str
    teacher_model: str
    student_model: str
    disagreement_score: float
    highlighted_differences: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "disagreement_type": self.disagreement_type.value,
            "teacher_output": self.teacher_output,
            "student_output": self.student_output,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "disagreement_score": self.disagreement_score,
            "highlighted_differences": self.highlighted_differences,
            "context": self.context,
        }


@dataclass
class DisagreementCase:
    """A single disagreement case."""

    case_id: str
    annotation_id: str
    video_id: str
    disagreement_type: DisagreementType
    teacher_output: str
    student_output: str
    teacher_model: str = "teacher"
    student_model: str = "student"
    disagreement_score: float = 0.0
    status: DisagreementStatus = DisagreementStatus.PENDING
    resolution: Optional[Resolution] = None
    resolution_notes: Optional[str] = None
    reviewer_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat()
    )
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "disagreement_type": self.disagreement_type.value,
            "teacher_output": self.teacher_output,
            "student_output": self.student_output,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "disagreement_score": self.disagreement_score,
            "status": self.status.value,
            "resolution": self.resolution.value if self.resolution else None,
            "resolution_notes": self.resolution_notes,
            "reviewer_id": self.reviewer_id,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }


class DisagreementReview:
    """Review system for handling teacher/student disagreements.

    Manages disagreement cases, provides visualization data,
    and handles the resolution workflow.
    """

    def __init__(self):
        self._cases: Dict[str, DisagreementCase] = {}

    def add_case(self, case: DisagreementCase) -> None:
        """Add a disagreement case.

        Args:
            case: DisagreementCase to add
        """
        self._cases[case.case_id] = case
        logger.info(
            "disagreement_case_added",
            case_id=case.case_id,
            type=case.disagreement_type.value,
            score=case.disagreement_score,
        )

    def get_case(self, case_id: str) -> Optional[DisagreementCase]:
        """Get a disagreement case by ID.

        Args:
            case_id: ID of the case

        Returns:
            DisagreementCase or None
        """
        return self._cases.get(case_id)

    def get_visualization(self, case_id: str) -> Optional[DisagreementVisualization]:
        """Get visualization data for a disagreement case.

        Args:
            case_id: ID of the case

        Returns:
            DisagreementVisualization or None
        """
        case = self._cases.get(case_id)
        if not case:
            return None

        differences = self._compute_differences(case)

        return DisagreementVisualization(
            annotation_id=case.annotation_id,
            video_id=case.video_id,
            disagreement_type=case.disagreement_type,
            teacher_output=case.teacher_output,
            student_output=case.student_output,
            teacher_model=case.teacher_model,
            student_model=case.student_model,
            disagreement_score=case.disagreement_score,
            highlighted_differences=differences,
        )

    def _compute_differences(self, case: DisagreementCase) -> List[Dict[str, Any]]:
        """Compute highlighted differences between outputs."""
        differences = []

        # Simple word-level diff for text outputs
        teacher_words = set(case.teacher_output.lower().split())
        student_words = set(case.student_output.lower().split())

        only_teacher = teacher_words - student_words
        only_student = student_words - teacher_words
        common = teacher_words & student_words

        if only_teacher:
            differences.append(
                {
                    "type": "teacher_only",
                    "words": list(only_teacher)[:20],
                    "description": f"{len(only_teacher)} words unique to teacher output",
                }
            )

        if only_student:
            differences.append(
                {
                    "type": "student_only",
                    "words": list(only_student)[:20],
                    "description": f"{len(only_student)} words unique to student output",
                }
            )

        differences.append(
            {
                "type": "common",
                "word_count": len(common),
                "description": f"{len(common)} common words",
            }
        )

        return differences

    def resolve_case(
        self,
        case_id: str,
        resolution: Resolution,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> Optional[DisagreementCase]:
        """Resolve a disagreement case.

        Args:
            case_id: ID of the case
            resolution: Resolution choice
            reviewer_id: ID of the resolving reviewer
            notes: Optional resolution notes

        Returns:
            Updated DisagreementCase or None
        """
        case = self._cases.get(case_id)
        if not case:
            return None

        case.status = DisagreementStatus.RESOLVED
        case.resolution = resolution
        case.reviewer_id = reviewer_id
        case.resolution_notes = notes
        case.resolved_at = __import__("datetime").datetime.utcnow().isoformat()

        logger.info(
            "disagreement_resolved",
            case_id=case_id,
            resolution=resolution.value,
            reviewer_id=reviewer_id,
        )
        return case

    def escalate_case(
        self, case_id: str, reason: Optional[str] = None
    ) -> Optional[DisagreementCase]:
        """Escalate a disagreement case.

        Args:
            case_id: ID of the case
            reason: Optional escalation reason

        Returns:
            Updated DisagreementCase or None
        """
        case = self._cases.get(case_id)
        if not case:
            return None

        case.status = DisagreementStatus.ESCALATED
        if reason:
            case.metadata["escalation_reason"] = reason

        logger.info("disagreement_escalated", case_id=case_id, reason=reason)
        return case

    def dismiss_case(
        self, case_id: str, reason: Optional[str] = None
    ) -> Optional[DisagreementCase]:
        """Dismiss a disagreement case.

        Args:
            case_id: ID of the case
            reason: Optional dismissal reason

        Returns:
            Updated DisagreementCase or None
        """
        case = self._cases.get(case_id)
        if not case:
            return None

        case.status = DisagreementStatus.DISMISSED
        if reason:
            case.metadata["dismissal_reason"] = reason

        logger.info("disagreement_dismissed", case_id=case_id, reason=reason)
        return case

    def get_cases_by_type(self, disagreement_type: DisagreementType) -> List[DisagreementCase]:
        """Get all cases of a specific type.

        Args:
            disagreement_type: Type of disagreement

        Returns:
            List of matching cases
        """
        return [c for c in self._cases.values() if c.disagreement_type == disagreement_type]

    def get_pending_cases(self) -> List[DisagreementCase]:
        """Get all pending cases.

        Returns:
            List of pending cases
        """
        return [c for c in self._cases.values() if c.status == DisagreementStatus.PENDING]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about disagreement cases.

        Returns:
            Dict with case statistics
        """
        stats = {
            "total_cases": len(self._cases),
            "pending": 0,
            "under_review": 0,
            "resolved": 0,
            "escalated": 0,
            "dismissed": 0,
            "by_type": {},
            "avg_disagreement_score": 0.0,
        }

        total_score = 0.0
        for case in self._cases.values():
            status = case.status.value
            if status in stats:
                stats[status] += 1

            type_name = case.disagreement_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1
            total_score += case.disagreement_score

        if self._cases:
            stats["avg_disagreement_score"] = total_score / len(self._cases)

        return stats

    def get_cases_by_annotation(self, annotation_id: str) -> List[DisagreementCase]:
        """Get all cases for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            List of matching cases
        """
        return [c for c in self._cases.values() if c.annotation_id == annotation_id]
