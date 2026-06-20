"""Approve/reject workflow for annotation acceptance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowStage(str, Enum):
    """Stages in the approval workflow."""
    INITIAL = "initial"
    AUTOMATED_REVIEW = "automated_review"
    HUMAN_REVIEW = "human_review"
    FINAL_APPROVAL = "final_approval"
    REJECTED = "rejected"
    APPROVED = "approved"


class RejectionReason(str, Enum):
    """Predefined rejection reasons."""
    QUALITY_SCORE = "quality_score"
    INCOMPLETE = "incomplete"
    INCORRECT = "incorrect"
    TEMPORAL_ISSUE = "temporal_issue"
    GROUNDING_ISSUE = "grounding_issue"
    LANGUAGE_ISSUE = "language_issue"
    OTHER = "other"


@dataclass
class StageTransition:
    """Record of a workflow stage transition."""
    from_stage: WorkflowStage
    to_stage: WorkflowStage
    actor: str
    timestamp: str = ""
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


@dataclass
class RejectionRecord:
    """Record of an annotation rejection."""
    annotation_id: str
    reason: RejectionReason
    details: str
    rejected_by: str
    rejected_at: str = ""
    stage_at_rejection: WorkflowStage = WorkflowStage.INITIAL
    can_be_resubmitted: bool = True

    def __post_init__(self):
        if not self.rejected_at:
            self.rejected_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "reason": self.reason.value,
            "details": self.details,
            "rejected_by": self.rejected_by,
            "rejected_at": self.rejected_at,
            "stage_at_rejection": self.stage_at_rejection.value,
            "can_be_resubmitted": self.can_be_resubmitted,
        }


@dataclass
class WorkflowAnnotation:
    """Annotation with workflow state."""
    annotation_id: str
    current_stage: WorkflowStage = WorkflowStage.INITIAL
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    transitions: List[StageTransition] = field(default_factory=list)
    rejection_history: List[RejectionRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    export_approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "current_stage": self.current_stage.value,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "transitions": [t.to_dict() for t in self.transitions],
            "rejection_history": [r.to_dict() for r in self.rejection_history],
            "metadata": self.metadata,
            "export_approved": self.export_approved,
        }


class ApprovalWorkflow:
    """Multi-stage approval workflow for annotation acceptance."""

    def __init__(self):
        self._annotations: Dict[str, WorkflowAnnotation] = {}
        self._stage_order = [
            WorkflowStage.INITIAL,
            WorkflowStage.AUTOMATED_REVIEW,
            WorkflowStage.HUMAN_REVIEW,
            WorkflowStage.FINAL_APPROVAL,
        ]

    def register_annotation(self, annotation_id: str) -> WorkflowAnnotation:
        wf = WorkflowAnnotation(annotation_id=annotation_id)
        self._annotations[annotation_id] = wf
        logger.info("annotation_registered", annotation_id=annotation_id)
        return wf

    def transition(self, annotation_id: str, to_stage: WorkflowStage, actor: str, notes: Optional[str] = None) -> Optional[WorkflowAnnotation]:
        wf = self._annotations.get(annotation_id)
        if not wf:
            return None

        from_stage = wf.current_stage
        transition = StageTransition(
            from_stage=from_stage,
            to_stage=to_stage,
            actor=actor,
            notes=notes,
        )
        wf.transitions.append(transition)
        wf.current_stage = to_stage

        logger.info("stage_transition", annotation_id=annotation_id, from_stage=from_stage.value, to_stage=to_stage.value, actor=actor)
        return wf

    def approve(self, annotation_id: str, approved_by: str, notes: Optional[str] = None) -> Optional[WorkflowAnnotation]:
        wf = self._annotations.get(annotation_id)
        if not wf:
            return None

        wf.approved = True
        wf.approved_by = approved_by
        wf.approved_at = datetime.utcnow().isoformat()
        wf.export_approved = True

        self.transition(annotation_id, WorkflowStage.APPROVED, approved_by, notes)
        logger.info("annotation_approved", annotation_id=annotation_id, approved_by=approved_by)
        return wf

    def reject(self, annotation_id: str, reason: RejectionReason, rejected_by: str, details: str = "") -> Optional[WorkflowAnnotation]:
        wf = self._annotations.get(annotation_id)
        if not wf:
            return None

        rejection = RejectionRecord(
            annotation_id=annotation_id,
            reason=reason,
            details=details,
            rejected_by=rejected_by,
            stage_at_rejection=wf.current_stage,
        )
        wf.rejection_history.append(rejection)

        self.transition(annotation_id, WorkflowStage.REJECTED, rejected_by, details)
        logger.info("annotation_rejected", annotation_id=annotation_id, reason=reason.value, rejected_by=rejected_by)
        return wf

    def can_approve(self, annotation_id: str) -> bool:
        wf = self._annotations.get(annotation_id)
        if not wf:
            return False
        return wf.current_stage in (WorkflowStage.HUMAN_REVIEW, WorkflowStage.FINAL_APPROVAL)

    def can_reject(self, annotation_id: str) -> bool:
        wf = self._annotations.get(annotation_id)
        if not wf:
            return False
        return wf.current_stage not in (WorkflowStage.APPROVED, WorkflowStage.REJECTED)

    def get_annotation_state(self, annotation_id: str) -> Optional[WorkflowAnnotation]:
        return self._annotations.get(annotation_id)

    def get_approved_annotations(self) -> List[WorkflowAnnotation]:
        return [wf for wf in self._annotations.values() if wf.approved]

    def get_rejected_annotations(self) -> List[WorkflowAnnotation]:
        return [wf for wf in self._annotations.values() if wf.current_stage == WorkflowStage.REJECTED]

    def get_rejection_reasons(self, annotation_id: str) -> List[RejectionRecord]:
        wf = self._annotations.get(annotation_id)
        return wf.rejection_history if wf else []

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self._annotations)
        approved = sum(1 for wf in self._annotations.values() if wf.approved)
        rejected = sum(1 for wf in self._annotations.values() if wf.current_stage == WorkflowStage.REJECTED)
        pending = total - approved - rejected
        return {
            "total_annotations": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": approved / total if total > 0 else 0.0,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "export_ready": sum(1 for wf in self._annotations.values() if wf.export_approved),
        }

    def check_export_gate(self, annotation_id: str) -> bool:
        wf = self._annotations.get(annotation_id)
        return wf.export_approved if wf else False

    def get_annotations_ready_for_export(self) -> List[str]:
        return [wf.annotation_id for wf in self._annotations.values() if wf.export_approved]
