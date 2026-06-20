"""Annotation approval workflows for DVAS governance.

Provides ApprovalWorkflow for managing multi-stage review workflows
with reviewer assignment strategies, queue management, escalation rules,
and audit trails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowStatus(str, Enum):
    """Status of an approval workflow."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class AssignmentStrategy(str, Enum):
    """Strategy for assigning reviewers."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    EXPERTISE_BASED = "expertise_based"
    RANDOM = "random"


@dataclass
class Reviewer:
    """A reviewer in the approval workflow."""

    id: str
    name: str
    expertise: List[str] = field(default_factory=list)
    max_queue_size: int = 10
    active_reviews: int = 0
    total_reviews: int = 0
    avg_review_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "expertise": self.expertise,
            "max_queue_size": self.max_queue_size,
            "active_reviews": self.active_reviews,
            "total_reviews": self.total_reviews,
            "avg_review_time": self.avg_review_time,
        }


@dataclass
class ApprovalRecord:
    """A single approval/rejection record."""

    annotation_id: str
    reviewer_id: str
    status: WorkflowStatus
    timestamp: float = field(default_factory=time.time)
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "reviewer_id": self.reviewer_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "comments": self.comments,
            "metadata": self.metadata,
        }


class ApprovalWorkflow:
    """Manages annotation approval workflows.

    Usage::

        workflow = ApprovalWorkflow(
            workflow_id="main",
            required_approvals=2,
            assignment_strategy=AssignmentStrategy.ROUND_ROBIN,
        )
        workflow.add_reviewer(Reviewer("r1", "Alice", ["kitchen"]))
        workflow.add_reviewer(Reviewer("r2", "Bob", ["robotics"]))

        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.approve("ann_001", "r1", "Looks good")
    """

    def __init__(
        self,
        workflow_id: str = "default",
        required_approvals: int = 1,
        assignment_strategy: AssignmentStrategy = AssignmentStrategy.ROUND_ROBIN,
        escalation_timeout: Optional[float] = None,
    ) -> None:
        """Initialize the approval workflow.

        Args:
            workflow_id: Unique workflow identifier.
            required_approvals: Number of approvals needed.
            assignment_strategy: How to assign reviewers.
            escalation_timeout: Seconds before auto-escalation.
        """
        self.workflow_id = workflow_id
        self.required_approvals = required_approvals
        self.assignment_strategy = assignment_strategy
        self.escalation_timeout = escalation_timeout or 86400.0  # Default 24h

        self._reviewers: Dict[str, Reviewer] = {}
        self._annotation_status: Dict[str, WorkflowStatus] = {}
        self._annotation_assignments: Dict[str, List[str]] = {}
        self._annotation_records: Dict[str, List[ApprovalRecord]] = {}
        self._annotation_submitted_at: Dict[str, float] = {}
        self._reviewer_queue: List[str] = []  # Global queue of annotation IDs
        self._reviewer_index = 0  # For round-robin
        self._audit_trail: List[Dict[str, Any]] = []

    def add_reviewer(self, reviewer: Reviewer) -> None:
        """Add a reviewer to the workflow.

        Args:
            reviewer: The reviewer to add.
        """
        self._reviewers[reviewer.id] = reviewer
        logger.info("reviewer_added", reviewer_id=reviewer.id, name=reviewer.name)

    def remove_reviewer(self, reviewer_id: str) -> bool:
        """Remove a reviewer from the workflow.

        Args:
            reviewer_id: The reviewer ID to remove.

        Returns:
            True if the reviewer was removed.
        """
        if reviewer_id in self._reviewers:
            del self._reviewers[reviewer_id]
            return True
        return False

    def submit_for_review(self, annotation_id: str) -> None:
        """Submit an annotation for review.

        Args:
            annotation_id: The annotation to review.
        """
        self._annotation_status[annotation_id] = WorkflowStatus.PENDING_REVIEW
        self._annotation_assignments[annotation_id] = []
        self._annotation_records[annotation_id] = []
        self._annotation_submitted_at[annotation_id] = time.time()
        self._reviewer_queue.append(annotation_id)

        self._log_event("submitted", annotation_id)
        logger.info("annotation_submitted", annotation_id=annotation_id)

    def assign_reviewer(self, annotation_id: str, reviewer_id: Optional[str] = None) -> Optional[str]:
        """Assign a reviewer to an annotation.

        Args:
            annotation_id: The annotation to assign.
            reviewer_id: Specific reviewer. If None, uses strategy.

        Returns:
            The assigned reviewer ID, or None if no reviewer available.
        """
        if annotation_id not in self._annotation_status:
            return None

        if reviewer_id:
            if reviewer_id not in self._reviewers:
                return None
            assigned = reviewer_id
        else:
            assigned = self._auto_assign_reviewer(annotation_id)

        if not assigned:
            return None

        reviewer = self._reviewers[assigned]
        if reviewer.active_reviews >= reviewer.max_queue_size:
            return None

        if annotation_id not in self._annotation_assignments:
            self._annotation_assignments[annotation_id] = []
        self._annotation_assignments[annotation_id].append(assigned)
        reviewer.active_reviews += 1

        self._annotation_status[annotation_id] = WorkflowStatus.IN_REVIEW

        self._log_event("assigned", annotation_id, reviewer_id=assigned)
        return assigned

    def _auto_assign_reviewer(self, annotation_id: str) -> Optional[str]:
        """Automatically assign a reviewer based on strategy."""
        available = list(self._reviewers.values())
        if not available:
            return None

        if self.assignment_strategy == AssignmentStrategy.ROUND_ROBIN:
            return self._round_robin_assign(available)
        elif self.assignment_strategy == AssignmentStrategy.LOAD_BALANCED:
            return self._load_balanced_assign(available)
        elif self.assignment_strategy == AssignmentStrategy.EXPERTISE_BASED:
            return self._expertise_assign(annotation_id, available)
        elif self.assignment_strategy == AssignmentStrategy.RANDOM:
            import random
            return random.choice([r.id for r in available]).id

        return available[0].id

    def _round_robin_assign(self, available: List[Reviewer]) -> Optional[str]:
        """Assign reviewer using round-robin."""
        for _ in range(len(available)):
            idx = self._reviewer_index % len(available)
            self._reviewer_index += 1
            reviewer = available[idx]
            if reviewer.active_reviews < reviewer.max_queue_size:
                return reviewer.id
        return None

    def _load_balanced_assign(self, available: List[Reviewer]) -> Optional[str]:
        """Assign reviewer with least load."""
        sorted_reviewers = sorted(
            available,
            key=lambda r: r.active_reviews / max(r.max_queue_size, 1),
        )
        for reviewer in sorted_reviewers:
            if reviewer.active_reviews < reviewer.max_queue_size:
                return reviewer.id
        return None

    def _expertise_assign(self, annotation_id: str, available: List[Reviewer]) -> Optional[str]:
        """Assign reviewer based on expertise."""
        # Simple: pick reviewer with least load (expertise matching would need annotation tags)
        return self._load_balanced_assign(available)

    def approve(self, annotation_id: str, reviewer_id: str, comments: str = "") -> WorkflowStatus:
        """Approve an annotation.

        Args:
            annotation_id: The annotation to approve.
            reviewer_id: The reviewer ID.
            comments: Optional comments.

        Returns:
            The updated workflow status.
        """
        if annotation_id not in self._annotation_records:
            return None  # type: ignore[return-value]

        record = ApprovalRecord(
            annotation_id=annotation_id,
            reviewer_id=reviewer_id,
            status=WorkflowStatus.APPROVED,
            comments=comments,
        )
        self._annotation_records[annotation_id].append(record)

        # Update reviewer stats
        if reviewer_id in self._reviewers:
            reviewer = self._reviewers[reviewer_id]
            reviewer.active_reviews = max(0, reviewer.active_reviews - 1)
            reviewer.total_reviews += 1

        # Check if enough approvals
        approvals = sum(
            1 for r in self._annotation_records[annotation_id]
            if r.status == WorkflowStatus.APPROVED
        )
        if approvals >= self.required_approvals:
            self._annotation_status[annotation_id] = WorkflowStatus.APPROVED

        self._log_event("approved", annotation_id, reviewer_id=reviewer_id, comments=comments)
        return self._annotation_status[annotation_id]

    def reject(self, annotation_id: str, reviewer_id: str, comments: str = "") -> WorkflowStatus:
        """Reject an annotation.

        Args:
            annotation_id: The annotation to reject.
            reviewer_id: The reviewer ID.
            comments: Rejection reason.

        Returns:
            The updated workflow status.
        """
        record = ApprovalRecord(
            annotation_id=annotation_id,
            reviewer_id=reviewer_id,
            status=WorkflowStatus.REJECTED,
            comments=comments,
        )
        self._annotation_records[annotation_id].append(record)
        self._annotation_status[annotation_id] = WorkflowStatus.REJECTED

        # Update reviewer stats
        if reviewer_id in self._reviewers:
            reviewer = self._reviewers[reviewer_id]
            reviewer.active_reviews = max(0, reviewer.active_reviews - 1)
            reviewer.total_reviews += 1

        self._log_event("rejected", annotation_id, reviewer_id=reviewer_id, comments=comments)
        return self._annotation_status[annotation_id]

    def get_status(self, annotation_id: str) -> Optional[WorkflowStatus]:
        """Get the status of an annotation."""
        return self._annotation_status.get(annotation_id)

    def get_assignments(self, annotation_id: str) -> List[str]:
        """Get all reviewers assigned to an annotation."""
        return self._annotation_assignments.get(annotation_id, [])

    def get_records(self, annotation_id: str) -> List[ApprovalRecord]:
        """Get all approval records for an annotation."""
        return self._annotation_records.get(annotation_id, [])

    def get_queue(self) -> List[str]:
        """Get all annotations in the review queue."""
        return [
            aid for aid in self._reviewer_queue
            if self._annotation_status.get(aid) in (WorkflowStatus.PENDING_REVIEW, WorkflowStatus.IN_REVIEW)
        ]

    def get_reviewer_queue_size(self, reviewer_id: str) -> int:
        """Get the number of annotations assigned to a reviewer."""
        return sum(
            1 for aid, reviewers in self._annotation_assignments.items()
            if reviewer_id in reviewers
            and self._annotation_status.get(aid) in (WorkflowStatus.PENDING_REVIEW, WorkflowStatus.IN_REVIEW)
        )

    def check_escalation(self, annotation_id: str) -> bool:
        """Check if an annotation should be escalated.

        Args:
            annotation_id: The annotation to check.

        Returns:
            True if the annotation should be escalated.
        """
        if self._annotation_status.get(annotation_id) not in (WorkflowStatus.PENDING_REVIEW, WorkflowStatus.IN_REVIEW):
            return False

        submitted_at = self._annotation_submitted_at.get(annotation_id, 0)
        if time.time() - submitted_at > self.escalation_timeout:
            self._annotation_status[annotation_id] = WorkflowStatus.ESCALATED
            self._log_event("escalated", annotation_id)
            return True
        return False

    def escalate(self, annotation_id: str, reason: str = "") -> WorkflowStatus:
        """Manually escalate an annotation.

        Args:
            annotation_id: The annotation to escalate.
            reason: Reason for escalation.

        Returns:
            The updated status.
        """
        self._annotation_status[annotation_id] = WorkflowStatus.ESCALATED
        self._log_event("escalated", annotation_id, reason=reason)
        return WorkflowStatus.ESCALATED

    def get_audit_trail(self, annotation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the audit trail.

        Args:
            annotation_id: Optional filter by annotation.

        Returns:
            List of audit trail entries.
        """
        if annotation_id:
            return [e for e in self._audit_trail if e.get("annotation_id") == annotation_id]
        return self._audit_trail.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        total = len(self._annotation_status)
        approved = sum(1 for s in self._annotation_status.values() if s == WorkflowStatus.APPROVED)
        rejected = sum(1 for s in self._annotation_status.values() if s == WorkflowStatus.REJECTED)
        pending = sum(1 for s in self._annotation_status.values() if s == WorkflowStatus.PENDING_REVIEW)
        in_review = sum(1 for s in self._annotation_status.values() if s == WorkflowStatus.IN_REVIEW)
        escalated = sum(1 for s in self._annotation_status.values() if s == WorkflowStatus.ESCALATED)

        return {
            "workflow_id": self.workflow_id,
            "total_annotations": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "in_review": in_review,
            "escalated": escalated,
            "reviewers": len(self._reviewers),
            "queue_length": len(self.get_queue()),
        }

    def _log_event(self, event_type: str, annotation_id: str, **kwargs: Any) -> None:
        """Log an event to the audit trail."""
        entry = {
            "event_type": event_type,
            "annotation_id": annotation_id,
            "timestamp": time.time(),
            "workflow_id": self.workflow_id,
        }
        entry.update(kwargs)
        self._audit_trail.append(entry)


__all__ = [
    "ApprovalWorkflow",
    "ApprovalRecord",
    "AssignmentStrategy",
    "Reviewer",
    "WorkflowStatus",
]
