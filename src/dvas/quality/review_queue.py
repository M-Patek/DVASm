"""Review queue management for quality control.

Manages queues for human review, disagreement cases, and low-quality
annotation quarantine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dvas.data.schemas import Annotation
from dvas.quality.schema import QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class ReviewPriority(str, Enum):
    """Priority levels for review queue items."""

    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"  # Review within 24 hours
    MEDIUM = "medium"  # Review within 3 days
    LOW = "low"  # Review within 7 days


class ReviewStatus(str, Enum):
    """Status of a review queue item."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class ReviewItem:
    """Item in a review queue."""

    annotation_id: str
    video_id: str
    priority: ReviewPriority
    status: ReviewStatus = ReviewStatus.PENDING

    # Quality information
    quality_scores: Optional[QualityScores] = None
    failed_dimensions: List[str] = field(default_factory=list)
    auto_analysis_issues: List[str] = field(default_factory=list)

    # Assignment
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_by: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Review result
    review_result: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None

    # Metadata
    source: str = "automatic"  # How it was added to queue
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set due date based on priority if not set."""
        if self.due_by is None:
            priority_days = {
                ReviewPriority.CRITICAL: 0,
                ReviewPriority.HIGH: 1,
                ReviewPriority.MEDIUM: 3,
                ReviewPriority.LOW: 7,
            }
            days = priority_days.get(self.priority, 3)
            self.due_by = self.created_at + timedelta(days=days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "quality_scores": self.quality_scores.to_dict() if self.quality_scores else None,
            "failed_dimensions": self.failed_dimensions,
            "auto_analysis_issues": self.auto_analysis_issues,
            "assigned_to": self.assigned_to,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "created_at": self.created_at.isoformat(),
            "due_by": self.due_by.isoformat() if self.due_by else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "review_result": self.review_result,
            "reviewer_notes": self.reviewer_notes,
            "source": self.source,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewItem":
        """Create from dictionary."""
        return cls(
            annotation_id=data["annotation_id"],
            video_id=data["video_id"],
            priority=ReviewPriority(data["priority"]),
            status=ReviewStatus(data.get("status", "pending")),
            quality_scores=QualityScores.from_dict(data["quality_scores"])
            if data.get("quality_scores")
            else None,
            failed_dimensions=data.get("failed_dimensions", []),
            auto_analysis_issues=data.get("auto_analysis_issues", []),
            assigned_to=data.get("assigned_to"),
            assigned_at=datetime.fromisoformat(data["assigned_at"])
            if data.get("assigned_at")
            else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            due_by=datetime.fromisoformat(data["due_by"]) if data.get("due_by") else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            review_result=data.get("review_result"),
            reviewer_notes=data.get("reviewer_notes"),
            source=data.get("source", "automatic"),
            tags=data.get("tags", []),
        )

    def is_overdue(self) -> bool:
        """Check if item is overdue."""
        if self.due_by is None:
            return False
        return datetime.utcnow() > self.due_by

    def assign(self, reviewer_id: str) -> None:
        """Assign item to a reviewer."""
        self.assigned_to = reviewer_id
        self.assigned_at = datetime.utcnow()
        self.status = ReviewStatus.ASSIGNED

    def complete(self, result: Dict[str, Any], notes: Optional[str] = None) -> None:
        """Mark item as completed."""
        self.status = ReviewStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.review_result = result
        if notes:
            self.reviewer_notes = notes


class HumanReviewQueue:
    """Queue for human review assignments.

    Manages the flow of annotations requiring human review,
    with priority-based ordering and assignment tracking.
    """

    def __init__(self):
        """Initialize the review queue."""
        self._items: Dict[str, ReviewItem] = {}
        self._lock = asyncio.Lock()
        self._assigned: Dict[str, Set[str]] = {}  # reviewer_id -> set of annotation_ids

    async def add(
        self,
        annotation: Annotation,
        quality_scores: Optional[QualityScores] = None,
        priority: ReviewPriority = ReviewPriority.MEDIUM,
        source: str = "automatic",
        tags: Optional[List[str]] = None,
    ) -> ReviewItem:
        """Add an annotation to the review queue.

        Args:
            annotation: Annotation to review
            quality_scores: Optional quality scores
            priority: Review priority
            source: Source of the review request
            tags: Optional tags for categorization

        Returns:
            Created ReviewItem
        """
        async with self._lock:
            # Determine priority from quality scores if not specified
            if quality_scores and priority == ReviewPriority.MEDIUM:
                priority = self._determine_priority(quality_scores)

            item = ReviewItem(
                annotation_id=annotation.id,
                video_id=annotation.video_id,
                priority=priority,
                quality_scores=quality_scores,
                failed_dimensions=[
                    d.value for d in quality_scores.failed_dimensions
                ] if quality_scores else [],
                source=source,
                tags=tags or [],
            )

            self._items[annotation.id] = item

            logger.info(
                "added_to_review_queue",
                annotation_id=annotation.id,
                priority=priority.value,
                source=source,
            )

            return item

    async def add_batch(
        self,
        annotations: List[Annotation],
        quality_scores_map: Optional[Dict[str, QualityScores]] = None,
        default_priority: ReviewPriority = ReviewPriority.MEDIUM,
    ) -> List[ReviewItem]:
        """Add multiple annotations to the review queue."""
        items = []
        for annotation in annotations:
            scores = quality_scores_map.get(annotation.id) if quality_scores_map else None
            item = await self.add(annotation, scores, default_priority)
            items.append(item)
        return items

    async def get_next(
        self,
        reviewer_id: str,
        priority_filter: Optional[List[ReviewPriority]] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> Optional[ReviewItem]:
        """Get the next item for review.

        Args:
            reviewer_id: ID of the reviewer
            priority_filter: Optional priority levels to consider
            tag_filter: Optional tags to filter by

        Returns:
            Next ReviewItem or None if queue is empty
        """
        async with self._lock:
            candidates = []

            for item in self._items.values():
                if item.status != ReviewStatus.PENDING:
                    continue

                if priority_filter and item.priority not in priority_filter:
                    continue

                if tag_filter and not any(t in item.tags for t in tag_filter):
                    continue

                candidates.append(item)

            if not candidates:
                return None

            # Sort by priority and creation time
            priority_order = {
                ReviewPriority.CRITICAL: 0,
                ReviewPriority.HIGH: 1,
                ReviewPriority.MEDIUM: 2,
                ReviewPriority.LOW: 3,
            }
            candidates.sort(key=lambda x: (priority_order[x.priority], x.created_at))

            # Assign the highest priority item
            item = candidates[0]
            item.assign(reviewer_id)

            if reviewer_id not in self._assigned:
                self._assigned[reviewer_id] = set()
            self._assigned[reviewer_id].add(item.annotation_id)

            logger.info(
                "assigned_review_item",
                annotation_id=item.annotation_id,
                reviewer_id=reviewer_id,
            )

            return item

    async def complete_review(
        self,
        annotation_id: str,
        result: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> Optional[ReviewItem]:
        """Mark a review as completed.

        Args:
            annotation_id: ID of the annotation
            result: Review result data
            notes: Optional reviewer notes

        Returns:
            Updated ReviewItem or None if not found
        """
        async with self._lock:
            item = self._items.get(annotation_id)
            if not item:
                return None

            item.complete(result, notes)

            # Remove from assigned
            if item.assigned_to and item.assigned_to in self._assigned:
                self._assigned[item.assigned_to].discard(annotation_id)

            logger.info(
                "review_completed",
                annotation_id=annotation_id,
                reviewer_id=item.assigned_to,
            )

            return item

    async def get_item(self, annotation_id: str) -> Optional[ReviewItem]:
        """Get a specific review item."""
        async with self._lock:
            return self._items.get(annotation_id)

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the review queue."""
        async with self._lock:
            stats = {
                "total_items": len(self._items),
                "by_status": {},
                "by_priority": {},
                "overdue": 0,
                "assigned": {},
            }

            for item in self._items.values():
                status = item.status.value
                priority = item.priority.value

                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

                if item.is_overdue():
                    stats["overdue"] += 1

            stats["assigned"] = {
                reviewer_id: len(items) for reviewer_id, items in self._assigned.items()
            }

            return stats

    def _determine_priority(self, scores: QualityScores) -> ReviewPriority:
        """Determine review priority from quality scores."""
        if scores.overall_score < 0.3:
            return ReviewPriority.CRITICAL
        elif scores.overall_score < 0.5:
            return ReviewPriority.HIGH
        elif scores.overall_score < 0.7:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW


@dataclass
class DisagreementCase:
    """Case of disagreement between teacher and student models."""

    annotation_id: str
    video_id: str

    # Model outputs
    teacher_output: str
    student_output: str
    teacher_model: str
    student_model: str

    # Disagreement analysis
    disagreement_type: str  # "action", "object", "caption", "temporal"
    disagreement_score: float  # 0.0 to 1.0, higher = more disagreement

    # Status
    status: str = "pending"  # pending, resolved, escalated
    resolution: Optional[str] = None
    correct_version: Optional[str] = None  # "teacher", "student", "neither", "both"

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "teacher_output": self.teacher_output,
            "student_output": self.student_output,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "disagreement_type": self.disagreement_type,
            "disagreement_score": self.disagreement_score,
            "status": self.status,
            "resolution": self.resolution,
            "correct_version": self.correct_version,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


class DisagreementQueue:
    """Queue for teacher/student disagreement cases.

    Tracks and manages cases where teacher and student models
    produce significantly different outputs for the same input.
    """

    def __init__(self):
        """Initialize the disagreement queue."""
        self._cases: Dict[str, DisagreementCase] = {}
        self._lock = asyncio.Lock()

    async def add_case(
        self,
        annotation_id: str,
        video_id: str,
        teacher_output: str,
        student_output: str,
        teacher_model: str = "teacher",
        student_model: str = "student",
        disagreement_type: str = "caption",
        disagreement_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DisagreementCase:
        """Add a disagreement case to the queue.

        Args:
            annotation_id: ID of the annotation
            video_id: ID of the video
            teacher_output: Teacher model output
            student_output: Student model output
            teacher_model: Teacher model identifier
            student_model: Student model identifier
            disagreement_type: Type of disagreement
            disagreement_score: Score indicating severity of disagreement
            metadata: Additional metadata

        Returns:
            Created DisagreementCase
        """
        async with self._lock:
            case = DisagreementCase(
                annotation_id=annotation_id,
                video_id=video_id,
                teacher_output=teacher_output,
                student_output=student_output,
                teacher_model=teacher_model,
                student_model=student_model,
                disagreement_type=disagreement_type,
                disagreement_score=disagreement_score,
                metadata=metadata or {},
            )

            self._cases[annotation_id] = case

            logger.info(
                "disagreement_case_added",
                annotation_id=annotation_id,
                disagreement_type=disagreement_type,
                score=disagreement_score,
            )

            return case

    async def resolve_case(
        self,
        annotation_id: str,
        correct_version: str,
        resolution_notes: Optional[str] = None,
    ) -> Optional[DisagreementCase]:
        """Resolve a disagreement case.

        Args:
            annotation_id: ID of the annotation
            correct_version: Which version is correct ("teacher", "student", "neither", "both")
            resolution_notes: Optional notes about the resolution

        Returns:
            Updated DisagreementCase or None if not found
        """
        async with self._lock:
            case = self._cases.get(annotation_id)
            if not case:
                return None

            case.status = "resolved"
            case.correct_version = correct_version
            case.resolution = resolution_notes
            case.resolved_at = datetime.utcnow()

            logger.info(
                "disagreement_case_resolved",
                annotation_id=annotation_id,
                correct_version=correct_version,
            )

            return case

    async def get_case(self, annotation_id: str) -> Optional[DisagreementCase]:
        """Get a specific disagreement case."""
        async with self._lock:
            return self._cases.get(annotation_id)

    async def get_pending_cases(
        self,
        min_disagreement: Optional[float] = None,
    ) -> List[DisagreementCase]:
        """Get all pending disagreement cases.

        Args:
            min_disagreement: Minimum disagreement score to include

        Returns:
            List of pending DisagreementCases
        """
        async with self._lock:
            cases = []
            for case in self._cases.values():
                if case.status == "pending":
                    if min_disagreement is None or case.disagreement_score >= min_disagreement:
                        cases.append(case)
            return cases

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about disagreement cases."""
        async with self._lock:
            stats = {
                "total_cases": len(self._cases),
                "pending": 0,
                "resolved": 0,
                "escalated": 0,
                "by_type": {},
                "avg_disagreement_score": 0.0,
            }

            total_score = 0.0
            for case in self._cases.values():
                stats[case.status] = stats.get(case.status, 0) + 1
                stats["by_type"][case.disagreement_type] = (
                    stats["by_type"].get(case.disagreement_type, 0) + 1
                )
                total_score += case.disagreement_score

            if self._cases:
                stats["avg_disagreement_score"] = total_score / len(self._cases)

            return stats


@dataclass
class QuarantineItem:
    """Item in the low-quality quarantine."""

    annotation_id: str
    video_id: str

    # Quality information
    quality_scores: QualityScores
    quarantine_reason: str

    # Quarantine status
    status: str = "quarantined"  # quarantined, fixed, rejected, approved

    # Fix attempts
    fix_attempts: int = 0
    last_fix_attempt: Optional[datetime] = None
    fix_results: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "quality_scores": self.quality_scores.to_dict(),
            "quarantine_reason": self.quarantine_reason,
            "status": self.status,
            "fix_attempts": self.fix_attempts,
            "last_fix_attempt": self.last_fix_attempt.isoformat()
            if self.last_fix_attempt
            else None,
            "fix_results": self.fix_results,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


class LowQualityQuarantine:
    """Quarantine for low-quality annotations pending fix.

    Holds annotations that failed quality thresholds,
    tracks fix attempts, and manages their lifecycle.
    """

    def __init__(self, max_fix_attempts: int = 3):
        """Initialize the quarantine.

        Args:
            max_fix_attempts: Maximum number of fix attempts before rejection
        """
        self._items: Dict[str, QuarantineItem] = {}
        self._max_fix_attempts = max_fix_attempts
        self._lock = asyncio.Lock()

    async def quarantine(
        self,
        annotation: Annotation,
        quality_scores: QualityScores,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantineItem:
        """Add an annotation to quarantine.

        Args:
            annotation: Annotation to quarantine
            quality_scores: Quality scores that led to quarantine
            reason: Reason for quarantine
            metadata: Additional metadata

        Returns:
            Created QuarantineItem
        """
        async with self._lock:
            item = QuarantineItem(
                annotation_id=annotation.id,
                video_id=annotation.video_id,
                quality_scores=quality_scores,
                quarantine_reason=reason,
                metadata=metadata or {},
            )

            self._items[annotation.id] = item

            logger.info(
                "annotation_quarantined",
                annotation_id=annotation.id,
                reason=reason,
                overall_score=quality_scores.overall_score,
            )

            return item

    async def record_fix_attempt(
        self,
        annotation_id: str,
        fix_result: Dict[str, Any],
        new_quality_scores: Optional[QualityScores] = None,
    ) -> Optional[QuarantineItem]:
        """Record a fix attempt for a quarantined item.

        Args:
            annotation_id: ID of the annotation
            fix_result: Result of the fix attempt
            new_quality_scores: Optional new quality scores after fix

        Returns:
            Updated QuarantineItem or None if not found
        """
        async with self._lock:
            item = self._items.get(annotation_id)
            if not item:
                return None

            item.fix_attempts += 1
            item.last_fix_attempt = datetime.utcnow()
            item.fix_results.append(fix_result)

            if new_quality_scores:
                item.quality_scores = new_quality_scores

            # Check if fixed
            if fix_result.get("success", False):
                item.status = "fixed"
                item.resolved_at = datetime.utcnow()
                logger.info(
                    "quarantine_item_fixed",
                    annotation_id=annotation_id,
                    attempts=item.fix_attempts,
                )
            elif item.fix_attempts >= self._max_fix_attempts:
                item.status = "rejected"
                item.resolved_at = datetime.utcnow()
                logger.info(
                    "quarantine_item_rejected",
                    annotation_id=annotation_id,
                    attempts=item.fix_attempts,
                )

            return item

    async def approve(
        self,
        annotation_id: str,
        approved_by: str,
        notes: Optional[str] = None,
    ) -> Optional[QuarantineItem]:
        """Manually approve a quarantined item.

        Args:
            annotation_id: ID of the annotation
            approved_by: ID of the approver
            notes: Optional approval notes

        Returns:
            Updated QuarantineItem or None if not found
        """
        async with self._lock:
            item = self._items.get(annotation_id)
            if not item:
                return None

            item.status = "approved"
            item.resolved_at = datetime.utcnow()
            item.metadata["approved_by"] = approved_by
            if notes:
                item.metadata["approval_notes"] = notes

            logger.info(
                "quarantine_item_approved",
                annotation_id=annotation_id,
                approved_by=approved_by,
            )

            return item

    async def get_item(self, annotation_id: str) -> Optional[QuarantineItem]:
        """Get a specific quarantine item."""
        async with self._lock:
            return self._items.get(annotation_id)

    async def get_active_items(self) -> List[QuarantineItem]:
        """Get all active (quarantined) items."""
        async with self._lock:
            return [item for item in self._items.values() if item.status == "quarantined"]

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the quarantine."""
        async with self._lock:
            stats = {
                "total_items": len(self._items),
                "quarantined": 0,
                "fixed": 0,
                "rejected": 0,
                "approved": 0,
                "avg_fix_attempts": 0.0,
            }

            total_attempts = 0
            for item in self._items.values():
                stats[item.status] = stats.get(item.status, 0) + 1
                total_attempts += item.fix_attempts

            if self._items:
                stats["avg_fix_attempts"] = total_attempts / len(self._items)

            return stats
