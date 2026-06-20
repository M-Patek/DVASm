"""Reviewer assignment with workload balancing and skill-based matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Reviewer:
    """A human reviewer with skills and workload."""
    reviewer_id: str
    name: str
    skills: List[str] = field(default_factory=list)
    max_workload: int = 10
    current_workload: int = 0
    active: bool = True
    avg_review_time_min: float = 15.0
    review_count: int = 0
    agreement_rate: float = 0.0

    @property
    def is_available(self) -> bool:
        return self.active and self.current_workload < self.max_workload

    @property
    def utilization(self) -> float:
        if self.max_workload <= 0:
            return 1.0
        return self.current_workload / self.max_workload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "name": self.name,
            "skills": self.skills,
            "max_workload": self.max_workload,
            "current_workload": self.current_workload,
            "active": self.active,
            "avg_review_time_min": self.avg_review_time_min,
            "review_count": self.review_count,
            "agreement_rate": self.agreement_rate,
        }


@dataclass
class Assignment:
    """An assignment of a review item to a reviewer."""
    item_id: str
    reviewer_id: str
    assigned_at: str = ""
    status: str = "pending"
    priority: str = "medium"
    required_skills: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "reviewer_id": self.reviewer_id,
            "assigned_at": self.assigned_at,
            "status": self.status,
            "priority": self.priority,
            "required_skills": self.required_skills,
            "notes": self.notes,
        }


class ReviewerAssignment:
    """Manages reviewer assignments with workload balancing and skill matching."""

    def __init__(self):
        self._reviewers: Dict[str, Reviewer] = {}
        self._assignments: Dict[str, Assignment] = {}
        self._reviewer_items: Dict[str, Set[str]] = {}

    def add_reviewer(self, reviewer: Reviewer) -> None:
        self._reviewers[reviewer.reviewer_id] = reviewer
        if reviewer.reviewer_id not in self._reviewer_items:
            self._reviewer_items[reviewer.reviewer_id] = set()
        logger.info("reviewer_added", reviewer_id=reviewer.reviewer_id, name=reviewer.name)

    def remove_reviewer(self, reviewer_id: str) -> bool:
        if reviewer_id in self._reviewers:
            del self._reviewers[reviewer_id]
            self._reviewer_items.pop(reviewer_id, None)
            logger.info("reviewer_removed", reviewer_id=reviewer_id)
            return True
        return False

    def get_reviewer_by_id(self, reviewer_id: str) -> Optional[Reviewer]:
        return self._reviewers.get(reviewer_id)

    def assign_item(self, item_id: str, required_skills: Optional[List[str]] = None, priority: str = "medium") -> Optional[Assignment]:
        reviewer = self._find_best_reviewer(required_skills or [])
        if not reviewer:
            logger.warning("no_reviewer_available", item_id=item_id)
            return None

        from datetime import datetime
        assignment = Assignment(
            item_id=item_id,
            reviewer_id=reviewer.reviewer_id,
            assigned_at=datetime.utcnow().isoformat(),
            priority=priority,
            required_skills=required_skills or [],
        )

        self._assignments[item_id] = assignment
        self._reviewer_items[reviewer.reviewer_id].add(item_id)
        reviewer.current_workload += 1
        reviewer.review_count += 1

        logger.info("item_assigned", item_id=item_id, reviewer_id=reviewer.reviewer_id, priority=priority)
        return assignment

    def _find_best_reviewer(self, required_skills: List[str]) -> Optional[Reviewer]:
        candidates = []
        for reviewer in self._reviewers.values():
            if not reviewer.is_available:
                continue

            skill_match = 0.0
            if required_skills:
                matched = sum(1 for skill in required_skills if skill in reviewer.skills)
                skill_match = matched / len(required_skills)
            else:
                skill_match = 1.0

            if skill_match > 0:
                candidates.append((reviewer, skill_match, reviewer.utilization))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[1], x[2]))
        return candidates[0][0]

    def batch_assign(self, item_ids: List[str], required_skills: Optional[List[str]] = None, priority: str = "medium") -> List[Assignment]:
        assignments = []
        for item_id in item_ids:
            assignment = self.assign_item(item_id, required_skills, priority)
            if assignment:
                assignments.append(assignment)
        return assignments

    def release_item(self, item_id: str) -> bool:
        assignment = self._assignments.get(item_id)
        if not assignment:
            return False

        reviewer = self._reviewers.get(assignment.reviewer_id)
        if reviewer:
            reviewer.current_workload = max(0, reviewer.current_workload - 1)
            self._reviewer_items[reviewer.reviewer_id].discard(item_id)

        del self._assignments[item_id]
        logger.info("item_released", item_id=item_id, reviewer_id=assignment.reviewer_id)
        return True

    def get_reviewer_workload(self, reviewer_id: str) -> int:
        reviewer = self._reviewers.get(reviewer_id)
        return reviewer.current_workload if reviewer else 0

    def get_reviewer_assignments(self, reviewer_id: str) -> List[Assignment]:
        item_ids = self._reviewer_items.get(reviewer_id, set())
        return [self._assignments[item_id] for item_id in item_ids if item_id in self._assignments]

    def get_pool_statistics(self) -> Dict[str, Any]:
        total_reviewers = len(self._reviewers)
        available = sum(1 for r in self._reviewers.values() if r.is_available)
        total_capacity = sum(r.max_workload for r in self._reviewers.values())
        current_load = sum(r.current_workload for r in self._reviewers.values())
        avg_agreement = sum(r.agreement_rate for r in self._reviewers.values()) / total_reviewers if total_reviewers > 0 else 0.0

        return {
            "total_reviewers": total_reviewers,
            "available_reviewers": available,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "utilization_rate": current_load / total_capacity if total_capacity > 0 else 0.0,
            "average_agreement_rate": avg_agreement,
        }

    def get_available_reviewers(self) -> List[Reviewer]:
        return [r for r in self._reviewers.values() if r.is_available]

    def get_assignment(self, item_id: str) -> Optional[Assignment]:
        return self._assignments.get(item_id)
