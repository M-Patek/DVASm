"""Review queue for managing pending reviews items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class QueuePriority(str, Enum):
    """Priority levels for review queue items."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueueItemStatus(str, Enum):
    """Status of a review queue item."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"


@dataclass
class QueueItem:
    """Item in the review queue."""

    item_id: str
    annotation_id: str
    video_id: str
    priority: QueuePriority
    status: QueueItemStatus = QueueItemStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_by: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.due_by is None:
            priority_days = {
                QueuePriority.CRITICAL: 0,
                QueuePriority.HIGH: 1,
                QueuePriority.MEDIUM: 3,
                QueuePriority.LOW: 7,
            }
            days = priority_days.get(self.priority, 3)
            self.due_by = self.created_at + timedelta(days=days)

    def is_overdue(self) -> bool:
        if self.due_by is None:
            return False
        now = datetime.now(timezone.utc)
        return now > self.due_by

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "due_by": self.due_by.isoformat() if self.due_by else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class ReviewQueue:
    """Queue for managing pending reviews with priority-based ordering."""

    def __init__(self):
        self._items: Dict[str, QueueItem] = {}
        self._priority_order = {
            QueuePriority.CRITICAL: 0,
            QueuePriority.HIGH: 1,
            QueuePriority.MEDIUM: 2,
            QueuePriority.LOW: 3,
        }

    def add_item(
        self,
        item_id: str,
        annotation_id: str,
        video_id: str,
        priority: QueuePriority = QueuePriority.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueueItem:
        item = QueueItem(
            item_id=item_id,
            annotation_id=annotation_id,
            video_id=video_id,
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._items[item_id] = item
        logger.info("item_added_to_queue", item_id=item_id, priority=priority.value)
        return item

    def remove_item(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            logger.info("item_removed_from_queue", item_id=item_id)
            return True
        return False

    def get_next_item(
        self,
        priority_filter: Optional[List[QueuePriority]] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> Optional[QueueItem]:
        candidates = self._get_pending_items(priority_filter, tag_filter)
        if not candidates:
            return None
        candidates.sort(key=lambda x: (self._priority_order[x.priority], x.created_at))
        return candidates[0]

    def _get_pending_items(
        self,
        priority_filter: Optional[List[QueuePriority]] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> List[QueueItem]:
        items = []
        for item in self._items.values():
            if item.status != QueueItemStatus.PENDING:
                continue
            if priority_filter and item.priority not in priority_filter:
                continue
            if tag_filter and not any(t in item.tags for t in tag_filter):
                continue
            items.append(item)
        return items

    def assign_item(self, item_id: str, reviewer_id: str) -> Optional[QueueItem]:
        item = self._items.get(item_id)
        if not item:
            return None
        item.status = QueueItemStatus.ASSIGNED
        item.assigned_to = reviewer_id
        logger.info("item_assigned", item_id=item_id, reviewer_id=reviewer_id)
        return item

    def complete_item(self, item_id: str) -> Optional[QueueItem]:
        item = self._items.get(item_id)
        if not item:
            return None
        item.status = QueueItemStatus.COMPLETED
        item.completed_at = datetime.utcnow()
        logger.info("item_completed", item_id=item_id)
        return item

    def reject_item(self, item_id: str, reason: Optional[str] = None) -> Optional[QueueItem]:
        item = self._items.get(item_id)
        if not item:
            return None
        item.status = QueueItemStatus.REJECTED
        item.completed_at = datetime.utcnow()
        if reason:
            item.metadata["rejection_reason"] = reason
        logger.info("item_rejected", item_id=item_id, reason=reason)
        return item

    def get_item(self, item_id: str) -> Optional[QueueItem]:
        return self._items.get(item_id)

    def get_pending_count(self) -> int:
        return sum(1 for item in self._items.values() if item.status == QueueItemStatus.PENDING)

    def get_items_by_status(self, status: QueueItemStatus) -> List[QueueItem]:
        return [item for item in self._items.values() if item.status == status]

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_items": len(self._items),
            "pending": 0,
            "assigned": 0,
            "in_progress": 0,
            "completed": 0,
            "rejected": 0,
            "overdue": 0,
            "by_priority": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        }
        for item in self._items.values():
            status = item.status.value
            if status in stats:
                stats[status] += 1
            priority = item.priority.value
            if priority in stats["by_priority"]:
                stats["by_priority"][priority] += 1
            if item.is_overdue():
                stats["overdue"] += 1
        return stats

    def batch_assign(
        self,
        reviewer_id: str,
        count: int = 5,
        priority_filter: Optional[List[QueuePriority]] = None,
    ) -> List[QueueItem]:
        assigned = []
        for _ in range(count):
            item = self.get_next_item(priority_filter)
            if not item:
                break
            self.assign_item(item.item_id, reviewer_id)
            assigned.append(item)
        logger.info("batch_assignment", reviewer_id=reviewer_id, assigned_count=len(assigned))
        return assigned

    def get_overdue_items(self) -> List[QueueItem]:
        return [item for item in self._items.values() if item.is_overdue()]

    def reorder_by_priority(self) -> List[QueueItem]:
        items = list(self._items.values())
        items.sort(key=lambda x: (self._priority_order[x.priority], x.created_at))
        return items
