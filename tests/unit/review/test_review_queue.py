"""Tests for review queue priority ordering."""

from datetime import datetime, timezone

import pytest

from dvas.review.review_queue import (
    QueueItem,
    QueueItemStatus,
    QueuePriority,
    ReviewQueue,
)


class TestReviewQueue:
    """Test suite for ReviewQueue."""

    def test_add_item(self):
        """Test adding items to the queue."""
        queue = ReviewQueue()
        item = queue.add_item("item1", "ann1", "vid1", QueuePriority.MEDIUM)

        assert item.item_id == "item1"
        assert item.annotation_id == "ann1"
        assert item.priority == QueuePriority.MEDIUM
        assert item.status == QueueItemStatus.PENDING

    def test_remove_item(self):
        """Test removing items from the queue."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")

        assert queue.remove_item("item1") is True
        assert queue.remove_item("item1") is False

    def test_get_next_item_priority_order(self):
        """Test that items are returned in priority order."""
        queue = ReviewQueue()
        queue.add_item("item_low", "ann1", "vid1", QueuePriority.LOW)
        queue.add_item("item_high", "ann2", "vid2", QueuePriority.HIGH)
        queue.add_item("item_critical", "ann3", "vid3", QueuePriority.CRITICAL)
        queue.add_item("item_medium", "ann4", "vid4", QueuePriority.MEDIUM)

        next_item = queue.get_next_item()
        assert next_item is not None
        assert next_item.item_id == "item_critical"

        # Mark as assigned and get next
        next_item.status = QueueItemStatus.ASSIGNED
        next_item = queue.get_next_item()
        assert next_item.item_id == "item_high"

    def test_get_next_item_priority_filter(self):
        """Test filtering by priority."""
        queue = ReviewQueue()
        queue.add_item("item_critical", "ann1", "vid1", QueuePriority.CRITICAL)
        queue.add_item("item_low", "ann2", "vid2", QueuePriority.LOW)

        next_item = queue.get_next_item(priority_filter=[QueuePriority.LOW])
        assert next_item is not None
        assert next_item.item_id == "item_low"

    def test_get_next_item_tag_filter(self):
        """Test filtering by tags."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1", tags=["batch1"])
        queue.add_item("item2", "ann2", "vid2", tags=["batch2"])

        next_item = queue.get_next_item(tag_filter=["batch2"])
        assert next_item is not None
        assert next_item.item_id == "item2"

    def test_get_next_item_empty_queue(self):
        """Test getting next item from empty queue."""
        queue = ReviewQueue()
        next_item = queue.get_next_item()
        assert next_item is None

    def test_assign_item(self):
        """Test assigning an item to a reviewer."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")

        result = queue.assign_item("item1", "reviewer1")
        assert result is not None
        assert result.status == QueueItemStatus.ASSIGNED
        assert result.assigned_to == "reviewer1"

    def test_assign_nonexistent_item(self):
        """Test assigning a non-existent item."""
        queue = ReviewQueue()
        result = queue.assign_item("nonexistent", "reviewer1")
        assert result is None

    def test_complete_item(self):
        """Test completing an item."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")

        result = queue.complete_item("item1")
        assert result is not None
        assert result.status == QueueItemStatus.COMPLETED
        assert result.completed_at is not None

    def test_reject_item(self):
        """Test rejecting an item."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")

        result = queue.reject_item("item1", "quality_too_low")
        assert result is not None
        assert result.status == QueueItemStatus.REJECTED
        assert result.metadata["rejection_reason"] == "quality_too_low"

    def test_get_pending_count(self):
        """Test getting pending count."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")
        queue.add_item("item2", "ann2", "vid2")
        queue.assign_item("item1", "reviewer1")

        assert queue.get_pending_count() == 1

    def test_get_items_by_status(self):
        """Test getting items by status."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1")
        queue.add_item("item2", "ann2", "vid2")
        queue.complete_item("item1")

        completed = queue.get_items_by_status(QueueItemStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].item_id == "item1"

        pending = queue.get_items_by_status(QueueItemStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].item_id == "item2"

    def test_get_statistics(self):
        """Test queue statistics."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1", QueuePriority.HIGH)
        queue.add_item("item2", "ann2", "vid2", QueuePriority.LOW)
        queue.add_item("item3", "ann3", "vid3", QueuePriority.MEDIUM)
        queue.complete_item("item1")

        stats = queue.get_statistics()
        assert stats["total_items"] == 3
        assert stats["completed"] == 1
        assert stats["pending"] == 2
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["low"] == 1
        assert stats["by_priority"]["medium"] == 1

    def test_batch_assign(self):
        """Test batch assignment."""
        queue = ReviewQueue()
        for i in range(5):
            queue.add_item(f"item{i}", f"ann{i}", f"vid{i}")

        assigned = queue.batch_assign("reviewer1", count=3)
        assert len(assigned) == 3
        assert all(a.assigned_to == "reviewer1" for a in assigned)

    def test_batch_assign_priority_filter(self):
        """Test batch assignment with priority filter."""
        queue = ReviewQueue()
        queue.add_item("item1", "ann1", "vid1", QueuePriority.HIGH)
        queue.add_item("item2", "ann2", "vid2", QueuePriority.LOW)
        queue.add_item("item3", "ann3", "vid3", QueuePriority.HIGH)

        assigned = queue.batch_assign(
            "reviewer1", count=5, priority_filter=[QueuePriority.HIGH]
        )
        assert len(assigned) == 2
        assert all(a.priority == QueuePriority.HIGH for a in assigned)

    def test_overdue_items(self):
        """Test overdue item detection."""
        queue = ReviewQueue()
        # Create an item with a past due date by manipulating created_at
        item = queue.add_item("item1", "ann1", "vid1", QueuePriority.LOW)
        # Set due_by to past by modifying the item directly
        item.due_by = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert item.is_overdue() is True

        # Create a future-dated item that is NOT overdue
        item2 = queue.add_item("item2", "ann2", "vid2", QueuePriority.LOW)
        item2.due_by = datetime(2099, 1, 1, tzinfo=timezone.utc)
        assert item2.is_overdue() is False

    def test_reorder_by_priority(self):
        """Test reordering by priority."""
        queue = ReviewQueue()
        queue.add_item("item_low", "ann1", "vid1", QueuePriority.LOW)
        queue.add_item("item_high", "ann2", "vid2", QueuePriority.HIGH)
        queue.add_item("item_critical", "ann3", "vid3", QueuePriority.CRITICAL)

        ordered = queue.reorder_by_priority()
        priorities = [i.priority for i in ordered]
        assert priorities == [QueuePriority.CRITICAL, QueuePriority.HIGH, QueuePriority.LOW]

    def test_queue_item_due_by(self):
        """Test automatic due_by calculation based on priority."""
        from datetime import timedelta

        item_critical = QueueItem("i1", "a1", "v1", QueuePriority.CRITICAL)
        item_high = QueueItem("i2", "a2", "v2", QueuePriority.HIGH)
        item_medium = QueueItem("i3", "a3", "v3", QueuePriority.MEDIUM)
        item_low = QueueItem("i4", "a4", "v4", QueuePriority.LOW)

        assert item_critical.due_by == item_critical.created_at + timedelta(days=0)
        assert item_high.due_by == item_high.created_at + timedelta(days=1)
        assert item_medium.due_by == item_medium.created_at + timedelta(days=3)
        assert item_low.due_by == item_low.created_at + timedelta(days=7)
