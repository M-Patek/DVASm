"""Tests for reviewer assignment logic."""

import pytest

from dvas.review.reviewer_assignment import (
    Assignment,
    Reviewer,
    ReviewerAssignment,
)


class TestReviewerAssignment:
    """Test suite for ReviewerAssignment."""

    def test_add_reviewer(self):
        """Test adding a reviewer to the pool."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(
            reviewer_id="rev1",
            name="Alice",
            skills=["action_verification", "temporal_analysis"],
            max_workload=5,
        )
        assignment.add_reviewer(reviewer)

        retrieved = assignment.get_reviewer_by_id("rev1")
        assert retrieved is not None
        assert retrieved.name == "Alice"
        assert retrieved.skills == ["action_verification", "temporal_analysis"]

    def test_remove_reviewer(self):
        """Test removing a reviewer."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice")
        assignment.add_reviewer(reviewer)

        assert assignment.remove_reviewer("rev1") is True
        assert assignment.get_reviewer_by_id("rev1") is None
        assert assignment.remove_reviewer("rev1") is False

    def test_assign_item_basic(self):
        """Test basic item assignment."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=5)
        assignment.add_reviewer(reviewer)

        result = assignment.assign_item("item1")
        assert result is not None
        assert result.item_id == "item1"
        assert result.reviewer_id == "rev1"
        assert result.status == "pending"

    def test_assign_item_no_reviewer(self):
        """Test assignment when no reviewers available."""
        assignment = ReviewerAssignment()
        result = assignment.assign_item("item1")
        assert result is None

    def test_assign_item_skill_match(self):
        """Test skill-based assignment."""
        assignment = ReviewerAssignment()
        reviewer_a = Reviewer(
            reviewer_id="rev_a",
            name="Alice",
            skills=["action_verification"],
            max_workload=5,
        )
        reviewer_b = Reviewer(
            reviewer_id="rev_b",
            name="Bob",
            skills=["temporal_analysis"],
            max_workload=5,
        )
        assignment.add_reviewer(reviewer_a)
        assignment.add_reviewer(reviewer_b)

        result = assignment.assign_item("item1", required_skills=["temporal_analysis"])
        assert result is not None
        assert result.reviewer_id == "rev_b"

    def test_assign_item_workload_balancing(self):
        """Test workload balancing between reviewers."""
        assignment = ReviewerAssignment()
        reviewer_a = Reviewer(reviewer_id="rev_a", name="Alice", max_workload=5)
        reviewer_b = Reviewer(reviewer_id="rev_b", name="Bob", max_workload=5)
        assignment.add_reviewer(reviewer_a)
        assignment.add_reviewer(reviewer_b)

        # Assign 3 items
        assignment.assign_item("item1")
        assignment.assign_item("item2")
        assignment.assign_item("item3")

        # Check workload distribution
        assert reviewer_a.current_workload + reviewer_b.current_workload == 3
        # Should be roughly balanced
        assert abs(reviewer_a.current_workload - reviewer_b.current_workload) <= 1

    def test_assign_item_full_workload(self):
        """Test assignment when reviewer is at capacity."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=2)
        assignment.add_reviewer(reviewer)

        assignment.assign_item("item1")
        assignment.assign_item("item2")

        # Third assignment should fail
        result = assignment.assign_item("item3")
        assert result is None

    def test_batch_assign(self):
        """Test batch assignment."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=10)
        assignment.add_reviewer(reviewer)

        results = assignment.batch_assign(["item1", "item2", "item3"])
        assert len(results) == 3
        assert all(r.reviewer_id == "rev1" for r in results)

    def test_release_item(self):
        """Test releasing an assigned item."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=5)
        assignment.add_reviewer(reviewer)

        assignment.assign_item("item1")
        assert reviewer.current_workload == 1

        assert assignment.release_item("item1") is True
        assert reviewer.current_workload == 0

    def test_release_nonexistent_item(self):
        """Test releasing a non-existent item."""
        assignment = ReviewerAssignment()
        assert assignment.release_item("nonexistent") is False

    def test_get_reviewer_workload(self):
        """Test getting reviewer workload."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=5)
        assignment.add_reviewer(reviewer)

        assert assignment.get_reviewer_workload("rev1") == 0
        assignment.assign_item("item1")
        assert assignment.get_reviewer_workload("rev1") == 1

    def test_get_reviewer_assignments(self):
        """Test getting reviewer assignments."""
        assignment = ReviewerAssignment()
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=5)
        assignment.add_reviewer(reviewer)

        assignment.assign_item("item1")
        assignment.assign_item("item2")

        assignments = assignment.get_reviewer_assignments("rev1")
        assert len(assignments) == 2
        assert {a.item_id for a in assignments} == {"item1", "item2"}

    def test_pool_statistics(self):
        """Test pool statistics."""
        assignment = ReviewerAssignment()
        reviewer_a = Reviewer(reviewer_id="rev_a", name="Alice", max_workload=5)
        reviewer_b = Reviewer(reviewer_id="rev_b", name="Bob", max_workload=10)
        assignment.add_reviewer(reviewer_a)
        assignment.add_reviewer(reviewer_b)

        assignment.assign_item("item1")
        assignment.assign_item("item2")

        stats = assignment.get_pool_statistics()
        assert stats["total_reviewers"] == 2
        assert stats["total_capacity"] == 15
        assert stats["current_load"] == 2
        assert stats["utilization_rate"] == pytest.approx(2 / 15, abs=0.01)

    def test_get_available_reviewers(self):
        """Test getting available reviewers."""
        assignment = ReviewerAssignment()
        reviewer_a = Reviewer(reviewer_id="rev_a", name="Alice", max_workload=1)
        reviewer_b = Reviewer(reviewer_id="rev_b", name="Bob", max_workload=5)
        assignment.add_reviewer(reviewer_a)
        assignment.add_reviewer(reviewer_b)

        # Fill reviewer_a
        assignment.assign_item("item1")

        available = assignment.get_available_reviewers()
        assert len(available) == 1
        assert available[0].reviewer_id == "rev_b"

    def test_reviewer_utilization(self):
        """Test reviewer utilization calculation."""
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=10)
        assert reviewer.utilization == 0.0

        reviewer.current_workload = 5
        assert reviewer.utilization == 0.5

        reviewer.current_workload = 10
        assert reviewer.utilization == 1.0

    def test_reviewer_availability(self):
        """Test reviewer availability."""
        reviewer = Reviewer(reviewer_id="rev1", name="Alice", max_workload=5)
        assert reviewer.is_available is True

        reviewer.current_workload = 5
        assert reviewer.is_available is False

        reviewer.active = False
        assert reviewer.is_available is False
