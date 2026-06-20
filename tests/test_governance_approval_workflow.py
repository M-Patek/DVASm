"""Tests for approval workflow module.

Tests for ApprovalWorkflow, ApprovalRecord, AssignmentStrategy,
Reviewer, and WorkflowStatus.
"""

import time

import pytest

from dvas.governance.approval_workflow import (
    ApprovalRecord,
    ApprovalWorkflow,
    AssignmentStrategy,
    Reviewer,
    WorkflowStatus,
)


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert WorkflowStatus.DRAFT.value == "draft"
        assert WorkflowStatus.PENDING_REVIEW.value == "pending_review"
        assert WorkflowStatus.IN_REVIEW.value == "in_review"
        assert WorkflowStatus.APPROVED.value == "approved"
        assert WorkflowStatus.REJECTED.value == "rejected"
        assert WorkflowStatus.ESCALATED.value == "escalated"


class TestAssignmentStrategy:
    """Test AssignmentStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert AssignmentStrategy.ROUND_ROBIN.value == "round_robin"
        assert AssignmentStrategy.LOAD_BALANCED.value == "load_balanced"
        assert AssignmentStrategy.EXPERTISE_BASED.value == "expertise_based"
        assert AssignmentStrategy.RANDOM.value == "random"


class TestReviewer:
    """Test Reviewer dataclass."""

    def test_reviewer_creation(self):
        """Test creating a reviewer."""
        reviewer = Reviewer(
            id="r1",
            name="Alice",
            expertise=["kitchen", "cooking"],
            max_queue_size=5,
        )
        assert reviewer.id == "r1"
        assert reviewer.name == "Alice"
        assert reviewer.expertise == ["kitchen", "cooking"]
        assert reviewer.max_queue_size == 5
        assert reviewer.active_reviews == 0

    def test_reviewer_to_dict(self):
        """Test converting reviewer to dict."""
        reviewer = Reviewer(id="r1", name="Alice")
        d = reviewer.to_dict()
        assert d["id"] == "r1"
        assert d["name"] == "Alice"


class TestApprovalRecord:
    """Test ApprovalRecord dataclass."""

    def test_record_creation(self):
        """Test creating an approval record."""
        record = ApprovalRecord(
            annotation_id="ann_001",
            reviewer_id="r1",
            status=WorkflowStatus.APPROVED,
            comments="Looks good",
        )
        assert record.annotation_id == "ann_001"
        assert record.reviewer_id == "r1"
        assert record.status == WorkflowStatus.APPROVED
        assert record.comments == "Looks good"

    def test_record_to_dict(self):
        """Test converting record to dict."""
        record = ApprovalRecord(
            annotation_id="ann_001",
            reviewer_id="r1",
            status=WorkflowStatus.APPROVED,
        )
        d = record.to_dict()
        assert d["annotation_id"] == "ann_001"
        assert d["status"] == "approved"


class TestApprovalWorkflow:
    """Test ApprovalWorkflow class."""

    def test_init(self):
        """Test initialization."""
        workflow = ApprovalWorkflow(
            workflow_id="test",
            required_approvals=2,
        )
        assert workflow.workflow_id == "test"
        assert workflow.required_approvals == 2

    def test_add_reviewer(self):
        """Test adding a reviewer."""
        workflow = ApprovalWorkflow()
        reviewer = Reviewer(id="r1", name="Alice")
        workflow.add_reviewer(reviewer)
        assert len(workflow._reviewers) == 1

    def test_remove_reviewer(self):
        """Test removing a reviewer."""
        workflow = ApprovalWorkflow()
        reviewer = Reviewer(id="r1", name="Alice")
        workflow.add_reviewer(reviewer)
        assert workflow.remove_reviewer("r1") is True
        assert len(workflow._reviewers) == 0

    def test_remove_reviewer_not_found(self):
        """Test removing non-existent reviewer."""
        workflow = ApprovalWorkflow()
        assert workflow.remove_reviewer("nonexistent") is False

    def test_submit_for_review(self):
        """Test submitting annotation for review."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        assert workflow.get_status("ann_001") == WorkflowStatus.PENDING_REVIEW

    def test_assign_reviewer_specific(self):
        """Test assigning a specific reviewer."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        assigned = workflow.assign_reviewer("ann_001", "r1")
        assert assigned == "r1"
        assert workflow.get_status("ann_001") == WorkflowStatus.IN_REVIEW

    def test_assign_reviewer_not_found(self):
        """Test assigning non-existent reviewer."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        assigned = workflow.assign_reviewer("ann_001", "nonexistent")
        assert assigned is None

    def test_assign_reviewer_no_reviewers(self):
        """Test auto-assign with no reviewers."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        assigned = workflow.assign_reviewer("ann_001")
        assert assigned is None

    def test_assign_reviewer_full_queue(self):
        """Test assigning reviewer with full queue."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice", max_queue_size=0))
        workflow.submit_for_review("ann_001")
        assigned = workflow.assign_reviewer("ann_001", "r1")
        assert assigned is None

    def test_approve_single(self):
        """Test single approval."""
        workflow = ApprovalWorkflow(required_approvals=1)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        status = workflow.approve("ann_001", "r1", "Good")
        assert status == WorkflowStatus.APPROVED

    def test_approve_multiple_required(self):
        """Test multiple approvals required."""
        workflow = ApprovalWorkflow(required_approvals=2)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.add_reviewer(Reviewer(id="r2", name="Bob"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.assign_reviewer("ann_001", "r2")

        status = workflow.approve("ann_001", "r1", "Good")
        assert status == WorkflowStatus.IN_REVIEW  # Still needs one more

        status = workflow.approve("ann_001", "r2", "Also good")
        assert status == WorkflowStatus.APPROVED

    def test_reject(self):
        """Test rejection."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        status = workflow.reject("ann_001", "r1", "Poor quality")
        assert status == WorkflowStatus.REJECTED

    def test_get_assignments(self):
        """Test getting assignments."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        assignments = workflow.get_assignments("ann_001")
        assert "r1" in assignments

    def test_get_records(self):
        """Test getting approval records."""
        workflow = ApprovalWorkflow(required_approvals=1)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.approve("ann_001", "r1", "Good")
        records = workflow.get_records("ann_001")
        assert len(records) == 1
        assert records[0].status == WorkflowStatus.APPROVED

    def test_get_queue(self):
        """Test getting review queue."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        workflow.submit_for_review("ann_002")
        queue = workflow.get_queue()
        assert len(queue) == 2
        assert "ann_001" in queue
        assert "ann_002" in queue

    def test_get_queue_after_approval(self):
        """Test queue after approval."""
        workflow = ApprovalWorkflow(required_approvals=1)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.approve("ann_001", "r1")
        queue = workflow.get_queue()
        assert "ann_001" not in queue

    def test_get_reviewer_queue_size(self):
        """Test getting reviewer queue size."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        assert workflow.get_reviewer_queue_size("r1") == 1

    def test_reviewer_stats_after_approval(self):
        """Test reviewer stats update after approval."""
        workflow = ApprovalWorkflow(required_approvals=1)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.approve("ann_001", "r1")
        reviewer = workflow._reviewers["r1"]
        assert reviewer.active_reviews == 0
        assert reviewer.total_reviews == 1

    def test_reviewer_stats_after_rejection(self):
        """Test reviewer stats update after rejection."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.reject("ann_001", "r1")
        reviewer = workflow._reviewers["r1"]
        assert reviewer.active_reviews == 0
        assert reviewer.total_reviews == 1

    def test_check_escalation_timeout(self):
        """Test escalation check with timeout."""
        workflow = ApprovalWorkflow(escalation_timeout=0.1)
        workflow.submit_for_review("ann_001")
        time.sleep(0.15)
        escalated = workflow.check_escalation("ann_001")
        assert escalated is True
        assert workflow.get_status("ann_001") == WorkflowStatus.ESCALATED

    def test_check_escalation_not_expired(self):
        """Test escalation check before timeout."""
        workflow = ApprovalWorkflow(escalation_timeout=3600)
        workflow.submit_for_review("ann_001")
        escalated = workflow.check_escalation("ann_001")
        assert escalated is False

    def test_escalate_manual(self):
        """Test manual escalation."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        status = workflow.escalate("ann_001", "Manual escalation")
        assert status == WorkflowStatus.ESCALATED

    def test_get_audit_trail(self):
        """Test getting audit trail."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        trail = workflow.get_audit_trail()
        assert len(trail) >= 1
        assert trail[0]["event_type"] == "submitted"

    def test_get_audit_trail_filtered(self):
        """Test getting filtered audit trail."""
        workflow = ApprovalWorkflow()
        workflow.submit_for_review("ann_001")
        workflow.submit_for_review("ann_002")
        trail = workflow.get_audit_trail("ann_001")
        assert all(e.get("annotation_id") == "ann_001" for e in trail)

    def test_get_stats(self):
        """Test getting workflow statistics."""
        workflow = ApprovalWorkflow(required_approvals=1)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        workflow.assign_reviewer("ann_001", "r1")
        workflow.approve("ann_001", "r1")
        stats = workflow.get_stats()
        assert stats["total_annotations"] == 1
        assert stats["approved"] == 1
        assert stats["reviewers"] == 1

    def test_get_stats_empty(self):
        """Test stats with empty workflow."""
        workflow = ApprovalWorkflow()
        stats = workflow.get_stats()
        assert stats["total_annotations"] == 0
        assert stats["approved"] == 0
        assert stats["reviewers"] == 0

    def test_round_robin_assignment(self):
        """Test round-robin reviewer assignment."""
        workflow = ApprovalWorkflow(assignment_strategy=AssignmentStrategy.ROUND_ROBIN)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.add_reviewer(Reviewer(id="r2", name="Bob"))
        workflow.submit_for_review("ann_001")
        workflow.submit_for_review("ann_002")
        assigned1 = workflow.assign_reviewer("ann_001")
        assigned2 = workflow.assign_reviewer("ann_002")
        assert assigned1 in ["r1", "r2"]
        assert assigned2 in ["r1", "r2"]

    def test_load_balanced_assignment(self):
        """Test load-balanced reviewer assignment."""
        workflow = ApprovalWorkflow(assignment_strategy=AssignmentStrategy.LOAD_BALANCED)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.add_reviewer(Reviewer(id="r2", name="Bob"))
        workflow.submit_for_review("ann_001")
        workflow.submit_for_review("ann_002")
        assigned1 = workflow.assign_reviewer("ann_001")
        assert assigned1 is not None

    def test_expertise_assignment(self):
        """Test expertise-based reviewer assignment."""
        workflow = ApprovalWorkflow(assignment_strategy=AssignmentStrategy.EXPERTISE_BASED)
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        workflow.submit_for_review("ann_001")
        assigned = workflow.assign_reviewer("ann_001")
        assert assigned is not None

    def test_approve_nonexistent_annotation(self):
        """Test approving non-existent annotation."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        # Should not raise, but won't change status
        status = workflow.approve("nonexistent", "r1")
        assert status is None

    def test_assign_reviewer_not_submitted(self):
        """Test assigning reviewer to non-submitted annotation."""
        workflow = ApprovalWorkflow()
        workflow.add_reviewer(Reviewer(id="r1", name="Alice"))
        assigned = workflow.assign_reviewer("ann_001")
        assert assigned is None
