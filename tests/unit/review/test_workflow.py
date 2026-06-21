"""Tests for approval workflow state transitions."""

import pytest

from dvas.review.workflow import (
    ApprovalWorkflow,
    RejectionReason,
    WorkflowStage,
)


class TestApprovalWorkflow:
    """Test suite for ApprovalWorkflow."""

    def test_register_annotation(self):
        """Test registering an annotation."""
        workflow = ApprovalWorkflow()
        wf = workflow.register_annotation("ann1")

        assert wf.annotation_id == "ann1"
        assert wf.current_stage == WorkflowStage.INITIAL
        assert wf.approved is False
        assert wf.export_approved is False

    def test_transition(self):
        """Test stage transitions."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        result = workflow.transition("ann1", WorkflowStage.AUTOMATED_REVIEW, "system")
        assert result is not None
        assert result.current_stage == WorkflowStage.AUTOMATED_REVIEW
        assert len(result.transitions) == 1
        assert result.transitions[0].from_stage == WorkflowStage.INITIAL
        assert result.transitions[0].to_stage == WorkflowStage.AUTOMATED_REVIEW
        assert result.transitions[0].actor == "system"

    def test_transition_nonexistent(self):
        """Test transition for non-existent annotation."""
        workflow = ApprovalWorkflow()
        result = workflow.transition("nonexistent", WorkflowStage.AUTOMATED_REVIEW, "system")
        assert result is None

    def test_approve(self):
        """Test approving an annotation."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")

        result = workflow.approve("ann1", "reviewer1", "Looks good")
        assert result is not None
        assert result.approved is True
        assert result.approved_by == "reviewer1"
        assert result.export_approved is True
        assert result.current_stage == WorkflowStage.APPROVED

    def test_reject(self):
        """Test rejecting an annotation."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")

        result = workflow.reject(
            "ann1",
            RejectionReason.QUALITY_SCORE,
            "reviewer1",
            "Quality below threshold",
        )
        assert result is not None
        assert result.current_stage == WorkflowStage.REJECTED
        assert len(result.rejection_history) == 1
        assert result.rejection_history[0].reason == RejectionReason.QUALITY_SCORE
        assert result.rejection_history[0].details == "Quality below threshold"
        assert result.rejection_history[0].rejected_by == "reviewer1"

    def test_multiple_rejections(self):
        """Test multiple rejections accumulate."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        workflow.reject("ann1", RejectionReason.INCOMPLETE, "rev1", "Missing actions")
        workflow.reject("ann1", RejectionReason.QUALITY_SCORE, "rev2", "Low score")

        wf = workflow.get_annotation_state("ann1")
        assert len(wf.rejection_history) == 2
        assert wf.rejection_history[0].reason == RejectionReason.INCOMPLETE
        assert wf.rejection_history[1].reason == RejectionReason.QUALITY_SCORE

    def test_can_approve(self):
        """Test can_approve check."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        # Cannot approve from INITIAL
        assert workflow.can_approve("ann1") is False

        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        assert workflow.can_approve("ann1") is True

        workflow.approve("ann1", "reviewer1")
        assert workflow.can_approve("ann1") is False

    def test_can_reject(self):
        """Test can_reject check."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        assert workflow.can_reject("ann1") is True

        workflow.approve("ann1", "reviewer1")
        assert workflow.can_reject("ann1") is False

    def test_get_approved_annotations(self):
        """Test getting approved annotations."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.register_annotation("ann2")
        workflow.register_annotation("ann3")

        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann1", "reviewer1")
        workflow.transition("ann2", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann2", "reviewer1")

        approved = workflow.get_approved_annotations()
        assert len(approved) == 2
        assert {a.annotation_id for a in approved} == {"ann1", "ann2"}

    def test_get_rejected_annotations(self):
        """Test getting rejected annotations."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.register_annotation("ann2")

        workflow.reject("ann1", RejectionReason.INCOMPLETE, "rev1")

        rejected = workflow.get_rejected_annotations()
        assert len(rejected) == 1
        assert rejected[0].annotation_id == "ann1"

    def test_get_rejection_reasons(self):
        """Test getting rejection reasons for an annotation."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        workflow.reject("ann1", RejectionReason.INCOMPLETE, "rev1", "Missing")
        workflow.reject("ann1", RejectionReason.QUALITY_SCORE, "rev2", "Low")

        reasons = workflow.get_rejection_reasons("ann1")
        assert len(reasons) == 2
        assert reasons[0].reason == RejectionReason.INCOMPLETE
        assert reasons[1].reason == RejectionReason.QUALITY_SCORE

    def test_get_statistics(self):
        """Test workflow statistics."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.register_annotation("ann2")
        workflow.register_annotation("ann3")

        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann1", "rev1")
        workflow.reject("ann2", RejectionReason.INCOMPLETE, "rev1")
        workflow.transition("ann3", WorkflowStage.AUTOMATED_REVIEW, "system")

        stats = workflow.get_statistics()
        assert stats["total_annotations"] == 3
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 1
        assert stats["approval_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert stats["rejection_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert stats["export_ready"] == 1

    def test_check_export_gate(self):
        """Test export gate check."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.register_annotation("ann2")

        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann1", "rev1")

        assert workflow.check_export_gate("ann1") is True
        assert workflow.check_export_gate("ann2") is False
        assert workflow.check_export_gate("nonexistent") is False

    def test_get_annotations_ready_for_export(self):
        """Test getting annotations ready for export."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")
        workflow.register_annotation("ann2")
        workflow.register_annotation("ann3")

        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann1", "rev1")
        workflow.transition("ann2", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann2", "rev1")

        ready = workflow.get_annotations_ready_for_export()
        assert len(ready) == 2
        assert "ann1" in ready
        assert "ann2" in ready

    def test_rejection_record_can_be_resubmitted(self):
        """Test rejection record resubmittable flag."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        workflow.reject("ann1", RejectionReason.INCOMPLETE, "rev1")
        reasons = workflow.get_rejection_reasons("ann1")
        assert reasons[0].can_be_resubmitted is True

    def test_transition_history(self):
        """Test transition history tracking."""
        workflow = ApprovalWorkflow()
        workflow.register_annotation("ann1")

        workflow.transition("ann1", WorkflowStage.AUTOMATED_REVIEW, "system")
        workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
        workflow.approve("ann1", "reviewer1")

        wf = workflow.get_annotation_state("ann1")
        assert len(wf.transitions) == 3
        assert wf.transitions[0].from_stage == WorkflowStage.INITIAL
        assert wf.transitions[0].to_stage == WorkflowStage.AUTOMATED_REVIEW
        assert wf.transitions[1].from_stage == WorkflowStage.AUTOMATED_REVIEW
        assert wf.transitions[1].to_stage == WorkflowStage.HUMAN_REVIEW
        assert wf.transitions[2].from_stage == WorkflowStage.HUMAN_REVIEW
        assert wf.transitions[2].to_stage == WorkflowStage.APPROVED
