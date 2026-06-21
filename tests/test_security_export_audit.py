"""Tests for export approval audit module.

Tests for ExportApprovalAudit, ExportApproval, and ExportStatus.
"""

import time

import pytest

from dvas.security.export_audit import (
    ExportApproval,
    ExportApprovalAudit,
    ExportStatus,
)


class TestExportStatus:
    """Test ExportStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert ExportStatus.PENDING.value == "pending"
        assert ExportStatus.APPROVED.value == "approved"
        assert ExportStatus.REJECTED.value == "rejected"
        assert ExportStatus.IN_PROGRESS.value == "in_progress"
        assert ExportStatus.COMPLETED.value == "completed"
        assert ExportStatus.FAILED.value == "failed"
        assert ExportStatus.CANCELLED.value == "cancelled"


class TestExportApproval:
    """Test ExportApproval dataclass."""

    def test_creation(self):
        """Test creating an export approval."""
        approval = ExportApproval(
            export_id="exp_001",
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001", "ann_002"],
            status=ExportStatus.PENDING,
            requested_at=time.time(),
            purpose="Monthly report",
        )
        assert approval.export_id == "exp_001"
        assert approval.status == ExportStatus.PENDING

    def test_to_dict(self):
        """Test converting to dict."""
        approval = ExportApproval(
            export_id="exp_001",
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            status=ExportStatus.PENDING,
            requested_at=time.time(),
            purpose="Test",
        )
        d = approval.to_dict()
        assert d["export_id"] == "exp_001"
        assert d["status"] == "pending"


class TestExportApprovalAudit:
    """Test ExportApprovalAudit class."""

    def test_init(self):
        """Test initialization."""
        audit = ExportApprovalAudit()
        assert audit is not None

    def test_request_export(self):
        """Test requesting an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001", "ann_002"],
            purpose="Monthly report",
        )
        assert export.requester_id == "user_001"
        assert export.status == ExportStatus.PENDING
        assert len(export.resource_ids) == 2

    def test_request_export_with_metadata(self):
        """Test requesting export with metadata."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
            data_classification="confidential",
            recipient_email="user@example.com",
            metadata={"department": "sales"},
        )
        assert export.data_classification == "confidential"
        assert export.recipient_email == "user@example.com"
        assert export.metadata["department"] == "sales"

    def test_approve_export(self):
        """Test approving an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        approved = audit.approve_export(export.export_id, "admin_001")
        assert approved.status == ExportStatus.APPROVED
        assert approved.approved_by == "admin_001"

    def test_approve_export_with_notes(self):
        """Test approving with notes."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        approved = audit.approve_export(
            export.export_id, "admin_001", notes="Approved for compliance"
        )
        assert approved.approver_notes == "Approved for compliance"

    def test_approve_non_pending_export(self):
        """Test approving non-pending export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        with pytest.raises(ValueError):
            audit.approve_export(export.export_id, "admin_002")

    def test_reject_export(self):
        """Test rejecting an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        rejected = audit.reject_export(export.export_id, "admin_001", "Insufficient justification")
        assert rejected.status == ExportStatus.REJECTED
        assert rejected.rejection_reason == "Insufficient justification"

    def test_reject_non_pending_export(self):
        """Test rejecting non-pending export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        with pytest.raises(ValueError):
            audit.reject_export(export.export_id, "admin_001", "Too late")

    def test_start_export(self):
        """Test starting an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        started = audit.start_export(export.export_id)
        assert started.status == ExportStatus.IN_PROGRESS

    def test_start_unapproved_export(self):
        """Test starting unapproved export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        with pytest.raises(ValueError):
            audit.start_export(export.export_id)

    def test_complete_export(self):
        """Test completing an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        audit.start_export(export.export_id)
        completed = audit.complete_export(export.export_id)
        assert completed.status == ExportStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_not_started_export(self):
        """Test completing export not in progress."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        with pytest.raises(ValueError):
            audit.complete_export(export.export_id)

    def test_fail_export(self):
        """Test failing an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        failed = audit.fail_export(export.export_id, "Network error")
        assert failed.status == ExportStatus.FAILED
        assert failed.rejection_reason == "Network error"

    def test_cancel_export(self):
        """Test cancelling an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        cancelled = audit.cancel_export(export.export_id, "user_001")
        assert cancelled.status == ExportStatus.CANCELLED

    def test_cancel_completed_export(self):
        """Test cancelling completed export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        audit.start_export(export.export_id)
        audit.complete_export(export.export_id)
        with pytest.raises(ValueError):
            audit.cancel_export(export.export_id, "user_001")

    def test_get_export(self):
        """Test getting an export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        retrieved = audit.get_export(export.export_id)
        assert retrieved is not None
        assert retrieved.requester_id == "user_001"

    def test_get_export_not_found(self):
        """Test getting non-existent export."""
        audit = ExportApprovalAudit()
        assert audit.get_export("nonexistent") is None

    def test_get_exports_by_status(self):
        """Test getting exports by status."""
        audit = ExportApprovalAudit()
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        export2 = audit.request_export(
            requester_id="user_002",
            resource_type="annotations",
            resource_ids=["ann_002"],
            purpose="Test",
        )
        audit.approve_export(export2.export_id, "admin_001")

        pending = audit.get_exports_by_status(ExportStatus.PENDING)
        approved = audit.get_exports_by_status(ExportStatus.APPROVED)
        assert len(pending) == 1
        assert len(approved) == 1

    def test_get_exports_by_requester(self):
        """Test getting exports by requester."""
        audit = ExportApprovalAudit()
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_002"],
            purpose="Test",
        )
        audit.request_export(
            requester_id="user_002",
            resource_type="annotations",
            resource_ids=["ann_003"],
            purpose="Test",
        )

        user_exports = audit.get_exports_by_requester("user_001")
        assert len(user_exports) == 2

    def test_get_pending_exports(self):
        """Test getting pending exports."""
        audit = ExportApprovalAudit()
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        export = audit.request_export(
            requester_id="user_002",
            resource_type="annotations",
            resource_ids=["ann_002"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")

        pending = audit.get_pending_exports()
        assert len(pending) == 1

    def test_get_audit_trail(self):
        """Test getting audit trail."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        trail = audit.get_audit_trail(export.export_id)
        assert len(trail) >= 2  # requested + approved

    def test_get_all_exports(self):
        """Test getting all exports."""
        audit = ExportApprovalAudit()
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.request_export(
            requester_id="user_002",
            resource_type="annotations",
            resource_ids=["ann_002"],
            purpose="Test",
        )
        all_exports = audit.get_all_exports()
        assert len(all_exports) == 2

    def test_get_export_stats(self):
        """Test getting export statistics."""
        audit = ExportApprovalAudit()
        audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        export = audit.request_export(
            requester_id="user_002",
            resource_type="annotations",
            resource_ids=["ann_002"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        audit.start_export(export.export_id)
        audit.complete_export(export.export_id)

        stats = audit.get_export_stats()
        assert stats["total_exports"] == 2
        assert stats["pending"] == 1
        assert stats["completed"] == 1
        assert stats["by_status"]["completed"] == 1

    def test_full_workflow(self):
        """Test complete export approval workflow."""
        audit = ExportApprovalAudit()

        # Request
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001", "ann_002"],
            purpose="Compliance report",
            data_classification="confidential",
        )
        assert export.status == ExportStatus.PENDING

        # Approve
        approved = audit.approve_export(export.export_id, "admin_001")
        assert approved.status == ExportStatus.APPROVED

        # Start
        started = audit.start_export(export.export_id)
        assert started.status == ExportStatus.IN_PROGRESS

        # Complete
        completed = audit.complete_export(export.export_id)
        assert completed.status == ExportStatus.COMPLETED

        # Verify audit trail
        trail = audit.get_audit_trail(export.export_id)
        assert len(trail) >= 4  # requested + approved + started + completed


class TestExportApprovalAuditEdgeCases:
    """Test edge cases for ExportApprovalAudit."""

    def test_empty_resource_ids(self):
        """Test with empty resource IDs."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=[],
            purpose="Test",
        )
        assert export.resource_ids == []

    def test_very_long_purpose(self):
        """Test with very long purpose."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="x" * 10000,
        )
        assert len(export.purpose) == 10000

    def test_multiple_approvals_same_export(self):
        """Test multiple attempts to approve same export."""
        audit = ExportApprovalAudit()
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001"],
            purpose="Test",
        )
        audit.approve_export(export.export_id, "admin_001")
        with pytest.raises(ValueError):
            audit.approve_export(export.export_id, "admin_002")
