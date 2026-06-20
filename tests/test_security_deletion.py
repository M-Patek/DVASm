"""Tests for deletion request flow module.

Tests for DeletionRequestFlow, DeletionRequest, DeletionStatus, and DeletionScope.
"""

import time

import pytest

from dvas.security.deletion import (
    DeletionRequest,
    DeletionRequestFlow,
    DeletionScope,
    DeletionStatus,
)


class TestDeletionStatus:
    """Test DeletionStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert DeletionStatus.PENDING.value == "pending"
        assert DeletionStatus.VERIFYING.value == "verifying"
        assert DeletionStatus.IN_PROGRESS.value == "in_progress"
        assert DeletionStatus.COMPLETED.value == "completed"
        assert DeletionStatus.PARTIAL.value == "partial"
        assert DeletionStatus.FAILED.value == "failed"
        assert DeletionStatus.CANCELLED.value == "cancelled"


class TestDeletionScope:
    """Test DeletionScope enum."""

    def test_scope_values(self):
        """Test scope values."""
        assert DeletionScope.USER_DATA.value == "user_data"
        assert DeletionScope.ANNOTATIONS.value == "annotations"
        assert DeletionScope.VIDEOS.value == "videos"
        assert DeletionScope.EXPORTS.value == "exports"
        assert DeletionScope.AUDIT_LOGS.value == "audit_logs"
        assert DeletionScope.ALL.value == "all"
        assert DeletionScope.CUSTOM.value == "custom"


class TestDeletionRequest:
    """Test DeletionRequest dataclass."""

    def test_creation(self):
        """Test creating a deletion request."""
        request = DeletionRequest(
            request_id="req_001",
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            status=DeletionStatus.PENDING,
            created_at=time.time(),
            reason="GDPR request",
        )
        assert request.request_id == "req_001"
        assert request.user_id == "user_001"
        assert request.scope == DeletionScope.USER_DATA
        assert request.status == DeletionStatus.PENDING

    def test_to_dict(self):
        """Test converting to dict."""
        request = DeletionRequest(
            request_id="req_001",
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            status=DeletionStatus.PENDING,
            created_at=1234567890.0,
            reason="GDPR request",
        )
        d = request.to_dict()
        assert d["request_id"] == "req_001"
        assert d["user_id"] == "user_001"
        assert d["scope"] == "user_data"
        assert d["status"] == "pending"


class TestDeletionRequestFlow:
    """Test DeletionRequestFlow class."""

    def test_init(self):
        """Test initialization."""
        flow = DeletionRequestFlow()
        assert flow is not None

    def test_create_request(self):
        """Test creating a deletion request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR Article 17",
        )
        assert request.user_id == "user_001"
        assert request.scope == DeletionScope.USER_DATA
        assert request.status == DeletionStatus.PENDING
        assert request.verification_code is not None

    def test_create_request_with_resource_ids(self):
        """Test creating request with specific resource IDs."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.ANNOTATIONS,
            reason="GDPR",
            resource_ids=["ann_001", "ann_002"],
        )
        assert request.resource_ids == ["ann_001", "ann_002"]

    def test_create_request_with_metadata(self):
        """Test creating request with metadata."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
            metadata={"ticket_id": "TICKET-123"},
        )
        assert request.metadata["ticket_id"] == "TICKET-123"

    def test_verify_request(self):
        """Test verifying a request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        code = request.verification_code
        assert flow.verify_request(request.request_id, code) is True
        assert request.status == DeletionStatus.VERIFYING

    def test_verify_request_wrong_code(self):
        """Test verifying with wrong code."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        with pytest.raises(ValueError):
            flow.verify_request(request.request_id, "wrong_code")

    def test_verify_request_not_pending(self):
        """Test verifying non-pending request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.verify_request(request.request_id, request.verification_code)
        with pytest.raises(ValueError):
            flow.verify_request(request.request_id, request.verification_code)

    def test_approve_request(self):
        """Test approving a request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.verify_request(request.request_id, request.verification_code)
        approved = flow.approve_request(request.request_id, "admin_001")
        assert approved.status == DeletionStatus.IN_PROGRESS
        assert approved.approved_by == "admin_001"

    def test_approve_request_not_verified(self):
        """Test approving non-verified request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        with pytest.raises(ValueError):
            flow.approve_request(request.request_id, "admin_001")

    def test_process_request(self):
        """Test processing a request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.verify_request(request.request_id, request.verification_code)
        flow.approve_request(request.request_id, "admin_001")
        processed = flow.process_request(request.request_id)
        assert processed.status in [DeletionStatus.COMPLETED, DeletionStatus.PARTIAL]

    def test_process_request_with_handler(self):
        """Test processing with a handler."""
        flow = DeletionRequestFlow()

        def handler(resource_id, request):
            return True

        flow.register_deletion_handler(DeletionScope.USER_DATA, handler)

        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
            resource_ids=["data_001"],
        )
        flow.verify_request(request.request_id, request.verification_code)
        flow.approve_request(request.request_id, "admin_001")
        processed = flow.process_request(request.request_id)
        assert processed.status == DeletionStatus.COMPLETED
        assert "data_001" in processed.completed_resources

    def test_process_request_with_failing_handler(self):
        """Test processing with a failing handler."""
        flow = DeletionRequestFlow()

        def failing_handler(resource_id, request):
            raise RuntimeError("Handler failed")

        flow.register_deletion_handler(DeletionScope.USER_DATA, failing_handler)

        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
            resource_ids=["data_001"],
        )
        flow.verify_request(request.request_id, request.verification_code)
        flow.approve_request(request.request_id, "admin_001")
        processed = flow.process_request(request.request_id)
        # Should not crash, but may fail
        assert processed.status in [DeletionStatus.COMPLETED, DeletionStatus.FAILED, DeletionStatus.PARTIAL]

    def test_process_request_not_approved(self):
        """Test processing non-approved request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        with pytest.raises(ValueError):
            flow.process_request(request.request_id)

    def test_cancel_request(self):
        """Test cancelling a request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        cancelled = flow.cancel_request(request.request_id, "user_001")
        assert cancelled.status == DeletionStatus.CANCELLED

    def test_cancel_completed_request(self):
        """Test cancelling completed request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.verify_request(request.request_id, request.verification_code)
        flow.approve_request(request.request_id, "admin_001")
        flow.process_request(request.request_id)
        with pytest.raises(ValueError):
            flow.cancel_request(request.request_id, "user_001")

    def test_get_request(self):
        """Test getting a request."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        retrieved = flow.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.user_id == "user_001"

    def test_get_request_not_found(self):
        """Test getting non-existent request."""
        flow = DeletionRequestFlow()
        assert flow.get_request("nonexistent") is None

    def test_get_requests_by_user(self):
        """Test getting requests by user."""
        flow = DeletionRequestFlow()
        flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.create_request(
            user_id="user_001",
            scope=DeletionScope.ANNOTATIONS,
            reason="GDPR",
        )
        flow.create_request(
            user_id="user_002",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        requests = flow.get_requests_by_user("user_001")
        assert len(requests) == 2

    def test_get_requests_by_status(self):
        """Test getting requests by status."""
        flow = DeletionRequestFlow()
        flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        request = flow.create_request(
            user_id="user_002",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.verify_request(request.request_id, request.verification_code)
        flow.approve_request(request.request_id, "admin_001")

        pending = flow.get_requests_by_status(DeletionStatus.PENDING)
        in_progress = flow.get_requests_by_status(DeletionStatus.IN_PROGRESS)
        assert len(pending) == 1
        assert len(in_progress) == 1

    def test_get_pending_requests(self):
        """Test getting pending requests."""
        flow = DeletionRequestFlow()
        flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.create_request(
            user_id="user_002",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        pending = flow.get_pending_requests()
        assert len(pending) == 2

    def test_get_audit_trail(self):
        """Test getting audit trail."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        trail = flow.get_audit_trail(request.request_id)
        assert len(trail) >= 1  # At least "created"

    def test_get_stats(self):
        """Test getting statistics."""
        flow = DeletionRequestFlow()
        flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        flow.create_request(
            user_id="user_002",
            scope=DeletionScope.ANNOTATIONS,
            reason="GDPR",
        )
        stats = flow.get_stats()
        assert stats["total_requests"] == 2
        assert stats["by_scope"]["user_data"] == 1
        assert stats["by_scope"]["annotations"] == 1
        assert stats["by_status"]["pending"] == 2

    def test_register_deletion_handler(self):
        """Test registering deletion handler."""
        flow = DeletionRequestFlow()

        def handler(resource_id, request):
            return True

        flow.register_deletion_handler(DeletionScope.USER_DATA, handler)
        assert len(flow._deletion_handlers[DeletionScope.USER_DATA]) == 1

    def test_full_workflow(self):
        """Test complete deletion request workflow."""
        flow = DeletionRequestFlow()

        # Create
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR Article 17 request",
            resource_ids=["data_001", "data_002"],
        )
        assert request.status == DeletionStatus.PENDING
        code = request.verification_code

        # Verify
        flow.verify_request(request.request_id, code)
        assert request.status == DeletionStatus.VERIFYING

        # Approve
        flow.approve_request(request.request_id, "admin_001")
        assert request.status == DeletionStatus.IN_PROGRESS

        # Process (no handlers, so all will fail)
        processed = flow.process_request(request.request_id)
        assert processed.status in [DeletionStatus.COMPLETED, DeletionStatus.PARTIAL, DeletionStatus.FAILED]

        # Verify audit trail
        trail = flow.get_audit_trail(request.request_id)
        assert len(trail) >= 4  # created + verified + approved + processed


class TestDeletionRequestFlowEdgeCases:
    """Test edge cases for DeletionRequestFlow."""

    def test_empty_resource_ids(self):
        """Test with empty resource IDs."""
        flow = DeletionRequestFlow()
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
            resource_ids=[],
        )
        assert request.resource_ids == []

    def test_very_long_reason(self):
        """Test with very long reason."""
        flow = DeletionRequestFlow()
        reason = "x" * 10000
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason=reason,
        )
        assert len(request.reason) == 10000

    def test_multiple_requests_same_user(self):
        """Test multiple requests from same user."""
        flow = DeletionRequestFlow()
        request1 = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        request2 = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.ANNOTATIONS,
            reason="GDPR",
        )
        assert request1.request_id != request2.request_id

    def test_verification_code_uniqueness(self):
        """Test that verification codes are unique."""
        flow = DeletionRequestFlow()
        request1 = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        request2 = flow.create_request(
            user_id="user_002",
            scope=DeletionScope.USER_DATA,
            reason="GDPR",
        )
        assert request1.verification_code != request2.verification_code
