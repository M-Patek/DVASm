"""GDPR-style data deletion request flow for DVAS.

Provides a workflow for handling data deletion requests
with verification, audit logging, and confirmation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DeletionStatus(str, Enum):
    """Status of a deletion request."""

    PENDING = "pending"
    VERIFYING = "verifying"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeletionScope(str, Enum):
    """Scope of data deletion."""

    USER_DATA = "user_data"
    ANNOTATIONS = "annotations"
    VIDEOS = "videos"
    EXPORTS = "exports"
    AUDIT_LOGS = "audit_logs"
    ALL = "all"
    CUSTOM = "custom"


@dataclass
class DeletionRequest:
    """A data deletion request."""

    request_id: str
    user_id: str
    scope: DeletionScope
    status: DeletionStatus
    created_at: float
    reason: str
    resource_ids: List[str] = field(default_factory=list)
    completed_at: Optional[float] = None
    completed_resources: List[str] = field(default_factory=list)
    failed_resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification_code: Optional[str] = None
    verified_at: Optional[float] = None
    approved_by: Optional[str] = None
    retention_exceptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "scope": self.scope.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "reason": self.reason,
            "resource_ids": self.resource_ids,
            "completed_at": self.completed_at,
            "completed_resources": self.completed_resources,
            "failed_resources": self.failed_resources,
            "metadata": self.metadata,
            "verification_code": self.verification_code,
            "verified_at": self.verified_at,
            "approved_by": self.approved_by,
            "retention_exceptions": self.retention_exceptions,
        }


class DeletionRequestFlow:
    """GDPR-style data deletion request workflow.

    Usage::

        flow = DeletionRequestFlow()

        # Create request
        request = flow.create_request(
            user_id="user_001",
            scope=DeletionScope.USER_DATA,
            reason="GDPR Article 17 request",
        )

        # Verify request
        flow.verify_request(request.request_id, verification_code="123456")

        # Approve and process
        flow.approve_request(request.request_id, approver_id="admin_001")
        flow.process_request(request.request_id)

        # Complete
        flow.complete_request(request.request_id)
    """

    def __init__(self) -> None:
        """Initialize the deletion request flow."""
        self._requests: Dict[str, DeletionRequest] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._deletion_handlers: Dict[DeletionScope, List[Callable]] = {}
        self._verification_ttl: int = 86400  # 24 hours

    def create_request(
        self,
        user_id: str,
        scope: DeletionScope,
        reason: str,
        resource_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeletionRequest:
        """Create a new deletion request.

        Args:
            user_id: The user requesting deletion.
            scope: The scope of deletion.
            reason: The reason for deletion.
            resource_ids: Optional specific resource IDs to delete.
            metadata: Optional additional metadata.

        Returns:
            The created DeletionRequest.
        """
        request_id = str(uuid.uuid4())
        verification_code = self._generate_verification_code()

        request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            scope=scope,
            status=DeletionStatus.PENDING,
            created_at=time.time(),
            reason=reason,
            resource_ids=resource_ids or [],
            metadata=metadata or {},
            verification_code=verification_code,
        )

        self._requests[request_id] = request
        self._log_event(request_id, "created", user_id)

        logger.info(
            "deletion_request_created",
            request_id=request_id,
            user_id=user_id,
            scope=scope.value,
        )

        return request

    def verify_request(self, request_id: str, verification_code: str) -> bool:
        """Verify a deletion request with a code.

        Args:
            request_id: The request ID.
            verification_code: The verification code.

        Returns:
            True if verified successfully.

        Raises:
            ValueError: If the request is not found or verification fails.
        """
        request = self._get_request(request_id)

        if request.status != DeletionStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending")

        if request.verification_code != verification_code:
            raise ValueError("Invalid verification code")

        # Check TTL
        if time.time() - request.created_at > self._verification_ttl:
            raise ValueError("Verification code has expired")

        request.status = DeletionStatus.VERIFYING
        request.verified_at = time.time()
        self._log_event(request_id, "verified", request.user_id)

        logger.info("deletion_request_verified", request_id=request_id)
        return True

    def approve_request(self, request_id: str, approver_id: str) -> DeletionRequest:
        """Approve a deletion request.

        Args:
            request_id: The request ID.
            approver_id: ID of the approving user.

        Returns:
            The updated DeletionRequest.

        Raises:
            ValueError: If the request is not found or not verified.
        """
        request = self._get_request(request_id)

        if request.status != DeletionStatus.VERIFYING:
            raise ValueError(f"Request {request_id} must be verified before approval")

        request.status = DeletionStatus.IN_PROGRESS
        request.approved_by = approver_id
        self._log_event(request_id, "approved", approver_id)

        logger.info("deletion_request_approved", request_id=request_id, approver=approver_id)
        return request

    def process_request(self, request_id: str) -> DeletionRequest:
        """Process a deletion request.

        Args:
            request_id: The request ID.

        Returns:
            The updated DeletionRequest.

        Raises:
            ValueError: If the request is not found or not approved.
        """
        request = self._get_request(request_id)

        if request.status != DeletionStatus.IN_PROGRESS:
            raise ValueError(f"Request {request_id} must be approved before processing")

        handlers = self._deletion_handlers.get(request.scope, [])
        resource_ids = request.resource_ids or []

        for resource_id in resource_ids:
            deleted = False
            for handler in handlers:
                try:
                    if handler(resource_id, request):
                        deleted = True
                        break
                except Exception as e:
                    logger.error(
                        "deletion_handler_failed",
                        request_id=request_id,
                        resource_id=resource_id,
                        error=str(e),
                    )

            if deleted:
                request.completed_resources.append(resource_id)
            else:
                request.failed_resources.append(resource_id)

        # Determine final status
        if not request.failed_resources:
            request.status = DeletionStatus.COMPLETED
        elif request.completed_resources:
            request.status = DeletionStatus.PARTIAL
        else:
            request.status = DeletionStatus.FAILED

        request.completed_at = time.time()
        self._log_event(request_id, "processed", request.user_id)

        logger.info(
            "deletion_request_processed",
            request_id=request_id,
            completed=len(request.completed_resources),
            failed=len(request.failed_resources),
        )

        return request

    def cancel_request(self, request_id: str, user_id: str) -> DeletionRequest:
        """Cancel a deletion request.

        Args:
            request_id: The request ID.
            user_id: ID of the user cancelling.

        Returns:
            The updated DeletionRequest.

        Raises:
            ValueError: If the request is not found or already completed.
        """
        request = self._get_request(request_id)

        if request.status in (DeletionStatus.COMPLETED, DeletionStatus.PARTIAL):
            raise ValueError(f"Cannot cancel completed request {request_id}")

        request.status = DeletionStatus.CANCELLED
        self._log_event(request_id, "cancelled", user_id)

        logger.info("deletion_request_cancelled", request_id=request_id, user_id=user_id)
        return request

    def get_request(self, request_id: str) -> Optional[DeletionRequest]:
        """Get a deletion request by ID.

        Args:
            request_id: The request ID.

        Returns:
            The DeletionRequest, or None if not found.
        """
        return self._requests.get(request_id)

    def get_requests_by_user(self, user_id: str) -> List[DeletionRequest]:
        """Get all deletion requests for a user.

        Args:
            user_id: The user ID.

        Returns:
            List of DeletionRequest.
        """
        return [r for r in self._requests.values() if r.user_id == user_id]

    def get_requests_by_status(self, status: DeletionStatus) -> List[DeletionRequest]:
        """Get all deletion requests with a specific status.

        Args:
            status: The status to filter by.

        Returns:
            List of DeletionRequest.
        """
        return [r for r in self._requests.values() if r.status == status]

    def get_pending_requests(self) -> List[DeletionRequest]:
        """Get all pending deletion requests.

        Returns:
            List of pending DeletionRequest.
        """
        return self.get_requests_by_status(DeletionStatus.PENDING)

    def get_audit_trail(self, request_id: str) -> List[Dict[str, Any]]:
        """Get the audit trail for a deletion request.

        Args:
            request_id: The request ID.

        Returns:
            List of audit log entries.
        """
        return [entry for entry in self._audit_log if entry["request_id"] == request_id]

    def register_deletion_handler(
        self,
        scope: DeletionScope,
        handler: Callable[[str, DeletionRequest], bool],
    ) -> None:
        """Register a deletion handler for a scope.

        Args:
            scope: The deletion scope.
            handler: Function that takes (resource_id, request) and returns True on success.
        """
        if scope not in self._deletion_handlers:
            self._deletion_handlers[scope] = []
        self._deletion_handlers[scope].append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get deletion request statistics.

        Returns:
            Dictionary with statistics.
        """
        total = len(self._requests)
        by_status: Dict[str, int] = {}
        by_scope: Dict[str, int] = {}

        for request in self._requests.values():
            by_status[request.status.value] = by_status.get(request.status.value, 0) + 1
            by_scope[request.scope.value] = by_scope.get(request.scope.value, 0) + 1

        return {
            "total_requests": total,
            "by_status": by_status,
            "by_scope": by_scope,
        }

    def _get_request(self, request_id: str) -> DeletionRequest:
        """Get a request, raising if not found."""
        if request_id not in self._requests:
            raise ValueError(f"Deletion request not found: {request_id}")
        return self._requests[request_id]

    def _generate_verification_code(self) -> str:
        """Generate a verification code."""
        import secrets

        return secrets.token_hex(3)  # 6 hex chars

    def _log_event(self, request_id: str, action: str, user_id: str) -> None:
        """Log an audit event."""
        self._audit_log.append(
            {
                "request_id": request_id,
                "action": action,
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )


__all__ = [
    "DeletionRequestFlow",
    "DeletionRequest",
    "DeletionStatus",
    "DeletionScope",
]
