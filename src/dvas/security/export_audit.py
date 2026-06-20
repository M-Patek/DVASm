"""Export approval audit for DVAS.

Provides audit trail for data exports with approval workflow,
tracked approvals, and compliance reporting.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class ExportStatus(str, Enum):
    """Status of an export request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportApproval:
    """An export approval record."""

    export_id: str
    requester_id: str
    resource_type: str
    resource_ids: List[str]
    status: ExportStatus
    requested_at: float
    approved_at: Optional[float] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    approver_notes: str = ""
    data_classification: str = "internal"
    purpose: str = ""
    recipient_email: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "export_id": self.export_id,
            "requester_id": self.requester_id,
            "resource_type": self.resource_type,
            "resource_ids": self.resource_ids,
            "status": self.status.value,
            "requested_at": self.requested_at,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "approver_notes": self.approver_notes,
            "data_classification": self.data_classification,
            "purpose": self.purpose,
            "recipient_email": self.recipient_email,
        }


class ExportApprovalAudit:
    """Audit trail for data exports with approval workflow.

    Usage::

        audit = ExportApprovalAudit()

        # Request export
        export = audit.request_export(
            requester_id="user_001",
            resource_type="annotations",
            resource_ids=["ann_001", "ann_002"],
            purpose="Monthly compliance report",
        )

        # Approve export
        audit.approve_export(export.export_id, approver_id="admin_001")

        # Complete export
        audit.complete_export(export.export_id)
    """

    def __init__(self) -> None:
        """Initialize the export approval audit."""
        self._exports: Dict[str, ExportApproval] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def request_export(
        self,
        requester_id: str,
        resource_type: str,
        resource_ids: List[str],
        purpose: str,
        data_classification: str = "internal",
        recipient_email: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportApproval:
        """Request a new export.

        Args:
            requester_id: ID of the requesting user.
            resource_type: Type of resources being exported.
            resource_ids: List of resource IDs to export.
            purpose: Purpose of the export.
            data_classification: Classification level of the data.
            recipient_email: Email of the recipient.
            metadata: Additional metadata.

        Returns:
            The created ExportApproval.
        """
        export_id = str(uuid.uuid4())
        export = ExportApproval(
            export_id=export_id,
            requester_id=requester_id,
            resource_type=resource_type,
            resource_ids=resource_ids,
            status=ExportStatus.PENDING,
            requested_at=time.time(),
            metadata=metadata or {},
            data_classification=data_classification,
            purpose=purpose,
            recipient_email=recipient_email,
        )

        self._exports[export_id] = export
        self._log_event(export_id, "requested", requester_id)

        logger.info(
            "export_requested",
            export_id=export_id,
            requester=requester_id,
            resource_count=len(resource_ids),
        )

        return export

    def approve_export(
        self,
        export_id: str,
        approver_id: str,
        notes: str = "",
    ) -> ExportApproval:
        """Approve an export request.

        Args:
            export_id: The export ID.
            approver_id: ID of the approving user.
            notes: Optional approval notes.

        Returns:
            The updated ExportApproval.

        Raises:
            ValueError: If the export is not found or not in pending status.
        """
        export = self._get_export(export_id)

        if export.status != ExportStatus.PENDING:
            raise ValueError(f"Export {export_id} is not pending (status: {export.status.value})")

        export.status = ExportStatus.APPROVED
        export.approved_at = time.time()
        export.approved_by = approver_id
        export.approver_notes = notes

        self._log_event(export_id, "approved", approver_id)
        logger.info("export_approved", export_id=export_id, approver=approver_id)

        return export

    def reject_export(
        self,
        export_id: str,
        approver_id: str,
        reason: str,
    ) -> ExportApproval:
        """Reject an export request.

        Args:
            export_id: The export ID.
            approver_id: ID of the rejecting user.
            reason: Reason for rejection.

        Returns:
            The updated ExportApproval.

        Raises:
            ValueError: If the export is not found or not in pending status.
        """
        export = self._get_export(export_id)

        if export.status != ExportStatus.PENDING:
            raise ValueError(f"Export {export_id} is not pending (status: {export.status.value})")

        export.status = ExportStatus.REJECTED
        export.approved_by = approver_id
        export.rejection_reason = reason

        self._log_event(export_id, "rejected", approver_id, reason=reason)
        logger.info("export_rejected", export_id=export_id, approver=approver_id, reason=reason)

        return export

    def start_export(self, export_id: str) -> ExportApproval:
        """Mark an export as in progress.

        Args:
            export_id: The export ID.

        Returns:
            The updated ExportApproval.

        Raises:
            ValueError: If the export is not found or not approved.
        """
        export = self._get_export(export_id)

        if export.status != ExportStatus.APPROVED:
            raise ValueError(
                f"Export {export_id} must be approved before starting (status: {export.status.value})"
            )

        export.status = ExportStatus.IN_PROGRESS
        self._log_event(export_id, "started", export.requester_id)
        logger.info("export_started", export_id=export_id)

        return export

    def complete_export(self, export_id: str) -> ExportApproval:
        """Mark an export as completed.

        Args:
            export_id: The export ID.

        Returns:
            The updated ExportApproval.

        Raises:
            ValueError: If the export is not found or not in progress.
        """
        export = self._get_export(export_id)

        if export.status != ExportStatus.IN_PROGRESS:
            raise ValueError(
                f"Export {export_id} is not in progress (status: {export.status.value})"
            )

        export.status = ExportStatus.COMPLETED
        export.completed_at = time.time()
        self._log_event(export_id, "completed", export.requester_id)
        logger.info("export_completed", export_id=export_id)

        return export

    def fail_export(self, export_id: str, error_message: str) -> ExportApproval:
        """Mark an export as failed.

        Args:
            export_id: The export ID.
            error_message: Error description.

        Returns:
            The updated ExportApproval.
        """
        export = self._get_export(export_id)
        export.status = ExportStatus.FAILED
        export.rejection_reason = error_message
        self._log_event(export_id, "failed", export.requester_id, reason=error_message)
        logger.error("export_failed", export_id=export_id, error=error_message)

        return export

    def cancel_export(self, export_id: str, user_id: str) -> ExportApproval:
        """Cancel an export request.

        Args:
            export_id: The export ID.
            user_id: ID of the user cancelling.

        Returns:
            The updated ExportApproval.

        Raises:
            ValueError: If the export is already completed.
        """
        export = self._get_export(export_id)

        if export.status == ExportStatus.COMPLETED:
            raise ValueError(f"Export {export_id} is already completed and cannot be cancelled")

        export.status = ExportStatus.CANCELLED
        self._log_event(export_id, "cancelled", user_id)
        logger.info("export_cancelled", export_id=export_id, user_id=user_id)

        return export

    def get_export(self, export_id: str) -> Optional[ExportApproval]:
        """Get an export by ID.

        Args:
            export_id: The export ID.

        Returns:
            The ExportApproval, or None if not found.
        """
        return self._exports.get(export_id)

    def get_exports_by_status(self, status: ExportStatus) -> List[ExportApproval]:
        """Get all exports with a specific status.

        Args:
            status: The status to filter by.

        Returns:
            List of matching ExportApprovals.
        """
        return [e for e in self._exports.values() if e.status == status]

    def get_exports_by_requester(self, requester_id: str) -> List[ExportApproval]:
        """Get all exports requested by a user.

        Args:
            requester_id: The requester user ID.

        Returns:
            List of matching ExportApprovals.
        """
        return [e for e in self._exports.values() if e.requester_id == requester_id]

    def get_pending_exports(self) -> List[ExportApproval]:
        """Get all pending exports.

        Returns:
            List of pending ExportApprovals.
        """
        return self.get_exports_by_status(ExportStatus.PENDING)

    def get_audit_trail(self, export_id: str) -> List[Dict[str, Any]]:
        """Get the audit trail for an export.

        Args:
            export_id: The export ID.

        Returns:
            List of audit log entries.
        """
        return [entry for entry in self._audit_log if entry["export_id"] == export_id]

    def get_all_exports(self) -> List[ExportApproval]:
        """Get all exports.

        Returns:
            List of all ExportApprovals.
        """
        return list(self._exports.values())

    def get_export_stats(self) -> Dict[str, Any]:
        """Get statistics about exports.

        Returns:
            Dictionary with export statistics.
        """
        total = len(self._exports)
        by_status: Dict[str, int] = {}

        for export in self._exports.values():
            status = export.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_exports": total,
            "by_status": by_status,
            "pending": by_status.get(ExportStatus.PENDING.value, 0),
            "approved": by_status.get(ExportStatus.APPROVED.value, 0),
            "rejected": by_status.get(ExportStatus.REJECTED.value, 0),
            "completed": by_status.get(ExportStatus.COMPLETED.value, 0),
            "failed": by_status.get(ExportStatus.FAILED.value, 0),
        }

    def _get_export(self, export_id: str) -> ExportApproval:
        """Get an export, raising if not found."""
        if export_id not in self._exports:
            raise ValueError(f"Export not found: {export_id}")
        return self._exports[export_id]

    def _log_event(
        self,
        export_id: str,
        action: str,
        user_id: str,
        reason: str = "",
    ) -> None:
        """Log an audit event."""
        self._audit_log.append({
            "export_id": export_id,
            "action": action,
            "user_id": user_id,
            "timestamp": time.time(),
            "reason": reason,
        })


__all__ = [
    "ExportApprovalAudit",
    "ExportApproval",
    "ExportStatus",
]
