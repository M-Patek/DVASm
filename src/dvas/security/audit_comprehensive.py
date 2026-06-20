"""Comprehensive audit logging for DVAS security operations.

Provides structured audit logging for all security-relevant operations
with persistence, querying, and compliance reporting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class AuditEventType(str, Enum):
    """Types of audit events."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    AUTHENTICATION = "authentication"
    PASSWORD_CHANGE = "password_change"
    MFA = "mfa"

    # Authorization
    AUTHORIZATION = "authorization"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # Data operations
    EXPORT = "export"
    IMPORT_ = "import"
    DOWNLOAD = "download"
    UPLOAD = "upload"

    # Security
    ENCRYPTION = "encryption"
    DECRYPTION = "decryption"
    KEY_ROTATION = "key_rotation"
    SECRET_ACCESS = "secret_access"
    PII_DETECTION = "pii_detection"
    PII_REDACTION = "pii_redaction"

    # Compliance
    RETENTION_POLICY = "retention_policy"
    DELETION_REQUEST = "deletion_request"
    DATA_ANONYMIZATION = "data_anonymization"
    COMPLIANCE_CHECK = "compliance_check"

    # System
    SYSTEM = "system"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"
    API_CALL = "api_call"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def _order(self) -> int:
        """Return numeric order for severity comparison."""
        return {
            AuditSeverity.DEBUG: 0,
            AuditSeverity.INFO: 1,
            AuditSeverity.WARNING: 2,
            AuditSeverity.ERROR: 3,
            AuditSeverity.CRITICAL: 4,
        }[self]

    def __ge__(self, other: "AuditSeverity") -> bool:
        return self._order >= other._order

    def __gt__(self, other: "AuditSeverity") -> bool:
        return self._order > other._order

    def __le__(self, other: "AuditSeverity") -> bool:
        return self._order <= other._order

    def __lt__(self, other: "AuditSeverity") -> bool:
        return self._order < other._order


@dataclass
class AuditEvent:
    """A security audit event record."""

    event_type: AuditEventType
    user_id: str
    resource_type: str
    resource_id: str
    action: str
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.INFO
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create an AuditEvent from a dictionary."""
        return cls(
            event_type=AuditEventType(data["event_type"]),
            user_id=data["user_id"],
            resource_type=data["resource_type"],
            resource_id=data["resource_id"],
            action=data["action"],
            timestamp=data["timestamp"],
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            details=data.get("details", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            severity=AuditSeverity(data.get("severity", "info")),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            tenant_id=data.get("tenant_id"),
        )


class AuditLog:
    """Comprehensive audit logging for security-relevant operations.

    Usage::

        audit = AuditLog(log_dir=Path("/var/log/dvas/audit"))
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        ))

        # Query events
        events = audit.get_events(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            start_time=time.time() - 86400,
        )
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_memory_events: int = 10000,
        flush_interval: int = 100,
    ) -> None:
        """Initialize the audit log.

        Args:
            log_dir: Directory for log files. If None, only in-memory.
            max_memory_events: Maximum events to keep in memory.
            flush_interval: Events between disk flushes.
        """
        self.log_dir = log_dir
        self._events: List[AuditEvent] = []
        self._max_memory_events = max_memory_events
        self._flush_interval = flush_interval
        self._events_since_flush = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: The audit event to log.
        """
        self._events.append(event)
        self._events_since_flush += 1

        # Keep only last N events in memory
        if len(self._events) > self._max_memory_events:
            self._events = self._events[-self._max_memory_events:]

        # Write to file if configured
        if self.log_dir and self._events_since_flush >= self._flush_interval:
            self._flush_to_disk()

        # Also log to structured logger
        logger.info(
            "audit_event",
            event_type=event.event_type.value,
            user_id=event.user_id,
            resource=event.resource_type,
            action=event.action,
            success=event.success,
            severity=event.severity.value,
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        success: bool = True,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Convenience method to log a security event.

        Args:
            event_type: The type of event.
            user_id: The user ID.
            resource_type: The resource type.
            resource_id: The resource ID.
            action: Description of the action.
            success: Whether the action succeeded.
            severity: Event severity.
            details: Additional details dictionary.
            **kwargs: Additional keyword arguments merged into details.
        """
        event_details = details or {}
        event_details.update(kwargs)
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            success=success,
            severity=severity,
            details=event_details,
        )
        self.log_event(event)

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        severity: Optional[AuditSeverity] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events matching filters.

        Args:
            event_type: Filter by event type.
            user_id: Filter by user ID.
            resource_type: Filter by resource type.
            resource_id: Filter by resource ID.
            start_time: Filter events after this timestamp.
            end_time: Filter events before this timestamp.
            severity: Filter by severity.
            success: Filter by success status.
            limit: Maximum number of events to return.

        Returns:
            List of matching audit events.
        """
        results = []

        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if resource_id and event.resource_id != resource_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if severity and event.severity != severity:
                continue
            if success is not None and event.success != success:
                continue

            results.append(event)

            if len(results) >= limit:
                break

        return results

    def get_user_activity(
        self,
        user_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get activity for a specific user.

        Args:
            user_id: The user ID.
            start_time: Filter events after this timestamp.
            end_time: Filter events before this timestamp.
            limit: Maximum number of events.

        Returns:
            List of user's audit events.
        """
        return self.get_events(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def get_resource_history(
        self,
        resource_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit history for a specific resource.

        Args:
            resource_id: The resource ID.
            limit: Maximum number of events.

        Returns:
            List of audit events for the resource.
        """
        return self.get_events(resource_id=resource_id, limit=limit)

    def get_failed_events(
        self,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get failed events.

        Args:
            severity: Optional minimum severity filter.
            limit: Maximum number of events.

        Returns:
            List of failed audit events.
        """
        events = self.get_events(success=False, limit=limit)
        if severity:
            events = [e for e in events if e.severity >= severity]
        return events

    def get_security_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get a security summary report.

        Args:
            start_time: Start of the reporting period.
            end_time: End of the reporting period.

        Returns:
            Dictionary with security statistics.
        """
        events = self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        total = len(events)
        failed = sum(1 for e in events if not e.success)
        by_type: Dict[str, int] = {}
        by_user: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_user[event.user_id] = by_user.get(event.user_id, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

        return {
            "total_events": total,
            "failed_events": failed,
            "success_rate": (total - failed) / total if total > 0 else 1.0,
            "events_by_type": by_type,
            "events_by_user": by_user,
            "events_by_severity": by_severity,
            "period_start": start_time,
            "period_end": end_time,
        }

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        if not self.log_dir:
            return

        date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except OSError as e:
            logger.error("failed_to_write_audit_log", file=str(log_file), error=str(e))

    def _flush_to_disk(self) -> None:
        """Flush in-memory events to disk."""
        if not self.log_dir or not self._events_since_flush:
            return

        # Group events by date
        events_by_date: Dict[str, List[AuditEvent]] = {}
        for event in self._events[-self._events_since_flush:]:
            date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event)

        for date_str, events in events_by_date.items():
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            try:
                with open(log_file, "a") as f:
                    for event in events:
                        f.write(json.dumps(event.to_dict(), default=str) + "\n")
            except OSError as e:
                logger.error("failed_to_flush_audit_log", file=str(log_file), error=str(e))

        self._events_since_flush = 0

    def flush(self) -> None:
        """Manually flush events to disk."""
        self._flush_to_disk()

    def close(self) -> None:
        """Close the audit log and flush remaining events."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AuditDecorator:
    """Decorator for automatically logging function calls."""

    def __init__(self, audit_log: AuditLog) -> None:
        """Initialize the decorator.

        Args:
            audit_log: The audit log instance.
        """
        self.audit_log = audit_log

    def log_access(
        self,
        event_type: AuditEventType,
        resource_type: str,
        action: str,
    ) -> Callable:
        """Decorator to log function access.

        Args:
            event_type: The event type.
            resource_type: The resource type.
            action: The action description.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Try to extract user_id from args or kwargs
                user_id = kwargs.get("user_id", "unknown")
                resource_id = kwargs.get("resource_id", "unknown")

                try:
                    result = func(*args, **kwargs)
                    self.audit_log.log_security_event(
                        event_type=event_type,
                        user_id=user_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=action,
                        success=True,
                    )
                    return result
                except Exception as e:
                    self.audit_log.log_security_event(
                        event_type=event_type,
                        user_id=user_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=action,
                        success=False,
                        error_message=str(e),
                    )
                    raise

            return wrapper
        return decorator


__all__ = [
    "AuditLog",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditDecorator",
]
