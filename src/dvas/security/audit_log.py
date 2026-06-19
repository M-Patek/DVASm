"""Audit logging for DVAS security events.

Provides structured audit event logging with filtering and querying capabilities.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    IMPORT_ = "import"
    API_CALL = "api_call"
    SYSTEM = "system"


@dataclass
class AuditEvent:
    """An audit event record."""

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
        }


class AuditLogger:
    """Audit logger for security events.

    Usage::

        audit = AuditLogger()
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        ))
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = log_dir
        self._events: List[AuditEvent] = []
        self._max_memory_events = 10000

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self._events.append(event)

        # Keep only last N events in memory
        if len(self._events) > self._max_memory_events:
            self._events = self._events[-self._max_memory_events :]

        # Write to file if configured
        if self.log_dir:
            self._write_event(event)

        # Also log to structured logger
        logger.info(
            "audit_event",
            event_type=event.event_type.value,
            user_id=event.user_id,
            resource=event.resource_type,
            action=event.action,
            success=event.success,
        )

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        if not self.log_dir:
            return

        # Use daily log files
        from datetime import datetime

        date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events matching filters."""
        results = []

        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)

            if len(results) >= limit:
                break

        return results

    def get_user_activity(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get activity for a specific user."""
        return self.get_events(user_id=user_id, limit=limit)


__all__ = ["AuditEventType", "AuditEvent", "AuditLogger"]
