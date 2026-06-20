"""Data retention policy for DVAS.

Provides configurable retention rules for managing data lifecycle,
including automatic expiration, archiving, and deletion.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class RetentionAction(str, Enum):
    """Actions to take when data reaches retention age."""

    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"
    MARK_FOR_REVIEW = "mark_for_review"


class DataType(str, Enum):
    """Types of data subject to retention policies."""

    ANNOTATION = "annotation"
    VIDEO = "video"
    EXPORT = "export"
    AUDIT_LOG = "audit_log"
    USER_DATA = "user_data"
    TEMPORARY = "temporary"
    SESSION = "session"
    BACKUP = "backup"


@dataclass
class RetentionRule:
    """A retention rule for a specific data type."""

    data_type: DataType
    retention_days: int
    action: RetentionAction
    description: str = ""
    grace_days: int = 7
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_type": self.data_type.value,
            "retention_days": self.retention_days,
            "action": self.action.value,
            "description": self.description,
            "grace_days": self.grace_days,
            "enabled": self.enabled,
            "conditions": self.conditions,
        }


@dataclass
class RetentionRecord:
    """A record tracked for retention."""

    record_id: str
    data_type: DataType
    created_at: float
    last_accessed: float
    owner_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retention_override_days: Optional[int] = None
    exempt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "data_type": self.data_type.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "owner_id": self.owner_id,
            "metadata": self.metadata,
            "retention_override_days": self.retention_override_days,
            "exempt": self.exempt,
        }


class DataRetentionPolicy:
    """Configurable data retention policy for DVAS.

    Usage::

        policy = DataRetentionPolicy()
        policy.add_rule(RetentionRule(
            data_type=DataType.ANNOTATION,
            retention_days=365,
            action=RetentionAction.ARCHIVE,
        ))

        # Track a record
        policy.track_record("ann_001", DataType.ANNOTATION, "user_001")

        # Check expired records
        expired = policy.get_expired_records()
        for record in expired:
            policy.apply_retention(record.record_id)
    """

    # Default retention rules
    DEFAULT_RULES = [
        RetentionRule(
            data_type=DataType.ANNOTATION,
            retention_days=365,
            action=RetentionAction.ARCHIVE,
            description="Annotations retained for 1 year",
        ),
        RetentionRule(
            data_type=DataType.VIDEO,
            retention_days=180,
            action=RetentionAction.ARCHIVE,
            description="Videos retained for 6 months",
        ),
        RetentionRule(
            data_type=DataType.EXPORT,
            retention_days=30,
            action=RetentionAction.DELETE,
            description="Exports deleted after 30 days",
        ),
        RetentionRule(
            data_type=DataType.AUDIT_LOG,
            retention_days=2555,  # ~7 years
            action=RetentionAction.ARCHIVE,
            description="Audit logs retained for 7 years",
        ),
        RetentionRule(
            data_type=DataType.TEMPORARY,
            retention_days=7,
            action=RetentionAction.DELETE,
            description="Temporary data deleted after 7 days",
        ),
        RetentionRule(
            data_type=DataType.SESSION,
            retention_days=1,
            action=RetentionAction.DELETE,
            description="Sessions deleted after 1 day",
        ),
        RetentionRule(
            data_type=DataType.BACKUP,
            retention_days=90,
            action=RetentionAction.DELETE,
            description="Backups deleted after 90 days",
        ),
    ]

    def __init__(self, rules: Optional[List[RetentionRule]] = None) -> None:
        """Initialize the retention policy.

        Args:
            rules: Optional list of retention rules. Uses defaults if not provided.
        """
        self._rules: Dict[DataType, RetentionRule] = {}
        self._records: Dict[str, RetentionRecord] = {}
        self._handlers: Dict[RetentionAction, List[Callable]] = {
            RetentionAction.DELETE: [],
            RetentionAction.ARCHIVE: [],
            RetentionAction.ANONYMIZE: [],
            RetentionAction.NOTIFY: [],
            RetentionAction.MARK_FOR_REVIEW: [],
        }

        # Add default or provided rules
        if rules is None:
            rules = [copy.deepcopy(r) for r in self.DEFAULT_RULES]
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule: RetentionRule) -> None:
        """Add a retention rule.

        Args:
            rule: The retention rule to add.
        """
        self._rules[rule.data_type] = rule
        logger.info("retention_rule_added", data_type=rule.data_type.value, days=rule.retention_days)

    def remove_rule(self, data_type: DataType) -> bool:
        """Remove a retention rule.

        Args:
            data_type: The data type to remove the rule for.

        Returns:
            True if removed, False if not found.
        """
        if data_type in self._rules:
            del self._rules[data_type]
            return True
        return False

    def get_rule(self, data_type: DataType) -> Optional[RetentionRule]:
        """Get the retention rule for a data type.

        Args:
            data_type: The data type.

        Returns:
            The RetentionRule, or None if not found.
        """
        return self._rules.get(data_type)

    def track_record(
        self,
        record_id: str,
        data_type: DataType,
        owner_id: str,
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetentionRecord:
        """Track a record for retention.

        Args:
            record_id: The record ID.
            data_type: The type of data.
            owner_id: The owner user ID.
            created_at: Optional creation timestamp. Uses current time if not provided.
            metadata: Optional metadata.

        Returns:
            The created RetentionRecord.
        """
        now = time.time()
        record = RetentionRecord(
            record_id=record_id,
            data_type=data_type,
            created_at=created_at or now,
            last_accessed=now,
            owner_id=owner_id,
            metadata=metadata or {},
        )
        self._records[record_id] = record
        return record

    def untrack_record(self, record_id: str) -> bool:
        """Stop tracking a record.

        Args:
            record_id: The record ID.

        Returns:
            True if removed, False if not found.
        """
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False

    def update_access_time(self, record_id: str) -> bool:
        """Update the last accessed time for a record.

        Args:
            record_id: The record ID.

        Returns:
            True if updated, False if not found.
        """
        if record_id not in self._records:
            return False

        self._records[record_id].last_accessed = time.time()
        return True

    def get_expired_records(self) -> List[RetentionRecord]:
        """Get all records that have exceeded their retention period.

        Returns:
            List of expired RetentionRecord.
        """
        now = time.time()
        expired = []

        for record in self._records.values():
            if record.exempt:
                continue

            rule = self._rules.get(record.data_type)
            if rule is None or not rule.enabled:
                continue

            retention_days = record.retention_override_days or rule.retention_days
            age_days = (now - record.created_at) / 86400

            if age_days >= retention_days:
                expired.append(record)

        return expired

    def apply_retention(self, record_id: str) -> bool:
        """Apply the retention policy to a record.

        Args:
            record_id: The record ID.

        Returns:
            True if applied, False if not found.
        """
        if record_id not in self._records:
            return False

        record = self._records[record_id]
        rule = self._rules.get(record.data_type)

        if rule is None or not rule.enabled:
            return False

        # Execute handlers
        for handler in self._handlers.get(rule.action, []):
            try:
                handler(record)
            except Exception as e:
                logger.error(
                    "retention_handler_failed",
                    record_id=record_id,
                    action=rule.action.value,
                    error=str(e),
                )

        logger.info(
            "retention_applied",
            record_id=record_id,
            action=rule.action.value,
            data_type=record.data_type.value,
        )

        return True

    def register_handler(self, action: RetentionAction, handler: Callable) -> None:
        """Register a handler for a retention action.

        Args:
            action: The action to handle.
            handler: The handler function.
        """
        if action not in self._handlers:
            self._handlers[action] = []
        self._handlers[action].append(handler)

    def set_retention_override(self, record_id: str, days: int) -> bool:
        """Set a retention override for a record.

        Args:
            record_id: The record ID.
            days: Number of days to retain.

        Returns:
            True if set, False if not found.
        """
        if record_id not in self._records:
            return False

        self._records[record_id].retention_override_days = days
        return True

    def set_exemption(self, record_id: str, exempt: bool = True) -> bool:
        """Set exemption status for a record.

        Args:
            record_id: The record ID.
            exempt: Whether to exempt from retention.

        Returns:
            True if set, False if not found.
        """
        if record_id not in self._records:
            return False

        self._records[record_id].exempt = exempt
        return True

    def get_record_age(self, record_id: str) -> Optional[float]:
        """Get the age of a record in days.

        Args:
            record_id: The record ID.

        Returns:
            Age in days, or None if not found.
        """
        if record_id not in self._records:
            return None

        record = self._records[record_id]
        return (time.time() - record.created_at) / 86400

    def get_records_by_type(self, data_type: DataType) -> List[RetentionRecord]:
        """Get all records of a specific data type.

        Args:
            data_type: The data type.

        Returns:
            List of matching RetentionRecord.
        """
        return [r for r in self._records.values() if r.data_type == data_type]

    def get_stats(self) -> Dict[str, Any]:
        """Get retention statistics.

        Returns:
            Dictionary with retention statistics.
        """
        total = len(self._records)
        by_type: Dict[str, int] = {}
        expired_count = len(self.get_expired_records())
        exempt_count = sum(1 for r in self._records.values() if r.exempt)

        for record in self._records.values():
            type_name = record.data_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_records": total,
            "expired_records": expired_count,
            "exempt_records": exempt_count,
            "records_by_type": by_type,
            "active_rules": len(self._rules),
        }

    def list_rules(self) -> List[RetentionRule]:
        """List all retention rules.

        Returns:
            List of RetentionRule.
        """
        return list(self._rules.values())


__all__ = [
    "DataRetentionPolicy",
    "RetentionRule",
    "RetentionRecord",
    "RetentionAction",
    "DataType",
]
