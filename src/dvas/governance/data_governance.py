"""Data governance and lineage for DVAS.

Provides DataGovernance for data management policies including retention,
access control, quality rules, lineage tracking, and GDPR/CCPA compliance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DataAccessLevel(str, Enum):
    """Data access levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class RetentionType(str, Enum):
    """Type of data retention policy."""

    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    INDEFINITE = "indefinite"


@dataclass
class DataAccessPolicy:
    """Policy defining data access permissions."""

    user_id: str
    data_id: str
    level: DataAccessLevel
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    granted_by: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "data_id": self.data_id,
            "level": self.level.value,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "granted_by": self.granted_by,
            "conditions": self.conditions,
        }

    def is_expired(self) -> bool:
        """Check if the access has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class RetentionPolicy:
    """Data retention policy definition."""

    data_type: str
    retention_type: RetentionType
    duration_days: Optional[int] = None
    trigger_event: Optional[str] = None
    description: str = ""
    action: str = "archive"  # archive, delete, anonymize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type,
            "retention_type": self.retention_type.value,
            "duration_days": self.duration_days,
            "trigger_event": self.trigger_event,
            "description": self.description,
            "action": self.action,
        }


@dataclass
class LineageRecord:
    """Record of data lineage."""

    record_id: str
    source_id: Optional[str] = None
    operation: str = ""
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


class DataGovernance:
    """Data governance and lineage management.

    Usage::

        governance = DataGovernance()
        governance.add_retention_policy(RetentionPolicy(
            data_type="annotation",
            retention_type=RetentionType.TIME_BASED,
            duration_days=365,
        ))
        governance.grant_access("user_001", "data_001", DataAccessLevel.RESTRICTED)
        governance.track_lineage("data_002", "data_001", "derived")
    """

    def __init__(self) -> None:
        """Initialize data governance."""
        self._access_policies: Dict[str, List[DataAccessPolicy]] = {}
        self._retention_policies: Dict[str, RetentionPolicy] = {}
        self._lineage: Dict[str, List[LineageRecord]] = {}
        self._data_quality_rules: Dict[str, List[Dict[str, Any]]] = {}
        self._gdpr_consent: Dict[str, Dict[str, Any]] = {}
        self._audit_log: List[Dict[str, Any]] = []

    # -- Access Control ----------------------------------------------------

    def grant_access(
        self,
        user_id: str,
        data_id: str,
        level: DataAccessLevel,
        granted_by: str = "",
        expires_at: Optional[float] = None,
    ) -> DataAccessPolicy:
        """Grant access to data.

        Args:
            user_id: The user to grant access.
            data_id: The data to access.
            level: Access level.
            granted_by: Who granted the access.
            expires_at: Optional expiration timestamp.

        Returns:
            The created access policy.
        """
        policy = DataAccessPolicy(
            user_id=user_id,
            data_id=data_id,
            level=level,
            granted_by=granted_by,
            expires_at=expires_at,
        )
        key = f"{user_id}:{data_id}"
        if key not in self._access_policies:
            self._access_policies[key] = []
        self._access_policies[key].append(policy)

        self._log_event("access_granted", user_id=user_id, data_id=data_id, level=level.value)
        return policy

    def revoke_access(self, user_id: str, data_id: str) -> bool:
        """Revoke access to data.

        Args:
            user_id: The user to revoke access for.
            data_id: The data to revoke access to.

        Returns:
            True if access was revoked.
        """
        key = f"{user_id}:{data_id}"
        if key in self._access_policies:
            del self._access_policies[key]
            self._log_event("access_revoked", user_id=user_id, data_id=data_id)
            return True
        return False

    def check_access(self, user_id: str, data_id: str, min_level: DataAccessLevel) -> bool:
        """Check if a user has access to data.

        Args:
            user_id: The user to check.
            data_id: The data to check.
            min_level: Minimum access level required.

        Returns:
            True if access is granted.
        """
        key = f"{user_id}:{data_id}"
        policies = self._access_policies.get(key, [])

        level_order = {
            DataAccessLevel.PUBLIC: 0,
            DataAccessLevel.INTERNAL: 1,
            DataAccessLevel.RESTRICTED: 2,
            DataAccessLevel.CONFIDENTIAL: 3,
        }
        min_value = level_order.get(min_level, 0)

        for policy in policies:
            if policy.is_expired():
                continue
            if level_order.get(policy.level, 0) >= min_value:
                return True
        return False

    def get_access_policies(self, data_id: str) -> List[DataAccessPolicy]:
        """Get all access policies for a data item."""
        policies = []
        for key, pols in self._access_policies.items():
            if key.endswith(f":{data_id}"):
                policies.extend(pols)
        return policies

    # -- Retention Policies ------------------------------------------------

    def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy.

        Args:
            policy: The retention policy to add.
        """
        self._retention_policies[policy.data_type] = policy
        self._log_event("retention_policy_added", data_type=policy.data_type)

    def get_retention_policy(self, data_type: str) -> Optional[RetentionPolicy]:
        """Get retention policy for a data type."""
        return self._retention_policies.get(data_type)

    def remove_retention_policy(self, data_type: str) -> bool:
        """Remove a retention policy.

        Args:
            data_type: The data type.

        Returns:
            True if a policy was removed.
        """
        if data_type in self._retention_policies:
            del self._retention_policies[data_type]
            return True
        return False

    def is_expired(self, data_type: str, created_at: float, event_triggered: Optional[str] = None) -> bool:
        """Check if data has expired under retention policy.

        Args:
            data_type: The data type.
            created_at: When the data was created.
            event_triggered: Optional event that triggered retention.

        Returns:
            True if the data has expired.
        """
        policy = self._retention_policies.get(data_type)
        if not policy:
            return False

        if policy.retention_type == RetentionType.INDEFINITE:
            return False
        elif policy.retention_type == RetentionType.TIME_BASED and policy.duration_days:
            return time.time() - created_at > policy.duration_days * 86400
        elif policy.retention_type == RetentionType.EVENT_BASED:
            return event_triggered == policy.trigger_event

        return False

    # -- Data Quality Rules ------------------------------------------------

    def add_quality_rule(self, data_type: str, rule: Dict[str, Any]) -> None:
        """Add a data quality rule.

        Args:
            data_type: The data type.
            rule: Rule definition.
        """
        if data_type not in self._data_quality_rules:
            self._data_quality_rules[data_type] = []
        self._data_quality_rules[data_type].append(rule)

    def validate_quality(self, data_type: str, data: Dict[str, Any]) -> List[str]:
        """Validate data against quality rules.

        Args:
            data_type: The data type.
            data: The data to validate.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        rules = self._data_quality_rules.get(data_type, [])

        for rule in rules:
            field = rule.get("field")
            condition = rule.get("condition")
            value = rule.get("value")

            if field and field not in data:
                errors.append(f"Missing required field: {field}")
                continue

            if field and condition and value is not None:
                actual = data.get(field)
                if condition == "gt" and actual is not None and actual <= value:  # type: ignore[operator]
                    errors.append(f"Field {field} must be > {value}")
                elif condition == "gte" and actual is not None and actual < value:  # type: ignore[operator]
                    errors.append(f"Field {field} must be >= {value}")
                elif condition == "exists" and actual is None:
                    errors.append(f"Field {field} must exist")

        return errors

    # -- Lineage Tracking --------------------------------------------------

    def track_lineage(
        self,
        record_id: str,
        source_id: Optional[str] = None,
        operation: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Track data lineage.

        Args:
            record_id: The record ID.
            source_id: The source record ID.
            operation: The operation performed.
            user_id: The user who performed the operation.
            metadata: Additional metadata.

        Returns:
            The created lineage record.
        """
        record = LineageRecord(
            record_id=record_id,
            source_id=source_id,
            operation=operation,
            user_id=user_id,
            metadata=metadata or {},
        )
        if record_id not in self._lineage:
            self._lineage[record_id] = []
        self._lineage[record_id].append(record)

        self._log_event("lineage_tracked", record_id=record_id, source_id=source_id, operation=operation)
        return record

    def get_lineage(self, record_id: str) -> List[LineageRecord]:
        """Get lineage records for a record."""
        return self._lineage.get(record_id, [])

    def get_ancestors(self, record_id: str) -> List[LineageRecord]:
        """Get all ancestor records in lineage."""
        ancestors: List[LineageRecord] = []
        visited: Set[str] = set()
        queue = [record_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for record in self._lineage.get(current, []):
                if record.source_id and record.source_id not in visited:
                    ancestors.append(record)
                    queue.append(record.source_id)

        return ancestors

    # -- GDPR / CCPA Compliance --------------------------------------------

    def record_consent(self, user_id: str, purpose: str, consented: bool = True) -> None:
        """Record user consent for data processing.

        Args:
            user_id: The user ID.
            purpose: The purpose of processing.
            consented: Whether consent was given.
        """
        if user_id not in self._gdpr_consent:
            self._gdpr_consent[user_id] = {}
        self._gdpr_consent[user_id][purpose] = {
            "consented": consented,
            "timestamp": time.time(),
        }
        self._log_event("consent_recorded", user_id=user_id, purpose=purpose, consented=consented)

    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if a user has given consent for a purpose.

        Args:
            user_id: The user ID.
            purpose: The purpose of processing.

        Returns:
            True if consent was given.
        """
        user_consent = self._gdpr_consent.get(user_id, {})
        return user_consent.get(purpose, {}).get("consented", False)

    def request_data_deletion(self, user_id: str, data_ids: List[str]) -> Dict[str, Any]:
        """Request deletion of user data (GDPR right to erasure).

        Args:
            user_id: The user ID.
            data_ids: List of data IDs to delete.

        Returns:
            Deletion request record.
        """
        record = {
            "user_id": user_id,
            "data_ids": data_ids,
            "requested_at": time.time(),
            "status": "requested",
        }
        self._log_event("data_deletion_requested", user_id=user_id, data_ids=data_ids)
        return record

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR right to data portability).

        Args:
            user_id: The user ID.

        Returns:
            User data export.
        """
        # Collect access policies for this user
        user_policies = {}
        for key, policies in self._access_policies.items():
            if key.startswith(f"{user_id}:"):
                user_policies[key] = [p.to_dict() for p in policies]

        return {
            "user_id": user_id,
            "access_policies": user_policies,
            "consent_records": self._gdpr_consent.get(user_id, {}),
            "exported_at": time.time(),
        }

    # -- Audit Logging -----------------------------------------------------

    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        """Log a governance event."""
        self._audit_log.append({
            "event_type": event_type,
            "timestamp": time.time(),
            **kwargs,
        })

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log.

        Args:
            limit: Maximum number of entries.

        Returns:
            List of audit log entries.
        """
        return self._audit_log[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "retention_policies": len(self._retention_policies),
            "access_policies": len(self._access_policies),
            "lineage_records": sum(len(v) for v in self._lineage.values()),
            "quality_rules": sum(len(v) for v in self._data_quality_rules.values()),
            "consent_records": len(self._gdpr_consent),
            "audit_log_entries": len(self._audit_log),
        }


__all__ = [
    "DataGovernance",
    "DataAccessPolicy",
    "DataAccessLevel",
    "RetentionPolicy",
    "RetentionType",
    "LineageRecord",
]
