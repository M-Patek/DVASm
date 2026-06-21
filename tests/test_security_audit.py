"""Tests for comprehensive audit logging module.

Tests for AuditLog, AuditEvent, AuditEventType, AuditSeverity, and AuditDecorator.
"""

import time

import pytest

from dvas.security.audit_comprehensive import (
    AuditDecorator,
    AuditEvent,
    AuditEventType,
    AuditLog,
    AuditSeverity,
)


class TestAuditSeverity:
    """Test AuditSeverity enum."""

    def test_severity_values(self):
        """Test severity values."""
        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"


class TestAuditEventType:
    """Test AuditEventType enum."""

    def test_event_type_values(self):
        """Test event type values."""
        assert AuditEventType.CREATE.value == "create"
        assert AuditEventType.READ.value == "read"
        assert AuditEventType.UPDATE.value == "update"
        assert AuditEventType.DELETE.value == "delete"
        assert AuditEventType.LOGIN.value == "login"
        assert AuditEventType.EXPORT.value == "export"
        assert AuditEventType.API_CALL.value == "api_call"
        assert AuditEventType.SYSTEM.value == "system"


class TestAuditEvent:
    """Test AuditEvent dataclass."""

    def test_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        )
        assert event.event_type == AuditEventType.CREATE
        assert event.user_id == "user_001"
        assert event.resource_type == "annotation"
        assert event.success is True
        assert event.severity == AuditSeverity.INFO

    def test_event_to_dict(self):
        """Test converting event to dict."""
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
            success=True,
        )
        d = event.to_dict()
        assert d["event_type"] == "create"
        assert d["user_id"] == "user_001"
        assert d["success"] is True
        assert d["severity"] == "info"

    def test_event_from_dict(self):
        """Test creating event from dict."""
        data = {
            "event_type": "create",
            "user_id": "user_001",
            "resource_type": "annotation",
            "resource_id": "ann_001",
            "action": "created annotation",
            "timestamp": 1234567890.0,
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "details": {"key": "value"},
            "success": True,
            "error_message": None,
            "severity": "warning",
            "session_id": "sess_001",
            "request_id": "req_001",
            "tenant_id": "tenant_001",
        }
        event = AuditEvent.from_dict(data)
        assert event.event_type == AuditEventType.CREATE
        assert event.user_id == "user_001"
        assert event.severity == AuditSeverity.WARNING


class TestAuditLog:
    """Test AuditLog class."""

    def test_init(self):
        """Test initialization."""
        audit = AuditLog()
        assert audit is not None
        assert len(audit._events) == 0

    def test_init_with_log_dir(self, tmp_path):
        """Test initialization with log directory."""
        log_dir = tmp_path / "audit"
        audit = AuditLog(log_dir=log_dir)
        assert audit.log_dir == log_dir
        assert log_dir.exists()

    def test_log_event(self):
        """Test logging an event."""
        audit = AuditLog()
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        )
        audit.log_event(event)
        assert len(audit._events) == 1

    def test_log_security_event(self):
        """Test logging a security event."""
        audit = AuditLog()
        audit.log_security_event(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
            details={"key": "value"},
        )
        assert len(audit._events) == 1
        assert audit._events[0].details["key"] == "value"

    def test_get_events_by_type(self):
        """Test getting events by type."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.READ,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="read",
            )
        )

        create_events = audit.get_events(event_type=AuditEventType.CREATE)
        assert len(create_events) == 1
        assert create_events[0].event_type == AuditEventType.CREATE

    def test_get_events_by_user(self):
        """Test getting events by user."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_002",
                resource_type="annotation",
                resource_id="ann_002",
                action="created",
            )
        )

        user_events = audit.get_events(user_id="user_001")
        assert len(user_events) == 1
        assert user_events[0].user_id == "user_001"

    def test_get_events_by_resource(self):
        """Test getting events by resource."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_002",
                action="created",
            )
        )

        resource_events = audit.get_events(resource_id="ann_001")
        assert len(resource_events) == 1
        assert resource_events[0].resource_id == "ann_001"

    def test_get_events_by_time_range(self):
        """Test getting events by time range."""
        audit = AuditLog()
        now = time.time()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
                timestamp=now - 100,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_002",
                action="created",
                timestamp=now - 50,
            )
        )

        events = audit.get_events(start_time=now - 60, end_time=now)
        assert len(events) == 1
        assert events[0].resource_id == "ann_002"

    def test_get_events_by_severity(self):
        """Test getting events by severity."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
                severity=AuditSeverity.INFO,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_002",
                action="created",
                severity=AuditSeverity.WARNING,
            )
        )

        warning_events = audit.get_events(severity=AuditSeverity.WARNING)
        assert len(warning_events) == 1
        assert warning_events[0].severity == AuditSeverity.WARNING

    def test_get_events_by_success(self):
        """Test getting events by success status."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
                success=True,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_002",
                action="failed",
                success=False,
            )
        )

        failed_events = audit.get_events(success=False)
        assert len(failed_events) == 1
        assert failed_events[0].resource_id == "ann_002"

    def test_get_user_activity(self):
        """Test getting user activity."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.READ,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="read",
            )
        )

        activity = audit.get_user_activity("user_001")
        assert len(activity) == 2

    def test_get_resource_history(self):
        """Test getting resource history."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.UPDATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="updated",
            )
        )

        history = audit.get_resource_history("ann_001")
        assert len(history) == 2

    def test_get_failed_events(self):
        """Test getting failed events."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
                success=True,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_002",
                action="failed",
                success=False,
                severity=AuditSeverity.ERROR,
            )
        )

        failed = audit.get_failed_events()
        assert len(failed) == 1
        assert failed[0].resource_id == "ann_002"

    def test_get_failed_events_with_severity(self):
        """Test getting failed events with severity filter."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="failed",
                success=False,
                severity=AuditSeverity.WARNING,
            )
        )

        # Filter for ERROR and above - should return empty
        failed = audit.get_failed_events(severity=AuditSeverity.ERROR)
        assert len(failed) == 0

    def test_get_security_summary(self):
        """Test getting security summary."""
        audit = AuditLog()
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
                success=True,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.READ,
                user_id="user_002",
                resource_type="annotation",
                resource_id="ann_002",
                action="read",
                success=True,
            )
        )
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_003",
                action="failed",
                success=False,
            )
        )

        summary = audit.get_security_summary()
        assert summary["total_events"] == 3
        assert summary["failed_events"] == 1
        assert summary["success_rate"] == 2 / 3

    def test_memory_limit(self):
        """Test that memory events are limited."""
        audit = AuditLog(max_memory_events=5)
        for i in range(10):
            audit.log_event(
                AuditEvent(
                    event_type=AuditEventType.CREATE,
                    user_id=f"user_{i}",
                    resource_type="annotation",
                    resource_id=f"ann_{i}",
                    action="created",
                )
            )

        assert len(audit._events) == 5

    def test_flush(self, tmp_path):
        """Test manual flush."""
        log_dir = tmp_path / "audit"
        audit = AuditLog(log_dir=log_dir, flush_interval=100)
        audit.log_event(
            AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created",
            )
        )
        audit.flush()
        # Should not raise

    def test_context_manager(self, tmp_path):
        """Test using as context manager."""
        log_dir = tmp_path / "audit"
        with AuditLog(log_dir=log_dir) as audit:
            audit.log_event(
                AuditEvent(
                    event_type=AuditEventType.CREATE,
                    user_id="user_001",
                    resource_type="annotation",
                    resource_id="ann_001",
                    action="created",
                )
            )
        # Should flush on exit


class TestAuditDecorator:
    """Test AuditDecorator class."""

    def test_init(self):
        """Test initialization."""
        audit = AuditLog()
        decorator = AuditDecorator(audit)
        assert decorator.audit_log is audit

    def test_log_access_decorator(self):
        """Test log access decorator."""
        audit = AuditLog()
        decorator = AuditDecorator(audit)

        @decorator.log_access(AuditEventType.CREATE, "annotation", "create_annotation")
        def create_annotation(user_id, **kwargs):
            return {"id": "ann_001"}

        result = create_annotation("user_001")
        assert result["id"] == "ann_001"
        assert len(audit._events) == 1

    def test_log_access_decorator_failure(self):
        """Test log access decorator with failure."""
        audit = AuditLog()
        decorator = AuditDecorator(audit)

        @decorator.log_access(AuditEventType.CREATE, "annotation", "create_annotation")
        def create_annotation(user_id, **kwargs):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            create_annotation("user_001")

        # Should have logged the failure
        assert len(audit._events) == 1
        assert audit._events[0].success is False


class TestAuditLogEdgeCases:
    """Test edge cases for AuditLog."""

    def test_get_events_empty(self):
        """Test getting events from empty log."""
        audit = AuditLog()
        events = audit.get_events(user_id="nonexistent")
        assert events == []

    def test_get_events_limit(self):
        """Test event limit."""
        audit = AuditLog()
        for i in range(20):
            audit.log_event(
                AuditEvent(
                    event_type=AuditEventType.CREATE,
                    user_id="user_001",
                    resource_type="annotation",
                    resource_id=f"ann_{i}",
                    action="created",
                )
            )

        events = audit.get_events(user_id="user_001", limit=5)
        assert len(events) == 5

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created",
            timestamp=1234567890.0,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            details={"key": "value"},
            success=False,
            error_message="Test error",
            severity=AuditSeverity.CRITICAL,
            session_id="sess_001",
            request_id="req_001",
            tenant_id="tenant_001",
        )
        assert event.ip_address == "192.168.1.1"
        assert event.error_message == "Test error"
        assert event.severity == AuditSeverity.CRITICAL
