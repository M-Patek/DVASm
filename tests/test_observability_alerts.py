"""Tests for configurable alerting rules."""

import time

import pytest

from dvas.observability.alerts import (
    AlertEvent,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackNotificationChannel,
    LogNotificationChannel,
)


class TestAlertSeverity:
    def test_values(self):
        assert AlertSeverity.INFO.name == "INFO"
        assert AlertSeverity.WARNING.name == "WARNING"
        assert AlertSeverity.CRITICAL.name == "CRITICAL"
        assert AlertSeverity.EMERGENCY.name == "EMERGENCY"


class TestAlertRule:
    def test_creation(self):
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric="test_metric",
            operator=">",
            threshold=10.0,
        )
        assert rule.name == "test_rule"
        assert rule.enabled is True

    def test_evaluate_greater_than(self):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator=">",
            threshold=10.0,
        )
        assert rule.evaluate(15.0) is True
        assert rule.evaluate(5.0) is False
        assert rule.evaluate(10.0) is False

    def test_evaluate_less_than(self):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator="<",
            threshold=10.0,
        )
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(15.0) is False

    def test_evaluate_equal(self):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator="==",
            threshold=10.0,
        )
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(5.0) is False

    def test_evaluate_greater_equal(self):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator=">=",
            threshold=10.0,
        )
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(15.0) is True
        assert rule.evaluate(5.0) is False

    def test_evaluate_unknown_operator(self):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator="invalid",
            threshold=10.0,
        )
        assert rule.evaluate(15.0) is False


class TestAlertEvent:
    def test_to_dict(self):
        event = AlertEvent(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Test message",
            value=15.0,
            threshold=10.0,
            timestamp=1234567890.0,
        )
        d = event.to_dict()
        assert d["rule_name"] == "test"
        assert d["severity"] == "WARNING"
        assert d["value"] == 15.0
        assert d["status"] == "FIRING"


class TestLogNotificationChannel:
    def test_creation(self):
        channel = LogNotificationChannel("test")
        assert channel.get_name() == "test"

    def test_send(self):
        channel = LogNotificationChannel()
        event = AlertEvent(
            rule_name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            value=1.0,
            threshold=0.0,
            timestamp=time.time(),
        )
        assert channel.send(event) is True


class TestCallbackNotificationChannel:
    def test_creation(self):
        def callback(event):
            return True

        channel = CallbackNotificationChannel("test", callback)
        assert channel.get_name() == "test"

    def test_send(self):
        def callback(event):
            return True

        channel = CallbackNotificationChannel("test", callback)
        event = AlertEvent(
            rule_name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            value=1.0,
            threshold=0.0,
            timestamp=time.time(),
        )
        assert channel.send(event) is True

    def test_send_error(self):
        def callback(event):
            raise ValueError("test error")

        channel = CallbackNotificationChannel("test", callback)
        event = AlertEvent(
            rule_name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            value=1.0,
            threshold=0.0,
            timestamp=time.time(),
        )
        assert channel.send(event) is False


class TestAlertManager:
    @pytest.fixture
    def manager(self):
        return AlertManager(rate_limit_seconds=0.0, max_alerts=100)

    def test_add_rule(self, manager):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator=">",
            threshold=10.0,
        )
        manager.add_rule(rule)
        assert manager.get_rule("test") is not None

    def test_remove_rule(self, manager):
        rule = AlertRule(
            name="test",
            description="test",
            metric="m",
            operator=">",
            threshold=10.0,
        )
        manager.add_rule(rule)
        assert manager.remove_rule("test") is True
        assert manager.remove_rule("nonexistent") is False

    def test_list_rules(self, manager):
        manager.add_rule(
            AlertRule(
                name="r1",
                description="test",
                metric="m1",
                operator=">",
                threshold=10.0,
            )
        )
        manager.add_rule(
            AlertRule(
                name="r2",
                description="test",
                metric="m2",
                operator="<",
                threshold=5.0,
            )
        )
        rules = manager.list_rules()
        assert len(rules) == 2

    def test_evaluate_triggered(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        alerts = manager.evaluate("cpu", 90.0)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "test"

    def test_evaluate_not_triggered(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        alerts = manager.evaluate("cpu", 50.0)
        assert len(alerts) == 0

    def test_evaluate_disabled_rule(self, manager):
        rule = AlertRule(
            name="test",
            description="test",
            metric="cpu",
            operator=">",
            threshold=80.0,
            enabled=False,
        )
        manager.add_rule(rule)
        alerts = manager.evaluate("cpu", 90.0)
        assert len(alerts) == 0

    def test_evaluate_wrong_metric(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        alerts = manager.evaluate("memory", 90.0)
        assert len(alerts) == 0

    def test_rate_limiting(self, manager):
        manager.rate_limit_seconds = 1.0
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        alerts1 = manager.evaluate("cpu", 90.0)
        assert len(alerts1) == 1
        # Second evaluation should be rate limited
        alerts2 = manager.evaluate("cpu", 95.0)
        assert len(alerts2) == 0

    def test_duration_requirement(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
                duration_seconds=1.0,
            )
        )
        # First trigger should not fire due to duration requirement
        alerts = manager.evaluate("cpu", 90.0)
        assert len(alerts) == 0

    def test_add_channel(self, manager):
        channel = LogNotificationChannel("test")
        manager.add_channel(channel)
        assert len(manager._channels) == 1

    def test_remove_channel(self, manager):
        channel = LogNotificationChannel("test")
        manager.add_channel(channel)
        assert manager.remove_channel("test") is True
        assert manager.remove_channel("nonexistent") is False

    def test_acknowledge(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        manager.evaluate("cpu", 90.0)
        assert manager.acknowledge("test") is True
        assert manager.acknowledge("nonexistent") is False

    def test_resolve(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        manager.evaluate("cpu", 90.0)
        assert manager.resolve("test") is True
        active = manager.get_active_alerts()
        assert len(active) == 0

    def test_get_active_alerts(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        manager.evaluate("cpu", 90.0)
        active = manager.get_active_alerts()
        assert len(active) == 1

    def test_get_alert_history(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
            )
        )
        manager.evaluate("cpu", 90.0)
        history = manager.get_alert_history(severity=AlertSeverity.WARNING)
        assert len(history) == 1

    def test_stats(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        manager.evaluate("cpu", 90.0)
        stats = manager.get_stats()
        assert stats["total_rules"] == 1
        assert stats["active_alerts"] == 1

    def test_create_default_rules(self, manager):
        manager.create_default_rules()
        rules = manager.list_rules()
        assert len(rules) == 8
        rule_names = [r.name for r in rules]
        assert "teacher_latency_high" in rule_names
        assert "queue_depth_critical" in rule_names
        assert "storage_critical" in rule_names

    def test_label_matching(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
                labels={"host": "server1"},
            )
        )
        alerts = manager.evaluate("cpu", 90.0, labels={"host": "server1"})
        assert len(alerts) == 1

    def test_label_mismatch(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
                labels={"host": "server1"},
            )
        )
        alerts = manager.evaluate("cpu", 90.0, labels={"host": "server2"})
        assert len(alerts) == 0

    def test_reset(self, manager):
        manager.add_rule(
            AlertRule(
                name="test",
                description="test",
                metric="cpu",
                operator=">",
                threshold=80.0,
            )
        )
        manager.evaluate("cpu", 90.0)
        manager.reset()
        assert len(manager.get_active_alerts()) == 0
        assert len(manager._alerts) == 0
