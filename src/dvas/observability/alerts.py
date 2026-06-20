"""Configurable alerting rules for DVAS.

Provides a rule-based alerting system with multiple severity levels,
notification channels, and rate limiting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class AlertStatus(Enum):
    """Alert lifecycle status."""

    PENDING = auto()
    FIRING = auto()
    RESOLVED = auto()
    ACKNOWLEDGED = auto()


@dataclass
class AlertRule:
    """A single alerting rule.

    Defines conditions for triggering an alert with
    configurable thresholds and severity.
    """

    name: str
    description: str
    metric: str
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    duration_seconds: float = 0.0  # How long condition must persist
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule is triggered.

        Args:
            value: Current metric value

        Returns:
            True if the rule condition is met
        """
        operators = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        op = operators.get(self.operator)
        if not op:
            logger.error("unknown_operator", operator=self.operator)
            return False
        return op(value, self.threshold)


@dataclass
class AlertEvent:
    """A triggered alert event."""

    rule_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.FIRING

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.name,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "status": self.status.name,
        }


class NotificationChannel:
    """Base class for alert notification channels."""

    def send(self, alert: AlertEvent) -> bool:
        """Send an alert notification.

        Args:
            alert: Alert event to send

        Returns:
            True if notification was sent successfully
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Get channel name."""
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Notification channel that logs alerts."""

    def __init__(self, name: str = "log") -> None:
        self._name = name

    def send(self, alert: AlertEvent) -> bool:
        level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.warning)

        level(
            "alert_notification",
            channel=self._name,
            **alert.to_dict(),
        )
        return True

    def get_name(self) -> str:
        return self._name


class CallbackNotificationChannel(NotificationChannel):
    """Notification channel that calls a callback function."""

    def __init__(self, name: str, callback: Callable[[AlertEvent], bool]) -> None:
        self._name = name
        self._callback = callback

    def send(self, alert: AlertEvent) -> bool:
        try:
            return self._callback(alert)
        except Exception as e:
            logger.error("callback_notification_failed", error=str(e))
            return False

    def get_name(self) -> str:
        return self._name


class AlertManager:
    """Configurable alerting system for DVAS.

    Manages alert rules, evaluates conditions, and sends notifications
    through configured channels with rate limiting.

    Usage::

        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high_latency",
            description="Teacher latency is too high",
            metric="teacher_latency_p95",
            operator=">",
            threshold=5000.0,
            severity=AlertSeverity.WARNING,
        ))
        manager.add_channel(LogNotificationChannel())
        manager.evaluate("teacher_latency_p95", 6000.0)
    """

    def __init__(
        self,
        rate_limit_seconds: float = 300.0,
        max_alerts: int = 1000,
    ) -> None:
        self.rate_limit_seconds = rate_limit_seconds
        self.max_alerts = max_alerts
        self._rules: Dict[str, AlertRule] = {}
        self._channels: List[NotificationChannel] = []
        self._alerts: List[AlertEvent] = []
        self._last_alert_time: Dict[str, float] = {}
        self._rule_first_trigger: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: AlertRule to add
        """
        with self._lock:
            self._rules[rule.name] = rule
        logger.info("alert_rule_added", name=rule.name, metric=rule.metric)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.

        Args:
            name: Name of rule to remove

        Returns:
            True if rule was found and removed
        """
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                self._rule_first_trigger.pop(name, None)
                return True
            return False

    def get_rule(self, name: str) -> Optional[AlertRule]:
        """Get an alert rule by name.

        Args:
            name: Rule name

        Returns:
            AlertRule if found, None otherwise
        """
        return self._rules.get(name)

    def list_rules(self) -> List[AlertRule]:
        """List all configured rules.

        Returns:
            List of AlertRule objects
        """
        return list(self._rules.values())

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.

        Args:
            channel: NotificationChannel to add
        """
        with self._lock:
            self._channels.append(channel)
        logger.info("notification_channel_added", name=channel.get_name())

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel by name.

        Args:
            name: Channel name

        Returns:
            True if channel was found and removed
        """
        with self._lock:
            for i, channel in enumerate(self._channels):
                if channel.get_name() == name:
                    self._channels.pop(i)
                    return True
            return False

    def evaluate(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> List[AlertEvent]:
        """Evaluate all rules for a metric.

        Args:
            metric_name: Name of the metric to evaluate
            value: Current metric value
            labels: Optional labels for the metric

        Returns:
            List of triggered AlertEvents
        """
        triggered: List[AlertEvent] = []
        now = time.time()

        for rule in self._rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue

            # Check label match if rule has labels
            if rule.labels and labels:
                if not all(labels.get(k) == v for k, v in rule.labels.items()):
                    continue

            if rule.evaluate(value):
                # Check duration requirement
                if rule.duration_seconds > 0:
                    first_trigger = self._rule_first_trigger.get(rule.name)
                    if first_trigger is None:
                        self._rule_first_trigger[rule.name] = now
                        continue
                    elif now - first_trigger < rule.duration_seconds:
                        continue
                else:
                    self._rule_first_trigger[rule.name] = now

                # Check rate limiting
                last_time = self._last_alert_time.get(rule.name, 0)
                if now - last_time < self.rate_limit_seconds:
                    continue

                alert = AlertEvent(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.description}: {metric_name}={value} (threshold: {rule.threshold})",
                    value=value,
                    threshold=rule.threshold,
                    timestamp=now,
                    labels=labels or {},
                )
                triggered.append(alert)
                self._fire_alert(alert)
            else:
                # Rule not triggered - reset first trigger
                self._rule_first_trigger.pop(rule.name, None)

        return triggered

    def _fire_alert(self, alert: AlertEvent) -> None:
        """Fire an alert through all channels."""
        self._last_alert_time[alert.rule_name] = alert.timestamp

        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > self.max_alerts:
                self._alerts = self._alerts[-self.max_alerts :]

        logger.warning(
            "alert_fired",
            rule=alert.rule_name,
            severity=alert.severity.name,
            value=alert.value,
            threshold=alert.threshold,
        )

        for channel in self._channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(
                    "notification_failed",
                    channel=channel.get_name(),
                    error=str(e),
                )

    def acknowledge(self, rule_name: str) -> bool:
        """Acknowledge an alert.

        Args:
            rule_name: Name of the rule to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        with self._lock:
            for alert in self._alerts:
                if alert.rule_name == rule_name and alert.status == AlertStatus.FIRING:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    logger.info("alert_acknowledged", rule=rule_name)
                    return True
            return False

    def resolve(self, rule_name: str) -> bool:
        """Resolve a firing alert.

        Args:
            rule_name: Name of the rule to resolve

        Returns:
            True if alert was found and resolved
        """
        with self._lock:
            for alert in self._alerts:
                if alert.rule_name == rule_name and alert.status in (
                    AlertStatus.FIRING,
                    AlertStatus.ACKNOWLEDGED,
                ):
                    alert.status = AlertStatus.RESOLVED
                    self._rule_first_trigger.pop(rule_name, None)
                    logger.info("alert_resolved", rule=rule_name)
                    return True
            return False

    def get_active_alerts(self) -> List[AlertEvent]:
        """Get currently firing alerts.

        Returns:
            List of active AlertEvents
        """
        with self._lock:
            return [
                a
                for a in self._alerts
                if a.status in (AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED)
            ]

    def get_alert_history(
        self,
        rule_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[AlertEvent]:
        """Get alert history.

        Args:
            rule_name: Optional rule filter
            severity: Optional severity filter

        Returns:
            List of matching AlertEvents
        """
        with self._lock:
            return [
                a
                for a in self._alerts
                if (rule_name is None or a.rule_name == rule_name)
                and (severity is None or a.severity == severity)
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics.

        Returns:
            Dict with rule count, active alerts, and history
        """
        active = self.get_active_alerts()
        by_severity: Dict[str, int] = {}
        for alert in active:
            sev = alert.severity.name
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_rules": len(self._rules),
            "active_alerts": len(active),
            "total_alerts_fired": len(self._alerts),
            "by_severity": by_severity,
            "rate_limit_seconds": self.rate_limit_seconds,
            "channels": [c.get_name() for c in self._channels],
        }

    def create_default_rules(self) -> None:
        """Create default DVAS alert rules."""
        defaults = [
            AlertRule(
                name="teacher_latency_high",
                description="Teacher model P95 latency is too high",
                metric="teacher_latency_p95",
                operator=">",
                threshold=5000.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0,
            ),
            AlertRule(
                name="teacher_cost_daily_exceeded",
                description="Daily teacher cost budget exceeded",
                metric="teacher_cost_daily",
                operator=">",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                name="parser_failure_rate_high",
                description="Parser failure rate is above threshold",
                metric="parser_failure_rate",
                operator=">",
                threshold=0.05,
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0,
            ),
            AlertRule(
                name="annotation_quality_low",
                description="Annotation quality score is below minimum",
                metric="annotation_quality_avg",
                operator="<",
                threshold=0.7,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=600.0,
            ),
            AlertRule(
                name="queue_depth_critical",
                description="Task queue depth is critically high",
                metric="task_queue_depth",
                operator=">",
                threshold=500.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60.0,
            ),
            AlertRule(
                name="export_slow",
                description="Export operations are too slow",
                metric="export_duration_p95",
                operator=">",
                threshold=30.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0,
            ),
            AlertRule(
                name="storage_critical",
                description="Storage utilization is critically high",
                metric="storage_utilization",
                operator=">",
                threshold=0.95,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=300.0,
            ),
            AlertRule(
                name="student_fallback_high",
                description="Student fallback rate is too high",
                metric="student_fallback_rate",
                operator=">",
                threshold=0.3,
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0,
            ),
        ]

        for rule in defaults:
            self.add_rule(rule)

    def reset(self) -> None:
        """Reset all alerts and state."""
        with self._lock:
            self._alerts.clear()
            self._last_alert_time.clear()
            self._rule_first_trigger.clear()
