"""Tests for task queue depth monitoring."""

import pytest

from dvas.observability.queue_depth import TaskQueueMonitor


class TestTaskQueueMonitor:
    @pytest.fixture
    def monitor(self):
        return TaskQueueMonitor(
            depth_threshold=10,
            critical_depth=50,
            max_wait_time_ms=5000.0,
        )

    def test_record_depth(self, monitor):
        monitor.record_depth("annotation_queue", 5)
        health = monitor.get_queue_health("annotation_queue")
        assert health["depth"] == 5
        assert health["status"] == "healthy"

    def test_warning_depth(self, monitor):
        monitor.record_depth("annotation_queue", 15)
        health = monitor.get_queue_health("annotation_queue")
        assert health["status"] == "warning"

    def test_critical_depth(self, monitor):
        monitor.record_depth("annotation_queue", 60)
        health = monitor.get_queue_health("annotation_queue")
        assert health["status"] == "critical"

    def test_task_completed(self, monitor):
        monitor.record_task_completed("annotation_queue", wait_time_ms=1000, processing_time_ms=500)
        health = monitor.get_queue_health("annotation_queue")
        # Queue health only returns metrics for queues that have had depth recorded
        # The processed_count is tracked internally
        assert health["status"] == "unknown"  # No depth was recorded

    def test_task_failed(self, monitor):
        monitor.record_task_failed("annotation_queue", "timeout")
        health = monitor.get_queue_health("annotation_queue")
        # Queue health only returns metrics for queues that have had depth recorded
        assert health["status"] == "unknown"

    def test_max_depth_tracking(self, monitor):
        monitor.record_depth("annotation_queue", 10)
        monitor.record_depth("annotation_queue", 20)
        monitor.record_depth("annotation_queue", 5)
        health = monitor.get_queue_health("annotation_queue")
        assert health["max_depth"] == 20

    def test_alert_on_critical_depth(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_depth("annotation_queue", 60)
        assert len(alerts) > 0
        assert alerts[0][0] == "queue_depth_critical"

    def test_alert_on_wait_time(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_task_completed("annotation_queue", wait_time_ms=10000)
        assert len(alerts) > 0
        assert alerts[0][0] == "queue_wait_time_exceeded"

    def test_is_healthy(self, monitor):
        monitor.record_depth("annotation_queue", 5)
        assert monitor.is_healthy("annotation_queue") is True

    def test_is_not_healthy(self, monitor):
        monitor.record_depth("annotation_queue", 60)
        assert monitor.is_healthy("annotation_queue") is False

    def test_all_queue_health(self, monitor):
        monitor.record_depth("queue1", 5)
        monitor.record_depth("queue2", 20)
        all_health = monitor.get_all_queue_health()
        assert "queue1" in all_health
        assert "queue2" in all_health

    def test_stats(self, monitor):
        monitor.record_depth("queue1", 5)
        monitor.record_depth("queue2", 20)
        stats = monitor.get_stats()
        assert stats["total_queues"] == 2
        assert stats["total_depth"] == 25
        assert "queue_health" in stats

    def test_reset_queue(self, monitor):
        monitor.record_depth("queue1", 10)
        monitor.reset("queue1")
        health = monitor.get_queue_health("queue1")
        assert health["status"] == "unknown"

    def test_reset_all(self, monitor):
        monitor.record_depth("queue1", 10)
        monitor.reset()
        assert monitor.get_all_queue_health() == {}

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False

    def test_task_enqueued_and_started(self, monitor):
        monitor.record_task_enqueued("annotation_queue")
        monitor.record_task_started("annotation_queue")
        # These just record metrics, no state to verify directly
        health = monitor.get_queue_health("annotation_queue")
        assert health["status"] == "unknown"  # No depth recorded yet
