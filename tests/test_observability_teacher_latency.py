"""Tests for teacher latency monitoring."""

import time

import pytest

from dvas.observability.teacher_latency import (
    LatencyThreshold,
    TeacherLatencyMonitor,
    TeacherLatencyTracker,
)


class TestLatencyThreshold:
    def test_default_values(self):
        threshold = LatencyThreshold()
        assert threshold.p50_ms == 1000.0
        assert threshold.p95_ms == 5000.0
        assert threshold.p99_ms == 10000.0

    def test_custom_values(self):
        threshold = LatencyThreshold(p50_ms=500.0, p95_ms=2000.0, p99_ms=5000.0)
        assert threshold.p50_ms == 500.0


class TestTeacherLatencyMonitor:
    @pytest.fixture
    def monitor(self):
        return TeacherLatencyMonitor(max_samples=100)

    def test_record_latency(self, monitor):
        monitor.record_latency("gpt-5.5", 2500.0)
        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["count"] == 1
        assert stats["avg_ms"] == 2500.0

    def test_multiple_latencies(self, monitor):
        for i in range(10):
            monitor.record_latency("gpt-5.5", float(i * 100))
        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["count"] == 10
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 900.0

    def test_percentile_calculation(self, monitor):
        for i in range(100):
            monitor.record_latency("gpt-5.5", float(i * 100))
        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["p50_ms"] == 5000.0
        assert stats["p95_ms"] == 9500.0

    def test_multiple_models(self, monitor):
        monitor.record_latency("gpt-5.5", 1000.0)
        monitor.record_latency("claude-opus", 2000.0)
        all_stats = monitor.get_all_stats()
        assert "gpt-5.5" in all_stats
        assert "claude-opus" in all_stats

    def test_alert_on_high_latency(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_latency("gpt-5.5", 15000.0)  # Above p99 threshold
        assert len(alerts) > 0
        assert alerts[0][0] == "latency_p99_exceeded"

    def test_is_healthy(self, monitor):
        monitor.record_latency("gpt-5.5", 500.0)
        assert monitor.is_healthy("gpt-5.5") is True

    def test_is_not_healthy(self, monitor):
        for _ in range(20):
            monitor.record_latency("gpt-5.5", 10000.0)
        assert monitor.is_healthy("gpt-5.5") is False

    def test_get_slowest_models(self, monitor):
        monitor.record_latency("model-a", 1000.0)
        monitor.record_latency("model-b", 2000.0)
        monitor.record_latency("model-c", 500.0)
        slowest = monitor.get_slowest_models(n=2)
        assert len(slowest) == 2
        assert slowest[0]["model"] == "model-b"

    def test_reset_model(self, monitor):
        monitor.record_latency("gpt-5.5", 1000.0)
        monitor.reset("gpt-5.5")
        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["count"] == 0

    def test_reset_all(self, monitor):
        monitor.record_latency("gpt-5.5", 1000.0)
        monitor.reset()
        all_stats = monitor.get_all_stats()
        assert len(all_stats) == 0

    def test_empty_stats(self, monitor):
        stats = monitor.get_latency_stats("nonexistent")
        assert stats["count"] == 0
        assert stats["avg_ms"] == 0.0

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False


class TestTeacherLatencyTracker:
    def test_context_manager(self):
        monitor = TeacherLatencyMonitor()
        with TeacherLatencyTracker("gpt-5.5", monitor) as tracker:
            time.sleep(0.01)
            assert tracker.model_name == "gpt-5.5"

        assert tracker.latency_ms is not None
        assert tracker.latency_ms >= 10

        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["count"] == 1

    def test_async_context_manager(self):
        import asyncio

        async def test_async():
            monitor = TeacherLatencyMonitor()
            async with TeacherLatencyTracker("gpt-5.5", monitor) as tracker:
                await asyncio.sleep(0.01)
            assert tracker.latency_ms is not None

        asyncio.run(test_async())

    def test_exception_handling(self):
        monitor = TeacherLatencyMonitor()
        try:
            with TeacherLatencyTracker("gpt-5.5", monitor):
                raise ValueError("test")
        except ValueError:
            pass
        # Should still record latency
        stats = monitor.get_latency_stats("gpt-5.5")
        assert stats["count"] == 1
