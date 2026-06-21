"""Tests for export throughput monitoring."""

import pytest

from dvas.observability.export_throughput import ExportThroughputMonitor


class TestExportThroughputMonitor:
    @pytest.fixture
    def monitor(self):
        return ExportThroughputMonitor(slow_threshold_ms=1000.0, max_records=100)

    def test_record_export(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        stats = monitor.get_export_stats("llava")
        assert stats["count"] == 1
        assert stats["total_bytes"] == 1024

    def test_multiple_exports(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        monitor.record_export("llava", bytes_written=2048, duration_ms=600)
        stats = monitor.get_export_stats("llava")
        assert stats["count"] == 2
        assert stats["total_bytes"] == 3072

    def test_export_failed(self, monitor):
        monitor.record_export(
            "llava", bytes_written=0, duration_ms=100, success=False, error="disk_full"
        )
        stats = monitor.get_export_stats("llava")
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 1

    def test_throughput_calculation(self, monitor):
        monitor.record_export("llava", bytes_written=1000, duration_ms=1000)
        throughput = monitor.get_throughput("llava", window_seconds=300)
        assert throughput > 0

    def test_format_comparison(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        monitor.record_export("openai", bytes_written=2048, duration_ms=600)
        comparison = monitor.get_format_comparison()
        assert "llava" in comparison
        assert "openai" in comparison

    def test_slow_exports(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=2000)
        slow = monitor.get_slow_exports(threshold_ms=1000, n=5)
        assert len(slow) == 1
        assert slow[0]["duration_ms"] == 2000

    def test_is_healthy(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        assert monitor.is_healthy("llava") is True

    def test_is_not_healthy(self, monitor):
        monitor.record_export("llava", bytes_written=0, duration_ms=100, success=False)
        assert monitor.is_healthy("llava") is False

    def test_alert_on_slow_export(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_export("llava", bytes_written=1024, duration_ms=2000)
        assert len(alerts) > 0
        assert alerts[0][0] == "export_slow"

    def test_alert_on_failed_export(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_export(
            "llava", bytes_written=0, duration_ms=100, success=False, error="disk_full"
        )
        assert len(alerts) > 0
        assert alerts[0][0] == "export_failed"

    def test_stats(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        stats = monitor.get_stats()
        assert "total_exports" in stats
        assert "overall" in stats
        assert "by_format" in stats

    def test_empty_stats(self, monitor):
        stats = monitor.get_export_stats("nonexistent")
        assert stats["count"] == 0
        assert stats["success_rate"] == 0.0

    def test_reset(self, monitor):
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        monitor.reset()
        assert monitor.get_export_stats("llava")["count"] == 0

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False
