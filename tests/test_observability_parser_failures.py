"""Tests for parser failure monitoring."""

import time

import pytest

from dvas.observability.parser_failures import ParserFailureMonitor


class TestParserFailureMonitor:
    @pytest.fixture
    def monitor(self):
        return ParserFailureMonitor(failure_threshold=0.1, max_records=100)

    def test_record_failure(self, monitor):
        monitor.record_failure("json_parser", "json_decode_error")
        counts = monitor.get_failure_counts()
        assert counts["json_decode_error"] == 1

    def test_record_failure_with_details(self, monitor):
        monitor.record_failure(
            "json_parser",
            "json_decode_error",
            video_id="vid_001",
            raw_text_preview="{invalid",
        )
        assert monitor.is_healthy() is False

    def test_record_success(self, monitor):
        monitor.record_success("json_parser")
        assert monitor._total_parsed == 1

    def test_failure_rate(self, monitor):
        monitor.record_success("json_parser")
        monitor.record_success("json_parser")
        monitor.record_failure("json_parser", "error")
        rate = monitor.get_failure_rate("json_parser")
        assert rate > 0

    def test_failure_rate_by_window(self, monitor):
        monitor.record_failure("json_parser", "error")
        rate_5m = monitor.get_failure_rate("json_parser", window_seconds=300)
        assert rate_5m > 0

    def test_failure_counts_by_type(self, monitor):
        monitor.record_failure("json_parser", "json_decode_error")
        monitor.record_failure("json_parser", "json_decode_error")
        monitor.record_failure("json_parser", "missing_field")
        counts = monitor.get_failure_counts("json_parser")
        assert counts["json_decode_error"] == 2
        assert counts["missing_field"] == 1

    def test_failure_trends(self, monitor):
        monitor.record_failure("json_parser", "error")
        time.sleep(0.01)
        monitor.record_failure("json_parser", "error")
        trends = monitor.get_failure_trends(interval_seconds=0.1)
        assert len(trends) >= 1

    def test_most_common_errors(self, monitor):
        monitor.record_failure("json_parser", "json_decode_error")
        monitor.record_failure("json_parser", "json_decode_error")
        monitor.record_failure("json_parser", "missing_field")
        top = monitor.get_most_common_errors(n=2)
        assert len(top) == 2
        assert top[0]["error_type"] == "json_decode_error"
        assert top[0]["count"] == 2

    def test_is_healthy(self, monitor):
        monitor.record_success("json_parser")
        monitor.record_success("json_parser")
        assert monitor.is_healthy("json_parser") is True

    def test_is_not_healthy(self, monitor):
        for _ in range(10):
            monitor.record_failure("json_parser", "error")
        assert monitor.is_healthy("json_parser") is False

    def test_stats(self, monitor):
        monitor.record_failure("json_parser", "error")
        monitor.record_success("json_parser")
        stats = monitor.get_stats()
        assert "total_failures" in stats
        assert "failure_rate_5m" in stats
        assert "top_errors" in stats

    def test_by_parser_stats(self, monitor):
        monitor.record_failure("json_parser", "error")
        monitor.record_failure("yaml_parser", "error")
        stats = monitor.get_stats()
        assert "json_parser" in stats["by_parser"]
        assert "yaml_parser" in stats["by_parser"]

    def test_alert_on_threshold(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        for _ in range(20):
            monitor.record_failure("json_parser", "error")
        assert len(alerts) > 0
        assert alerts[0][0] == "parser_failure_rate_exceeded"

    def test_reset(self, monitor):
        monitor.record_failure("json_parser", "error")
        monitor.reset()
        assert monitor.get_failure_counts() == {}
        assert monitor._total_parsed == 0

    def test_empty_trends(self, monitor):
        trends = monitor.get_failure_trends()
        assert trends == []

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False
