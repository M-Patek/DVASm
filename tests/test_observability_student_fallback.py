"""Tests for student fallback rate monitoring."""

import time

import pytest

from dvas.observability.student_fallback import StudentFallbackMonitor


class TestStudentFallbackMonitor:
    @pytest.fixture
    def monitor(self):
        return StudentFallbackMonitor(
            fallback_threshold=0.1,
            critical_threshold=0.3,
            max_records=100,
        )

    def test_record_fallback(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        by_reason = monitor.get_fallback_by_reason()
        assert by_reason["low_confidence"] == 1

    def test_record_fallback_with_details(self, monitor):
        monitor.record_fallback(
            "error",
            "student_v1",
            "gpt-5.5",
            video_id="vid_001",
            metadata={"confidence": 0.3},
        )
        assert monitor.is_healthy("student_v1") is False

    def test_record_student_call(self, monitor):
        monitor.record_student_call("student_v1")
        assert monitor._total_student_calls == 1

    def test_fallback_rate(self, monitor):
        monitor.record_student_call("student_v1")
        monitor.record_student_call("student_v1")
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        rate = monitor.get_fallback_rate("student_v1")
        assert rate > 0

    def test_fallback_by_reason(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_fallback("error", "student_v1", "gpt-5.5")
        counts = monitor.get_fallback_by_reason("student_v1")
        assert counts["low_confidence"] == 2
        assert counts["error"] == 1

    def test_fallback_trends(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        time.sleep(0.01)
        monitor.record_fallback("error", "student_v1", "gpt-5.5")
        trends = monitor.get_fallback_trends(interval_seconds=0.1)
        assert len(trends) >= 1

    def test_stats(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_student_call("student_v1")
        stats = monitor.get_stats("student_v1")
        assert "total_fallbacks" in stats
        assert "fallback_rate_1h" in stats
        assert "by_reason" in stats

    def test_top_fallback_reasons(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_fallback("error", "student_v1", "gpt-5.5")
        top = monitor.get_top_fallback_reasons(n=2)
        assert len(top) == 2
        assert top[0]["reason"] == "low_confidence"
        assert top[0]["count"] == 2

    def test_is_healthy(self, monitor):
        monitor.record_student_call("student_v1")
        monitor.record_student_call("student_v1")
        assert monitor.is_healthy("student_v1") is True

    def test_is_not_healthy(self, monitor):
        for _ in range(20):
            monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        assert monitor.is_healthy("student_v1") is False

    def test_model_comparison(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.record_fallback("error", "student_v2", "claude")
        comparison = monitor.get_model_comparison()
        assert "student_v1" in comparison
        assert "student_v2" in comparison

    def test_alert_on_threshold(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        for _ in range(20):
            monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        assert len(alerts) > 0
        assert alerts[0][0] == "fallback_rate_critical"

    def test_reset_model(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.reset("student_v1")
        assert monitor.get_fallback_by_reason("student_v1") == {}

    def test_reset_all(self, monitor):
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        monitor.reset()
        assert monitor.get_fallback_by_reason() == {}
        assert monitor._total_student_calls == 0

    def test_empty_trends(self, monitor):
        trends = monitor.get_fallback_trends()
        assert trends == []

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False
