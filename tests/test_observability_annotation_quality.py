"""Tests for annotation quality monitoring."""

import time

import pytest

from dvas.observability.annotation_quality import AnnotationQualityMonitor


class TestAnnotationQualityMonitor:
    @pytest.fixture
    def monitor(self):
        return AnnotationQualityMonitor(min_score=0.7, alert_threshold=0.6, max_records=100)

    def test_record_score(self, monitor):
        monitor.record_score("vid_001", 0.85, model_name="gpt-5.5")
        avg = monitor.get_average_score()
        assert avg == 0.85

    def test_score_clamping(self, monitor):
        monitor.record_score("vid_001", 1.5)
        assert monitor.get_average_score() == 1.0

        monitor.record_score("vid_002", -0.5)
        assert monitor.get_average_score() == 0.5  # avg of 1.0 and 0.0

    def test_average_by_model(self, monitor):
        monitor.record_score("vid_001", 0.9, model_name="gpt-5.5")
        monitor.record_score("vid_002", 0.7, model_name="claude")
        assert monitor.get_average_score("gpt-5.5") == 0.9
        assert monitor.get_average_score("claude") == 0.7

    def test_score_distribution(self, monitor):
        monitor.record_score("vid_001", 0.95)
        monitor.record_score("vid_002", 0.85)
        monitor.record_score("vid_003", 0.65)
        dist = monitor.get_score_distribution()
        assert dist["excellent (0.9-1.0)"] == 1
        assert dist["good (0.8-0.9)"] == 1
        assert dist["poor (0.6-0.7)"] == 1

    def test_quality_trends(self, monitor):
        monitor.record_score("vid_001", 0.9)
        time.sleep(0.01)
        monitor.record_score("vid_002", 0.8)
        trends = monitor.get_quality_trends(interval_seconds=0.1)
        assert len(trends) >= 1

    def test_low_quality_videos(self, monitor):
        monitor.record_score("vid_001", 0.5)
        monitor.record_score("vid_002", 0.4)
        monitor.record_score("vid_003", 0.9)
        low = monitor.get_low_quality_videos(threshold=0.6, n=2)
        assert len(low) == 2
        assert low[0]["score"] == 0.4

    def test_model_comparison(self, monitor):
        monitor.record_score("vid_001", 0.9, model_name="gpt-5.5")
        monitor.record_score("vid_002", 0.8, model_name="gpt-5.5")
        monitor.record_score("vid_003", 0.7, model_name="claude")
        comparison = monitor.get_model_comparison()
        assert "gpt-5.5" in comparison
        assert "claude" in comparison
        assert abs(comparison["gpt-5.5"]["avg_score"] - 0.85) < 0.01

    def test_is_quality_acceptable(self, monitor):
        monitor.record_score("vid_001", 0.8)
        assert monitor.is_quality_acceptable() is True

    def test_is_not_quality_acceptable(self, monitor):
        monitor.record_score("vid_001", 0.5)
        assert monitor.is_quality_acceptable() is False

    def test_detect_degradation(self, monitor):
        # Simulate baseline period
        monitor.record_score("vid_001", 0.9)
        time.sleep(0.02)
        # Simulate recent degradation
        monitor.record_score("vid_002", 0.5)
        degradation = monitor.detect_degradation(
            recent_window=0.015,  # Only vid_002 (10ms+ after vid_001)
            baseline_window=0.1,  # Both scores
        )
        assert degradation is not None
        assert degradation["degraded"] is True

    def test_no_degradation(self, monitor):
        monitor.record_score("vid_001", 0.9)
        time.sleep(0.01)
        monitor.record_score("vid_002", 0.88)
        degradation = monitor.detect_degradation(
            recent_window=0.05,  # Large enough to include both scores after 10ms sleep
            baseline_window=0.1,
        )
        assert degradation is None

    def test_alert_on_low_quality(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_score("vid_001", 0.5)  # Below threshold of 0.6
        assert len(alerts) > 0
        assert alerts[0][0] == "quality_below_threshold"

    def test_stats(self, monitor):
        monitor.record_score("vid_001", 0.85)
        stats = monitor.get_stats()
        assert "total_scored" in stats
        assert "average_score" in stats
        assert "score_distribution" in stats
        assert "model_comparison" in stats

    def test_reset(self, monitor):
        monitor.record_score("vid_001", 0.85)
        monitor.reset()
        assert monitor.get_average_score() == 0.0

    def test_empty_trends(self, monitor):
        trends = monitor.get_quality_trends()
        assert trends == []

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False
