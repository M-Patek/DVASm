"""Tests for teacher cost monitoring."""

import time

import pytest

from dvas.observability.teacher_cost import CostBudget, TeacherCostMonitor


class TestCostBudget:
    def test_default_values(self):
        budget = CostBudget()
        assert budget.daily_usd == 100.0
        assert budget.hourly_usd == 10.0
        assert budget.per_request_usd == 1.0

    def test_custom_values(self):
        budget = CostBudget(daily_usd=500.0, hourly_usd=50.0)
        assert budget.daily_usd == 500.0
        assert budget.hourly_usd == 50.0


class TestTeacherCostMonitor:
    @pytest.fixture
    def monitor(self):
        return TeacherCostMonitor(CostBudget(daily_usd=100.0, hourly_usd=10.0))

    def test_record_cost(self, monitor):
        monitor.record_cost("gpt-5.5", 0.05)
        by_model = monitor.get_cost_by_model()
        assert "gpt-5.5" in by_model
        assert by_model["gpt-5.5"] == 0.05

    def test_hourly_cost(self, monitor):
        monitor.record_cost("gpt-5.5", 5.0)
        hourly = monitor.get_hourly_cost("gpt-5.5")
        assert hourly == 5.0

    def test_daily_cost(self, monitor):
        monitor.record_cost("gpt-5.5", 5.0)
        monitor.record_cost("gpt-5.5", 3.0)
        daily = monitor.get_daily_cost("gpt-5.5")
        assert daily == 8.0

    def test_cost_by_model(self, monitor):
        monitor.record_cost("gpt-5.5", 1.0)
        monitor.record_cost("claude-opus", 2.0)
        by_model = monitor.get_cost_by_model()
        assert by_model["gpt-5.5"] == 1.0
        assert by_model["claude-opus"] == 2.0

    def test_cost_breakdown(self, monitor):
        monitor.record_cost("gpt-5.5", 0.5, request_type="annotation")
        monitor.record_cost("gpt-5.5", 0.3, request_type="batch")
        breakdown = monitor.get_cost_breakdown("gpt-5.5")
        assert breakdown["annotation"] == 0.5
        assert breakdown["batch"] == 0.3

    def test_within_budget(self, monitor):
        monitor.record_cost("gpt-5.5", 5.0)
        assert monitor.is_within_budget("gpt-5.5") is True

    def test_exceeds_budget(self, monitor):
        monitor.record_cost("gpt-5.5", 150.0)
        assert monitor.is_within_budget("gpt-5.5") is False

    def test_most_expensive_models(self, monitor):
        monitor.record_cost("model-a", 10.0)
        monitor.record_cost("model-b", 20.0)
        monitor.record_cost("model-c", 5.0)
        expensive = monitor.get_most_expensive_models(n=2)
        assert len(expensive) == 2
        assert expensive[0]["model"] == "model-b"
        assert expensive[0]["total_cost_usd"] == 20.0

    def test_estimate_remaining_budget(self, monitor):
        monitor.record_cost("gpt-5.5", 50.0)
        estimate = monitor.estimate_remaining_budget()
        assert estimate["daily_budget_usd"] == 100.0
        assert estimate["daily_spent_usd"] == 50.0
        assert estimate["remaining_usd"] == 50.0

    def test_stats(self, monitor):
        monitor.record_cost("gpt-5.5", 10.0)
        stats = monitor.get_stats()
        assert "hourly_cost_usd" in stats
        assert "daily_cost_usd" in stats
        assert "by_model" in stats

    def test_alert_on_budget_exceeded(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_cost("gpt-5.5", 15.0)  # Exceeds hourly budget of 10
        assert len(alerts) > 0
        assert alerts[0][0] == "hourly_budget_exceeded"

    def test_reset_model(self, monitor):
        monitor.record_cost("gpt-5.5", 10.0)
        monitor.reset("gpt-5.5")
        assert monitor.get_cost_by_model().get("gpt-5.5", 0) == 0

    def test_reset_all(self, monitor):
        monitor.record_cost("gpt-5.5", 10.0)
        monitor.reset()
        assert len(monitor.get_cost_by_model()) == 0

    def test_per_request_alert(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_cost("gpt-5.5", 5.0)  # Exceeds per_request of 1.0
        assert any(a[0] == "per_request_budget_exceeded" for a in alerts)

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False
