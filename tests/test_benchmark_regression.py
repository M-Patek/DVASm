"""Tests for regression benchmark."""

import tempfile
from datetime import datetime, timedelta

import pytest

from dvas.benchmarks.regression import RegressionAlert, RegressionBenchmark
from dvas.benchmarks.base import BenchmarkResult


class TestRegressionAlert:
    """Test RegressionAlert dataclass."""

    def test_creation(self):
        """Test basic creation."""
        alert = RegressionAlert(
            metric_name="bleu",
            baseline_value=0.80,
            current_value=0.72,
            change_percent=-10.0,
            severity="critical",
        )
        assert alert.metric_name == "bleu"
        assert alert.baseline_value == 0.80
        assert alert.current_value == 0.72
        assert alert.severity == "critical"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = RegressionAlert(
            metric_name="rouge",
            baseline_value=0.75,
            current_value=0.70,
            change_percent=-6.67,
        )
        data = alert.to_dict()
        assert data["metric_name"] == "rouge"
        assert data["severity"] == "warning"


class TestRegressionBenchmark:
    """Test RegressionBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RegressionBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "regression"
        assert temp_benchmark.results_dir.exists()
        assert temp_benchmark.regression_threshold == 0.05
        assert temp_benchmark.critical_threshold == 0.10

    def test_set_baseline(self, temp_benchmark):
        """Test baseline setting."""
        temp_benchmark.set_baseline(
            "epic_kitchens",
            "gpt-4",
            {"bleu": 0.80, "rouge": 0.75},
        )
        baseline = temp_benchmark.load_baseline("epic_kitchens", "gpt-4")
        assert baseline is not None
        assert baseline["bleu"] == 0.80

    def test_load_baseline_not_found(self, temp_benchmark):
        """Test loading non-existent baseline."""
        baseline = temp_benchmark.load_baseline("nonexistent", "model")
        assert baseline is None

    def test_detect_regression_no_change(self, temp_benchmark):
        """Test no regression when metrics stable."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.80}
        alerts = temp_benchmark.detect_regression(current, baseline)
        assert len(alerts) == 0

    def test_detect_regression_warning(self, temp_benchmark):
        """Test warning regression detection."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.75}
        alerts = temp_benchmark.detect_regression(current, baseline)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_detect_regression_critical(self, temp_benchmark):
        """Test critical regression detection."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.70}
        alerts = temp_benchmark.detect_regression(current, baseline)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_detect_improvement(self, temp_benchmark):
        """Test improvement detection."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.85}
        improvements = temp_benchmark.detect_improvement(current, baseline)
        assert len(improvements) == 1
        assert improvements[0]["change_percent"] > 0

    def test_detect_improvement_no_change(self, temp_benchmark):
        """Test no improvement when metrics stable."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.80}
        improvements = temp_benchmark.detect_improvement(current, baseline)
        assert len(improvements) == 0

    def test_compute_trend(self, temp_benchmark):
        """Test trend computation."""
        result = temp_benchmark.compute_trend("bleu", "test", "model")
        assert "slope" in result
        assert "trend" in result

    def test_get_historical_results_empty(self, temp_benchmark):
        """Test getting historical results when empty."""
        results = temp_benchmark.get_historical_results("test", "model")
        assert results == []

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        result = temp_benchmark.run_benchmark(
            "gpt-4",
            "epic",
            {"bleu": 0.82},
        )
        assert result.benchmark_name == "regression_epic"
        assert result.model_id == "gpt-4"

    def test_generate_summary_report(self, temp_benchmark):
        """Test summary report generation."""
        temp_benchmark.set_baseline("test", "model", {"bleu": 0.80})
        report = temp_benchmark.generate_summary_report("model")
        assert isinstance(report, dict)

    def test_empty_current_metrics(self, temp_benchmark):
        """Test with empty current metrics."""
        baseline = {"bleu": 0.80}
        current = {"bleu": 0.80}
        alerts = temp_benchmark.detect_regression(current, baseline)
        assert len(alerts) == 0
