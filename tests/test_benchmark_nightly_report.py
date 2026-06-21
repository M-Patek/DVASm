"""Tests for nightly benchmark report."""

import tempfile
from datetime import datetime

import pytest

from dvas.benchmarks.nightly_report import NightlySummary, NightlyBenchmarkReport
from dvas.benchmarks.base import BenchmarkResult


class TestNightlySummary:
    """Test NightlySummary dataclass."""

    def test_creation(self):
        """Test basic creation."""
        summary = NightlySummary(
            benchmark_name="epic_kitchens",
            model_id="gpt-4",
            status="pass",
            metrics={"bleu": 0.80},
            duration_seconds=120.0,
        )
        assert summary.benchmark_name == "epic_kitchens"
        assert summary.status == "pass"
        assert summary.duration_seconds == 120.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = NightlySummary(
            benchmark_name="ego4d",
            model_id="claude",
            status="fail",
            metrics={"accuracy": 0.75},
            alerts=["regression detected"],
        )
        data = summary.to_dict()
        assert data["benchmark_name"] == "ego4d"
        assert data["status"] == "fail"
        assert len(data["alerts"]) == 1


class TestNightlyBenchmarkReport:
    """Test NightlyBenchmarkReport."""

    @pytest.fixture
    def temp_report(self):
        """Create temporary report directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield NightlyBenchmarkReport(tmpdir)

    def test_init(self, temp_report):
        """Test initialization."""
        assert temp_report.name == "nightly_report"
        assert temp_report.report_dir.exists()

    def test_add_summary(self, temp_report):
        """Test adding a summary."""
        summary = NightlySummary("epic", "gpt-4", "pass", {"bleu": 0.80})
        temp_report.add_summary(summary)
        assert len(temp_report._summaries) == 1

    def test_add_result(self, temp_report):
        """Test adding a BenchmarkResult."""
        result = BenchmarkResult(
            benchmark_name="epic",
            model_id="gpt-4",
            timestamp=datetime.utcnow(),
            metrics={"bleu": 0.80},
            predictions=[],
            references=[],
        )
        temp_report.add_result(result, duration_seconds=60.0)
        assert len(temp_report._summaries) == 1
        assert temp_report._summaries[0].duration_seconds == 60.0

    def test_compute_overall_status_pass(self, temp_report):
        """Test overall status when all pass."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        temp_report.add_summary(NightlySummary("ego4d", "m2", "pass"))
        assert temp_report.compute_overall_status() == "pass"

    def test_compute_overall_status_regression(self, temp_report):
        """Test overall status with regression."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        temp_report.add_summary(NightlySummary("ego4d", "m2", "regression"))
        assert temp_report.compute_overall_status() == "regression"

    def test_compute_overall_status_fail(self, temp_report):
        """Test overall status with failure."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        temp_report.add_summary(NightlySummary("ego4d", "m2", "fail"))
        assert temp_report.compute_overall_status() == "fail"

    def test_compute_summary_stats(self, temp_report):
        """Test summary statistics."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass", duration_seconds=60.0))
        temp_report.add_summary(NightlySummary("ego4d", "m2", "fail", duration_seconds=120.0))
        stats = temp_report.compute_summary_stats()
        assert stats["total_benchmarks"] == 2
        assert stats["passed"] == 1
        assert stats["failed"] == 1
        assert stats["total_duration_seconds"] == 180.0

    def test_generate_markdown_report(self, temp_report):
        """Test Markdown report generation."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass", {"bleu": 0.80}))
        report = temp_report.generate_markdown_report()
        assert isinstance(report, str)
        assert "# Nightly Benchmark Report" in report
        assert "epic" in report

    def test_generate_json_report(self, temp_report):
        """Test JSON report generation."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        report = temp_report.generate_json_report()
        assert isinstance(report, dict)
        assert "overall_status" in report
        assert "summary" in report

    def test_save_report(self, temp_report):
        """Test report saving."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        path = temp_report.save_report()
        assert path.exists()
        assert path.suffix == ".md"

    def test_get_recent_reports(self, temp_report):
        """Test getting recent reports."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        temp_report.save_report()
        reports = temp_report.get_recent_reports(days=7)
        assert isinstance(reports, list)

    def test_compare_with_previous_no_previous(self, temp_report):
        """Test comparison when no previous report exists."""
        comparison = temp_report.compare_with_previous()
        assert comparison["comparison_available"] is False

    def test_run_benchmark(self, temp_report):
        """Test full benchmark run."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass", {"bleu": 0.80}))
        result = temp_report.run_benchmark("nightly_run")
        assert result.benchmark_name == "nightly_report"
        assert result.model_id == "nightly_run"

    def test_should_alert_pass(self, temp_report):
        """Test alert check when all pass."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        assert temp_report.should_alert() is False

    def test_should_alert_fail(self, temp_report):
        """Test alert check when failure."""
        temp_report.add_summary(NightlySummary("epic", "m1", "fail"))
        assert temp_report.should_alert() is True

    def test_get_alert_summary(self, temp_report):
        """Test alert summary generation."""
        temp_report.add_summary(NightlySummary("epic", "m1", "fail", alerts=["BLEU dropped"]))
        summary = temp_report.get_alert_summary()
        assert summary is not None
        assert "Nightly Benchmark Alert" in summary
        assert "BLEU dropped" in summary

    def test_get_alert_summary_no_alerts(self, temp_report):
        """Test alert summary when no alerts."""
        temp_report.add_summary(NightlySummary("epic", "m1", "pass"))
        summary = temp_report.get_alert_summary()
        assert summary is None

    def test_empty_report(self, temp_report):
        """Test with empty report."""
        assert temp_report.compute_overall_status() == "pass"
        stats = temp_report.compute_summary_stats()
        assert stats["total_benchmarks"] == 0
