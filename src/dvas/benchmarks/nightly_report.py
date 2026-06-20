"""Nightly benchmark report for automated reporting.

Generates comprehensive nightly benchmark reports aggregating
results from all benchmark suites and sending notifications.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NightlySummary:
    """Summary of a single benchmark from the nightly run.

    Attributes:
        benchmark_name: Name of the benchmark
        model_id: Model identifier
        status: Run status ("pass", "fail", "regression")
        metrics: Key metrics from the run
        alerts: Any alerts generated
        duration_seconds: How long the benchmark took
    """

    benchmark_name: str
    model_id: str
    status: str = "pass"
    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_id": self.model_id,
            "status": self.status,
            "metrics": self.metrics,
            "alerts": self.alerts,
            "duration_seconds": self.duration_seconds,
        }


class NightlyBenchmarkReport(BaseBenchmark):
    """Nightly benchmark report generator.

    Aggregates results from all benchmark suites and generates
    a comprehensive nightly report with alerts and trends.

    Args:
        benchmark_dir: Directory for storing benchmark data
        report_dir: Directory for storing generated reports
    """

    STATUS_PASS = "pass"
    STATUS_FAIL = "fail"
    STATUS_REGRESSION = "regression"

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        report_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(benchmark_dir, "nightly_report")
        self.report_dir = Path(report_dir) if report_dir else self.benchmark_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: List[NightlySummary] = []

    def add_summary(self, summary: NightlySummary) -> None:
        """Add a benchmark summary to the nightly report.

        Args:
            summary: NightlySummary to add
        """
        self._summaries.append(summary)
        logger.info(
            "Added nightly summary",
            benchmark=summary.benchmark_name,
            model=summary.model_id,
            status=summary.status,
        )

    def add_result(self, result: BenchmarkResult, duration_seconds: float = 0.0) -> None:
        """Add a BenchmarkResult to the nightly report.

        Args:
            result: BenchmarkResult to add
            duration_seconds: How long the benchmark took
        """
        # Determine status
        status = self.STATUS_PASS
        alerts = []

        if "n_regressions" in result.metrics and result.metrics["n_regressions"] > 0:
            status = self.STATUS_REGRESSION
            alerts.append(f"{result.metrics['n_regressions']} regression(s) detected")

        # Extract key metrics (top-level numeric metrics)
        key_metrics = {
            k: v for k, v in result.metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        summary = NightlySummary(
            benchmark_name=result.benchmark_name,
            model_id=result.model_id,
            status=status,
            metrics=key_metrics,
            alerts=alerts,
            duration_seconds=duration_seconds,
        )
        self.add_summary(summary)

    def compute_overall_status(self) -> str:
        """Compute overall status for the nightly run.

        Returns:
            Overall status ("pass", "fail", or "regression")
        """
        if not self._summaries:
            return self.STATUS_PASS

        statuses = [s.status for s in self._summaries]
        if self.STATUS_FAIL in statuses:
            return self.STATUS_FAIL
        if self.STATUS_REGRESSION in statuses:
            return self.STATUS_REGRESSION
        return self.STATUS_PASS

    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the nightly run.

        Returns:
            Dictionary with summary statistics
        """
        if not self._summaries:
            return {
                "total_benchmarks": 0,
                "passed": 0,
                "failed": 0,
                "regressions": 0,
                "total_duration_seconds": 0.0,
            }

        total = len(self._summaries)
        passed = sum(1 for s in self._summaries if s.status == self.STATUS_PASS)
        failed = sum(1 for s in self._summaries if s.status == self.STATUS_FAIL)
        regressions = sum(1 for s in self._summaries if s.status == self.STATUS_REGRESSION)
        total_duration = sum(s.duration_seconds for s in self._summaries)
        total_alerts = sum(len(s.alerts) for s in self._summaries)

        return {
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "regressions": regressions,
            "total_duration_seconds": total_duration,
            "total_alerts": total_alerts,
            "pass_rate": passed / total if total > 0 else 0.0,
        }

    def generate_markdown_report(self) -> str:
        """Generate a Markdown formatted nightly report.

        Returns:
            Markdown report string
        """
        now = datetime.utcnow()
        stats = self.compute_summary_stats()
        overall_status = self.compute_overall_status()

        lines = [
            "# Nightly Benchmark Report",
            "",
            f"**Date:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Overall Status:** {overall_status.upper()}",
            "",
            "## Summary",
            "",
            f"- Total Benchmarks: {stats['total_benchmarks']}",
            f"- Passed: {stats['passed']}",
            f"- Failed: {stats['failed']}",
            f"- Regressions: {stats['regressions']}",
            f"- Total Duration: {stats['total_duration_seconds']:.1f}s",
            f"- Pass Rate: {stats['pass_rate']:.1%}",
            "",
            "## Benchmark Results",
            "",
            "| Benchmark | Model | Status | Duration | Alerts |",
            "|-----------|-------|--------|----------|--------|",
        ]

        for summary in self._summaries:
            alert_str = "; ".join(summary.alerts) if summary.alerts else "-"
            lines.append(
                f"| {summary.benchmark_name} | {summary.model_id} | "
                f"{summary.status} | {summary.duration_seconds:.1f}s | {alert_str} |"
            )

        lines.extend([
            "",
            "## Key Metrics",
            "",
        ])

        for summary in self._summaries:
            lines.append(f"### {summary.benchmark_name} ({summary.model_id})")
            lines.append("")
            for metric_name, value in summary.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {metric_name}: {value:.4f}")
                else:
                    lines.append(f"- {metric_name}: {value}")
            lines.append("")

        if overall_status != self.STATUS_PASS:
            lines.extend([
                "## Alerts",
                "",
            ])
            for summary in self._summaries:
                if summary.alerts:
                    lines.append(f"### {summary.benchmark_name}")
                    for alert in summary.alerts:
                        lines.append(f"- {alert}")
                    lines.append("")

        lines.extend([
            "",
            "---",
            "*Generated by DVAS Nightly Benchmark Report*",
        ])

        return "\n".join(lines)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a JSON formatted nightly report.

        Returns:
            Dictionary with report data
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": self.compute_overall_status(),
            "summary": self.compute_summary_stats(),
            "benchmarks": [s.to_dict() for s in self._summaries],
        }

    def save_report(self, report: Optional[str] = None) -> Path:
        """Save the nightly report to disk.

        Args:
            report: Optional pre-generated report string

        Returns:
            Path to saved report
        """
        if report is None:
            report = self.generate_markdown_report()

        timestamp = datetime.utcnow().strftime("%Y%m%d")
        report_path = self.report_dir / f"nightly_report_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Also save JSON version
        json_path = self.report_dir / f"nightly_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.generate_json_report(), f, indent=2)

        logger.info("Saved nightly report", path=str(report_path))
        return report_path

    def get_recent_reports(self, days: int = 7) -> List[Path]:
        """Get recent nightly reports.

        Args:
            days: Number of days to look back

        Returns:
            List of report file paths
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        reports = []

        for report_path in self.report_dir.glob("nightly_report_*.md"):
            # Extract date from filename
            try:
                date_str = report_path.stem.split("_")[-1]
                report_date = datetime.strptime(date_str, "%Y%m%d")
                if report_date >= cutoff:
                    reports.append(report_path)
            except ValueError:
                continue

        return sorted(reports)

    def compare_with_previous(
        self,
        days_back: int = 1,
    ) -> Dict[str, Any]:
        """Compare current results with previous nightly run.

        Args:
            days_back: Number of days to look back for comparison

        Returns:
            Comparison dictionary
        """
        # This is a simplified comparison
        # In practice, you'd load previous JSON reports
        previous_date = datetime.utcnow() - timedelta(days=days_back)
        prev_path = self.report_dir / f"nightly_report_{previous_date.strftime('%Y%m%d')}.json"

        if not prev_path.exists():
            return {"comparison_available": False, "reason": "No previous report found"}

        with open(prev_path, "r", encoding="utf-8") as f:
            previous = json.load(f)

        current = self.generate_json_report()

        # Compare summary stats
        prev_summary = previous.get("summary", {})
        curr_summary = current.get("summary", {})

        comparison = {
            "comparison_available": True,
            "previous_date": previous_date.isoformat(),
            "current_date": datetime.utcnow().isoformat(),
            "total_benchmarks_change": (
                curr_summary.get("total_benchmarks", 0) - prev_summary.get("total_benchmarks", 0)
            ),
            "pass_rate_change": (
                curr_summary.get("pass_rate", 0.0) - prev_summary.get("pass_rate", 0.0)
            ),
            "regressions_change": (
                curr_summary.get("regressions", 0) - prev_summary.get("regressions", 0)
            ),
        }

        return comparison

    def run_benchmark(
        self,
        model_id: str = "nightly_report",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the nightly benchmark report.

        Args:
            model_id: Identifier for this report
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with report data
        """
        logger.info("Generating nightly benchmark report")

        stats = self.compute_summary_stats()
        overall_status = self.compute_overall_status()

        metrics = {
            "total_benchmarks": stats["total_benchmarks"],
            "passed": stats["passed"],
            "failed": stats["failed"],
            "regressions": stats["regressions"],
            "pass_rate": stats["pass_rate"],
            "total_duration_seconds": stats["total_duration_seconds"],
            "total_alerts": stats["total_alerts"],
        }

        # Generate and save report
        report = self.generate_markdown_report()
        report_path = self.save_report(report)

        # Compare with previous
        comparison = self.compare_with_previous()
        if comparison.get("comparison_available"):
            metrics["pass_rate_change"] = comparison["pass_rate_change"]
            metrics["regressions_change"] = comparison["regressions_change"]

        predictions = [report]
        references = predictions

        result = BenchmarkResult(
            benchmark_name="nightly_report",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata={
                "report_path": str(report_path),
                "overall_status": overall_status,
                "comparison": comparison,
                **(metadata or {}),
            },
        )

        self._save_result(result)
        logger.info("Nightly benchmark report complete", metrics=metrics)
        return result

    def should_alert(self) -> bool:
        """Check if any alerts should be sent.

        Returns:
            True if there are failures or regressions
        """
        overall = self.compute_overall_status()
        return overall in (self.STATUS_FAIL, self.STATUS_REGRESSION)

    def get_alert_summary(self) -> Optional[str]:
        """Get a summary of alerts for notification.

        Returns:
            Alert summary string, or None if no alerts
        """
        if not self.should_alert():
            return None

        stats = self.compute_summary_stats()
        lines = [
            f"Nightly Benchmark Alert - {self.compute_overall_status().upper()}",
            "",
            f"Total: {stats['total_benchmarks']} | "
            f"Passed: {stats['passed']} | "
            f"Failed: {stats['failed']} | "
            f"Regressions: {stats['regressions']}",
            "",
            "Affected benchmarks:",
        ]

        for summary in self._summaries:
            if summary.status != self.STATUS_PASS:
                lines.append(f"- {summary.benchmark_name}: {summary.status}")
                for alert in summary.alerts:
                    lines.append(f"  * {alert}")

        return "\n".join(lines)
