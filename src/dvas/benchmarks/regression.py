"""Regression benchmark for tracking model performance over time.

Detects performance regressions by comparing current model
results against historical baselines.
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
class RegressionAlert:
    """Alert for detected regression.

    Attributes:
        metric_name: Name of regressed metric
        baseline_value: Baseline metric value
        current_value: Current metric value
        change_percent: Percentage change
        severity: Alert severity ("warning", "critical")
        timestamp: When the regression was detected
    """

    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    severity: str = "warning"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change_percent": self.change_percent,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
        }


class RegressionBenchmark(BaseBenchmark):
    """Regression tracking benchmark.

    Tracks model performance over time and detects regressions
    by comparing against historical baselines.

    Args:
        benchmark_dir: Directory for storing benchmark data
        regression_threshold: Threshold for flagging regression (default 0.05 = 5%)
        critical_threshold: Threshold for critical alerts (default 0.10 = 10%)
    """

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        regression_threshold: float = 0.05,
        critical_threshold: float = 0.10,
    ):
        super().__init__(benchmark_dir, "regression")
        self.regression_threshold = regression_threshold
        self.critical_threshold = critical_threshold
        self.baselines_dir = self.benchmark_dir / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def set_baseline(
        self,
        benchmark_name: str,
        model_id: str,
        metrics: Dict[str, float],
    ) -> None:
        """Set baseline metrics for a model.

        Args:
            benchmark_name: Name of the benchmark
            model_id: Model identifier
            metrics: Baseline metric values
        """
        baseline_data = {
            "benchmark_name": benchmark_name,
            "model_id": model_id,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        baseline_path = self.baselines_dir / f"{benchmark_name}_{model_id}_baseline.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)

        logger.info(
            "Set baseline",
            benchmark=benchmark_name,
            model=model_id,
            metrics=list(metrics.keys()),
        )

    def load_baseline(
        self,
        benchmark_name: str,
        model_id: str,
    ) -> Optional[Dict[str, float]]:
        """Load baseline metrics for a model.

        Args:
            benchmark_name: Name of the benchmark
            model_id: Model identifier

        Returns:
            Baseline metrics, or None if not found
        """
        baseline_path = self.baselines_dir / f"{benchmark_name}_{model_id}_baseline.json"
        if not baseline_path.exists():
            return None

        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("metrics")

    def detect_regression(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> List[RegressionAlert]:
        """Detect regressions by comparing current to baseline.

        Args:
            current_metrics: Current metric values
            baseline_metrics: Baseline metric values

        Returns:
            List of regression alerts
        """
        alerts = []

        all_metrics = set(current_metrics.keys()) | set(baseline_metrics.keys())

        for metric in all_metrics:
            current = current_metrics.get(metric, 0.0)
            baseline = baseline_metrics.get(metric, 0.0)

            if baseline == 0:
                continue

            change = (current - baseline) / baseline

            if change < -self.regression_threshold:
                severity = "critical" if abs(change) > self.critical_threshold else "warning"
                alerts.append(
                    RegressionAlert(
                        metric_name=metric,
                        baseline_value=baseline,
                        current_value=current,
                        change_percent=change * 100,
                        severity=severity,
                    )
                )

        return alerts

    def detect_improvement(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Detect significant improvements.

        Args:
            current_metrics: Current metric values
            baseline_metrics: Baseline metric values

        Returns:
            List of improvement records
        """
        improvements = []

        all_metrics = set(current_metrics.keys()) | set(baseline_metrics.keys())

        for metric in all_metrics:
            current = current_metrics.get(metric, 0.0)
            baseline = baseline_metrics.get(metric, 0.0)

            if baseline == 0:
                continue

            change = (current - baseline) / baseline

            if change > self.regression_threshold:
                improvements.append(
                    {
                        "metric_name": metric,
                        "baseline_value": baseline,
                        "current_value": current,
                        "change_percent": change * 100,
                    }
                )

        return improvements

    def compute_trend(
        self,
        metric_name: str,
        benchmark_name: str,
        model_id: str,
        window_size: int = 7,
    ) -> Dict[str, float]:
        """Compute trend for a metric over time.

        Args:
            metric_name: Name of the metric
            benchmark_name: Name of the benchmark
            model_id: Model identifier
            window_size: Number of recent results to include

        Returns:
            Dictionary with trend statistics
        """
        results = self._load_results(benchmark_name)
        model_results = [r for r in results if r.model_id == model_id]

        if len(model_results) < 2:
            return {"slope": 0.0, "trend": "insufficient_data"}

        # Sort by timestamp
        model_results.sort(key=lambda r: r.timestamp)

        # Take last window_size results
        recent = model_results[-window_size:]

        values = [r.get_metric(metric_name) for r in recent]
        x = np.arange(len(values))

        # Linear regression
        if len(values) < 2 or np.std(values) == 0:
            return {"slope": 0.0, "trend": "stable"}

        slope = np.polyfit(x, values, 1)[0]

        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "degrading"

        return {
            "slope": float(slope),
            "trend": trend,
            "latest_value": values[-1],
            "earliest_value": values[0],
            "change": values[-1] - values[0],
            "n_points": len(values),
        }

    def get_historical_results(
        self,
        benchmark_name: str,
        model_id: str,
        days: int = 30,
    ) -> List[BenchmarkResult]:
        """Get historical results for a model.

        Args:
            benchmark_name: Name of the benchmark
            model_id: Model identifier
            days: Number of days to look back

        Returns:
            List of historical results
        """
        results = self._load_results(benchmark_name)
        cutoff = datetime.utcnow() - timedelta(days=days)

        model_results = [r for r in results if r.model_id == model_id and r.timestamp >= cutoff]

        model_results.sort(key=lambda r: r.timestamp)
        return model_results

    def run_benchmark(
        self,
        model_id: str,
        benchmark_name: str,
        current_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the regression benchmark.

        Args:
            model_id: Identifier for the model
            benchmark_name: Name of the benchmark
            current_metrics: Current metric values
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with regression analysis
        """
        logger.info(
            "Running regression benchmark",
            model=model_id,
            benchmark=benchmark_name,
        )

        # Load baseline
        baseline = self.load_baseline(benchmark_name, model_id)

        if baseline is None:
            # Set baseline if not exists
            self.set_baseline(benchmark_name, model_id, current_metrics)
            baseline = current_metrics.copy()
            logger.info("Set initial baseline", model=model_id, benchmark=benchmark_name)

        # Detect regressions and improvements
        regressions = self.detect_regression(current_metrics, baseline)
        improvements = self.detect_improvement(current_metrics, baseline)

        # Build metrics
        metrics = {
            "n_regressions": len(regressions),
            "n_improvements": len(improvements),
        }

        for alert in regressions:
            metrics[f"regression_{alert.metric_name}"] = alert.change_percent

        for imp in improvements:
            metrics[f"improvement_{imp['metric_name']}"] = imp["change_percent"]

        # Add current values
        for metric_name, value in current_metrics.items():
            metrics[f"current_{metric_name}"] = value
            metrics[f"baseline_{metric_name}"] = baseline.get(metric_name, 0.0)

        # Compute trends
        for metric_name in current_metrics.keys():
            trend = self.compute_trend(metric_name, benchmark_name, model_id)
            metrics[f"trend_{metric_name}_slope"] = trend["slope"]
            metrics[f"trend_{metric_name}_direction"] = trend["trend"]

        predictions = [json.dumps(current_metrics)]
        references = [json.dumps(baseline)]

        result = BenchmarkResult(
            benchmark_name=f"regression_{benchmark_name}",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata={
                "regressions": [a.to_dict() for a in regressions],
                "improvements": improvements,
                **(metadata or {}),
            },
        )

        self._save_result(result)

        if regressions:
            logger.warning(
                "Regressions detected",
                n_regressions=len(regressions),
                critical=sum(1 for a in regressions if a.severity == "critical"),
            )

        logger.info("Regression benchmark complete", metrics=metrics)
        return result

    def generate_summary_report(
        self,
        model_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Generate a summary report of recent regressions.

        Args:
            model_id: Model identifier
            days: Number of days to include

        Returns:
            Summary report dictionary
        """
        results = self._load_results()
        cutoff = datetime.utcnow() - timedelta(days=days)

        model_results = [r for r in results if r.model_id == model_id and r.timestamp >= cutoff]

        if not model_results:
            return {
                "model_id": model_id,
                "period_days": days,
                "n_results": 0,
                "regressions": [],
            }

        # Collect regressions
        all_regressions = []
        for result in model_results:
            if "regressions" in result.metadata:
                all_regressions.extend(result.metadata["regressions"])

        return {
            "model_id": model_id,
            "period_days": days,
            "n_results": len(model_results),
            "regressions": all_regressions,
            "latest_run": model_results[-1].timestamp.isoformat() if model_results else None,
        }
