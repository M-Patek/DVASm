"""Automated validation gate for student model deployment.

Provides quality gates that run benchmarks and block deployment
if quality thresholds are not met.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.models.student.benchmark import (
    BenchmarkResult,
    RegressionReport,
    StudentRegressionBenchmark,
)
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Default quality thresholds
DEFAULT_THRESHOLDS = {
    "bleu": 0.30,  # BLEU-4 >= 30%
    "rouge_l": 0.25,  # ROUGE-L >= 25%
    "cider": 1.0,  # CIDEr >= 1.0
    "success_rate": 0.95,  # 95% inference success rate
}


@dataclass
class ValidationGateResult:
    """Result of a validation gate check."""

    passed: bool
    gate_name: str
    metrics: Dict[str, float]
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_name": self.gate_name,
            "metrics": self.metrics,
            "failures": self.failures,
            "warnings": self.warnings,
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'=' * 60}")
        print(f"VALIDATION GATE: {self.gate_name} — {status}")
        print(f"{'=' * 60}")

        print("\n--- METRICS ---")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")

        if self.failures:
            print("\n--- FAILURES ---")
            for failure in self.failures:
                print(f"  ❌ {failure}")

        if self.warnings:
            print("\n--- WARNINGS ---")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        print(f"{'=' * 60}\n")


class ValidationGate:
    """Automated validation gate for model deployment.

    Runs benchmarks and quality checks, blocking deployment if
    thresholds are not met.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        benchmark_dir: Optional[Path] = None,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.benchmark_dir = benchmark_dir or Path("outputs/benchmarks")
        self._benchmark = StudentRegressionBenchmark(
            self.benchmark_dir,
            regression_threshold=0.05,
        )

    def check_benchmark_gate(
        self,
        benchmark_result: BenchmarkResult,
    ) -> ValidationGateResult:
        """Check if benchmark result meets quality thresholds.

        Args:
            benchmark_result: Result from benchmark evaluation

        Returns:
            ValidationGateResult with pass/fail status
        """
        metrics = benchmark_result.metrics
        failures = []
        warnings_list = []

        # Check each threshold
        for metric_name, threshold in self.thresholds.items():
            if metric_name not in metrics:
                warnings_list.append(f"Metric '{metric_name}' not found in benchmark result")
                continue

            value = metrics[metric_name]
            if value < threshold:
                failures.append(f"{metric_name}: {value:.4f} < threshold {threshold:.4f}")

        return ValidationGateResult(
            passed=len(failures) == 0,
            gate_name="benchmark_quality",
            metrics=metrics,
            failures=failures,
            warnings=warnings_list,
        )

    def check_regression_gate(
        self,
        regression_report: RegressionReport,
    ) -> ValidationGateResult:
        """Check if regression report shows acceptable changes.

        Args:
            regression_report: Report comparing current to baseline

        Returns:
            ValidationGateResult with pass/fail status
        """
        failures = []
        warnings_list = []

        if regression_report.significant_regressions:
            for metric in regression_report.significant_regressions:
                change = regression_report.metric_changes.get(metric, 0)
                failures.append(
                    f"{metric}: regressed by {change:+.1%} "
                    f"(threshold: {regression_report.threshold:.1%})"
                )

        if regression_report.significant_improvements:
            for metric in regression_report.significant_improvements:
                change = regression_report.metric_changes.get(metric, 0)
                warnings_list.append(f"{metric}: improved by {change:+.1%}")

        return ValidationGateResult(
            passed=len(failures) == 0,
            gate_name="regression",
            metrics=regression_report.current_metrics,
            failures=failures,
            warnings=warnings_list,
        )

    def run_full_validation(
        self,
        model_path: Path,
        benchmark_name: str = "student_regression",
    ) -> List[ValidationGateResult]:
        """Run full validation suite on a model.

        Args:
            model_path: Path to model checkpoint
            benchmark_name: Name of benchmark to run

        Returns:
            List of validation results (one per gate)
        """
        results = []

        # Gate 1: Benchmark quality
        logger.info("Running benchmark quality gate", benchmark=benchmark_name)
        # Note: In practice, this would load the model and run inference
        # For now, we check if a benchmark result exists
        benchmark_result = self._load_latest_benchmark_result(benchmark_name)
        if benchmark_result:
            gate_result = self.check_benchmark_gate(benchmark_result)
            results.append(gate_result)
        else:
            logger.warning("No benchmark result found", benchmark=benchmark_name)
            results.append(
                ValidationGateResult(
                    passed=False,
                    gate_name="benchmark_quality",
                    metrics={},
                    failures=["No benchmark result found"],
                )
            )

        # Gate 2: Regression check (if baseline exists)
        logger.info("Running regression gate")
        baseline = self._benchmark.load_baseline(benchmark_name)
        if baseline and benchmark_result:
            regression_report = self._benchmark.compare_to_baseline(benchmark_result)
            gate_result = self.check_regression_gate(regression_report)
            results.append(gate_result)
        else:
            logger.info("Skipping regression gate (no baseline)")

        return results

    def _load_latest_benchmark_result(
        self,
        benchmark_name: str,
    ) -> Optional[BenchmarkResult]:
        """Load the most recent benchmark result."""
        results = self._benchmark.get_benchmark_history(benchmark_name)
        if results:
            return results[-1]
        return None

    def save_validation_report(
        self,
        results: List[ValidationGateResult],
        output_path: Path,
    ) -> None:
        """Save validation results to a JSON report.

        Args:
            results: List of validation gate results
            output_path: Path to save report
        """
        report = {
            "overall_passed": all(r.passed for r in results),
            "gates": [r.to_dict() for r in results],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Validation report saved", path=str(output_path))


def run_validation_cli(
    model_path: str,
    benchmark_name: str = "student_regression",
    thresholds: Optional[Dict[str, float]] = None,
    benchmark_dir: Optional[str] = None,
) -> int:
    """Run validation gates from CLI and exit with appropriate code.

    Args:
        model_path: Path to model checkpoint
        benchmark_name: Name of benchmark to evaluate against
        thresholds: Optional custom thresholds
        benchmark_dir: Optional benchmark directory

    Returns:
        Exit code (0 for pass, 1 for fail)
    """
    gate = ValidationGate(
        thresholds=thresholds,
        benchmark_dir=Path(benchmark_dir) if benchmark_dir else None,
    )

    print(f"Running validation gates for: {model_path}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Thresholds: {gate.thresholds}")
    print("-" * 60)

    results = gate.run_full_validation(
        Path(model_path),
        benchmark_name=benchmark_name,
    )

    # Print results
    for result in results:
        result.print_summary()

    # Save report
    report_path = Path(model_path) / "validation_report.json"
    gate.save_validation_report(results, report_path)

    # Exit with appropriate code
    overall_passed = all(r.passed for r in results)
    if overall_passed:
        print("All validation gates PASSED")
        return 0
    else:
        print("Some validation gates FAILED")
        return 1


def main() -> None:
    """CLI entry point for validation gate."""
    import argparse

    parser = argparse.ArgumentParser(description="Run validation gates for model deployment")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--benchmark", type=str, default="student_regression", help="Benchmark name"
    )
    parser.add_argument("--benchmark-dir", type=str, help="Benchmark directory")
    parser.add_argument("--threshold-bleu", type=float, help="BLEU threshold")
    parser.add_argument("--threshold-rouge", type=float, help="ROUGE-L threshold")
    parser.add_argument("--threshold-cider", type=float, help="CIDEr threshold")
    parser.add_argument("--threshold-success", type=float, help="Success rate threshold")

    args = parser.parse_args()

    # Build custom thresholds from CLI args
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.threshold_bleu is not None:
        thresholds["bleu"] = args.threshold_bleu
    if args.threshold_rouge is not None:
        thresholds["rouge_l"] = args.threshold_rouge
    if args.threshold_cider is not None:
        thresholds["cider"] = args.threshold_cider
    if args.threshold_success is not None:
        thresholds["success_rate"] = args.threshold_success

    exit_code = run_validation_cli(
        model_path=args.model_path,
        benchmark_name=args.benchmark,
        thresholds=thresholds,
        benchmark_dir=args.benchmark_dir,
    )
    sys.exit(exit_code)
