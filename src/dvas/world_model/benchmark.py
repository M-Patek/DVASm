"""Benchmarks for World Model annotation capabilities.

Provides standardized benchmarks for:
- State prediction accuracy
- Causal relation extraction
- Counterfactual generation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dvas.data.schemas import Action, Annotation, Segment
from dvas.utils.logging import get_logger
from dvas.world_model.annotator import WorldModelAnnotator
from dvas.world_model.quality_evaluator import WorldModelQualityEvaluator
from dvas.world_model.state_repr import WorldState

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        benchmark_name: Name of the benchmark
        metric_name: Specific metric measured
        value: Metric value
        unit: Unit of measurement
        metadata: Additional benchmark metadata
        timestamp: When benchmark was run
    """

    benchmark_name: str
    metric_name: str
    value: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkSuiteResult:
    """Complete results from a benchmark suite run.

    Attributes:
        suite_name: Name of the benchmark suite
        results: List of individual benchmark results
        summary: Aggregated statistics
        duration_seconds: Total runtime
        timestamp: When suite was run
    """

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class StatePredictionBenchmark:
    """Benchmark for state prediction capabilities.

    Tests the annotator's ability to predict future states
    given current state and action.
    """

    def __init__(
        self,
        annotator: Optional[WorldModelAnnotator] = None,
    ):
        self.annotator = annotator or WorldModelAnnotator()
        self.evaluator = WorldModelQualityEvaluator()

    async def run(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[BenchmarkResult]:
        """Run state prediction benchmark.

        Args:
            test_cases: List of test cases with:
                - initial_state: WorldState
                - action: Action
                - expected_state: WorldState (ground truth)

        Returns:
            List of benchmark results
        """
        results = []
        errors = []

        for i, test_case in enumerate(test_cases):
            try:
                initial_state = test_case["initial_state"]
                action = test_case["action"]
                expected = test_case["expected_state"]

                # Generate prediction
                predicted = await self.annotator.predict_next_state(initial_state, action)

                # Evaluate accuracy
                metrics = self.evaluator.evaluate_state_predictions([predicted], [expected])

                results.append(
                    BenchmarkResult(
                        benchmark_name="state_prediction",
                        metric_name=f"test_{i}_mae",
                        value=metrics.mae,
                        unit="meters",
                        metadata={"test_id": i},
                    )
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="state_prediction",
                        metric_name=f"test_{i}_rmse",
                        value=metrics.rmse,
                        unit="meters",
                        metadata={"test_id": i},
                    )
                )

            except Exception as e:
                logger.error("benchmark_test_failed", test_id=i, error=str(e))
                errors.append(i)

        # Add aggregate results
        if results:
            mae_values = [r.value for r in results if r.metric_name.endswith("_mae")]
            if mae_values:
                results.append(
                    BenchmarkResult(
                        benchmark_name="state_prediction",
                        metric_name="mean_mae",
                        value=float(np.mean(mae_values)),
                        unit="meters",
                        metadata={"error_count": len(errors)},
                    )
                )

        return results


class CausalRelationBenchmark:
    """Benchmark for causal relation extraction.

    Tests the annotator's ability to identify causal
    relationships between actions and outcomes.
    """

    def __init__(
        self,
        annotator: Optional[WorldModelAnnotator] = None,
    ):
        self.annotator = annotator or WorldModelAnnotator()

    async def run(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[BenchmarkResult]:
        """Run causal relation extraction benchmark.

        Args:
            test_cases: List of test cases with:
                - segment: Video segment
                - expected_causes: List of expected causal relations

        Returns:
            List of benchmark results
        """
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                segment = test_case["segment"]
                expected_causes = test_case["expected_causes"]

                # Extract causal relations
                causal_relations = await self.annotator.extract_causal_relations(segment)

                # Compute precision and recall
                extracted = set(causal_relations)
                expected = set(expected_causes)

                true_positives = len(extracted & expected)
                false_positives = len(extracted - expected)
                false_negatives = len(expected - extracted)

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="causal_relation",
                        metric_name=f"test_{i}_precision",
                        value=precision,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="causal_relation",
                        metric_name=f"test_{i}_recall",
                        value=recall,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="causal_relation",
                        metric_name=f"test_{i}_f1",
                        value=f1,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

            except Exception as e:
                logger.error("causal_benchmark_failed", test_id=i, error=str(e))

        # Add aggregate results
        f1_values = [r.value for r in results if r.metric_name.endswith("_f1")]
        if f1_values:
            results.append(
                BenchmarkResult(
                    benchmark_name="causal_relation",
                    metric_name="mean_f1",
                    value=float(np.mean(f1_values)),
                    unit="ratio",
                )
            )

        return results


class CounterfactualBenchmark:
    """Benchmark for counterfactual generation.

    Tests the annotator's ability to generate valid
    counterfactual scenarios.
    """

    def __init__(
        self,
        annotator: Optional[WorldModelAnnotator] = None,
    ):
        self.annotator = annotator or WorldModelAnnotator()
        self.evaluator = WorldModelQualityEvaluator()

    async def run(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[BenchmarkResult]:
        """Run counterfactual generation benchmark.

        Args:
            test_cases: List of test cases with:
                - segment: Video segment
                - actual_action: Action that occurred
                - alternative_actions: List of alternative actions
                - validity_annotations: Human judgments of validity

        Returns:
            List of benchmark results
        """
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                segment = test_case["segment"]
                actual_action = test_case["actual_action"]
                alternatives = test_case["alternative_actions"]

                generated_counterfactuals = []

                for alt_action in alternatives:
                    counterfactuals = await self.annotator.generate_counterfactuals(
                        segment, actual_action, [alt_action]
                    )
                    generated_counterfactuals.extend(counterfactuals)

                # Evaluate counterfactuals
                metrics = self.evaluator.evaluate_counterfactuals(
                    generated_counterfactuals,
                    Annotation(
                        id=f"benchmark_{i}",
                        video_id="benchmark",
                        video_path="benchmark.mp4",
                        metadata={
                            "fps": 30,
                            "resolution": [1920, 1080],
                            "duration": 10.0,
                            "total_frames": 300,
                        },
                    ),
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="counterfactual",
                        metric_name=f"test_{i}_plausibility",
                        value=metrics.physical_plausibility,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="counterfactual",
                        metric_name=f"test_{i}_coherence",
                        value=metrics.semantic_coherence,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

                results.append(
                    BenchmarkResult(
                        benchmark_name="counterfactual",
                        metric_name=f"test_{i}_diversity",
                        value=metrics.diversity,
                        unit="ratio",
                        metadata={"test_id": i},
                    )
                )

            except Exception as e:
                logger.error("counterfactual_benchmark_failed", test_id=i, error=str(e))

        # Add aggregate results
        plausibility_values = [r.value for r in results if r.metric_name.endswith("_plausibility")]
        if plausibility_values:
            results.append(
                BenchmarkResult(
                    benchmark_name="counterfactual",
                    metric_name="mean_plausibility",
                    value=float(np.mean(plausibility_values)),
                    unit="ratio",
                )
            )

        return results


class WorldModelAnnotationBenchmark:
    """Comprehensive benchmark suite for World Model annotation.

    Runs all world model benchmarks and generates a complete report.
    """

    def __init__(
        self,
        annotator: Optional[WorldModelAnnotator] = None,
        output_dir: Optional[Path] = None,
    ):
        self.annotator = annotator or WorldModelAnnotator()
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-benchmarks
        self.state_benchmark = StatePredictionBenchmark(self.annotator)
        self.causal_benchmark = CausalRelationBenchmark(self.annotator)
        self.counterfactual_benchmark = CounterfactualBenchmark(self.annotator)

    async def run_all(
        self,
        state_test_cases: Optional[List[Dict]] = None,
        causal_test_cases: Optional[List[Dict]] = None,
        counterfactual_test_cases: Optional[List[Dict]] = None,
    ) -> BenchmarkSuiteResult:
        """Run all benchmarks.

        Args:
            state_test_cases: Test cases for state prediction
            causal_test_cases: Test cases for causal extraction
            counterfactual_test_cases: Test cases for counterfactuals

        Returns:
            Complete benchmark suite results
        """
        start_time = time.time()
        all_results: List[BenchmarkResult] = []

        # Run state prediction benchmark
        if state_test_cases:
            logger.info("running_state_prediction_benchmark", count=len(state_test_cases))
            state_results = await self.state_benchmark.run(state_test_cases)
            all_results.extend(state_results)

        # Run causal relation benchmark
        if causal_test_cases:
            logger.info("running_causal_relation_benchmark", count=len(causal_test_cases))
            causal_results = await self.causal_benchmark.run(causal_test_cases)
            all_results.extend(causal_results)

        # Run counterfactual benchmark
        if counterfactual_test_cases:
            logger.info("running_counterfactual_benchmark", count=len(counterfactual_test_cases))
            cf_results = await self.counterfactual_benchmark.run(counterfactual_test_cases)
            all_results.extend(cf_results)

        duration = time.time() - start_time

        # Compute summary statistics
        summary = self._compute_summary(all_results)

        suite_result = BenchmarkSuiteResult(
            suite_name="world_model_annotation",
            results=all_results,
            summary=summary,
            duration_seconds=duration,
        )

        logger.info(
            "benchmark_suite_complete",
            duration=duration,
            result_count=len(all_results),
        )

        return suite_result

    async def run_quick_test(self) -> BenchmarkSuiteResult:
        """Run quick test with synthetic data.

        Useful for verifying the benchmark pipeline works.
        """
        # Create minimal test cases
        state_test_cases = [
            {
                "initial_state": WorldState(timestamp=0.0),
                "action": Action(verb="push", noun="block"),
                "expected_state": WorldState(timestamp=1.0),
            }
        ]

        causal_test_cases = [
            {
                "segment": Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Push the block",
                    actions=[Action(verb="push", noun="block")],
                ),
                "expected_causes": ["push causes movement"],
            }
        ]

        counterfactual_test_cases = [
            {
                "segment": Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Push the block",
                ),
                "actual_action": Action(verb="push", noun="block"),
                "alternative_actions": [Action(verb="pull", noun="block")],
            }
        ]

        return await self.run_all(
            state_test_cases=state_test_cases,
            causal_test_cases=causal_test_cases,
            counterfactual_test_cases=counterfactual_test_cases,
        )

    def save_results(
        self,
        results: BenchmarkSuiteResult,
        filename: Optional[str] = None,
    ) -> Path:
        """Save benchmark results to file.

        Args:
            results: Benchmark results to save
            filename: Output filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_{results.suite_name}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results.to_json())

        logger.info("benchmark_results_saved", path=str(output_path))
        return output_path

    def _compute_summary(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        summary = {
            "total_benchmarks": len(results),
            "benchmarks_by_type": {},
            "overall_scores": {},
        }

        # Group by benchmark type
        by_type: Dict[str, List[float]] = {}
        for r in results:
            if r.benchmark_name not in by_type:
                by_type[r.benchmark_name] = []
            by_type[r.benchmark_name].append(r.value)

        # Compute statistics per type
        for name, values in by_type.items():
            summary["benchmarks_by_type"][name] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Overall score (average of means)
        if by_type:
            overall = np.mean([np.mean(values) for values in by_type.values()])
            summary["overall_score"] = float(overall)

        return summary

    def generate_report(
        self,
        results: BenchmarkSuiteResult,
    ) -> str:
        """Generate human-readable benchmark report.

        Args:
            results: Benchmark results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "World Model Annotation Benchmark Report",
            "=" * 60,
            f"Suite: {results.suite_name}",
            f"Duration: {results.duration_seconds:.2f}s",
            f"Total Results: {len(results.results)}",
            "",
            "Summary:",
            "-" * 40,
        ]

        for metric, value in results.summary.items():
            if isinstance(value, dict):
                lines.append(f"\n{metric}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for kk, vv in v.items():
                            lines.append(
                                f"    {kk}: {vv:.4f}"
                                if isinstance(vv, float)
                                else f"    {kk}: {vv}"
                            )
                    else:
                        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                lines.append(
                    f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}"
                )

        lines.extend(
            [
                "",
                "Detailed Results:",
                "-" * 40,
            ]
        )

        # Group by benchmark
        by_benchmark: Dict[str, List[BenchmarkResult]] = {}
        for r in results.results:
            if r.benchmark_name not in by_benchmark:
                by_benchmark[r.benchmark_name] = []
            by_benchmark[r.benchmark_name].append(r)

        for name, benchmarks in by_benchmark.items():
            lines.append(f"\n{name}:")
            for b in benchmarks[:5]:  # Show first 5
                lines.append(f"  {b.metric_name}: {b.value:.4f} {b.unit}")
            if len(benchmarks) > 5:
                lines.append(f"  ... and {len(benchmarks) - 5} more")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


# Convenience functions for running benchmarks


async def run_benchmarks(
    annotator: Optional[WorldModelAnnotator] = None,
    output_dir: Optional[Path] = None,
    quick_test: bool = False,
) -> BenchmarkSuiteResult:
    """Run world model benchmarks.

    Args:
        annotator: Annotator to benchmark (default: new instance)
        output_dir: Directory for output files
        quick_test: Run with synthetic test data

    Returns:
        Benchmark results
    """
    benchmark = WorldModelAnnotationBenchmark(
        annotator=annotator,
        output_dir=output_dir,
    )

    if quick_test:
        results = await benchmark.run_quick_test()
    else:
        results = await benchmark.run_all()

    # Save results
    benchmark.save_results(results)

    # Print report
    print(benchmark.generate_report(results))

    return results


def load_benchmark_results(path: Path) -> BenchmarkSuiteResult:
    """Load benchmark results from file.

    Args:
        path: Path to JSON results file

    Returns:
        BenchmarkSuiteResult
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [
        BenchmarkResult(
            benchmark_name=r["benchmark_name"],
            metric_name=r["metric_name"],
            value=r["value"],
            unit=r.get("unit", ""),
            metadata=r.get("metadata", {}),
            timestamp=r.get("timestamp", 0),
        )
        for r in data["results"]
    ]

    return BenchmarkSuiteResult(
        suite_name=data["suite_name"],
        results=results,
        summary=data.get("summary", {}),
        duration_seconds=data.get("duration_seconds", 0),
        timestamp=data.get("timestamp", 0),
    )
