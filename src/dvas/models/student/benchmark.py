"""Regression benchmark for student models.

Provides standardized evaluation to detect performance regressions
across model versions and training runs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    benchmark_name: str
    model_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    predictions: List[str]
    references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "predictions": self.predictions,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        return cls(
            benchmark_name=data["benchmark_name"],
            model_id=data["model_id"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            predictions=data["predictions"],
            references=data["references"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegressionReport:
    """Report comparing model to baseline."""

    benchmark_name: str
    current_model: str
    baseline_model: str
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    metric_changes: Dict[str, float]
    significant_regressions: List[str]
    significant_improvements: List[str]
    threshold: float = 0.05  # 5% change threshold

    def has_regression(self) -> bool:
        """Check if any significant regressions exist."""
        return len(self.significant_regressions) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "current_model": self.current_model,
            "baseline_model": self.baseline_model,
            "current_metrics": self.current_metrics,
            "baseline_metrics": self.baseline_metrics,
            "metric_changes": self.metric_changes,
            "significant_regressions": self.significant_regressions,
            "significant_improvements": self.significant_improvements,
            "threshold": self.threshold,
            "has_regression": self.has_regression(),
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("REGRESSION REPORT")
        print("=" * 60)
        print(f"Benchmark: {self.benchmark_name}")
        print(f"Current: {self.current_model}")
        print(f"Baseline: {self.baseline_model}")

        print("\n--- METRIC COMPARISON ---")
        for metric, change in self.metric_changes.items():
            current = self.current_metrics.get(metric, 0)
            baseline = self.baseline_metrics.get(metric, 0)
            symbol = "!" if metric in self.significant_regressions else (
                "+" if metric in self.significant_improvements else " "
            )
            print(f"{symbol} {metric}: {baseline:.3f} -> {current:.3f} ({change:+.1%})")

        if self.significant_regressions:
            print("\n--- REGRESSIONS DETECTED ---")
            for metric in self.significant_regressions:
                print(f"  - {metric}: {self.metric_changes[metric]:+.1%}")

        if self.significant_improvements:
            print("\n--- IMPROVEMENTS ---")
            for metric in self.significant_improvements:
                print(f"  + {metric}: {self.metric_changes[metric]:+.1%}")

        print("=" * 60)


class StudentRegressionBenchmark:
    """Regression benchmark suite for student models.

    Maintains a suite of test cases and compares model performance
    against baselines to detect regressions.
    """

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        regression_threshold: float = 0.05,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.regression_threshold = regression_threshold
        self.benchmarks_dir = self.benchmark_dir / "benchmarks"
        self.results_dir = self.benchmark_dir / "results"
        self.baselines_dir = self.benchmark_dir / "baselines"

        # Create directories
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def create_benchmark(
        self,
        name: str,
        test_cases: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> Path:
        """Create a new benchmark suite.

        Args:
            name: Benchmark name
            test_cases: List of test cases with video paths and references
            description: Optional description

        Returns:
            Path to saved benchmark
        """
        benchmark_data = {
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "test_cases": test_cases,
        }

        benchmark_path = self.benchmarks_dir / f"{name}.json"
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info("Created benchmark", name=name, cases=len(test_cases))
        return benchmark_path

    def load_benchmark(self, name: str) -> Dict[str, Any]:
        """Load a benchmark suite."""
        benchmark_path = self.benchmarks_dir / f"{name}.json"
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark not found: {name}")

        with open(benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(
        self,
        benchmark_name: str,
        model,
        model_id: str,
        metrics_fn=None,
    ) -> BenchmarkResult:
        """Run benchmark on a model.

        Args:
            benchmark_name: Name of benchmark to run
            model: Model to evaluate (with generate method)
            model_id: Identifier for the model
            metrics_fn: Function to compute metrics(predictions, references)

        Returns:
            BenchmarkResult with metrics
        """
        import asyncio

        benchmark = self.load_benchmark(benchmark_name)
        test_cases = benchmark["test_cases"]

        logger.info(
            "Running benchmark",
            benchmark=benchmark_name,
            model=model_id,
            cases=len(test_cases),
        )

        # Run inference
        predictions = []
        references = []

        async def run_inference():
            results = []
            for case in test_cases:
                result = await model.generate(
                    video_path=Path(case["video_path"]),
                    prompt=case.get("prompt"),
                )
                results.append(result)
            return results

        results = asyncio.run(run_inference())

        # Extract predictions and references
        pred_texts = []
        for i, result in enumerate(results):
            pred_texts.append(result.text if result.is_success() else "")
            references.append(test_cases[i].get("reference", ""))

        predictions = pred_texts

        # Compute metrics
        if metrics_fn:
            metrics = metrics_fn(predictions, references)
        else:
            metrics = self._default_metrics(predictions, references)

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata={
                "n_test_cases": len(test_cases),
                "success_rate": sum(r.is_success() for r in results) / len(results),
            },
        )

        # Save result
        self._save_result(result)

        return result

    def _default_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute default metrics."""
        from dvas.models.evaluator.metrics import MetricsCalculator

        metrics_calc = MetricsCalculator()

        if not predictions or not references:
            return {"bleu": 0.0, "rouge_l": 0.0}

        # BLEU
        try:
            bleu_scores = [metrics_calc.bleu(ref, pred) for ref, pred in zip(references, predictions)]
            bleu = np.mean([s.get("bleu_4", 0.0) for s in bleu_scores])
        except Exception:
            bleu = 0.0

        # ROUGE-L
        try:
            rouge_scores = [
                metrics_calc.rouge(ref, pred)
                for ref, pred in zip(references, predictions)
            ]
            rouge_l = np.mean([s.get("rougeL_f", 0.0) for s in rouge_scores])
        except Exception:
            rouge_l = 0.0

        return {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "avg_length": np.mean([len(p.split()) for p in predictions]),
        }

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        result_path = (
            self.results_dir /
            f"{result.benchmark_name}_{result.model_id}_{timestamp}.json"
        )

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Saved benchmark result", path=str(result_path))

    def set_baseline(
        self,
        benchmark_name: str,
        model_id: str,
        result: Optional[BenchmarkResult] = None,
    ) -> None:
        """Set baseline for a benchmark.

        Args:
            benchmark_name: Benchmark name
            model_id: Model to use as baseline
            result: Optional result to use (loads from saved if not provided)
        """
        if result is None:
            # Load most recent result for this model
            result = self._load_most_recent_result(benchmark_name, model_id)

        if result is None:
            raise ValueError(f"No results found for {model_id} on {benchmark_name}")

        baseline_path = self.baselines_dir / f"{benchmark_name}_baseline.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(
            "Set baseline",
            benchmark=benchmark_name,
            model=model_id,
        )

    def _load_most_recent_result(
        self,
        benchmark_name: str,
        model_id: str,
    ) -> Optional[BenchmarkResult]:
        """Load most recent result for model on benchmark."""
        pattern = f"{benchmark_name}_{model_id}_*.json"
        results = list(self.results_dir.glob(pattern))

        if not results:
            return None

        # Sort by modification time (newest first)
        results.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        with open(results[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        return BenchmarkResult.from_dict(data)

    def load_baseline(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Load baseline for a benchmark."""
        baseline_path = self.baselines_dir / f"{benchmark_name}_baseline.json"
        if not baseline_path.exists():
            return None

        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BenchmarkResult.from_dict(data)

    def compare_to_baseline(
        self,
        current_result: BenchmarkResult,
    ) -> RegressionReport:
        """Compare current result to baseline.

        Args:
            current_result: Current benchmark result

        Returns:
            RegressionReport with comparison
        """
        baseline = self.load_baseline(current_result.benchmark_name)

        if baseline is None:
            raise ValueError(
                f"No baseline set for {current_result.benchmark_name}"
            )

        # Compare metrics
        metric_changes = {}
        regressions = []
        improvements = []

        all_metrics = set(current_result.metrics.keys()) | set(baseline.metrics.keys())

        for metric in all_metrics:
            current = current_result.metrics.get(metric, 0)
            base = baseline.metrics.get(metric, 0)

            if base > 0:
                change = (current - base) / base
            else:
                change = 0 if current == 0 else float("inf")

            metric_changes[metric] = change

            # Check significance
            if change < -self.regression_threshold:
                regressions.append(metric)
            elif change > self.regression_threshold:
                improvements.append(metric)

        report = RegressionReport(
            benchmark_name=current_result.benchmark_name,
            current_model=current_result.model_id,
            baseline_model=baseline.model_id,
            current_metrics=current_result.metrics,
            baseline_metrics=baseline.metrics,
            metric_changes=metric_changes,
            significant_regressions=regressions,
            significant_improvements=improvements,
            threshold=self.regression_threshold,
        )

        logger.info(
            "Regression comparison complete",
            regressions=len(regressions),
            improvements=len(improvements),
        )

        return report

    def get_benchmark_history(
        self,
        benchmark_name: str,
        model_id: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """Get history of benchmark results.

        Args:
            benchmark_name: Benchmark name
            model_id: Optional model filter

        Returns:
            List of historical results
        """
        pattern = f"{benchmark_name}_*.json"
        if model_id:
            pattern = f"{benchmark_name}_{model_id}_*.json"

        result_paths = list(self.results_dir.glob(pattern))
        result_paths.sort(key=lambda p: p.stat().st_mtime)

        results = []
        for path in result_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(BenchmarkResult.from_dict(data))

        return results

    def create_default_benchmark(
        self,
        name: str = "student_regression",
        video_dir: Optional[Path] = None,
    ) -> Path:
        """Create a default regression benchmark.

        Args:
            name: Benchmark name
            video_dir: Directory with test videos

        Returns:
            Path to created benchmark
        """
        # Default test cases focusing on key capabilities
        test_cases = [
            {
                "id": "simple_action",
                "video_path": str(video_dir / "simple_action.mp4") if video_dir else "test_videos/simple_action.mp4",
                "prompt": "Describe the action in this video.",
                "reference": "A person is cutting vegetables on a cutting board.",
                "category": "action_recognition",
            },
            {
                "id": "object_interaction",
                "video_path": str(video_dir / "object_interaction.mp4") if video_dir else "test_videos/object_interaction.mp4",
                "prompt": "What objects are being used and how?",
                "reference": "The person is using a knife to chop carrots and onions.",
                "category": "object_interaction",
            },
            {
                "id": "temporal_sequence",
                "video_path": str(video_dir / "temporal_sequence.mp4") if video_dir else "test_videos/temporal_sequence.mp4",
                "prompt": "Describe the sequence of actions.",
                "reference": "First washing hands, then preparing ingredients, then cooking.",
                "category": "temporal",
            },
        ]

        return self.create_benchmark(
            name=name,
            test_cases=test_cases,
            description="Default student model regression benchmark",
        )

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        training_run_id: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint files
            training_run_id: Unique training run identifier
            step: Training step number
            metrics: Current metrics

        Returns:
            Path to saved checkpoint
        """
        from datetime import datetime

        run_dir = self.results_dir.parent / "checkpoints" / training_run_id
        target_dir = run_dir / f"checkpoint-{step}"
        target_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        if checkpoint_path.is_dir():
            for item in checkpoint_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)

        # Save checkpoint metadata
        metadata = {
            "training_run_id": training_run_id,
            "step": step,
            "metrics": metrics or {},
            "saved_at": datetime.utcnow().isoformat(),
        }

        metadata_path = target_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, indent=2)

        return target_dir

    def list_checkpoints(self, training_run_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a training run."""
        import json

        run_dir = self.results_dir.parent / "checkpoints" / training_run_id
        if not run_dir.exists():
            return []

        checkpoints = []
        for checkpoint_dir in sorted(run_dir.glob("checkpoint-*")):
            metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)

        return checkpoints


def quick_benchmark(
    model,
    test_videos: List[Path],
    references: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Quick benchmark without persistence.

    Args:
        model: Model to evaluate
        test_videos: List of test video paths
        references: Optional reference annotations

    Returns:
        Dictionary of metrics
    """
    import asyncio

    async def run():
        results = []
        for video_path in test_videos:
            result = await model.generate(video_path=video_path)
            results.append(result)
        return results

    predictions = asyncio.run(run())
    pred_texts = [p.text if p.is_success() else "" for p in predictions]

    # Simple metrics
    from dvas.models.evaluator.metrics import MetricsCalculator

    metrics_calc = MetricsCalculator()

    if references:
        bleu_scores = [metrics_calc.bleu(ref, pred) for ref, pred in zip(references, pred_texts)]
        bleu = np.mean([s.get("bleu_4", 0.0) for s in bleu_scores])
        rouge_scores = [metrics_calc.rouge(ref, pred) for ref, pred in zip(references, pred_texts)]
        rouge_l = np.mean([s.get("rougeL_f", 0.0) for s in rouge_scores])
    else:
        bleu = 0.0
        rouge_l = 0.0

    return {
        "bleu": bleu,
        "rouge_l": rouge_l,
        "success_rate": sum(p.is_success() for p in predictions) / len(predictions),
        "avg_length": np.mean([len(p.split()) for p in pred_texts]),
    }
