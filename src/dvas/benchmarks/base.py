"""Base classes for benchmark infrastructure.

Provides shared data structures and utilities used across all benchmark modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        benchmark_name: Name of the benchmark
        model_id: Identifier for the model evaluated
        timestamp: When the benchmark was run
        metrics: Dictionary of metric names to scores
        predictions: List of model predictions
        references: List of ground truth references
        metadata: Additional benchmark metadata
    """

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
        """Convert to dictionary for serialization."""
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
        """Create from dictionary."""
        return cls(
            benchmark_name=data["benchmark_name"],
            model_id=data["model_id"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            predictions=data["predictions"],
            references=data["references"],
            metadata=data.get("metadata", {}),
        )

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a specific metric by name."""
        return self.metrics.get(name, default)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results for a model.

    Attributes:
        model_id: Identifier for the model
        results: List of benchmark results
        aggregated_metrics: Combined metrics across benchmarks
    """

    model_id: str
    results: List[BenchmarkResult] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
        self._recompute_aggregates()

    def _recompute_aggregates(self) -> None:
        """Recompute aggregated metrics from all results."""
        if not self.results:
            self.aggregated_metrics = {}
            return

        all_metrics: Dict[str, List[float]] = {}
        for result in self.results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        self.aggregated_metrics = {
            name: float(np.mean(values)) for name, values in all_metrics.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "results": [r.to_dict() for r in self.results],
            "aggregated_metrics": self.aggregated_metrics,
        }


class BaseBenchmark:
    """Base class for all benchmarks.

    Provides common functionality for running benchmarks,
    computing metrics, and persisting results.

    Args:
        benchmark_dir: Directory for storing benchmark data
        name: Name of the benchmark
    """

    def __init__(self, benchmark_dir: Union[str, Path], name: str):
        self.benchmark_dir = Path(benchmark_dir)
        self.name = name
        self.results_dir = self.benchmark_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _save_result(self, result: BenchmarkResult) -> Path:
        """Save a benchmark result to disk.

        Args:
            result: Benchmark result to save

        Returns:
            Path to saved file
        """
        import json

        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        result_path = (
            self.results_dir / f"{result.benchmark_name}_{result.model_id}_{timestamp}.json"
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        return result_path

    def _load_results(self, benchmark_name: Optional[str] = None) -> List[BenchmarkResult]:
        """Load benchmark results from disk.

        Args:
            benchmark_name: Optional filter by benchmark name

        Returns:
            List of benchmark results
        """
        import json

        results = []
        for result_path in self.results_dir.glob("*.json"):
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = BenchmarkResult.from_dict(data)
            if benchmark_name is None or result.benchmark_name == benchmark_name:
                results.append(result)
        return results

    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute average BLEU-4 score.

        Args:
            predictions: Predicted texts
            references: Reference texts

        Returns:
            Average BLEU-4 score
        """
        from dvas.models.evaluator.metrics import MetricsCalculator

        if not predictions or not references:
            return 0.0

        calc = MetricsCalculator()
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                bleu_scores = calc.bleu(ref, pred)
                scores.append(bleu_scores.get("bleu_4", 0.0))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    def compute_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """Compute average ROUGE-L score.

        Args:
            predictions: Predicted texts
            references: Reference texts

        Returns:
            Average ROUGE-L F1 score
        """
        from dvas.models.evaluator.metrics import MetricsCalculator

        if not predictions or not references:
            return 0.0

        calc = MetricsCalculator()
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                rouge_scores = calc.rouge(ref, pred)
                scores.append(rouge_scores.get("rougeL_f", 0.0))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    def compute_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy.

        Args:
            predictions: Predicted texts
            references: Reference texts

        Returns:
            Exact match accuracy
        """
        if not predictions or not references:
            return 0.0
        matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return matches / len(predictions)
