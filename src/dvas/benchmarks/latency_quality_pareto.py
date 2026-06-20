"""Latency/quality Pareto chart for latency-quality tradeoff analysis.

Analyzes the tradeoff between inference latency and annotation quality,
identifying Pareto-optimal models and generating frontier data.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyQualityPoint:
    """Data point for latency-quality analysis.

    Attributes:
        model_id: Model identifier
        latency_ms: Latency in milliseconds per sample
        quality_score: Quality score (0-100)
        cost_per_sample: Optional cost in USD per sample
        metadata: Additional metadata
    """

    model_id: str
    latency_ms: float
    quality_score: float
    cost_per_sample: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "quality_score": self.quality_score,
            "cost_per_sample": self.cost_per_sample,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencyQualityPoint":
        return cls(
            model_id=data["model_id"],
            latency_ms=data["latency_ms"],
            quality_score=data["quality_score"],
            cost_per_sample=data.get("cost_per_sample", 0.0),
            metadata=data.get("metadata", {}),
        )


class LatencyQualityPareto(BaseBenchmark):
    """Latency/quality Pareto analysis.

    Analyzes the tradeoff between inference latency and annotation quality,
    identifying Pareto-optimal models and generating frontier data.

    Args:
        benchmark_dir: Directory for storing benchmark data
    """

    def __init__(self, benchmark_dir: Union[str, Path]):
        super().__init__(benchmark_dir, "latency_quality_pareto")
        self._points: List[LatencyQualityPoint] = []

    def add_point(self, point: LatencyQualityPoint) -> None:
        """Add a data point to the analysis.

        Args:
            point: LatencyQualityPoint to add
        """
        self._points.append(point)
        logger.info(
            "Added latency/quality point",
            model=point.model_id,
            latency=point.latency_ms,
            quality=point.quality_score,
        )

    def add_model(
        self,
        model_id: str,
        latency_ms: float,
        quality_score: float,
        cost_per_sample: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a model to the analysis.

        Args:
            model_id: Model identifier
            latency_ms: Latency in milliseconds per sample
            quality_score: Quality score (0-100)
            cost_per_sample: Optional cost in USD per sample
            metadata: Additional metadata
        """
        point = LatencyQualityPoint(
            model_id=model_id,
            latency_ms=latency_ms,
            quality_score=quality_score,
            cost_per_sample=cost_per_sample,
            metadata=metadata or {},
        )
        self.add_point(point)

    def compute_pareto_frontier(self) -> List[LatencyQualityPoint]:
        """Compute the Pareto-optimal frontier.

        A model is Pareto-optimal if no other model has both
        lower latency and higher quality.

        Returns:
            List of Pareto-optimal points sorted by latency
        """
        if not self._points:
            return []

        # Sort by latency (ascending)
        sorted_points = sorted(self._points, key=lambda p: p.latency_ms)

        pareto = []
        max_quality = -1.0

        for point in sorted_points:
            if point.quality_score > max_quality:
                pareto.append(point)
                max_quality = point.quality_score

        logger.info("Computed latency/quality Pareto frontier", n_points=len(pareto))
        return pareto

    def compute_speed_quality_ratio(self) -> Dict[str, float]:
        """Compute quality-per-millisecond ratio for each model.

        Returns:
            Dictionary mapping model ID to speed-quality ratio
        """
        if not self._points:
            return {}

        ratios = {}
        for point in self._points:
            if point.latency_ms > 0:
                ratios[point.model_id] = point.quality_score / point.latency_ms
            else:
                ratios[point.model_id] = float("inf")

        return ratios

    def compute_auc(self) -> float:
        """Compute area under the Pareto frontier curve.

        Higher AUC indicates better overall latency-quality tradeoff.

        Returns:
            Area under the Pareto frontier
        """
        frontier = self.compute_pareto_frontier()
        if len(frontier) < 2:
            return 0.0

        # Compute area using trapezoidal rule
        auc = 0.0
        for i in range(len(frontier) - 1):
            latency_diff = frontier[i + 1].latency_ms - frontier[i].latency_ms
            avg_quality = (frontier[i].quality_score + frontier[i + 1].quality_score) / 2
            auc += latency_diff * avg_quality

        return auc

    def find_best_model_for_latency_budget(
        self,
        max_latency_ms: float,
        min_quality: float = 0.0,
    ) -> Optional[str]:
        """Find the best quality model within a latency budget.

        Args:
            max_latency_ms: Maximum latency in milliseconds
            min_quality: Minimum quality score required

        Returns:
            Best model ID within latency budget, or None if none qualify
        """
        candidates = [
            p for p in self._points
            if p.latency_ms <= max_latency_ms and p.quality_score >= min_quality
        ]

        if not candidates:
            return None

        best = max(candidates, key=lambda p: p.quality_score)
        return best.model_id

    def find_fastest_model_for_quality(
        self,
        target_quality: float,
    ) -> Optional[str]:
        """Find the fastest model meeting a quality target.

        Args:
            target_quality: Minimum quality score required

        Returns:
            Fastest model ID meeting quality target, or None if none qualify
        """
        candidates = [p for p in self._points if p.quality_score >= target_quality]

        if not candidates:
            return None

        fastest = min(candidates, key=lambda p: p.latency_ms)
        return fastest.model_id

    def compute_tradeoff_slope(
        self,
        model_a: str,
        model_b: str,
    ) -> Optional[float]:
        """Compute the marginal quality gain per ms between two models.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Returns:
            Slope (delta_quality / delta_latency), or None if models not found
        """
        point_a = next((p for p in self._points if p.model_id == model_a), None)
        point_b = next((p for p in self._points if p.model_id == model_b), None)

        if point_a is None or point_b is None:
            return None

        delta_latency = point_b.latency_ms - point_a.latency_ms
        delta_quality = point_b.quality_score - point_a.quality_score

        if delta_latency == 0:
            return float("inf") if delta_quality != 0 else 0.0

        return delta_quality / delta_latency

    def compute_speedup_vs_quality_loss(
        self,
        baseline_model: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute speedup and quality loss relative to a baseline.

        Args:
            baseline_model: Baseline model ID

        Returns:
            Dictionary mapping model ID to speedup and quality loss
        """
        baseline = next((p for p in self._points if p.model_id == baseline_model), None)
        if baseline is None:
            return {}

        results = {}
        for point in self._points:
            if point.model_id == baseline_model:
                continue

            speedup = baseline.latency_ms / point.latency_ms if point.latency_ms > 0 else float("inf")
            quality_loss = baseline.quality_score - point.quality_score

            results[point.model_id] = {
                "speedup": speedup,
                "quality_loss": quality_loss,
                "quality_loss_percent": (quality_loss / baseline.quality_score * 100) if baseline.quality_score > 0 else 0.0,
            }

        return results

    def generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for plotting a latency/quality Pareto chart.

        Returns:
            Dictionary with all points and Pareto frontier for plotting
        """
        frontier = self.compute_pareto_frontier()

        return {
            "all_points": [p.to_dict() for p in self._points],
            "pareto_frontier": [p.to_dict() for p in frontier],
            "pareto_model_ids": [p.model_id for p in frontier],
            "speed_quality_ratios": self.compute_speed_quality_ratio(),
            "auc": self.compute_auc(),
        }

    def run_benchmark(
        self,
        model_id: str = "latency_quality_analysis",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the latency/quality Pareto benchmark.

        Args:
            model_id: Identifier for this benchmark run
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with Pareto analysis data
        """
        if not self._points:
            raise ValueError("No data points added. Use add_point() or add_model() first.")

        logger.info(
            "Running latency/quality Pareto benchmark",
            n_points=len(self._points),
        )

        frontier = self.compute_pareto_frontier()
        ratios = self.compute_speed_quality_ratio()
        auc = self.compute_auc()

        metrics = {
            "n_models": len(self._points),
            "n_pareto_optimal": len(frontier),
            "pareto_auc": auc,
            "min_latency_ms": min(p.latency_ms for p in self._points),
            "max_latency_ms": max(p.latency_ms for p in self._points),
            "min_quality": min(p.quality_score for p in self._points),
            "max_quality": max(p.quality_score for p in self._points),
        }

        # Add speed-quality ratios
        for model_id_key, ratio in ratios.items():
            metrics[f"ratio_{model_id_key}"] = ratio if ratio != float("inf") else 9999.0

        # Add frontier point metrics
        for i, point in enumerate(frontier):
            metrics[f"pareto_{i + 1}_model"] = point.model_id
            metrics[f"pareto_{i + 1}_latency_ms"] = point.latency_ms
            metrics[f"pareto_{i + 1}_quality"] = point.quality_score

        chart_data = self.generate_chart_data()
        predictions = [json.dumps(chart_data)]
        references = predictions

        result = BenchmarkResult(
            benchmark_name="latency_quality_pareto",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Latency/quality Pareto benchmark complete", metrics=metrics)
        return result

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """Export Pareto analysis to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = self.generate_chart_data()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported latency/quality Pareto chart data", path=str(output_path))
