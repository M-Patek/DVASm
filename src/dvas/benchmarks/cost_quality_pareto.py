"""Cost/quality Pareto chart for cost-quality tradeoff analysis.

Analyzes the tradeoff between inference cost and annotation quality,
identifying Pareto-optimal models and generating frontier data.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CostQualityPoint:
    """Data point for cost-quality analysis.

    Attributes:
        model_id: Model identifier
        cost_per_sample: Cost in USD per sample
        quality_score: Quality score (0-100)
        latency_ms: Optional latency in milliseconds
        metadata: Additional metadata
    """

    model_id: str
    cost_per_sample: float
    quality_score: float
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "cost_per_sample": self.cost_per_sample,
            "quality_score": self.quality_score,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostQualityPoint":
        return cls(
            model_id=data["model_id"],
            cost_per_sample=data["cost_per_sample"],
            quality_score=data["quality_score"],
            latency_ms=data.get("latency_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


class CostQualityPareto(BaseBenchmark):
    """Cost/quality Pareto analysis.

    Analyzes the tradeoff between inference cost and annotation quality,
    identifying Pareto-optimal models and generating frontier data.

    Args:
        benchmark_dir: Directory for storing benchmark data
    """

    def __init__(self, benchmark_dir: Union[str, Path]):
        super().__init__(benchmark_dir, "cost_quality_pareto")
        self._points: List[CostQualityPoint] = []

    def add_point(self, point: CostQualityPoint) -> None:
        """Add a data point to the analysis.

        Args:
            point: CostQualityPoint to add
        """
        self._points.append(point)
        logger.info(
            "Added cost/quality point",
            model=point.model_id,
            cost=point.cost_per_sample,
            quality=point.quality_score,
        )

    def add_model(
        self,
        model_id: str,
        cost_per_sample: float,
        quality_score: float,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a model to the analysis.

        Args:
            model_id: Model identifier
            cost_per_sample: Cost in USD per sample
            quality_score: Quality score (0-100)
            latency_ms: Optional latency in milliseconds
            metadata: Additional metadata
        """
        point = CostQualityPoint(
            model_id=model_id,
            cost_per_sample=cost_per_sample,
            quality_score=quality_score,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.add_point(point)

    def compute_pareto_frontier(self) -> List[CostQualityPoint]:
        """Compute the Pareto-optimal frontier.

        A model is Pareto-optimal if no other model has both
        lower cost and higher quality.

        Returns:
            List of Pareto-optimal points sorted by cost
        """
        if not self._points:
            return []

        # Sort by cost (ascending)
        sorted_points = sorted(self._points, key=lambda p: p.cost_per_sample)

        pareto = []
        max_quality = -1.0

        for point in sorted_points:
            if point.quality_score > max_quality:
                pareto.append(point)
                max_quality = point.quality_score

        logger.info("Computed Pareto frontier", n_points=len(pareto))
        return pareto

    def compute_efficiency_ratio(self) -> Dict[str, float]:
        """Compute quality-per-dollar efficiency ratio for each model.

        Returns:
            Dictionary mapping model ID to efficiency ratio
        """
        if not self._points:
            return {}

        ratios = {}
        for point in self._points:
            if point.cost_per_sample > 0:
                ratios[point.model_id] = point.quality_score / point.cost_per_sample
            else:
                ratios[point.model_id] = float("inf")

        return ratios

    def compute_auc(self) -> float:
        """Compute area under the Pareto frontier curve.

        Higher AUC indicates better overall cost-quality tradeoff.

        Returns:
            Area under the Pareto frontier
        """
        frontier = self.compute_pareto_frontier()
        if len(frontier) < 2:
            return 0.0

        # Compute area using trapezoidal rule
        auc = 0.0
        for i in range(len(frontier) - 1):
            cost_diff = frontier[i + 1].cost_per_sample - frontier[i].cost_per_sample
            avg_quality = (frontier[i].quality_score + frontier[i + 1].quality_score) / 2
            auc += cost_diff * avg_quality

        return auc

    def find_best_model_for_budget(
        self,
        budget: float,
        min_quality: float = 0.0,
    ) -> Optional[str]:
        """Find the best quality model within a budget.

        Args:
            budget: Maximum cost per sample
            min_quality: Minimum quality score required

        Returns:
            Best model ID within budget, or None if none qualify
        """
        candidates = [
            p
            for p in self._points
            if p.cost_per_sample <= budget and p.quality_score >= min_quality
        ]

        if not candidates:
            return None

        best = max(candidates, key=lambda p: p.quality_score)
        return best.model_id

    def find_cheapest_model_for_quality(
        self,
        target_quality: float,
    ) -> Optional[str]:
        """Find the cheapest model meeting a quality target.

        Args:
            target_quality: Minimum quality score required

        Returns:
            Cheapest model ID meeting quality target, or None if none qualify
        """
        candidates = [p for p in self._points if p.quality_score >= target_quality]

        if not candidates:
            return None

        cheapest = min(candidates, key=lambda p: p.cost_per_sample)
        return cheapest.model_id

    def compute_tradeoff_slope(
        self,
        model_a: str,
        model_b: str,
    ) -> Optional[float]:
        """Compute the marginal quality gain per dollar between two models.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Returns:
            Slope (delta_quality / delta_cost), or None if models not found
        """
        point_a = next((p for p in self._points if p.model_id == model_a), None)
        point_b = next((p for p in self._points if p.model_id == model_b), None)

        if point_a is None or point_b is None:
            return None

        delta_cost = point_b.cost_per_sample - point_a.cost_per_sample
        delta_quality = point_b.quality_score - point_a.quality_score

        if delta_cost == 0:
            return float("inf") if delta_quality != 0 else 0.0

        return delta_quality / delta_cost

    def generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for plotting a cost/quality Pareto chart.

        Returns:
            Dictionary with all points and Pareto frontier for plotting
        """
        frontier = self.compute_pareto_frontier()

        return {
            "all_points": [p.to_dict() for p in self._points],
            "pareto_frontier": [p.to_dict() for p in frontier],
            "pareto_model_ids": [p.model_id for p in frontier],
            "efficiency_ratios": self.compute_efficiency_ratio(),
            "auc": self.compute_auc(),
        }

    def run_benchmark(
        self,
        model_id: str = "cost_quality_analysis",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the cost/quality Pareto benchmark.

        Args:
            model_id: Identifier for this benchmark run
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with Pareto analysis data
        """
        if not self._points:
            raise ValueError("No data points added. Use add_point() or add_model() first.")

        logger.info(
            "Running cost/quality Pareto benchmark",
            n_points=len(self._points),
        )

        frontier = self.compute_pareto_frontier()
        efficiency = self.compute_efficiency_ratio()
        auc = self.compute_auc()

        metrics = {
            "n_models": len(self._points),
            "n_pareto_optimal": len(frontier),
            "pareto_auc": auc,
            "min_cost": min(p.cost_per_sample for p in self._points),
            "max_cost": max(p.cost_per_sample for p in self._points),
            "min_quality": min(p.quality_score for p in self._points),
            "max_quality": max(p.quality_score for p in self._points),
        }

        # Add efficiency ratios
        for model_id_key, ratio in efficiency.items():
            metrics[f"efficiency_{model_id_key}"] = ratio if ratio != float("inf") else 9999.0

        # Add frontier point metrics
        for i, point in enumerate(frontier):
            metrics[f"pareto_{i + 1}_model"] = point.model_id
            metrics[f"pareto_{i + 1}_cost"] = point.cost_per_sample
            metrics[f"pareto_{i + 1}_quality"] = point.quality_score

        chart_data = self.generate_chart_data()
        predictions = [json.dumps(chart_data)]
        references = predictions

        result = BenchmarkResult(
            benchmark_name="cost_quality_pareto",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Cost/quality Pareto benchmark complete", metrics=metrics)
        return result

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """Export Pareto analysis to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = self.generate_chart_data()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported Pareto chart data", path=str(output_path))
