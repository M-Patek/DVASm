"""Teacher comparison leaderboard.

Compares teacher models across multiple dimensions:
quality, cost, latency, and feature support.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.models.teacher.pricing import get_pricing_registry
from dvas.models.teacher.profiles import (
    get_profile_manager,
)
from dvas.models.teacher.registry import MODEL_REGISTRY, get_registry
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TeacherScore:
    """Score for a single teacher model.

    Attributes:
        model_name: Model identifier
        quality_score: Overall quality score (0-100)
        cost_score: Cost efficiency score (0-100)
        latency_score: Latency score (0-100)
        feature_score: Feature support score (0-100)
        overall_score: Weighted overall score
        rankings: Per-dimension rankings
    """

    model_name: str
    quality_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    feature_score: float = 0.0
    overall_score: float = 0.0
    rankings: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quality_score": self.quality_score,
            "cost_score": self.cost_score,
            "latency_score": self.latency_score,
            "feature_score": self.feature_score,
            "overall_score": self.overall_score,
            "rankings": self.rankings,
        }


class TeacherLeaderboard(BaseBenchmark):
    """Teacher model comparison leaderboard.

    Compares teacher models across quality, cost, latency,
    and feature support dimensions.

    Args:
        benchmark_dir: Directory for storing benchmark data
    """

    # Default weights for overall score
    DEFAULT_WEIGHTS = {
        "quality": 0.4,
        "cost": 0.2,
        "latency": 0.2,
        "features": 0.2,
    }

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(benchmark_dir, "teacher_leaderboard")
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.pricing_registry = get_pricing_registry()
        self.profile_manager = get_profile_manager()
        self.model_registry = get_registry()

    def compute_quality_scores(self, model_names: List[str]) -> Dict[str, float]:
        """Compute quality scores for models.

        Args:
            model_names: List of model identifiers

        Returns:
            Dictionary mapping model name to quality score
        """
        scores = {}
        for name in model_names:
            profile = self.profile_manager.get_quality_profile(name)
            if profile:
                scores[name] = profile.overall_score
            else:
                scores[name] = 50.0  # Default mid-range score
        return scores

    def compute_cost_scores(
        self,
        model_names: List[str],
        input_tokens: int = 2000,
        output_tokens: int = 500,
        num_images: int = 16,
    ) -> Dict[str, float]:
        """Compute cost efficiency scores for models.

        Lower cost = higher score (inverse normalized).

        Args:
            model_names: List of model identifiers
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_images: Number of images

        Returns:
            Dictionary mapping model name to cost score
        """
        costs = {}
        for name in model_names:
            pricing = self.pricing_registry.get_pricing(name)
            if pricing:
                cost = pricing.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    num_images=num_images,
                )
                costs[name] = cost
            else:
                costs[name] = 0.05  # Default estimate

        if not costs:
            return {}

        max_cost = max(costs.values())
        min_cost = min(costs.values())
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

        scores = {}
        for name, cost in costs.items():
            # Inverse normalized: lower cost = higher score
            scores[name] = 100.0 * (max_cost - cost) / cost_range if cost_range > 0 else 50.0

        return scores

    def compute_latency_scores(self, model_names: List[str]) -> Dict[str, float]:
        """Compute latency scores for models.

        Lower latency = higher score (inverse normalized).

        Args:
            model_names: List of model identifiers

        Returns:
            Dictionary mapping model name to latency score
        """
        latencies = {}
        for name in model_names:
            profile = self.profile_manager.get_latency_profile(name)
            if profile:
                latencies[name] = profile.p50_latency_ms
            else:
                latencies[name] = 1000.0  # Default estimate

        if not latencies:
            return {}

        max_latency = max(latencies.values())
        min_latency = min(latencies.values())
        latency_range = max_latency - min_latency if max_latency > min_latency else 1.0

        scores = {}
        for name, latency in latencies.items():
            scores[name] = 100.0 * (max_latency - latency) / latency_range if latency_range > 0 else 50.0

        return scores

    def compute_feature_scores(self, model_names: List[str]) -> Dict[str, float]:
        """Compute feature support scores for models.

        Args:
            model_names: List of model identifiers

        Returns:
            Dictionary mapping model name to feature score
        """
        # Define feature weights
        feature_weights = {
            "vision": 1.0,
            "video": 1.0,
            "streaming": 0.5,
            "json_mode": 0.5,
            "function_calling": 0.5,
            "tool_use": 0.5,
            "reasoning": 0.5,
            "structured_output": 0.5,
        }

        max_possible = sum(feature_weights.values())

        scores = {}
        for name in model_names:
            spec = self.model_registry.get_model(name)
            if spec and spec.features:
                feature_score = sum(feature_weights.get(f, 0.0) for f in spec.features)
                scores[name] = 100.0 * feature_score / max_possible
            else:
                scores[name] = 50.0

        return scores

    def compute_overall_scores(
        self,
        quality_scores: Dict[str, float],
        cost_scores: Dict[str, float],
        latency_scores: Dict[str, float],
        feature_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute weighted overall scores.

        Args:
            quality_scores: Quality scores per model
            cost_scores: Cost scores per model
            latency_scores: Latency scores per model
            feature_scores: Feature scores per model

        Returns:
            Dictionary mapping model name to overall score
        """
        model_names = set(quality_scores.keys()) | set(cost_scores.keys()) | set(latency_scores.keys()) | set(feature_scores.keys())

        scores = {}
        for name in model_names:
            q = quality_scores.get(name, 0.0)
            c = cost_scores.get(name, 0.0)
            lat = latency_scores.get(name, 0.0)
            f = feature_scores.get(name, 0.0)

            overall = (
                self.weights["quality"] * q +
                self.weights["cost"] * c +
                self.weights["latency"] * lat +
                self.weights["features"] * f
            )
            scores[name] = overall

        return scores

    def compute_rankings(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Compute rankings from scores.

        Args:
            scores: Dictionary of model scores

        Returns:
            Dictionary mapping model name to rank (1 = best)
        """
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {name: i + 1 for i, (name, _) in enumerate(sorted_models)}

    def generate_leaderboard(
        self,
        model_names: Optional[List[str]] = None,
    ) -> List[TeacherScore]:
        """Generate the full teacher leaderboard.

        Args:
            model_names: Optional list of models to compare (default: all registered)

        Returns:
            List of TeacherScore objects sorted by overall score
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        logger.info("Generating teacher leaderboard", n_models=len(model_names))

        # Compute scores
        quality_scores = self.compute_quality_scores(model_names)
        cost_scores = self.compute_cost_scores(model_names)
        latency_scores = self.compute_latency_scores(model_names)
        feature_scores = self.compute_feature_scores(model_names)
        overall_scores = self.compute_overall_scores(
            quality_scores, cost_scores, latency_scores, feature_scores
        )

        # Compute rankings
        quality_rankings = self.compute_rankings(quality_scores)
        cost_rankings = self.compute_rankings(cost_scores)
        latency_rankings = self.compute_rankings(latency_scores)
        feature_rankings = self.compute_rankings(feature_scores)
        overall_rankings = self.compute_rankings(overall_scores)

        # Build leaderboard
        leaderboard = []
        for name in model_names:
            score = TeacherScore(
                model_name=name,
                quality_score=quality_scores.get(name, 0.0),
                cost_score=cost_scores.get(name, 0.0),
                latency_score=latency_scores.get(name, 0.0),
                feature_score=feature_scores.get(name, 0.0),
                overall_score=overall_scores.get(name, 0.0),
                rankings={
                    "quality": quality_rankings.get(name, 0),
                    "cost": cost_rankings.get(name, 0),
                    "latency": latency_rankings.get(name, 0),
                    "features": feature_rankings.get(name, 0),
                    "overall": overall_rankings.get(name, 0),
                },
            )
            leaderboard.append(score)

        # Sort by overall score
        leaderboard.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info("Teacher leaderboard generated", n_models=len(leaderboard))
        return leaderboard

    def run_benchmark(
        self,
        model_id: str = "teacher_comparison",
        model_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the teacher comparison benchmark.

        Args:
            model_id: Identifier for this benchmark run
            model_names: Optional list of models to compare
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with leaderboard data
        """
        leaderboard = self.generate_leaderboard(model_names)

        # Extract metrics from leaderboard
        metrics = {}
        for i, score in enumerate(leaderboard):
            prefix = f"rank_{i + 1}_{score.model_name}"
            metrics[f"{prefix}_overall"] = score.overall_score
            metrics[f"{prefix}_quality"] = score.quality_score
            metrics[f"{prefix}_cost"] = score.cost_score
            metrics[f"{prefix}_latency"] = score.latency_score
            metrics[f"{prefix}_features"] = score.feature_score

        # Overall stats
        if leaderboard:
            metrics["top_score"] = leaderboard[0].overall_score
            metrics["avg_score"] = np.mean([s.overall_score for s in leaderboard])
            metrics["n_models"] = len(leaderboard)

        predictions = [json.dumps(s.to_dict()) for s in leaderboard]
        references = predictions  # Self-referential for leaderboard

        result = BenchmarkResult(
            benchmark_name="teacher_leaderboard",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Teacher leaderboard benchmark complete", n_models=len(leaderboard))
        return result

    def get_best_model_for_task(
        self,
        task: str,
        model_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Get the best model for a specific task.

        Args:
            task: Task type (e.g., "fine_grained", "action_recognition")
            model_names: Optional list of models to consider

        Returns:
            Best model name for the task
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        best_model = None
        best_score = -1.0

        for name in model_names:
            profile = self.profile_manager.get_quality_profile(name)
            if profile:
                score = profile.get_task_score(task)
                if score > best_score:
                    best_score = score
                    best_model = name

        return best_model

    def get_cheapest_model(
        self,
        model_names: Optional[List[str]] = None,
        min_quality: float = 70.0,
    ) -> Optional[str]:
        """Get the cheapest model meeting quality threshold.

        Args:
            model_names: Optional list of models to consider
            min_quality: Minimum quality score required

        Returns:
            Cheapest model name meeting quality threshold
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        cheapest = None
        lowest_cost = float("inf")

        for name in model_names:
            profile = self.profile_manager.get_quality_profile(name)
            if profile and profile.overall_score >= min_quality:
                pricing = self.pricing_registry.get_pricing(name)
                if pricing:
                    cost = pricing.calculate_cost(input_tokens=2000, output_tokens=500, num_images=16)
                    if cost < lowest_cost:
                        lowest_cost = cost
                        cheapest = name

        return cheapest
