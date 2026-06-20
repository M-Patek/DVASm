"""Student comparison leaderboard.

Compares student models across multiple dimensions:
quality, size, latency, and training efficiency.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StudentScore:
    """Score for a single student model.

    Attributes:
        model_id: Model identifier
        model_size: Model size in millions of parameters
        quality_score: Overall quality score (0-100)
        latency_ms: Average inference latency in milliseconds
        throughput: Throughput in samples per second
        training_cost: Training cost in USD
        memory_mb: Memory footprint in MB
        overall_score: Weighted overall score
        rankings: Per-dimension rankings
    """

    model_id: str
    model_size: float = 0.0
    quality_score: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    training_cost: float = 0.0
    memory_mb: float = 0.0
    overall_score: float = 0.0
    rankings: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_size": self.model_size,
            "quality_score": self.quality_score,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "training_cost": self.training_cost,
            "memory_mb": self.memory_mb,
            "overall_score": self.overall_score,
            "rankings": self.rankings,
        }


class StudentLeaderboard(BaseBenchmark):
    """Student model comparison leaderboard.

    Compares student models across quality, size, latency,
    throughput, and training efficiency dimensions.

    Args:
        benchmark_dir: Directory for storing benchmark data
        weights: Optional custom weights for overall score
    """

    DEFAULT_WEIGHTS = {
        "quality": 0.35,
        "size": 0.15,
        "latency": 0.20,
        "throughput": 0.15,
        "memory": 0.15,
    }

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(benchmark_dir, "student_leaderboard")
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._student_data: Dict[str, Dict[str, Any]] = {}

    def register_student(
        self,
        model_id: str,
        model_size: float,
        quality_score: float,
        latency_ms: float,
        throughput: float,
        training_cost: float = 0.0,
        memory_mb: float = 0.0,
    ) -> None:
        """Register a student model for comparison.

        Args:
            model_id: Unique model identifier
            model_size: Model size in millions of parameters
            quality_score: Overall quality score (0-100)
            latency_ms: Average inference latency in milliseconds
            throughput: Throughput in samples per second
            training_cost: Training cost in USD
            memory_mb: Memory footprint in MB
        """
        self._student_data[model_id] = {
            "model_size": model_size,
            "quality_score": quality_score,
            "latency_ms": latency_ms,
            "throughput": throughput,
            "training_cost": training_cost,
            "memory_mb": memory_mb,
        }
        logger.info("Registered student model", model_id=model_id, size=model_size)

    def compute_size_scores(self) -> Dict[str, float]:
        """Compute size efficiency scores.

        Smaller models get higher scores.

        Returns:
            Dictionary mapping model ID to size score
        """
        if not self._student_data:
            return {}

        sizes = {mid: data["model_size"] for mid, data in self._student_data.items()}
        max_size = max(sizes.values())
        min_size = min(sizes.values())
        size_range = max_size - min_size if max_size > min_size else 1.0

        scores = {}
        for model_id, size in sizes.items():
            scores[model_id] = 100.0 * (max_size - size) / size_range if size_range > 0 else 50.0

        return scores

    def compute_latency_scores(self) -> Dict[str, float]:
        """Compute latency scores.

        Lower latency = higher score.

        Returns:
            Dictionary mapping model ID to latency score
        """
        if not self._student_data:
            return {}

        latencies = {mid: data["latency_ms"] for mid, data in self._student_data.items()}
        max_latency = max(latencies.values())
        min_latency = min(latencies.values())
        latency_range = max_latency - min_latency if max_latency > min_latency else 1.0

        scores = {}
        for model_id, latency in latencies.items():
            scores[model_id] = 100.0 * (max_latency - latency) / latency_range if latency_range > 0 else 50.0

        return scores

    def compute_throughput_scores(self) -> Dict[str, float]:
        """Compute throughput scores.

        Higher throughput = higher score.

        Returns:
            Dictionary mapping model ID to throughput score
        """
        if not self._student_data:
            return {}

        throughputs = {mid: data["throughput"] for mid, data in self._student_data.items()}
        max_throughput = max(throughputs.values())
        min_throughput = min(throughputs.values())
        throughput_range = max_throughput - min_throughput if max_throughput > min_throughput else 1.0

        scores = {}
        for model_id, throughput in throughputs.items():
            scores[model_id] = 100.0 * (throughput - min_throughput) / throughput_range if throughput_range > 0 else 50.0

        return scores

    def compute_memory_scores(self) -> Dict[str, float]:
        """Compute memory efficiency scores.

        Lower memory = higher score.

        Returns:
            Dictionary mapping model ID to memory score
        """
        if not self._student_data:
            return {}

        memories = {mid: data["memory_mb"] for mid, data in self._student_data.items()}
        max_memory = max(memories.values())
        min_memory = min(memories.values())
        memory_range = max_memory - min_memory if max_memory > min_memory else 1.0

        scores = {}
        for model_id, memory in memories.items():
            scores[model_id] = 100.0 * (max_memory - memory) / memory_range if memory_range > 0 else 50.0

        return scores

    def compute_quality_scores(self) -> Dict[str, float]:
        """Compute quality scores.

        Returns:
            Dictionary mapping model ID to quality score
        """
        if not self._student_data:
            return {}

        return {
            model_id: data["quality_score"]
            for model_id, data in self._student_data.items()
        }

    def compute_overall_scores(
        self,
        quality_scores: Dict[str, float],
        size_scores: Dict[str, float],
        latency_scores: Dict[str, float],
        throughput_scores: Dict[str, float],
        memory_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute weighted overall scores.

        Args:
            quality_scores: Quality scores per model
            size_scores: Size scores per model
            latency_scores: Latency scores per model
            throughput_scores: Throughput scores per model
            memory_scores: Memory scores per model

        Returns:
            Dictionary mapping model ID to overall score
        """
        model_ids = set(quality_scores.keys())

        scores = {}
        for model_id in model_ids:
            q = quality_scores.get(model_id, 0.0)
            s = size_scores.get(model_id, 0.0)
            lat = latency_scores.get(model_id, 0.0)
            t = throughput_scores.get(model_id, 0.0)
            m = memory_scores.get(model_id, 0.0)

            overall = (
                self.weights["quality"] * q +
                self.weights["size"] * s +
                self.weights["latency"] * lat +
                self.weights["throughput"] * t +
                self.weights["memory"] * m
            )
            scores[model_id] = overall

        return scores

    def compute_rankings(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Compute rankings from scores.

        Args:
            scores: Dictionary of model scores

        Returns:
            Dictionary mapping model ID to rank (1 = best)
        """
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {name: i + 1 for i, (name, _) in enumerate(sorted_models)}

    def generate_leaderboard(self) -> List[StudentScore]:
        """Generate the full student leaderboard.

        Returns:
            List of StudentScore objects sorted by overall score
        """
        if not self._student_data:
            logger.warning("No student models registered")
            return []

        logger.info("Generating student leaderboard", n_models=len(self._student_data))

        # Compute scores
        quality_scores = self.compute_quality_scores()
        size_scores = self.compute_size_scores()
        latency_scores = self.compute_latency_scores()
        throughput_scores = self.compute_throughput_scores()
        memory_scores = self.compute_memory_scores()
        overall_scores = self.compute_overall_scores(
            quality_scores, size_scores, latency_scores, throughput_scores, memory_scores
        )

        # Compute rankings
        quality_rankings = self.compute_rankings(quality_scores)
        size_rankings = self.compute_rankings(size_scores)
        latency_rankings = self.compute_rankings(latency_scores)
        throughput_rankings = self.compute_rankings(throughput_scores)
        memory_rankings = self.compute_rankings(memory_scores)
        overall_rankings = self.compute_rankings(overall_scores)

        # Build leaderboard
        leaderboard = []
        for model_id, data in self._student_data.items():
            score = StudentScore(
                model_id=model_id,
                model_size=data["model_size"],
                quality_score=data["quality_score"],
                latency_ms=data["latency_ms"],
                throughput=data["throughput"],
                training_cost=data["training_cost"],
                memory_mb=data["memory_mb"],
                overall_score=overall_scores.get(model_id, 0.0),
                rankings={
                    "quality": quality_rankings.get(model_id, 0),
                    "size": size_rankings.get(model_id, 0),
                    "latency": latency_rankings.get(model_id, 0),
                    "throughput": throughput_rankings.get(model_id, 0),
                    "memory": memory_rankings.get(model_id, 0),
                    "overall": overall_rankings.get(model_id, 0),
                },
            )
            leaderboard.append(score)

        # Sort by overall score
        leaderboard.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info("Student leaderboard generated", n_models=len(leaderboard))
        return leaderboard

    def run_benchmark(
        self,
        model_id: str = "student_comparison",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the student comparison benchmark.

        Args:
            model_id: Identifier for this benchmark run
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with leaderboard data
        """
        leaderboard = self.generate_leaderboard()

        # Extract metrics from leaderboard
        metrics = {}
        for i, score in enumerate(leaderboard):
            prefix = f"rank_{i + 1}_{score.model_id}"
            metrics[f"{prefix}_overall"] = score.overall_score
            metrics[f"{prefix}_quality"] = score.quality_score
            metrics[f"{prefix}_size"] = score.model_size
            metrics[f"{prefix}_latency"] = score.latency_ms
            metrics[f"{prefix}_throughput"] = score.throughput
            metrics[f"{prefix}_memory"] = score.memory_mb

        # Overall stats
        if leaderboard:
            metrics["top_score"] = leaderboard[0].overall_score
            metrics["avg_score"] = np.mean([s.overall_score for s in leaderboard])
            metrics["n_models"] = len(leaderboard)

        predictions = [json.dumps(s.to_dict()) for s in leaderboard]
        references = predictions

        result = BenchmarkResult(
            benchmark_name="student_leaderboard",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=predictions,
            references=references,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Student leaderboard benchmark complete", n_models=len(leaderboard))
        return result

    def get_pareto_optimal_models(self) -> List[str]:
        """Get Pareto-optimal models (best quality for given size).

        Returns:
            List of model IDs that are Pareto-optimal
        """
        if not self._student_data:
            return []

        pareto = []
        for model_id, data in self._student_data.items():
            is_dominated = False
            for other_id, other_data in self._student_data.items():
                if model_id == other_id:
                    continue
                # Other dominates if it has better or equal quality and smaller or equal size
                if (other_data["quality_score"] >= data["quality_score"] and
                    other_data["model_size"] <= data["model_size"] and
                    (other_data["quality_score"] > data["quality_score"] or
                     other_data["model_size"] < data["model_size"])):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(model_id)

        return pareto
