"""Latency and quality profiles for teacher models.

Profiles track model performance characteristics and quality metrics.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LatencyProfile:
    """Performance/latency characteristics for a model.

    Attributes:
        model_name: Model identifier
        avg_first_token_ms: Average time to first token
        avg_token_ms: Average time per token after first
        p50_latency_ms: 50th percentile latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        throughput_tokens_per_sec: Sustained throughput
        sample_count: Number of samples for statistics
    """

    model_name: str
    avg_first_token_ms: float = 0.0
    avg_token_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    sample_count: int = 0

    def estimate_latency_ms(self, num_tokens: int) -> float:
        """Estimate latency for a given number of tokens."""
        return self.avg_first_token_ms + (num_tokens * self.avg_token_ms)

    def update_from_samples(self, latencies_ms: List[float]) -> None:
        """Update profile from new latency samples."""
        if not latencies_ms:
            return

        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)

        self.p50_latency_ms = statistics.median(sorted_latencies)
        self.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n >= 20 else self.p50_latency_ms
        self.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n >= 100 else self.p95_latency_ms
        self.avg_first_token_ms = statistics.mean(sorted_latencies)

        self.sample_count += n


@dataclass
class QualityProfile:
    """Quality assessment profile for a model.

    Attributes:
        model_name: Model identifier
        task_scores: Quality scores by task type
        overall_score: Overall quality score (0-100)
        consistency_score: Consistency score (0-100)
        error_rate: Error rate (0-1)
        human_preference_score: Human preference score if available
        benchmark_scores: Scores on standard benchmarks
    """

    model_name: str
    task_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    consistency_score: float = 0.0
    error_rate: float = 0.0
    human_preference_score: Optional[float] = None
    benchmark_scores: Dict[str, float] = field(default_factory=dict)

    def get_task_score(self, task: str) -> float:
        """Get quality score for a specific task."""
        return self.task_scores.get(task, self.overall_score)

    def update_task_score(self, task: str, score: float) -> None:
        """Update quality score for a task."""
        self.task_scores[task] = score
        # Recalculate overall score as average
        if self.task_scores:
            self.overall_score = statistics.mean(self.task_scores.values())


# Estimated baseline latency profiles (based on typical performance)
BASELINE_LATENCY_PROFILES: Dict[str, LatencyProfile] = {
    "gpt-5.5": LatencyProfile(
        model_name="gpt-5.5",
        avg_first_token_ms=300,
        avg_token_ms=15,
        p50_latency_ms=800,
        p95_latency_ms=1500,
        throughput_tokens_per_sec=1200,
    ),
    "gpt-5": LatencyProfile(
        model_name="gpt-5",
        avg_first_token_ms=250,
        avg_token_ms=12,
        p50_latency_ms=600,
        p95_latency_ms=1200,
        throughput_tokens_per_sec=1500,
    ),
    "claude-opus-4-8": LatencyProfile(
        model_name="claude-opus-4-8",
        avg_first_token_ms=500,
        avg_token_ms=20,
        p50_latency_ms=1200,
        p95_latency_ms=2500,
        throughput_tokens_per_sec=800,
    ),
    "claude-sonnet-4-6": LatencyProfile(
        model_name="claude-sonnet-4-6",
        avg_first_token_ms=200,
        avg_token_ms=10,
        p50_latency_ms=500,
        p95_latency_ms=1000,
        throughput_tokens_per_sec=2000,
    ),
    "meta-llama/Llama-3.2-90B-Vision-Instruct": LatencyProfile(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
        avg_first_token_ms=400,
        avg_token_ms=25,
        p50_latency_ms=1000,
        p95_latency_ms=2000,
        throughput_tokens_per_sec=600,
    ),
    "Qwen/Qwen2.5-VL-72B-Instruct": LatencyProfile(
        model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        avg_first_token_ms=350,
        avg_token_ms=20,
        p50_latency_ms=900,
        p95_latency_ms=1800,
        throughput_tokens_per_sec=700,
    ),
}


# Estimated baseline quality profiles
BASELINE_QUALITY_PROFILES: Dict[str, QualityProfile] = {
    "gpt-5.5": QualityProfile(
        model_name="gpt-5.5",
        overall_score=92.0,
        consistency_score=90.0,
        error_rate=0.02,
        task_scores={
            "fine_grained": 94.0,
            "action_recognition": 93.0,
            "object_detection": 91.0,
            "qa": 92.0,
        },
    ),
    "gpt-5": QualityProfile(
        model_name="gpt-5",
        overall_score=88.0,
        consistency_score=86.0,
        error_rate=0.03,
        task_scores={
            "fine_grained": 90.0,
            "action_recognition": 89.0,
            "object_detection": 87.0,
            "qa": 88.0,
        },
    ),
    "claude-opus-4-8": QualityProfile(
        model_name="claude-opus-4-8",
        overall_score=94.0,
        consistency_score=92.0,
        error_rate=0.015,
        task_scores={
            "fine_grained": 95.0,
            "action_recognition": 94.0,
            "object_detection": 93.0,
            "qa": 95.0,
        },
    ),
    "claude-sonnet-4-6": QualityProfile(
        model_name="claude-sonnet-4-6",
        overall_score=89.0,
        consistency_score=88.0,
        error_rate=0.025,
        task_scores={
            "fine_grained": 90.0,
            "action_recognition": 89.0,
            "object_detection": 88.0,
            "qa": 90.0,
        },
    ),
    "meta-llama/Llama-3.2-90B-Vision-Instruct": QualityProfile(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
        overall_score=82.0,
        consistency_score=80.0,
        error_rate=0.05,
        task_scores={
            "fine_grained": 83.0,
            "action_recognition": 84.0,
            "object_detection": 81.0,
            "qa": 82.0,
        },
    ),
}


class ProfileManager:
    """Manages latency and quality profiles for models."""

    def __init__(self):
        self._latency_profiles: Dict[str, LatencyProfile] = dict(BASELINE_LATENCY_PROFILES)
        self._quality_profiles: Dict[str, QualityProfile] = dict(BASELINE_QUALITY_PROFILES)
        self._latency_samples: Dict[str, List[float]] = {}

    def get_latency_profile(self, model_name: str) -> Optional[LatencyProfile]:
        """Get latency profile for a model."""
        return self._latency_profiles.get(model_name)

    def get_quality_profile(self, model_name: str) -> Optional[QualityProfile]:
        """Get quality profile for a model."""
        return self._quality_profiles.get(model_name)

    def update_latency_profile(
        self,
        model_name: str,
        latency_ms: float,
    ) -> None:
        """Update latency profile with a new sample."""
        if model_name not in self._latency_samples:
            self._latency_samples[model_name] = []

        self._latency_samples[model_name].append(latency_ms)

        # Update profile every 10 samples
        if len(self._latency_samples[model_name]) >= 10:
            profile = self._latency_profiles.get(model_name)
            if profile is None:
                profile = LatencyProfile(model_name=model_name)
                self._latency_profiles[model_name] = profile

            profile.update_from_samples(self._latency_samples[model_name])
            self._latency_samples[model_name] = []

    def set_latency_profile(self, profile: LatencyProfile) -> None:
        """Set latency profile for a model."""
        self._latency_profiles[profile.model_name] = profile

    def set_quality_profile(self, profile: QualityProfile) -> None:
        """Set quality profile for a model."""
        self._quality_profiles[profile.model_name] = profile

    def estimate_latency(self, model_name: str, num_tokens: int) -> Optional[float]:
        """Estimate latency for a model and token count."""
        profile = self._latency_profiles.get(model_name)
        if profile:
            return profile.estimate_latency_ms(num_tokens)
        return None

    def get_quality_ranking(self, task: Optional[str] = None) -> List[tuple]:
        """Get models ranked by quality for a task."""
        scores = []
        for model_name, profile in self._quality_profiles.items():
            if task:
                score = profile.get_task_score(task)
            else:
                score = profile.overall_score
            scores.append((model_name, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_fastest_models(self) -> List[tuple]:
        """Get models ranked by median latency."""
        latencies = []
        for model_name, profile in self._latency_profiles.items():
            latencies.append((model_name, profile.p50_latency_ms))

        return sorted(latencies, key=lambda x: x[1])

    def find_quality_efficiency_tradeoff(
        self,
        min_quality_score: float = 80.0,
    ) -> List[tuple]:
        """Find models that balance quality and cost.

        Returns models sorted by quality/cost ratio.
        """
        from dvas.models.teacher.pricing import get_pricing_registry

        pricing_registry = get_pricing_registry()
        results = []

        for model_name, quality in self._quality_profiles.items():
            if quality.overall_score < min_quality_score:
                continue

            pricing = pricing_registry.get_pricing(model_name)
            if pricing:
                # Estimate cost for typical usage
                estimated_cost = pricing.calculate_cost(
                    input_tokens=2000,
                    output_tokens=500,
                    num_images=16,
                )
                if estimated_cost > 0:
                    efficiency = quality.overall_score / estimated_cost
                    results.append((model_name, efficiency, quality.overall_score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_profile_summary(self, model_name: str) -> Dict[str, Any]:
        """Get a summary of all profiles for a model."""
        latency = self._latency_profiles.get(model_name)
        quality = self._quality_profiles.get(model_name)

        summary = {"model": model_name}

        if latency:
            summary["latency"] = {
                "p50_ms": latency.p50_latency_ms,
                "p95_ms": latency.p95_latency_ms,
                "throughput": latency.throughput_tokens_per_sec,
            }

        if quality:
            summary["quality"] = {
                "overall": quality.overall_score,
                "consistency": quality.consistency_score,
                "error_rate": quality.error_rate,
                "task_scores": quality.task_scores,
            }

        return summary


# Global manager instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get the global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


def reset_profile_manager() -> None:
    """Reset the global profile manager (useful for testing)."""
    global _profile_manager
    _profile_manager = None
