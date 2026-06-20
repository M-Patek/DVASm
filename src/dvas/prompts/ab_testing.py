"""A/B testing framework for prompt templates.

Provides random assignment, metrics collection, and statistical
significance testing for comparing prompt performance.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AssignmentMethod(str, Enum):
    """Methods for assigning users/videos to A/B test groups."""

    RANDOM = "random"
    HASH = "hash"
    ROUND_ROBIN = "round_robin"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""

    test_name: str
    variant_a_id: str
    variant_b_id: str
    traffic_split: float = 0.5  # Fraction for variant A
    assignment_method: AssignmentMethod = AssignmentMethod.RANDOM
    min_sample_size: int = 100
    significance_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.traffic_split < 1:
            raise ValueError("traffic_split must be between 0 and 1")
        if self.min_sample_size < 10:
            raise ValueError("min_sample_size must be at least 10")


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""

    variant_a_id: str
    variant_b_id: str
    metric_name: str
    a_mean: float
    b_mean: float
    difference: float
    percent_change: float
    p_value: float
    is_significant: bool
    sample_size_a: int
    sample_size_b: int
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_a_id": self.variant_a_id,
            "variant_b_id": self.variant_b_id,
            "metric_name": self.metric_name,
            "a_mean": self.a_mean,
            "b_mean": self.b_mean,
            "difference": self.difference,
            "percent_change": self.percent_change,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "confidence_interval": self.confidence_interval,
        }


@dataclass
class ABTestMetrics:
    """Metrics collected for a single A/B test variant."""

    quality_scores: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)

    def add_quality(self, score: float) -> None:
        self.quality_scores.append(score)

    def add_latency(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)

    def add_cost(self, cost: float) -> None:
        self.costs.append(cost)

    @property
    def avg_quality(self) -> float:
        if not self.quality_scores:
            return 0.0
        return statistics.mean(self.quality_scores)

    @property
    def avg_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def avg_cost(self) -> float:
        if not self.costs:
            return 0.0
        return statistics.mean(self.costs)


class ABTestRunner:
    """Runs A/B tests between prompt variants."""

    def __init__(self) -> None:
        self._configs: Dict[str, ABTestConfig] = {}
        self._metrics: Dict[str, Dict[str, ABTestMetrics]] = {}
        self._assignments: Dict[str, str] = {}
        self._round_robin_counter = 0

    def register_test(self, config: ABTestConfig) -> None:
        """Register a new A/B test."""
        self._configs[config.test_name] = config
        self._metrics[config.test_name] = {
            config.variant_a_id: ABTestMetrics(),
            config.variant_b_id: ABTestMetrics(),
        }
        logger.info(
            "ab_test_registered",
            test_name=config.test_name,
            variant_a=config.variant_a_id,
            variant_b=config.variant_b_id,
        )

    def assign_variant(
        self,
        test_name: str,
        entity_id: str,
    ) -> Optional[str]:
        """Assign an entity to a variant.

        Args:
            test_name: Name of the A/B test.
            entity_id: Unique identifier for the entity (video, user, etc.).

        Returns:
            Variant ID (prompt_id) or None if test not found.
        """
        if test_name not in self._configs:
            return None

        config = self._configs[test_name]

        # Check if already assigned
        key = f"{test_name}:{entity_id}"
        if key in self._assignments:
            return self._assignments[key]

        if config.assignment_method == AssignmentMethod.RANDOM:
            variant = (
                config.variant_a_id
                if random.random() < config.traffic_split
                else config.variant_b_id
            )
        elif config.assignment_method == AssignmentMethod.HASH:
            hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
            variant = (
                config.variant_a_id
                if (hash_val % 100) < (config.traffic_split * 100)
                else config.variant_b_id
            )
        elif config.assignment_method == AssignmentMethod.ROUND_ROBIN:
            self._round_robin_counter += 1
            total = int(1 / config.traffic_split)
            variant = (
                config.variant_a_id
                if (self._round_robin_counter % total) < 1
                else config.variant_b_id
            )
        else:
            variant = config.variant_a_id

        self._assignments[key] = variant
        return variant

    def record_metric(
        self,
        test_name: str,
        variant_id: str,
        quality: Optional[float] = None,
        latency_ms: Optional[float] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Record metrics for a variant."""
        if test_name not in self._metrics:
            return
        if variant_id not in self._metrics[test_name]:
            return

        metrics = self._metrics[test_name][variant_id]
        if quality is not None:
            metrics.add_quality(quality)
        if latency_ms is not None:
            metrics.add_latency(latency_ms)
        if cost is not None:
            metrics.add_cost(cost)

    def get_metrics(self, test_name: str, variant_id: str) -> Optional[ABTestMetrics]:
        """Get metrics for a variant."""
        if test_name not in self._metrics:
            return None
        return self._metrics[test_name].get(variant_id)

    def compare(
        self,
        test_name: str,
        metric: str = "quality",
    ) -> Optional[ABTestResult]:
        """Compare variants and return statistical result.

        Uses a simple t-test approximation for significance.
        """
        if test_name not in self._configs:
            return None

        config = self._configs[test_name]
        a_metrics = self._metrics[test_name][config.variant_a_id]
        b_metrics = self._metrics[test_name][config.variant_b_id]

        if metric == "quality":
            a_data = a_metrics.quality_scores
            b_data = b_metrics.quality_scores
        elif metric == "latency":
            a_data = a_metrics.latency_ms
            b_data = b_metrics.latency_ms
        elif metric == "cost":
            a_data = a_metrics.costs
            b_data = b_metrics.costs
        else:
            return None

        if len(a_data) < 2 or len(b_data) < 2:
            return None

        a_mean = statistics.mean(a_data)
        b_mean = statistics.mean(b_data)
        diff = b_mean - a_mean
        percent_change = (diff / a_mean * 100) if a_mean != 0 else 0.0

        # Simple t-statistic approximation
        a_var = statistics.variance(a_data) if len(a_data) > 1 else 0
        b_var = statistics.variance(b_data) if len(b_data) > 1 else 0
        n_a, n_b = len(a_data), len(b_data)
        pooled_se = ((a_var / n_a) + (b_var / n_b)) ** 0.5

        if pooled_se == 0:
            p_value = 1.0
        else:
            t_stat = diff / pooled_se
            # Approximate p-value (simplified)
            p_value = max(0.001, min(1.0, 1.0 / (1.0 + abs(t_stat) ** 2)))

        is_significant = p_value < config.significance_threshold

        # Confidence interval (95%)
        if pooled_se > 0:
            margin = 1.96 * pooled_se
            ci = (diff - margin, diff + margin)
        else:
            ci = (diff, diff)

        return ABTestResult(
            variant_a_id=config.variant_a_id,
            variant_b_id=config.variant_b_id,
            metric_name=metric,
            a_mean=a_mean,
            b_mean=b_mean,
            difference=diff,
            percent_change=percent_change,
            p_value=p_value,
            is_significant=is_significant,
            sample_size_a=len(a_data),
            sample_size_b=len(b_data),
            confidence_interval=ci,
        )

    def get_winner(self, test_name: str, metric: str = "quality") -> Optional[str]:
        """Determine the winning variant based on a metric.

        Returns:
            The variant_id of the winner, or None if no clear winner.
        """
        result = self.compare(test_name, metric)
        if result is None:
            return None

        if not result.is_significant:
            return None

        if result.difference > 0:
            return result.variant_b_id
        else:
            return result.variant_a_id
