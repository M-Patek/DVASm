"""Canary and shadow deployment for safe model releases.

Provides CanaryRouter for traffic splitting and ShadowDeployment
for parallel model evaluation without affecting production results.

Usage::

    from dvas.routing.canary import CanaryRouter, ShadowDeployment

    # Canary deployment: 10% traffic to new model
    router = CanaryRouter()
    router.register_model("v1.0", old_model)
    router.register_model("v1.1", new_model)
    router.set_traffic_split({"v1.0": 0.9, "v1.1": 0.1})

    model = router.route(request_id)
    result = model.predict(request)

    # Shadow deployment: parallel evaluation
    shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
    result = shadow.predict(request)  # Returns primary result
    comparison = shadow.compare_last()  # Compares primary vs shadow
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DeploymentMode(str, Enum):
    """Deployment mode for canary/shadow."""

    CANARY = "canary"  # Traffic split between versions
    SHADOW = "shadow"  # Parallel execution, primary result returned
    BLUE_GREEN = "blue_green"  # Instant switch
    AB_TEST = "ab_test"  # Statistical comparison


class RollbackReason(str, Enum):
    """Reason for automatic rollback."""

    QUALITY_REGRESSION = "quality_regression"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_REGRESSION = "latency_regression"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ModelVersion:
    """Registered model version.

    Attributes:
        version_id: Unique version identifier
        model: Model instance or callable
        metadata: Version metadata
        registered_at: Registration timestamp
        traffic_weight: Current traffic weight (0-1)
        is_active: Whether this version is active
        error_count: Count of errors
        total_requests: Total requests served
        avg_latency_ms: Average latency
        avg_quality_score: Average quality score
    """

    version_id: str
    model: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    traffic_weight: float = 0.0
    is_active: bool = True
    error_count: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.0


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        version_id: Selected version ID
        mode: Deployment mode
        is_shadow: Whether this is a shadow request
        shadow_version: Shadow version (if shadow mode)
    """

    version_id: str
    mode: DeploymentMode
    is_shadow: bool = False
    shadow_version: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison between two model versions.

    Attributes:
        primary_version: Primary version ID
        shadow_version: Shadow version ID
        primary_result: Primary model output
        shadow_result: Shadow model output
        similarity_score: Similarity between outputs (0-1)
        latency_diff_ms: Latency difference
        quality_diff: Quality score difference
        is_significant: Whether difference is significant
        timestamp: Comparison timestamp
    """

    primary_version: str
    shadow_version: str
    primary_result: Any
    shadow_result: Any
    similarity_score: float = 0.0
    latency_diff_ms: float = 0.0
    quality_diff: float = 0.0
    is_significant: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RollbackConfig:
    """Configuration for automatic rollback.

    Attributes:
        enabled: Whether auto-rollback is enabled
        quality_threshold: Minimum quality score
        error_rate_threshold: Maximum error rate (0-1)
        latency_threshold_ms: Maximum latency in ms
        min_samples: Minimum samples before rollback check
        cooldown_seconds: Cooldown between rollbacks
        max_rollbacks: Maximum rollbacks per hour
    """

    enabled: bool = True
    quality_threshold: float = 0.6
    error_rate_threshold: float = 0.1
    latency_threshold_ms: float = 10000.0
    min_samples: int = 50
    cooldown_seconds: int = 30
    max_rollbacks: int = 5


@dataclass
class DeploymentMetrics:
    """Metrics for deployment monitoring.

    Attributes:
        version_id: Version being monitored
        total_requests: Total requests
        successful_requests: Successful requests
        failed_requests: Failed requests
        error_rate: Current error rate
        avg_latency_ms: Average latency
        avg_quality: Average quality score
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
    """

    version_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_quality: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class CanaryRouter:
    """Canary deployment router with traffic splitting.

        Routes requests between model versions based on configurable
    traffic weights. Supports automatic rollback on quality degradation.

        Usage::

            router = CanaryRouter()
            router.register_model("v1.0", model_v1)
            router.register_model("v1.1", model_v2)
            router.set_traffic_split({"v1.0": 0.9, "v1.1": 0.1})

            version_id = router.route(request_id="req-123")
            model = router.get_model(version_id)
            result = model.predict(request)

            # Record metrics for rollback monitoring
            router.record_metrics(version_id, latency_ms=500, quality=0.85)
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.CANARY,
        rollback_config: Optional[RollbackConfig] = None,
    ):
        self.mode = mode
        self.rollback_config = rollback_config or RollbackConfig()
        self._versions: Dict[str, ModelVersion] = {}
        self._traffic_split: Dict[str, float] = {}
        self._metrics: Dict[str, DeploymentMetrics] = {}
        self._last_rollback: Optional[Tuple[str, str, float]] = None  # (version, reason, time)
        self._rollback_count: int = 0
        self._rollback_window_start: float = time.time()

    def register_model(
        self,
        version_id: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model version.

        Args:
            version_id: Unique version identifier
            model: Model instance or callable
            metadata: Optional version metadata
        """
        self._versions[version_id] = ModelVersion(
            version_id=version_id,
            model=model,
            metadata=metadata or {},
        )
        self._metrics[version_id] = DeploymentMetrics(version_id=version_id)
        logger.info("model_registered", version=version_id)

    def unregister_model(self, version_id: str) -> bool:
        """Unregister a model version.

        Args:
            version_id: Version to unregister

        Returns:
            True if unregistered, False if not found
        """
        if version_id in self._versions:
            del self._versions[version_id]
            del self._metrics[version_id]
            if version_id in self._traffic_split:
                del self._traffic_split[version_id]
            logger.info("model_unregistered", version=version_id)
            return True
        return False

    def set_traffic_split(self, weights: Dict[str, float]) -> None:
        """Set traffic weights for each version.

        Args:
            weights: Dict of version_id -> weight (0-1, sums to ~1)
        """
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Traffic weights must sum to > 0")

        # Normalize weights
        self._traffic_split = {k: v / total for k, v in weights.items()}

        # Update version weights
        for version_id, weight in self._traffic_split.items():
            if version_id in self._versions:
                self._versions[version_id].traffic_weight = weight

        logger.info(
            "traffic_split_updated",
            weights={k: f"{v:.1%}" for k, v in self._traffic_split.items()},
        )

    def route(self, request_id: Optional[str] = None) -> str:
        """Route a request to a model version.

        Uses deterministic hashing for consistent routing.

        Args:
            request_id: Request identifier for consistent hashing

        Returns:
            Selected version ID
        """
        if not self._traffic_split:
            if self._versions:
                return list(self._versions.keys())[0]
            raise ValueError("No models registered")

        if request_id:
            # Deterministic routing based on request ID
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            rand = (hash_val % 1000) / 1000
        else:
            # Random routing
            rand = np.random.random()

        # Cumulative probability routing
        cumulative = 0.0
        for version_id, weight in self._traffic_split.items():
            cumulative += weight
            if rand <= cumulative:
                return version_id

        # Fallback to last version
        return list(self._traffic_split.keys())[-1]

    def get_model(self, version_id: str) -> Any:
        """Get model instance by version ID.

        Args:
            version_id: Version identifier

        Returns:
            Model instance
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not registered")
        return self._versions[version_id].model

    def get_routing_decision(self, request_id: Optional[str] = None) -> RoutingDecision:
        """Get full routing decision with metadata.

        Args:
            request_id: Request identifier

        Returns:
            RoutingDecision with version and mode info
        """
        version_id = self.route(request_id)
        return RoutingDecision(
            version_id=version_id,
            mode=self.mode,
            is_shadow=False,
        )

    def record_metrics(
        self,
        version_id: str,
        latency_ms: float = 0.0,
        quality: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record metrics for a version.

        Args:
            version_id: Version that served the request
            latency_ms: Request latency in milliseconds
            quality: Quality score (0-1)
            success: Whether the request succeeded
        """
        if version_id not in self._metrics:
            self._metrics[version_id] = DeploymentMetrics(version_id=version_id)

        metrics = self._metrics[version_id]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        metrics.error_rate = metrics.failed_requests / metrics.total_requests

        # Update latency (exponential moving average)
        alpha = 0.1
        metrics.avg_latency_ms = (1 - alpha) * metrics.avg_latency_ms + alpha * latency_ms

        # Update quality
        metrics.avg_quality = (1 - alpha) * metrics.avg_quality + alpha * quality

        # Update version stats
        if version_id in self._versions:
            version = self._versions[version_id]
            version.total_requests += 1
            if not success:
                version.error_count += 1
            version.avg_latency_ms = metrics.avg_latency_ms
            version.avg_quality_score = metrics.avg_quality

    def should_rollback(self, version_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a version should be rolled back.

        Args:
            version_id: Version to check

        Returns:
            Tuple of (should_rollback, reason)
        """
        if not self.rollback_config.enabled:
            return False, None

        if version_id not in self._metrics:
            return False, None

        metrics = self._metrics[version_id]

        if metrics.total_requests < self.rollback_config.min_samples:
            return False, None

        # Check cooldown
        if self._last_rollback:
            elapsed = time.time() - self._last_rollback[2]
            if elapsed < self.rollback_config.cooldown_seconds:
                return False, None

        # Check max rollbacks per hour
        if time.time() - self._rollback_window_start > 3600:
            self._rollback_count = 0
            self._rollback_window_start = time.time()

        if self._rollback_count >= self.rollback_config.max_rollbacks:
            return False, None

        # Check quality regression
        if metrics.avg_quality < self.rollback_config.quality_threshold:
            return True, RollbackReason.QUALITY_REGRESSION

        # Check error rate spike
        if metrics.error_rate > self.rollback_config.error_rate_threshold:
            return True, RollbackReason.ERROR_RATE_SPIKE

        # Check latency regression
        if metrics.avg_latency_ms > self.rollback_config.latency_threshold_ms:
            return True, RollbackReason.LATENCY_REGRESSION

        return False, None

    def rollback(self, version_id: str, reason: str) -> None:
        """Rollback a version by removing it from traffic.

        Args:
            version_id: Version to rollback
            reason: Rollback reason
        """
        if version_id in self._traffic_split:
            # Remove version from traffic
            del self._traffic_split[version_id]

            # Redistribute traffic to remaining versions
            remaining = {k: v for k, v in self._traffic_split.items()}
            if remaining:
                total = sum(remaining.values())
                self._traffic_split = {k: v / total for k, v in remaining.items()}

            # Mark version as inactive
            if version_id in self._versions:
                self._versions[version_id].is_active = False

        self._last_rollback = (version_id, reason, time.time())
        self._rollback_count += 1

        logger.warning(
            "version_rolled_back",
            version=version_id,
            reason=reason,
        )

    def get_metrics(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a version or all versions.

        Args:
            version_id: Optional version filter

        Returns:
            Dict of metrics
        """
        if version_id:
            if version_id in self._metrics:
                m = self._metrics[version_id]
                return {
                    "version_id": m.version_id,
                    "total_requests": m.total_requests,
                    "error_rate": m.error_rate,
                    "avg_latency_ms": m.avg_latency_ms,
                    "avg_quality": m.avg_quality,
                }
            return {}

        return {
            vid: {
                "total_requests": m.total_requests,
                "error_rate": m.error_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "avg_quality": m.avg_quality,
            }
            for vid, m in self._metrics.items()
        }

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status.

        Returns:
            Dict with deployment status
        """
        return {
            "mode": self.mode.value,
            "versions": {
                vid: {
                    "traffic_weight": v.traffic_weight,
                    "is_active": v.is_active,
                    "total_requests": v.total_requests,
                    "error_count": v.error_count,
                    "avg_quality": v.avg_quality_score,
                }
                for vid, v in self._versions.items()
            },
            "traffic_split": self._traffic_split,
            "rollback_config": {
                "enabled": self.rollback_config.enabled,
                "quality_threshold": self.rollback_config.quality_threshold,
                "error_rate_threshold": self.rollback_config.error_rate_threshold,
            },
            "last_rollback": self._last_rollback,
        }


class ShadowDeployment:
    """Shadow deployment for safe model evaluation.

    Runs primary and shadow models in parallel, returning primary results
    while recording shadow results for comparison.

    Usage::

        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        shadow.register_model("v1.0", old_model)
        shadow.register_model("v1.1", new_model)

        # Primary result is returned, shadow runs in background
        result = shadow.predict(request)

        # Compare results
        comparison = shadow.compare_last()
        if comparison.is_significant:
            print(f"Warning: models diverge by {comparison.similarity_score:.2%}")
    """

    def __init__(
        self,
        primary: str,
        shadow: str,
        similarity_threshold: float = 0.95,
    ):
        self.primary_version = primary
        self.shadow_version = shadow
        self.similarity_threshold = similarity_threshold
        self._models: Dict[str, Any] = {}
        self._last_comparison: Optional[ComparisonResult] = None
        self._comparisons: List[ComparisonResult] = []
        self._max_comparisons: int = 10000

    def register_model(self, version_id: str, model: Any) -> None:
        """Register a model version.

        Args:
            version_id: Version identifier
            model: Model instance
        """
        self._models[version_id] = model
        logger.info("shadow_model_registered", version=version_id)

    def predict(self, request: Any) -> Any:
        """Run prediction with shadow deployment.

        Runs primary model and returns its result. Shadow model
        runs in parallel for comparison.

        Args:
            request: Input request

        Returns:
            Primary model result
        """
        if self.primary_version not in self._models:
            raise ValueError(f"Primary model {self.primary_version} not registered")

        # Run primary model
        primary_model = self._models[self.primary_version]
        primary_start = time.time()
        primary_result = primary_model(request)
        primary_latency = (time.time() - primary_start) * 1000

        # Run shadow model (if registered)
        shadow_result = None
        shadow_latency = 0.0

        if self.shadow_version in self._models:
            shadow_model = self._models[self.shadow_version]
            shadow_start = time.time()
            try:
                shadow_result = shadow_model(request)
            except Exception as e:
                logger.warning("shadow_model_error", error=str(e))
            shadow_latency = (time.time() - shadow_start) * 1000

        # Record comparison
        if shadow_result is not None:
            comparison = self._compare_results(
                primary_result,
                shadow_result,
                primary_latency,
                shadow_latency,
            )
            self._last_comparison = comparison
            self._comparisons.append(comparison)

            # Trim history
            if len(self._comparisons) > self._max_comparisons:
                self._comparisons = self._comparisons[-self._max_comparisons :]

        return primary_result

    def _compare_results(
        self,
        primary: Any,
        shadow: Any,
        primary_latency: float,
        shadow_latency: float,
    ) -> ComparisonResult:
        """Compare primary and shadow results.

        Args:
            primary: Primary result
            shadow: Shadow result
            primary_latency: Primary latency in ms
            shadow_latency: Shadow latency in ms

        Returns:
            ComparisonResult
        """
        # Compute similarity
        similarity = self._compute_similarity(primary, shadow)

        # Latency difference
        latency_diff = shadow_latency - primary_latency

        # Quality difference (if results have quality scores)
        quality_diff = 0.0
        if isinstance(primary, dict) and isinstance(shadow, dict):
            p_quality = primary.get("quality_score", 0)
            s_quality = shadow.get("quality_score", 0)
            quality_diff = s_quality - p_quality

        # Significance check
        is_significant = similarity < self.similarity_threshold

        if is_significant:
            logger.warning(
                "shadow_divergence_detected",
                similarity=similarity,
                threshold=self.similarity_threshold,
            )

        return ComparisonResult(
            primary_version=self.primary_version,
            shadow_version=self.shadow_version,
            primary_result=primary,
            shadow_result=shadow,
            similarity_score=similarity,
            latency_diff_ms=latency_diff,
            quality_diff=quality_diff,
            is_significant=is_significant,
        )

    def _compute_similarity(self, a: Any, b: Any) -> float:
        """Compute similarity between two results.

        Args:
            a: First result
            b: Second result

        Returns:
            Similarity score (0-1)
        """
        if not isinstance(a, type(b)):
            return 0.0

        if isinstance(a, str):
            # Text similarity (Jaccard on words)
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a and not words_b:
                return 1.0
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            return intersection / union if union > 0 else 0.0

        if isinstance(a, (list, tuple)):
            # Sequence similarity
            if len(a) == 0 and len(b) == 0:
                return 1.0
            matches = sum(1 for x, y in zip(a, b) if x == y)
            max_len = max(len(a), len(b))
            return matches / max_len if max_len > 0 else 0.0

        if isinstance(a, dict):
            # Dict similarity (key overlap)
            keys_a = set(a.keys())
            keys_b = set(b.keys())
            if not keys_a and not keys_b:
                return 1.0
            common = keys_a & keys_b
            total = keys_a | keys_b
            return len(common) / len(total) if total else 0.0

        # Default: exact match
        return 1.0 if a == b else 0.0

    def compare_last(self) -> Optional[ComparisonResult]:
        """Get the most recent comparison.

        Returns:
            Last ComparisonResult or None
        """
        return self._last_comparison

    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get statistics on shadow comparisons.

        Returns:
            Dict with comparison statistics
        """
        if not self._comparisons:
            return {"comparisons": 0}

        similarities = [c.similarity_score for c in self._comparisons]
        latencies = [c.latency_diff_ms for c in self._comparisons]
        significant = sum(1 for c in self._comparisons if c.is_significant)

        return {
            "comparisons": len(self._comparisons),
            "avg_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "significant_divergences": significant,
            "divergence_rate": significant / len(self._comparisons),
            "avg_latency_diff_ms": np.mean(latencies),
            "primary_version": self.primary_version,
            "shadow_version": self.shadow_version,
        }

    def get_divergent_samples(self, limit: int = 10) -> List[ComparisonResult]:
        """Get most divergent comparisons.

        Args:
            limit: Maximum number of samples

        Returns:
            List of divergent comparisons
        """
        divergent = [c for c in self._comparisons if c.is_significant]
        divergent.sort(key=lambda x: x.similarity_score)
        return divergent[:limit]

    def reset(self) -> None:
        """Reset comparison history."""
        self._comparisons.clear()
        self._last_comparison = None
        logger.info("shadow_deployment_reset")


class BlueGreenDeployment:
    """Blue-green deployment for zero-downtime releases.

        Maintains two identical production environments (blue and green),
    switching traffic instantly between them.

        Usage::

            deploy = BlueGreenDeployment()
            deploy.deploy_blue(model_v1)
            deploy.deploy_green(model_v2)

            # Switch traffic to green
            deploy.switch_to("green")

            # Rollback if needed
            deploy.switch_to("blue")
    """

    def __init__(self):
        self._environments: Dict[str, Any] = {"blue": None, "green": None}
        self._active: str = "blue"
        self._switch_count: int = 0
        self._last_switch: Optional[float] = None

    def deploy_blue(self, model: Any) -> None:
        """Deploy to blue environment.

        Args:
            model: Model to deploy
        """
        self._environments["blue"] = model
        logger.info("deployed_to_blue")

    def deploy_green(self, model: Any) -> None:
        """Deploy to green environment.

        Args:
            model: Model to deploy
        """
        self._environments["green"] = model
        logger.info("deployed_to_green")

    def switch_to(self, environment: str) -> None:
        """Switch traffic to an environment.

        Args:
            environment: "blue" or "green"
        """
        if environment not in self._environments:
            raise ValueError(f"Unknown environment: {environment}")

        if self._environments[environment] is None:
            raise ValueError(f"Environment {environment} is not deployed")

        self._active = environment
        self._switch_count += 1
        self._last_switch = time.time()

        logger.info("traffic_switched", environment=environment, switches=self._switch_count)

    def get_active(self) -> Any:
        """Get the currently active model.

        Returns:
            Active model instance
        """
        return self._environments[self._active]

    def get_active_environment(self) -> str:
        """Get the currently active environment name.

        Returns:
            "blue" or "green"
        """
        return self._active

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status.

        Returns:
            Dict with status information
        """
        return {
            "active": self._active,
            "blue_deployed": self._environments["blue"] is not None,
            "green_deployed": self._environments["green"] is not None,
            "switch_count": self._switch_count,
            "last_switch": self._last_switch,
        }
