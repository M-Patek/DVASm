"""Teacher latency monitoring for DVAS.

Tracks and alerts on teacher model response latency with
percentile calculations and threshold-based alerting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dvas.observability.collector import MetricsCollector, get_metrics
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyThreshold:
    """Latency threshold configuration."""

    p50_ms: float = 1000.0
    p95_ms: float = 5000.0
    p99_ms: float = 10000.0


class TeacherLatencyMonitor:
    """Monitor teacher model latency with alerting.

    Tracks response times for teacher models and triggers alerts
    when latency exceeds configured thresholds.

    Usage::

        monitor = TeacherLatencyMonitor()
        monitor.record_latency("gpt-5.5", 2500.0)
        stats = monitor.get_latency_stats("gpt-5.5")
    """

    def __init__(
        self,
        thresholds: Optional[LatencyThreshold] = None,
        max_samples: int = 10000,
    ) -> None:
        self.thresholds = thresholds or LatencyThreshold()
        self.max_samples = max_samples
        self._latencies: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_latency(self, model_name: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            model_name: Name of the teacher model
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if model_name not in self._latencies:
                self._latencies[model_name] = []
            self._latencies[model_name].append(latency_ms)
            # Keep only last max_samples
            if len(self._latencies[model_name]) > self.max_samples:
                self._latencies[model_name] = self._latencies[model_name][-self.max_samples :]

        # Also record in global metrics
        get_metrics().observe(
            "teacher_latency",
            latency_ms / 1000.0,  # Convert to seconds for Prometheus
            labels={"model": model_name},
        )
        get_metrics().increment(
            "teacher_requests_total",
            labels={"model": model_name},
        )

        # Check thresholds
        self._check_thresholds(model_name, latency_ms)

    def _check_thresholds(self, model_name: str, latency_ms: float) -> None:
        """Check if latency exceeds thresholds and trigger alerts."""
        if latency_ms > self.thresholds.p99_ms:
            self._trigger_alert(
                "latency_p99_exceeded",
                {
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "threshold_ms": self.thresholds.p99_ms,
                    "severity": "critical",
                },
            )
        elif latency_ms > self.thresholds.p95_ms:
            self._trigger_alert(
                "latency_p95_exceeded",
                {
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "threshold_ms": self.thresholds.p95_ms,
                    "severity": "warning",
                },
            )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "teacher_latency_alert",
            alert_type=alert_type,
            **details,
        )
        for handler in self._alert_handlers:
            try:
                handler(alert_type, details)
            except Exception as e:
                logger.error("alert_handler_failed", error=str(e))

    def add_alert_handler(
        self, handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Add an alert handler callback.

        Args:
            handler: Function called with (alert_type, details) when threshold exceeded
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(
        self, handler: Callable[[str, Dict[str, Any]], None]
    ) -> bool:
        """Remove an alert handler.

        Returns:
            True if handler was found and removed
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            return True
        return False

    def get_latency_stats(self, model_name: str) -> Dict[str, Any]:
        """Get latency statistics for a model.

        Args:
            model_name: Name of the teacher model

        Returns:
            Dict with count, p50, p95, p99, min, max, avg
        """
        with self._lock:
            values = self._latencies.get(model_name, [])

        if not values:
            return {
                "model": model_name,
                "count": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "avg_ms": 0.0,
            }

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "model": model_name,
            "count": n,
            "p50_ms": sorted_values[int(n * 0.50)],
            "p95_ms": sorted_values[int(n * 0.95)],
            "p99_ms": sorted_values[min(int(n * 0.99), n - 1)],
            "min_ms": sorted_values[0],
            "max_ms": sorted_values[-1],
            "avg_ms": sum(sorted_values) / n,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get latency statistics for all models."""
        with self._lock:
            models = list(self._latencies.keys())
        return {model: self.get_latency_stats(model) for model in models}

    def is_healthy(self, model_name: str) -> bool:
        """Check if a model's latency is within acceptable thresholds.

        Args:
            model_name: Name of the teacher model

        Returns:
            True if p95 latency is below threshold
        """
        stats = self.get_latency_stats(model_name)
        if stats["count"] == 0:
            return True
        return stats["p95_ms"] <= self.thresholds.p95_ms

    def get_slowest_models(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the n slowest models by average latency.

        Args:
            n: Number of models to return

        Returns:
            List of model stats sorted by avg latency descending
        """
        all_stats = self.get_all_stats().values()
        return sorted(all_stats, key=lambda x: x["avg_ms"], reverse=True)[:n]

    def reset(self, model_name: Optional[str] = None) -> None:
        """Reset latency data.

        Args:
            model_name: Optional model to reset (all if None)
        """
        with self._lock:
            if model_name:
                self._latencies.pop(model_name, None)
            else:
                self._latencies.clear()


class TeacherLatencyTracker:
    """Context manager for tracking latency of teacher model calls.

    Usage::

        with TeacherLatencyTracker("gpt-5.5") as tracker:
            result = await teacher.annotate(frames)
            # Latency automatically recorded on exit
    """

    def __init__(
        self,
        model_name: str,
        monitor: Optional[TeacherLatencyMonitor] = None,
    ) -> None:
        self.model_name = model_name
        self.monitor = monitor or TeacherLatencyMonitor()
        self.start_time: Optional[float] = None
        self.latency_ms: Optional[float] = None

    def __enter__(self) -> "TeacherLatencyTracker":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            self.latency_ms = (time.perf_counter() - self.start_time) * 1000
            self.monitor.record_latency(self.model_name, self.latency_ms)

    async def __aenter__(self) -> "TeacherLatencyTracker":
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            self.latency_ms = (time.perf_counter() - self.start_time) * 1000
            self.monitor.record_latency(self.model_name, self.latency_ms)
