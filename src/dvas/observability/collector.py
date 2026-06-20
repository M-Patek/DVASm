"""Unified metrics collector for DVAS.

Centralized metrics collection supporting counters, gauges, histograms,
and summaries with label support for dimensional monitoring.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class MetricValue:
    """A single metric value with optional labels."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Centralized metrics collector with dimensional support.

    Provides counters, gauges, histograms, and summaries for monitoring
    application performance. All operations are thread-safe.

    Usage::

        collector = MetricsCollector()
        collector.increment("requests_total", labels={"method": "GET"})
        collector.gauge("active_connections", 42.0)
        collector.observe("request_duration_seconds", 0.5)
    """

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: {"value": 0})
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: {"value": 0.0})
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"values": []}
        )
        self._summaries: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"values": []}
        )
        self._lock = threading.RLock()
        self._registered_callbacks: Dict[str, Callable[[], float]] = {}

    def increment(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment by (default: 1)
            labels: Optional dimension labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key]["value"] = self._counters[key].get("value", 0) + value

    def decrement(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Decrement a counter metric.

        Args:
            name: Metric name
            value: Amount to decrement by (default: 1)
            labels: Optional dimension labels
        """
        self.increment(name, -value, labels)

    def gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional dimension labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key]["value"] = value

    def observe(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a value in a histogram.

        Args:
            name: Metric name
            value: Value to observe
            labels: Optional dimension labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key]["values"].append(value)
            # Keep only last 10,000 values to prevent unbounded growth
            if len(self._histograms[key]["values"]) > 10000:
                self._histograms[key]["values"] = self._histograms[key]["values"][-10000:]

    def summary(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a summary value.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional dimension labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._summaries[key]["values"].append(value)
            # Keep only last 10,000 values
            if len(self._summaries[key]["values"]) > 10000:
                self._summaries[key]["values"] = self._summaries[key]["values"][-10000:]

    def register_callback(self, name: str, callback: Callable[[], float]) -> None:
        """Register a callback for dynamic gauge values.

        Args:
            name: Metric name
            callback: Function that returns the current value
        """
        self._registered_callbacks[name] = callback

    def unregister_callback(self, name: str) -> None:
        """Unregister a callback."""
        self._registered_callbacks.pop(name, None)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, {}).get("value", 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key, {}).get("value", 0.0)

    def get_histogram(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get histogram statistics.

        Returns a dict with count, sum, avg, min, max, p50, p95, p99.
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, {}).get("values", [])

        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(sorted_values),
            "avg": sum(sorted_values) / n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(n * 0.50)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[min(int(n * 0.99), n - 1)],
        }

    def get_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        key = self._make_key(name, labels)
        with self._lock:
            values = self._summaries.get(key, {}).get("values", [])

        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns a string in Prometheus exposition format.
        """
        lines: List[str] = []

        with self._lock:
            # Counters
            for key, data in self._counters.items():
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{key} {data['value']}")

            # Gauges
            for key, data in self._gauges.items():
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{key} {data['value']}")

            # Histograms
            for key, data in self._histograms.items():
                values = data["values"]
                if values:
                    name = key.split("{")[0]
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")
                    # Calculate buckets
                    sorted_vals = sorted(values)
                    for bucket in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:
                        count = sum(1 for v in sorted_vals if v <= bucket)
                        lines.append(f'{key}_bucket{{le="{bucket}"}} {count}')
                    lines.append(f'{key}_bucket{{le="+Inf"}} {len(values)}')

            # Summaries
            for key, data in self._summaries.items():
                values = data["values"]
                if values:
                    name = key.split("{")[0]
                    lines.append(f"# TYPE {name} summary")
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")

        return "\n".join(lines)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": {k: v["value"] for k, v in self._counters.items()},
                "gauges": {k: v["value"] for k, v in self._gauges.items()},
                "histograms": {
                    k: self.get_histogram(k.split("{")[0]) for k in self._histograms.keys()
                },
                "summaries": {
                    k: self.get_summary(k.split("{")[0]) for k in self._summaries.keys()
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()

    def get_metric_names(self) -> List[str]:
        """Get all registered metric names."""
        with self._lock:
            names: set[str] = set()
            for key in self._counters:
                names.add(key.split("{")[0])
            for key in self._gauges:
                names.add(key.split("{")[0])
            for key in self._histograms:
                names.add(key.split("{")[0])
            for key in self._summaries:
                names.add(key.split("{")[0])
            return sorted(names)


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()
    return _metrics_instance


def reset_metrics() -> None:
    """Reset all metrics."""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = MetricsCollector()
