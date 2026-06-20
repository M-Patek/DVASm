"""Prometheus-compatible metrics export for DVAS.

Provides Prometheus text format exposition for all collected metrics,
including counters, gauges, histograms, and summaries.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from dvas.observability.collector import MetricsCollector, get_metrics
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Default histogram buckets matching Prometheus defaults
DEFAULT_BUCKETS = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
]


class PrometheusExporter:
    """Prometheus metrics exporter.

    Converts internal metrics to Prometheus text format for scraping.

    Usage::

        exporter = PrometheusExporter()
        text = exporter.export()  # Returns Prometheus text format
    """

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        namespace: str = "dvas",
        buckets: Optional[List[float]] = None,
    ) -> None:
        self.metrics = metrics or get_metrics()
        self.namespace = namespace
        self.buckets = buckets or DEFAULT_BUCKETS.copy()
        self._custom_collectors: List[Callable[[], str]] = []
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        """Create full metric name with namespace."""
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name

    def export(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format
        """
        lines: List[str] = []

        # Export counters
        lines.extend(self._export_counters())

        # Export gauges
        lines.extend(self._export_gauges())

        # Export histograms
        lines.extend(self._export_histograms())

        # Export summaries
        lines.extend(self._export_summaries())

        # Export custom collectors
        for collector in self._custom_collectors:
            try:
                lines.append(collector())
            except Exception as e:
                logger.error("custom_collector_failed", error=str(e))

        return "\n".join(lines)

    def _export_counters(self) -> List[str]:
        """Export counter metrics."""
        lines: List[str] = []
        stats = self.metrics.get_all_stats()

        for key, value in stats.get("counters", {}).items():
            name = self._full_name(key.split("{")[0])
            lines.append(f"# TYPE {name} counter")
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"{key.replace(key.split('{')[0], name)} {value}")

        return lines

    def _export_gauges(self) -> List[str]:
        """Export gauge metrics."""
        lines: List[str] = []
        stats = self.metrics.get_all_stats()

        for key, value in stats.get("gauges", {}).items():
            name = self._full_name(key.split("{")[0])
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"{key.replace(key.split('{')[0], name)} {value}")

        return lines

    def _export_histograms(self) -> List[str]:
        """Export histogram metrics with buckets."""
        lines: List[str] = []

        for key, data in self.metrics._histograms.items():
            values = data["values"]
            if not values:
                continue

            name = self._full_name(key.split("{")[0])
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"# HELP {name} Histogram metric")

            sorted_vals = sorted(values)
            for bucket in self.buckets:
                count = sum(1 for v in sorted_vals if v <= bucket)
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
            lines.append(f"{name}_count {len(values)}")
            lines.append(f"{name}_sum {sum(values)}")

        return lines

    def _export_summaries(self) -> List[str]:
        """Export summary metrics."""
        lines: List[str] = []

        for key, data in self.metrics._summaries.items():
            values = data["values"]
            if not values:
                continue

            name = self._full_name(key.split("{")[0])
            lines.append(f"# TYPE {name} summary")
            lines.append(f"# HELP {name} Summary metric")

            sorted_vals = sorted(values)
            n = len(sorted_vals)

            # Quantiles
            for quantile in [0.5, 0.9, 0.95, 0.99]:
                idx = min(int(n * quantile), n - 1)
                lines.append(f'{name}{{quantile="{quantile}"}} {sorted_vals[idx]}')

            lines.append(f"{name}_count {n}")
            lines.append(f"{name}_sum {sum(values)}")

        return lines

    def register_custom_collector(self, collector: Callable[[], str]) -> None:
        """Register a custom metric collector.

        Args:
            collector: Function that returns Prometheus text format lines
        """
        with self._lock:
            self._custom_collectors.append(collector)

    def unregister_custom_collector(self, collector: Callable[[], str]) -> bool:
        """Unregister a custom collector.

        Returns:
            True if collector was found and removed
        """
        with self._lock:
            if collector in self._custom_collectors:
                self._custom_collectors.remove(collector)
                return True
            return False

    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current value of a metric.

        Args:
            name: Metric name (without namespace prefix)
            labels: Optional labels

        Returns:
            Current metric value
        """
        counter = self.metrics.get_counter(name, labels)
        if counter:
            return float(counter)
        gauge = self.metrics.get_gauge(name, labels)
        if gauge:
            return gauge
        hist = self.metrics.get_histogram(name, labels)
        if hist["count"] > 0:
            return hist["avg"]
        return 0.0


class PrometheusHTTPHandler:
    """Simple HTTP handler for Prometheus metrics endpoint.

    Can be integrated with FastAPI or other ASGI/WSGI frameworks.

    Usage::

        from fastapi import FastAPI
        app = FastAPI()
        handler = PrometheusHTTPHandler()
        app.add_route("/metrics", handler.handle)
    """

    def __init__(self, exporter: Optional[PrometheusExporter] = None) -> None:
        self.exporter = exporter or PrometheusExporter()

    def handle(self) -> str:
        """Handle metrics request.

        Returns:
            Prometheus text format metrics
        """
        return self.exporter.export()

    def get_content_type(self) -> str:
        """Get the Prometheus content type."""
        return "text/plain; version=0.0.4; charset=utf-8"


def create_prometheus_metrics(
    namespace: str = "dvas",
    buckets: Optional[List[float]] = None,
) -> PrometheusExporter:
    """Create a Prometheus metrics exporter with default configuration.

    Args:
        namespace: Metric namespace prefix
        buckets: Custom histogram buckets

    Returns:
        Configured PrometheusExporter
    """
    return PrometheusExporter(namespace=namespace, buckets=buckets)
