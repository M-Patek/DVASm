"""Observability and monitoring for DVAS.

Provides Prometheus metrics, distributed tracing, and structured logging
for production monitoring.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Simple metrics collector (Prometheus-compatible).

    Provides counters, gauges, histograms, and summaries
    for monitoring application performance.

    Usage::

        metrics = MetricsCollector()
        metrics.increment("requests_total", labels={"method": "GET"})
        metrics.observe("request_duration_seconds", 0.5)
    """

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, int]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, List[float]]] = {}
        self._summaries: Dict[str, Dict[str, List[float]]] = {}

    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = {}
        self._counters[key]["value"] = self._counters[key].get("value", 0) + value

    def decrement(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement a counter."""
        self.increment(name, -value, labels)

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = {}
        self._gauges[key]["value"] = value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a histogram."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = {"values": []}
        self._histograms[key]["values"].append(value)

    def summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a summary value."""
        key = self._make_key(name, labels)
        if key not in self._summaries:
            self._summaries[key] = {"values": []}
        self._summaries[key]["values"].append(value)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, {}).get("value", 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, {}).get("value", 0.0)

    def get_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, {}).get("values", [])
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
        """Export metrics in Prometheus text format."""
        lines = []

        # Counters
        for key, data in self._counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {data['value']}")

        # Gauges
        for key, data in self._gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {data['value']}")

        # Histograms
        for key, data in self._histograms.items():
            values = data["values"]
            if values:
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")

        return "\n".join(lines)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": {k: v["value"] for k, v in self._counters.items()},
            "gauges": {k: v["value"] for k, v in self._gauges.items()},
            "histograms": {k: self.get_histogram(k.split("{")[0]) for k in self._histograms.keys()},
        }


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


def reset_metrics() -> None:
    """Reset all metrics."""
    global _metrics_instance
    _metrics_instance = MetricsCollector()


# ---------------------------------------------------------------------------
# Timing Decorator
# ---------------------------------------------------------------------------


def timed(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution.

    Usage::

        @timed("process_video_duration", labels={"model": "gpt55"})
        async def process_video(video_path):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                name = metric_name or func.__name__
                get_metrics().observe(name, duration, labels)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                name = metric_name or func.__name__
                get_metrics().observe(name, duration, labels)

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Counter Decorator
# ---------------------------------------------------------------------------


def counted(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls.

    Usage::

        @counted("annotations_generated", labels={"model": "gpt55"})
        async def generate_annotation(video_path):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            name = metric_name or func.__name__
            get_metrics().increment(name, labels=labels)
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            name = metric_name or func.__name__
            get_metrics().increment(name, labels=labels)
            return await func(*args, **kwargs)

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Distributed Tracing (simplified)
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single span in a distributed trace."""

    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def finish(self) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the span."""
        self.tags[key] = value


class Tracer:
    """Simple distributed tracer.

    Usage::

        tracer = Tracer()
        span = tracer.start_span("process_video", trace_id="abc123")
        try:
            # ... do work ...
            span.set_tag("video_id", "vid_001")
        finally:
            span.finish()
    """

    def __init__(self) -> None:
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}

    def start_span(self, name: str, trace_id: Optional[str] = None) -> Span:
        """Start a new span."""
        import uuid

        span = Span(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
            name=name,
            start_time=time.time(),
        )
        self._spans.append(span)
        self._active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span) -> None:
        """Finish a span."""
        span.finish()
        self._active_spans.pop(span.span_id, None)

    def get_spans(self) -> List[Span]:
        """Get all spans."""
        return self._spans.copy()

    def get_active_spans(self) -> List[Span]:
        """Get currently active spans."""
        return list(self._active_spans.values())

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return [s for s in self._spans if s.trace_id == trace_id]

    def reset(self) -> None:
        """Clear all spans."""
        self._spans = []
        self._active_spans = {}


# Global tracer instance
_tracer_instance: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = Tracer()
    return _tracer_instance


# ---------------------------------------------------------------------------
# Context manager for tracing
# ---------------------------------------------------------------------------


@contextmanager
def trace_span(name: str, **tags: str):
    """Context manager for tracing a span.

    Usage::

        with trace_span("process_video", video_id="vid_001"):
            # ... do work ...
    """
    tracer = get_tracer()
    span = tracer.start_span(name)
    for key, value in tags.items():
        span.set_tag(key, value)

    try:
        yield span
    finally:
        span.finish()
        logger.debug(
            "span_finished",
            name=span.name,
            trace_id=span.trace_id,
            duration_ms=round(span.duration_ms, 2),
        )


# ---------------------------------------------------------------------------
# Performance Monitoring
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Monitor application performance.

    Tracks throughput, latency, and error rates.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._requests: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def record_request(self, name: str, duration_ms: float, status: str = "success") -> None:
        """Record a request."""
        self._requests.append(
            {
                "name": name,
                "duration_ms": duration_ms,
                "status": status,
                "timestamp": time.time(),
            }
        )

        # Keep only last window_size requests
        if len(self._requests) > self.window_size:
            self._requests = self._requests[-self.window_size :]

    def record_error(self, name: str, error: str) -> None:
        """Record an error."""
        self._errors.append(
            {
                "name": name,
                "error": error,
                "timestamp": time.time(),
            }
        )

        # Keep only last window_size errors
        if len(self._errors) > self.window_size:
            self._errors = self._errors[-self.window_size :]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._requests:
            return {
                "total_requests": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "throughput_rps": 0.0,
            }

        durations = [r["duration_ms"] for r in self._requests]
        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        total_time = time.time() - self._start_time
        error_count = len(self._errors)
        total_requests = len(self._requests)

        return {
            "total_requests": total_requests,
            "error_count": error_count,
            "error_rate": error_count / max(total_requests, 1),
            "avg_latency_ms": sum(durations) / n,
            "p50_latency_ms": sorted_durations[n // 2],
            "p95_latency_ms": sorted_durations[int(n * 0.95)],
            "p99_latency_ms": sorted_durations[int(n * 0.99)],
            "min_latency_ms": sorted_durations[0],
            "max_latency_ms": sorted_durations[-1],
            "throughput_rps": total_requests / max(total_time, 1),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._requests = []
        self._errors = []
        self._start_time = time.time()


# Global performance monitor
_perf_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = PerformanceMonitor()
    return _perf_monitor
