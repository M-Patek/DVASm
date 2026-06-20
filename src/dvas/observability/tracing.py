"""Unified distributed tracing for DVAS.

Provides span-based distributed tracing with context propagation
for tracking requests across service boundaries.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class Span:
    """A single span in a distributed trace.

    Represents one unit of work within a distributed trace.
    Spans can have parent-child relationships forming a trace tree.
    """

    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    service_name: str = "dvas"
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished.

        Args:
            status: Span status - "ok" or "error"
        """
        self.end_time = time.time()
        self.status = status

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the span.

        Args:
            key: Tag name
            value: Tag value
        """
        self.tags[key] = value

    def set_error(self, error_type: str, message: str) -> None:
        """Mark span as errored with details.

        Args:
            error_type: Type of error (e.g., "timeout", "validation")
            message: Error message
        """
        self.status = "error"
        self.tags["error.type"] = error_type
        self.tags["error.message"] = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "service_name": self.service_name,
            "status": self.status,
        }


class Tracer:
    """Distributed tracer for tracking requests across services.

    Usage::

        tracer = Tracer()
        span = tracer.start_span("process_video", trace_id="abc123")
        try:
            span.set_tag("video_id", "vid_001")
            # ... do work ...
        except Exception as e:
            span.set_error("processing_error", str(e))
        finally:
            span.finish()
    """

    def __init__(self, service_name: str = "dvas") -> None:
        self.service_name = service_name
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
        self._span_context = threading.local()

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **tags: str,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name describing the operation
            trace_id: Optional trace ID (generated if not provided)
            parent_id: Optional parent span ID
            **tags: Initial tags for the span

        Returns:
            The created Span
        """
        span = Span(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            service_name=self.service_name,
        )
        for key, value in tags.items():
            span.set_tag(key, value)

        with self._lock:
            self._spans.append(span)
            self._active_spans[span.span_id] = span

        # Track current span in thread-local for child spans
        self._span_context.current_span_id = span.span_id

        return span

    def finish_span(self, span: Span, status: str = "ok") -> None:
        """Finish a span.

        Args:
            span: Span to finish
            status: Completion status
        """
        span.finish(status)
        with self._lock:
            self._active_spans.pop(span.span_id, None)

    def get_spans(self) -> List[Span]:
        """Get all completed and active spans."""
        with self._lock:
            return self._spans.copy()

    def get_active_spans(self) -> List[Span]:
        """Get currently active (unfinished) spans."""
        with self._lock:
            return list(self._active_spans.values())

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace.

        Args:
            trace_id: Trace ID to look up

        Returns:
            List of spans in the trace
        """
        with self._lock:
            return [s for s in self._spans if s.trace_id == trace_id]

    def get_current_span(self) -> Optional[Span]:
        """Get the current span for this thread."""
        span_id = getattr(self._span_context, "current_span_id", None)
        if span_id:
            with self._lock:
                return self._active_spans.get(span_id)
        return None

    def reset(self) -> None:
        """Clear all spans."""
        with self._lock:
            self._spans = []
            self._active_spans = {}

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of a trace.

        Returns:
            Dict with trace statistics and span list
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "span_count": 0, "total_duration_ms": 0.0}

        durations = [s.duration_ms for s in spans if s.end_time is not None]
        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration_ms": sum(durations) if durations else 0.0,
            "root_span": spans[0].name if spans else None,
            "spans": [s.to_dict() for s in spans],
        }

    def inject_context(self) -> Dict[str, str]:
        """Extract trace context for propagation across service boundaries.

        Returns:
            Dict with trace_id, span_id for downstream services
        """
        current = self.get_current_span()
        if current:
            return {
                "trace_id": current.trace_id,
                "span_id": current.span_id,
                "service": self.service_name,
            }
        return {}

    def extract_context(self, context: Dict[str, str]) -> Optional[str]:
        """Extract trace context from incoming request.

        Args:
            context: Dict with trace context from upstream service

        Returns:
            Trace ID if found, None otherwise
        """
        return context.get("trace_id")


# Global tracer instance
_tracer_instance: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer(service_name: str = "dvas") -> Tracer:
    """Get the global tracer."""
    global _tracer_instance
    if _tracer_instance is None:
        with _tracer_lock:
            if _tracer_instance is None:
                _tracer_instance = Tracer(service_name=service_name)
    return _tracer_instance


@contextmanager
def trace_span(name: str, **tags: str) -> Generator[Span, None, None]:
    """Context manager for tracing a span.

    Usage::

        with trace_span("process_video", video_id="vid_001") as span:
            # ... do work ...
            span.set_tag("status", "success")
    """
    tracer = get_tracer()
    span = tracer.start_span(name, **tags)

    try:
        yield span
        span.finish("ok")
    except Exception as e:
        span.set_error(type(e).__name__, str(e))
        span.finish("error")
        raise
    finally:
        logger.debug(
            "span_finished",
            name=span.name,
            trace_id=span.trace_id,
            duration_ms=round(span.duration_ms, 2),
            status=span.status,
        )
