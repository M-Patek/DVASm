"""Tests for distributed tracing."""

import time

import pytest

from dvas.observability.tracing import Span, Tracer, get_tracer, trace_span


class TestSpan:
    def test_span_creation(self):
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test_operation",
            start_time=time.time(),
        )
        assert span.trace_id == "trace-1"
        assert span.span_id == "span-1"
        assert span.name == "test_operation"
        assert span.end_time is None

    def test_span_duration(self):
        start = time.time()
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time=start,
        )
        time.sleep(0.01)
        span.finish()
        assert span.duration_ms >= 10

    def test_span_tags(self):
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time=time.time(),
        )
        span.set_tag("key", "value")
        assert span.tags["key"] == "value"

    def test_span_error(self):
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time=time.time(),
        )
        span.set_error("TimeoutError", "Request timed out")
        assert span.status == "error"
        assert span.tags["error.type"] == "TimeoutError"

    def test_span_to_dict(self):
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time=time.time(),
        )
        span.finish()
        d = span.to_dict()
        assert d["trace_id"] == "trace-1"
        assert d["name"] == "test"
        assert d["status"] == "ok"


class TestTracer:
    @pytest.fixture
    def tracer(self):
        return Tracer(service_name="test")

    def test_start_span(self, tracer):
        span = tracer.start_span("operation")
        assert span.name == "operation"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span in tracer.get_active_spans()

    def test_finish_span(self, tracer):
        span = tracer.start_span("operation")
        tracer.finish_span(span)
        assert span.end_time is not None
        assert span not in tracer.get_active_spans()

    def test_get_trace(self, tracer):
        trace_id = "test-trace"
        span1 = tracer.start_span("op1", trace_id=trace_id)
        span2 = tracer.start_span("op2", trace_id=trace_id, parent_id=span1.span_id)
        tracer.finish_span(span1)
        tracer.finish_span(span2)

        trace = tracer.get_trace(trace_id)
        assert len(trace) == 2

    def test_get_current_span(self, tracer):
        span = tracer.start_span("operation")
        current = tracer.get_current_span()
        assert current is not None
        assert current.span_id == span.span_id
        tracer.finish_span(span)

    def test_trace_summary(self, tracer):
        trace_id = "summary-trace"
        span = tracer.start_span("op", trace_id=trace_id)
        time.sleep(0.01)
        tracer.finish_span(span)

        summary = tracer.get_trace_summary(trace_id)
        assert summary["trace_id"] == trace_id
        assert summary["span_count"] == 1
        assert summary["total_duration_ms"] >= 10

    def test_context_propagation(self, tracer):
        span = tracer.start_span("parent")
        context = tracer.inject_context()
        assert context["trace_id"] == span.trace_id
        assert context["span_id"] == span.span_id
        tracer.finish_span(span)

    def test_extract_context(self, tracer):
        trace_id = "extracted-trace"
        result = tracer.extract_context({"trace_id": trace_id})
        assert result == trace_id

    def test_reset(self, tracer):
        span = tracer.start_span("op")
        tracer.finish_span(span)
        assert len(tracer.get_spans()) == 1
        tracer.reset()
        assert len(tracer.get_spans()) == 0


class TestTraceSpan:
    def test_trace_span_context_manager(self):
        with trace_span("test_operation", key="value") as span:
            assert span.name == "test_operation"
            assert span.tags["key"] == "value"
            assert span in get_tracer().get_active_spans()

    def test_trace_span_exception(self):
        tracer = Tracer()
        try:
            with trace_span("failing_op"):
                raise ValueError("test error")
        except ValueError:
            pass

    def test_trace_span_sets_status(self):
        with trace_span("success_op") as span:
            pass
        assert span.status == "ok"


class TestGlobalTracer:
    def test_get_tracer_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
