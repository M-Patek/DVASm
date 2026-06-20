"""Tests for structured logging."""

import pytest

from dvas.observability.logging import (
    BoundStructuredLogger,
    StructuredLogger,
    get_correlation_id,
    get_request_context,
    get_structured_logger,
    logging_context,
    set_correlation_id,
    set_request_context,
)


class TestStructuredLogger:
    def test_logger_creation(self):
        logger = StructuredLogger("test.module")
        assert logger.name == "test.module"

    def test_bind(self):
        logger = StructuredLogger("test")
        bound = logger.bind(video_id="vid_001")
        assert isinstance(bound, BoundStructuredLogger)


class TestCorrelationId:
    def test_set_and_get(self):
        set_correlation_id("test-cid-123")
        assert get_correlation_id() == "test-cid-123"
        set_correlation_id(None)

    def test_default_is_none(self):
        set_correlation_id(None)
        assert get_correlation_id() is None


class TestRequestContext:
    def test_set_and_get(self):
        set_request_context({"video_id": "vid_001"})
        ctx = get_request_context()
        assert ctx["video_id"] == "vid_001"
        set_request_context({})

    def test_default_empty(self):
        set_request_context({})
        assert get_request_context() == {}


class TestLoggingContext:
    def test_context_manager(self):
        with logging_context(correlation_id="ctx-123", video_id="vid_001"):
            assert get_correlation_id() == "ctx-123"
            assert get_request_context()["video_id"] == "vid_001"

    def test_context_cleanup(self):
        set_correlation_id("before")
        with logging_context(correlation_id="during"):
            pass
        assert get_correlation_id() == "before"

    def test_nested_context(self):
        with logging_context(correlation_id="outer"):
            assert get_correlation_id() == "outer"
            with logging_context(correlation_id="inner"):
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"

    def test_auto_correlation_id(self):
        set_correlation_id(None)
        with logging_context() as ctx:
            cid = get_correlation_id()
            assert cid is not None
            assert cid.startswith("auto-")
        set_correlation_id(None)


class TestGetStructuredLogger:
    def test_returns_logger(self):
        logger = get_structured_logger("test")
        assert isinstance(logger, StructuredLogger)
