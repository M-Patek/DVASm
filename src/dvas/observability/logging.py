"""Unified structured logging with context for DVAS.

Provides structured logging with automatic context injection,
correlation IDs, and configurable output formats.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

import structlog
from structlog.types import EventDict, Processor

from dvas.utils.logging import get_logger

T = TypeVar("T")

# Context variable for correlation IDs
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: Optional[str]) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def get_request_context() -> Dict[str, Any]:
    """Get the current request context."""
    return _request_context.get()


def set_request_context(context: Dict[str, Any]) -> None:
    """Set the request context."""
    _request_context.set(context)


def clear_logging_context() -> None:
    """Clear all logging context variables."""
    _correlation_id.set(None)
    _request_context.set({})


class StructuredLogger:
    """Structured logger with context injection.

    Wraps structlog to provide automatic context injection
    and correlation ID tracking.

    Usage::

        logger = StructuredLogger("dvas.pipeline")
        logger.info("processing_started", video_id="vid_001")
    """

    def __init__(self, name: str) -> None:
        self._logger = get_logger(name)
        self.name = name

    def _inject_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject correlation ID and context into log kwargs."""
        cid = get_correlation_id()
        if cid and "correlation_id" not in kwargs:
            kwargs["correlation_id"] = cid

        context = get_request_context()
        for key, value in context.items():
            if key not in kwargs:
                kwargs[key] = value

        return kwargs

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(msg, **self._inject_context(kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(msg, **self._inject_context(kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(msg, **self._inject_context(kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(msg, **self._inject_context(kwargs))

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(msg, **self._inject_context(kwargs))

    def exception(self, msg: str, exc_info: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log exception with traceback."""
        kwargs = self._inject_context(kwargs)
        if exc_info:
            kwargs["error"] = str(exc_info)
            kwargs["error_type"] = type(exc_info).__name__
        self._logger.exception(msg, **kwargs)

    def bind(self, **kwargs: Any) -> "BoundStructuredLogger":
        """Create a bound logger with additional context."""
        return BoundStructuredLogger(self, kwargs)


class BoundStructuredLogger:
    """Logger with pre-bound context."""

    def __init__(self, logger: StructuredLogger, bound_context: Dict[str, Any]) -> None:
        self._logger = logger
        self._bound_context = bound_context

    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._bound_context.copy()
        merged.update(kwargs)
        return merged

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **self._merge_context(kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, **self._merge_context(kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **self._merge_context(kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **self._merge_context(kwargs))

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._logger.critical(msg, **self._merge_context(kwargs))

    def exception(self, msg: str, exc_info: Optional[Exception] = None, **kwargs: Any) -> None:
        self._logger.exception(msg, exc_info, **self._merge_context(kwargs))


@contextmanager
def logging_context(
    correlation_id: Optional[str] = None,
    **context: Any,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for scoped logging context.

    Usage::

        with logging_context(correlation_id="abc123", video_id="vid_001"):
            logger.info("processing")  # Automatically includes context
    """
    old_cid = get_correlation_id()
    old_context = get_request_context().copy()

    cid = correlation_id or old_cid or f"auto-{threading.current_thread().ident}-{time.time()}"
    set_correlation_id(cid)

    new_context = old_context.copy()
    new_context.update(context)
    set_request_context(new_context)

    try:
        yield new_context
    finally:
        set_correlation_id(old_cid)
        set_request_context(old_context)


def log_execution_time(
    logger: Optional[StructuredLogger] = None,
    level: str = "info",
    msg: str = "execution_completed",
) -> Callable:
    """Decorator to log function execution time.

    Usage::

        @log_execution_time()
        def process_video(video_path: str) -> None:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logger or StructuredLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                getattr(log, level)(
                    msg,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration = time.time() - start
                log.error(
                    msg,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="error",
                    error=str(e),
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                getattr(log, level)(
                    msg,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration = time.time() - start
                log.error(
                    msg,
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    status="error",
                    error=str(e),
                )
                raise

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def format_log_event(event_dict: EventDict) -> str:
    """Format a log event as JSON string.

    Args:
        event_dict: Structured log event dictionary

    Returns:
        JSON string representation
    """
    return json.dumps(event_dict, ensure_ascii=False, default=str)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
