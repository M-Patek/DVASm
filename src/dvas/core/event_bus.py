"""Event bus for decoupled, event-driven architecture.

Provides both synchronous and asynchronous event publishing with
support for typed events, middleware, and persistent subscriptions.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

T = TypeVar("T")
EventType = TypeVar("EventType", bound="Event")


class EventPriority(Enum):
    """Priority levels for event processing."""

    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@dataclass(frozen=True, slots=True)
class Event:
    """Base event with metadata.

    Events are immutable and hashable for use in sets/dicts.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.event_id)


@dataclass(frozen=True, slots=True)
class VideoLoadedEvent(Event):
    """Emitted when a video is loaded."""

    video_id: str = ""
    video_path: str = ""
    duration: float = 0.0
    fps: float = 0.0


@dataclass(frozen=True, slots=True)
class SceneDetectedEvent(Event):
    """Emitted when scenes are detected."""

    video_id: str = ""
    num_scenes: int = 0
    scene_boundaries: List[tuple] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AnnotationStartedEvent(Event):
    """Emitted when annotation begins."""

    video_id: str = ""
    model_type: str = ""
    num_frames: int = 0


@dataclass(frozen=True, slots=True)
class AnnotationCompletedEvent(Event):
    """Emitted when annotation finishes."""

    video_id: str = ""
    model_type: str = ""
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class AnnotationFailedEvent(Event):
    """Emitted when annotation fails."""

    video_id: str = ""
    model_type: str = ""
    error: str = ""
    retry_count: int = 0


@dataclass(frozen=True, slots=True)
class PipelineStageCompletedEvent(Event):
    """Emitted when a pipeline stage completes."""

    video_id: str = ""
    stage_name: str = ""
    duration_ms: float = 0.0
    success: bool = True


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class EventMiddleware(ABC):
    """Base class for event middleware."""

    @abstractmethod
    async def before_publish(self, event: Event) -> Event:
        """Called before event is published. Return modified event."""
        pass

    @abstractmethod
    async def after_publish(self, event: Event) -> None:
        """Called after event is published."""
        pass


class LoggingMiddleware(EventMiddleware):
    """Logs all events at appropriate levels."""

    async def before_publish(self, event: Event) -> Event:
        logger.debug("event_publishing", event_type=type(event).__name__, event_id=event.event_id)
        return event

    async def after_publish(self, event: Event) -> None:
        logger.debug("event_published", event_type=type(event).__name__, event_id=event.event_id)


class MetricsMiddleware(EventMiddleware):
    """Collects metrics on event throughput."""

    def __init__(self) -> None:
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.event_latencies: Dict[str, List[float]] = defaultdict(list)

    async def before_publish(self, event: Event) -> Event:
        self.event_counts[type(event).__name__] += 1
        return event

    async def after_publish(self, event: Event) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return collected metrics."""
        return {
            "event_counts": dict(self.event_counts),
            "total_events": sum(self.event_counts.values()),
        }


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """Central event bus for decoupled communication.

    Supports typed subscriptions, async/sync handlers, middleware,
    and priority-based processing.

    Usage::

        bus = EventBus()

        @bus.subscribe(VideoLoadedEvent)
        async def on_video_loaded(event: VideoLoadedEvent) -> None:
            print(f"Video loaded: {event.video_id}")

        await bus.publish(VideoLoadedEvent(video_id="vid_001"))
    """

    def __init__(self) -> None:
        self._handlers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        self._middleware: List[EventMiddleware] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._queue: asyncio.Queue[tuple] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("event_bus_stopped")

    # -- Middleware ----------------------------------------------------------

    def add_middleware(self, middleware: EventMiddleware) -> None:
        """Add middleware to the event pipeline."""
        self._middleware.append(middleware)

    # -- Subscription --------------------------------------------------------

    def subscribe(
        self,
        event_type: Type[EventType],
        handler: Optional[Callable[[EventType], Any]] = None,
    ) -> Callable:
        """Subscribe a handler to an event type.

        Can be used as a decorator::

            @bus.subscribe(VideoLoadedEvent)
            async def handler(event: VideoLoadedEvent) -> None:
                ...
        """

        def decorator(func: Callable) -> Callable:
            self._handlers[event_type].append(func)
            logger.debug(
                "handler_registered",
                event_type=event_type.__name__,
                handler=func.__name__,
            )
            return func

        if handler is not None:
            return decorator(handler)
        return decorator

    def unsubscribe(
        self,
        event_type: Type[EventType],
        handler: Callable[[EventType], Any],
    ) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            return True
        return False

    # -- Publishing ----------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Events are processed asynchronously through the event loop.
        """
        # Run middleware before publish
        for mw in self._middleware:
            event = await mw.before_publish(event)

        if self._running:
            await self._queue.put(event)
        else:
            # Process immediately if not running
            await self._dispatch(event)

        # Run middleware after publish
        for mw in self._middleware:
            await mw.after_publish(event)

    async def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (blocking).

        Useful for ensuring events are processed before continuing.
        """
        await self._dispatch(event)

    # -- Internal ------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all matching handlers."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug("no_handlers", event_type=event_type.__name__)
            return

        # Run all handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(asyncio.create_task(handler(event)))
            else:
                # Run sync handlers in thread pool
                loop = asyncio.get_event_loop()
                tasks.append(asyncio.create_task(loop.run_in_executor(None, handler, event)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- Utilities -----------------------------------------------------------

    def get_subscriber_count(self, event_type: Optional[Type[Event]] = None) -> int:
        """Get number of subscribers for an event type."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values())


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_default_bus: Optional[EventBus] = None
_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the default event bus instance (singleton)."""
    global _default_bus
    if _default_bus is None:
        with _lock:
            if _default_bus is None:
                _default_bus = EventBus()
    return _default_bus


def reset_event_bus() -> None:
    """Reset the default event bus (useful for testing)."""
    global _default_bus
    _default_bus = None
