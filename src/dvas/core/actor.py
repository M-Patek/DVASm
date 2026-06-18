"""Actor model implementation for concurrent state management.

Provides actor-based concurrency where each actor has its own
mailbox and processes messages sequentially, eliminating shared state.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Actor types
# ---------------------------------------------------------------------------

class ActorStatus(Enum):
    """Status of an actor."""

    IDLE = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass
class ActorMessage:
    """A message sent to an actor."""

    sender: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))


class Actor(ABC):
    """Base class for actors.

    Actors process messages from a mailbox sequentially,
    ensuring no shared state conflicts.

    Usage::

        class VideoProcessor(Actor):
            async def handle_message(self, message: ActorMessage) -> None:
                video_path = message.payload
                await self.process_video(video_path)

        actor = VideoProcessor("video_processor")
        await actor.start()
        await actor.send(ActorMessage(sender="pipeline", payload="/path/to/video.mp4"))
    """

    def __init__(self, name: str, mailbox_size: int = 1000) -> None:
        self.name = name
        self.status = ActorStatus.IDLE
        self._mailbox: asyncio.Queue[ActorMessage] = asyncio.Queue(maxsize=mailbox_size)
        self._task: Optional[asyncio.Task] = None
        self._message_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the actor's message processing loop."""
        if self.status != ActorStatus.IDLE:
            return

        self.status = ActorStatus.RUNNING
        self._task = asyncio.create_task(self._process_loop())
        logger.info("actor_started", name=self.name)

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the actor gracefully."""
        if self.status != ActorStatus.RUNNING:
            return

        self.status = ActorStatus.STOPPING

        # Wait for mailbox to drain
        try:
            await asyncio.wait_for(self._mailbox.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("actor_stop_timeout", name=self.name, pending=self._mailbox.qsize())

        # Cancel processing task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self.status = ActorStatus.STOPPED
        logger.info("actor_stopped", name=self.name, messages=self._message_count)

    async def send(self, message: ActorMessage, timeout: Optional[float] = None) -> bool:
        """Send a message to the actor's mailbox.

        Returns True if the message was queued, False if the mailbox is full.
        """
        try:
            if timeout:
                await asyncio.wait_for(self._mailbox.put(message), timeout=timeout)
            else:
                self._mailbox.put_nowait(message)
            return True
        except (asyncio.QueueFull, asyncio.TimeoutError):
            logger.warning("actor_mailbox_full", name=self.name, sender=message.sender)
            return False

    async def _process_loop(self) -> None:
        """Main message processing loop."""
        while self.status == ActorStatus.RUNNING:
            try:
                message = await asyncio.wait_for(self._mailbox.get(), timeout=1.0)
                self._message_count += 1

                try:
                    await self.handle_message(message)
                except Exception as e:
                    self._error_count += 1
                    logger.error(
                        "actor_message_error",
                        name=self.name,
                        message_id=message.message_id,
                        error=str(e),
                    )

                self._mailbox.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def handle_message(self, message: ActorMessage) -> None:
        """Handle a message. Must be implemented by subclasses."""
        pass

    @property
    def stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "name": self.name,
            "status": self.status.name,
            "messages_processed": self._message_count,
            "errors": self._error_count,
            "mailbox_size": self._mailbox.qsize(),
        }


class ActorSystem:
    """Manager for a collection of actors.

    Provides supervision, routing, and lifecycle management.
    """

    def __init__(self) -> None:
        self._actors: Dict[str, Actor] = {}
        self._running = False

    def register(self, actor: Actor) -> None:
        """Register an actor with the system."""
        self._actors[actor.name] = actor

    async def start_all(self) -> None:
        """Start all registered actors."""
        self._running = True
        await asyncio.gather(*[actor.start() for actor in self._actors.values()])
        logger.info("actor_system_started", actors=len(self._actors))

    async def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all registered actors."""
        self._running = False
        await asyncio.gather(*[actor.stop(timeout) for actor in self._actors.values()])
        logger.info("actor_system_stopped")

    async def send(self, actor_name: str, message: ActorMessage) -> bool:
        """Send a message to a specific actor."""
        actor = self._actors.get(actor_name)
        if actor is None:
            logger.error("actor_not_found", name=actor_name)
            return False
        return await actor.send(message)

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all actors."""
        return {name: actor.stats for name, actor in self._actors.items()}


# ---------------------------------------------------------------------------
# Pre-built actors for annotation pipeline
# ---------------------------------------------------------------------------

class VideoLoaderActor(Actor):
    """Actor for loading videos."""

    def __init__(self, name: str = "video_loader") -> None:
        super().__init__(name)
        self._loaded_videos: Dict[str, Any] = {}

    async def handle_message(self, message: ActorMessage) -> None:
        video_path = message.payload
        # Load video logic here
        self._loaded_videos[video_path] = {"loaded_at": time.time()}
        logger.info("video_loaded_by_actor", path=video_path)


class AnnotationWorkerActor(Actor):
    """Actor for processing annotations."""

    def __init__(self, name: str = "annotation_worker") -> None:
        super().__init__(name)
        self._processed_count = 0

    async def handle_message(self, message: ActorMessage) -> None:
        video_id = message.payload
        # Annotation logic here
        self._processed_count += 1
        logger.info("annotation_processed_by_actor", video_id=video_id)
