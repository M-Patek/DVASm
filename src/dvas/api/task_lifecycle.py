"""Task lifecycle management for DVAS API.

Provides task retry with exponential backoff, cancellation, resume from checkpoint,
and progress tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from dvas.api.task_store import Task, TaskStatus, TaskStore
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class RetryPolicy(Enum):
    """Retry policy types."""

    EXPONENTIAL_BACKOFF = auto()
    FIXED_DELAY = auto()
    LINEAR_BACKOFF = auto()
    NO_RETRY = auto()


@dataclass
class RetryConfig:
    """Configuration for task retry behavior."""

    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for a given retry count."""
        if self.policy == RetryPolicy.NO_RETRY:
            return 0.0
        if self.policy == RetryPolicy.FIXED_DELAY:
            delay = self.base_delay_seconds
        elif self.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.base_delay_seconds * retry_count
        else:
            delay = self.base_delay_seconds * (self.backoff_multiplier ** retry_count)
        delay = min(delay, self.max_delay_seconds)
        if self.jitter:
            import random
            jitter_factor = 0.75 + random.random() * 0.5
            delay *= jitter_factor
        return delay


@dataclass
class Checkpoint:
    """Task checkpoint for resume capability."""

    task_id: str
    state: Dict[str, Any]
    progress: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    segment_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "progress": self.progress,
            "timestamp": self.timestamp.isoformat(),
            "segment_index": self.segment_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        return cls(
            task_id=data["task_id"],
            state=data.get("state", {}),
            progress=data.get("progress", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            segment_index=data.get("segment_index"),
        )


class TaskLifecycleManager:
    """Manages task lifecycle: retry, cancel, resume, progress."""

    def __init__(
        self,
        task_store: TaskStore,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self._store = task_store
        self._retry_config = retry_config or RetryConfig()
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._progress_callbacks: Dict[str, List[Callable[[str, float], None]]] = {}
        self._cancelled_tasks: set = set()
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        task_type: Any,
        payload: Dict[str, Any],
        priority: int = 5,
        tenant_id: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> Task:
        from dvas.api.task_store import TaskType
        task = Task(
            type=task_type if isinstance(task_type, TaskType) else TaskType.ANNOTATION,
            payload=payload,
            priority=priority,
            tenant_id=tenant_id,
            status=TaskStatus.PENDING,
            max_retries=max_retries or self._retry_config.max_retries,
        )
        await self._store.create(task)
        logger.info("lifecycle_task_created", task_id=task.id, type=task.type.value)
        return task

    async def start_processing(self, task_id: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        if task_id in self._cancelled_tasks:
            logger.warning("task_start_cancelled", task_id=task_id)
            return None
        task.status = TaskStatus.PROCESSING
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        logger.info("lifecycle_task_processing", task_id=task_id)
        return task

    async def update_progress(self, task_id: str, progress: float) -> None:
        task = await self._store.get(task_id)
        if task is None:
            return
        task.progress = max(0.0, min(100.0, progress))
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        for callback in self._progress_callbacks.get(task_id, []):
            try:
                callback(task_id, task.progress)
            except Exception as e:
                logger.warning("progress_callback_error", task_id=task_id, error=str(e))
        logger.debug("lifecycle_progress_updated", task_id=task_id, progress=task.progress)

    async def save_checkpoint(
        self,
        task_id: str,
        state: Dict[str, Any],
        progress: float,
        segment_index: Optional[int] = None,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            task_id=task_id,
            state=state,
            progress=progress,
            segment_index=segment_index,
        )
        self._checkpoints[task_id] = checkpoint
        await self.update_progress(task_id, progress)
        logger.info(
            "lifecycle_checkpoint_saved",
            task_id=task_id,
            progress=progress,
            segment_index=segment_index,
        )
        return checkpoint

    async def get_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        return self._checkpoints.get(task_id)

    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.progress = 100.0
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        self._checkpoints.pop(task_id, None)
        self._progress_callbacks.pop(task_id, None)
        self._cancelled_tasks.discard(task_id)
        logger.info("lifecycle_task_completed", task_id=task_id)
        return task

    async def fail_task(self, task_id: str, error: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.FAILED
        task.error = error
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        logger.warning("lifecycle_task_failed", task_id=task_id, error=error)
        return task

    async def retry_task(self, task_id: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            logger.warning("retry_task_not_found", task_id=task_id)
            return None
        if not task.can_retry():
            logger.warning(
                "retry_exhausted",
                task_id=task_id,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
            )
            return None
        if self._retry_config.policy == RetryPolicy.NO_RETRY:
            logger.warning("retry_disabled", task_id=task_id)
            return None
        delay = self._retry_config.calculate_delay(task.retry_count)
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        task.error = None
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        logger.info(
            "lifecycle_task_retrying",
            task_id=task_id,
            retry_count=task.retry_count,
            delay_seconds=delay,
        )
        if delay > 0:
            await asyncio.sleep(delay)
        if task_id in self._cancelled_tasks:
            logger.warning("retry_cancelled_during_wait", task_id=task_id)
            return None
        task.status = TaskStatus.PENDING
        await self._store.update(task)
        logger.info("lifecycle_task_retry_ready", task_id=task_id)
        return task

    async def cancel_task(self, task_id: str) -> bool:
        async with self._lock:
            task = await self._store.get(task_id)
            if task is None:
                return False
            if task.is_terminal():
                logger.warning("cancel_task_terminal", task_id=task_id, status=task.status.name)
                return False
            self._cancelled_tasks.add(task_id)
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.now(timezone.utc)
            await self._store.update(task)
            logger.info("lifecycle_task_cancelled", task_id=task_id)
            return True

    async def resume_task(self, task_id: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        checkpoint = self._checkpoints.get(task_id)
        if checkpoint is None:
            logger.warning("resume_no_checkpoint", task_id=task_id)
            task.status = TaskStatus.PENDING
            task.progress = 0.0
            await self._store.update(task)
            return task
        task.status = TaskStatus.PENDING
        task.progress = checkpoint.progress
        task.error = None
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        logger.info(
            "lifecycle_task_resumed",
            task_id=task_id,
            progress=checkpoint.progress,
            segment_index=checkpoint.segment_index,
        )
        return task

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        checkpoint = self._checkpoints.get(task_id)
        return {
            "task_id": task.id,
            "status": task.status.name,
            "type": task.type.value,
            "progress": task.progress,
            "error": task.error,
            "result": task.result,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "can_retry": task.can_retry(),
            "is_terminal": task.is_terminal(),
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    def register_progress_callback(
        self, task_id: str, callback: Callable[[str, float], None]
    ) -> None:
        if task_id not in self._progress_callbacks:
            self._progress_callbacks[task_id] = []
        self._progress_callbacks[task_id].append(callback)

    def unregister_progress_callback(
        self, task_id: str, callback: Callable[[str, float], None]
    ) -> None:
        if task_id in self._progress_callbacks:
            self._progress_callbacks[task_id] = [
                cb for cb in self._progress_callbacks[task_id] if cb is not callback
            ]

    async def stream_progress(self, task_id: str) -> AsyncIterator[Dict[str, Any]]:
        task = await self._store.get(task_id)
        if task is None:
            yield {"error": "Task not found", "task_id": task_id}
            return
        last_progress = -1.0
        while True:
            task = await self._store.get(task_id)
            if task is None:
                yield {"error": "Task disappeared", "task_id": task_id}
                break
            if task.progress != last_progress or task.is_terminal():
                last_progress = task.progress
                yield {
                    "task_id": task_id,
                    "status": task.status.name,
                    "progress": task.progress,
                    "error": task.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            if task.is_terminal():
                break
            await asyncio.sleep(0.5)

    async def get_stats(self) -> Dict[str, Any]:
        total = await self._store.count()
        pending = await self._store.count(status=TaskStatus.PENDING)
        processing = await self._store.count(status=TaskStatus.PROCESSING)
        completed = await self._store.count(status=TaskStatus.COMPLETED)
        failed = await self._store.count(status=TaskStatus.FAILED)
        cancelled = await self._store.count(status=TaskStatus.CANCELLED)
        retrying = await self._store.count(status=TaskStatus.RETRYING)
        return {
            "total_tasks": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "retrying": retrying,
            "checkpoints_stored": len(self._checkpoints),
            "active_callbacks": sum(len(cbs) for cbs in self._progress_callbacks.values()),
        }
