"""Task queue abstraction for DVAS API.

Provides abstract TaskQueue interface with Celery-compatible wrappers.
Supports task enqueue, dequeue, retry logic, and priority handling.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.api.task_store import InMemoryTaskStore, Task, TaskStatus, TaskStore, TaskType
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class QueueBackend(Enum):
    """Supported queue backends."""

    IN_MEMORY = "in_memory"
    CELERY = "celery"


@dataclass
class QueueConfig:
    """Configuration for task queue."""

    backend: QueueBackend = QueueBackend.IN_MEMORY
    max_retries: int = 3
    retry_delay_base_seconds: float = 2.0
    retry_delay_max_seconds: float = 300.0
    retry_backoff_multiplier: float = 2.0
    default_timeout_seconds: float = 300.0
    max_concurrent: int = 10
    poll_interval_seconds: float = 1.0


@dataclass
class EnqueueResult:
    """Result of enqueuing a task."""

    task_id: str
    status: str
    queue_position: Optional[int] = None
    estimated_wait_seconds: Optional[float] = None


class TaskQueue(ABC):
    """Abstract task queue interface."""

    @abstractmethod
    async def enqueue(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: int = 5,
        tenant_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EnqueueResult:
        pass

    @abstractmethod
    async def dequeue(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        pass

    @abstractmethod
    async def ack(self, task_id: str, success: bool = True) -> None:
        pass

    @abstractmethod
    async def retry(self, task_id: str) -> Optional[Task]:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass

    @abstractmethod
    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_queue_length(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue implementation."""

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
        task_store: Optional[TaskStore] = None,
    ) -> None:
        self.config = config or QueueConfig()
        self._store = task_store or InMemoryTaskStore()
        self._processing: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False

    def _calculate_retry_delay(self, retry_count: int) -> float:
        delay = (
            self.config.retry_delay_base_seconds
            * (self.config.retry_backoff_multiplier ** retry_count)
        )
        return min(delay, self.config.retry_delay_max_seconds)

    async def enqueue(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: int = 5,
        tenant_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EnqueueResult:
        task = Task(
            type=task_type,
            payload=payload,
            priority=priority,
            tenant_id=tenant_id,
            status=TaskStatus.PENDING,
            max_retries=self.config.max_retries,
        )
        await self._store.create(task)
        queue_len = await self.get_queue_length(task_type, tenant_id)
        logger.info(
            "task_enqueued",
            task_id=task.id,
            type=task_type.value,
            priority=priority,
        )
        return EnqueueResult(
            task_id=task.id,
            status="pending",
            queue_position=queue_len,
            estimated_wait_seconds=queue_len * self.config.poll_interval_seconds,
        )

    async def dequeue(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        async with self._lock:
            if self._shutdown:
                return None
            if len(self._processing) >= self.config.max_concurrent:
                return None
            task = await self._store.get_next_pending(task_type, tenant_id)
            if task is None:
                return None
            task.status = TaskStatus.PROCESSING
            task.updated_at = datetime.now(timezone.utc)
            await self._store.update(task)
            self._processing[task.id] = task
            logger.info("task_dequeued", task_id=task.id, type=task.type.value)
            return task

    async def ack(self, task_id: str, success: bool = True) -> None:
        async with self._lock:
            self._processing.pop(task_id, None)
        task = await self._store.get(task_id)
        if task is None:
            logger.warning("task_ack_not_found", task_id=task_id)
            return
        if success:
            task.status = TaskStatus.COMPLETED
            logger.info("task_completed", task_id=task_id)
        else:
            task.status = TaskStatus.FAILED
            logger.warning("task_failed", task_id=task_id)
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)

    async def retry(self, task_id: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task is None:
            logger.warning("task_retry_not_found", task_id=task_id)
            return None
        if not task.can_retry():
            logger.warning(
                "task_retry_exhausted",
                task_id=task_id,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
            )
            return None
        async with self._lock:
            self._processing.pop(task_id, None)
        task.retry_count += 1
        task.status = TaskStatus.PENDING
        task.error = None
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        delay = self._calculate_retry_delay(task.retry_count)
        logger.info(
            "task_retry_scheduled",
            task_id=task_id,
            retry_count=task.retry_count,
            delay_seconds=delay,
        )
        await asyncio.sleep(delay)
        return task

    async def cancel(self, task_id: str) -> bool:
        task = await self._store.get(task_id)
        if task is None:
            return False
        if task.status not in {TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RETRYING}:
            logger.warning(
                "task_cancel_invalid_state",
                task_id=task_id,
                status=task.status.name,
            )
            return False
        async with self._lock:
            self._processing.pop(task_id, None)
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)
        await self._store.update(task)
        logger.info("task_cancelled", task_id=task_id)
        return True

    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        return {
            "task_id": task.id,
            "status": task.status.name,
            "type": task.type.value,
            "progress": task.progress,
            "error": task.error,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    async def get_queue_length(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        return await self._store.count(
            status=TaskStatus.PENDING,
            task_type=task_type,
            tenant_id=tenant_id,
        )

    async def health_check(self) -> bool:
        return await self._store.health_check()

    async def shutdown(self) -> None:
        self._shutdown = True
        async with self._lock:
            for task in list(self._processing.values()):
                task.status = TaskStatus.FAILED
                task.error = "Queue shutdown"
                await self._store.update(task)
            self._processing.clear()
        logger.info("queue_shutdown")


class CeleryTaskQueue(TaskQueue):
    """Celery-compatible task queue wrapper."""

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
        broker_url: str = "redis://localhost:6379/0",
        backend_url: Optional[str] = None,
    ) -> None:
        self.config = config or QueueConfig()
        self.broker_url = broker_url
        self.backend_url = backend_url or broker_url
        self._celery_app = None
        self._store = InMemoryTaskStore()

    def _get_celery(self):
        if self._celery_app is None:
            try:
                from celery import Celery
                self._celery_app = Celery(
                    "dvas",
                    broker=self.broker_url,
                    backend=self.backend_url,
                )
            except ImportError:
                raise ImportError("celery required for CeleryTaskQueue")
        return self._celery_app

    async def enqueue(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: int = 5,
        tenant_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EnqueueResult:
        task = Task(
            type=task_type,
            payload=payload,
            priority=priority,
            tenant_id=tenant_id,
            status=TaskStatus.QUEUED,
        )
        await self._store.create(task)
        app = self._get_celery()
        result = app.send_task(
            f"dvas.tasks.{task_type.value}",
            args=[task.id, payload],
            countdown=0,
            priority=priority,
        )
        logger.info("task_enqueued_celery", task_id=task.id, celery_task_id=result.id)
        return EnqueueResult(
            task_id=task.id,
            status="queued",
            queue_position=None,
        )

    async def dequeue(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        return await self._store.get_next_pending(task_type, tenant_id)

    async def ack(self, task_id: str, success: bool = True) -> None:
        task = await self._store.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            await self._store.update(task)

    async def retry(self, task_id: str) -> Optional[Task]:
        task = await self._store.get(task_id)
        if task and task.can_retry():
            task.retry_count += 1
            task.status = TaskStatus.QUEUED
            await self._store.update(task)
            app = self._get_celery()
            app.send_task(
                f"dvas.tasks.{task.type.value}",
                args=[task.id, task.payload],
                countdown=self._calculate_retry_delay(task.retry_count),
            )
            return task
        return None

    def _calculate_retry_delay(self, retry_count: int) -> float:
        delay = (
            self.config.retry_delay_base_seconds
            * (self.config.retry_backoff_multiplier ** retry_count)
        )
        return min(delay, self.config.retry_delay_max_seconds)

    async def cancel(self, task_id: str) -> bool:
        task = await self._store.get(task_id)
        if task is None:
            return False
        task.status = TaskStatus.CANCELLED
        await self._store.update(task)
        return True

    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = await self._store.get(task_id)
        if task is None:
            return None
        return {
            "task_id": task.id,
            "status": task.status.name,
            "type": task.type.value,
            "progress": task.progress,
            "error": task.error,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
        }

    async def get_queue_length(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        return await self._store.count(
            status=TaskStatus.QUEUED,
            task_type=task_type,
            tenant_id=tenant_id,
        )

    async def health_check(self) -> bool:
        try:
            app = self._get_celery()
            return app.control.ping() is not None
        except Exception:
            return False


def create_task_queue(
    backend: QueueBackend = QueueBackend.IN_MEMORY,
    config: Optional[QueueConfig] = None,
    **kwargs: Any,
) -> TaskQueue:
    config = config or QueueConfig(backend=backend)
    if backend == QueueBackend.IN_MEMORY:
        return InMemoryTaskQueue(config, **kwargs)
    elif backend == QueueBackend.CELERY:
        return CeleryTaskQueue(config, **kwargs)
    else:
        logger.warning("unsupported_backend", backend=backend.value, falling_back="in_memory")
        return InMemoryTaskQueue(config)
