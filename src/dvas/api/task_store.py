"""Task storage backends for DVAS API.

Provides an abstract TaskStore interface and concrete implementations
for development (in-memory) and production (Redis, PostgreSQL).
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = auto()
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class TaskType(Enum):
    """Types of tasks."""

    ANNOTATION = "annotation"
    EXPORT = "export"
    BATCH = "batch"
    REVIEW = "review"
    TRAINING = "training"
    EVALUATION = "evaluation"


@dataclass
class Task:
    """Task model with full lifecycle tracking."""

    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    status: TaskStatus = TaskStatus.PENDING
    type: TaskType = TaskType.ANNOTATION
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: Optional[str] = None
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.name,
            "type": self.type.value,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        return cls(
            id=data["id"],
            status=TaskStatus[data.get("status", "PENDING")],
            type=TaskType(data.get("type", "annotation")),
            payload=data.get("payload", {}),
            result=data.get("result"),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tenant_id=data.get("tenant_id"),
            priority=data.get("priority", 5),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )

    def update_status(self, status: TaskStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def is_terminal(self) -> bool:
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.status in {
            TaskStatus.FAILED,
            TaskStatus.RETRYING,
        }


class TaskStore(ABC):
    """Abstract task storage backend interface."""

    @abstractmethod
    async def create(self, task: Task) -> Task:
        pass

    @abstractmethod
    async def get(self, task_id: str, tenant_id: Optional[str] = None) -> Optional[Task]:
        pass

    @abstractmethod
    async def update(self, task: Task) -> Task:
        pass

    @abstractmethod
    async def delete(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    async def list(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Task]:
        pass

    @abstractmethod
    async def count(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        pass

    @abstractmethod
    async def get_next_pending(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class InMemoryTaskStore(TaskStore):
    """In-memory task store for development and testing."""

    def __init__(
        self,
        max_tasks: int = 10000,
        finished_task_ttl_seconds: float = 3600.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._max_tasks = max_tasks
        self._finished_task_ttl = finished_task_ttl_seconds
        self._lock = None

    def _get_lock(self):
        if self._lock is None:
            import asyncio

            self._lock = asyncio.Lock()
        return self._lock

    def _prune_old_tasks(self) -> None:
        now = time.monotonic()
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        expired = [
            tid
            for tid, task in self._tasks.items()
            if task.status in terminal
            and (now - task.updated_at.timestamp()) > self._finished_task_ttl
        ]
        for tid in expired:
            self._tasks.pop(tid, None)
        if len(self._tasks) > self._max_tasks:
            terminal_tasks = [(tid, t) for tid, t in self._tasks.items() if t.status in terminal]
            terminal_tasks.sort(key=lambda x: x[1].updated_at.timestamp())
            overflow = len(self._tasks) - self._max_tasks
            for tid, _ in terminal_tasks[:overflow]:
                self._tasks.pop(tid, None)

    async def create(self, task: Task) -> Task:
        async with self._get_lock():
            self._prune_old_tasks()
            self._tasks[task.id] = task
            logger.info("task_created", task_id=task.id, type=task.type.value)
            return task

    async def get(self, task_id: str, tenant_id: Optional[str] = None) -> Optional[Task]:
        async with self._get_lock():
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if tenant_id is not None and task.tenant_id != tenant_id:
                return None
            return task

    async def update(self, task: Task) -> Task:
        async with self._get_lock():
            if task.id not in self._tasks:
                raise KeyError(f"Task not found: {task.id}")
            task.updated_at = datetime.now(timezone.utc)
            self._tasks[task.id] = task
            return task

    async def delete(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        async with self._get_lock():
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if tenant_id is not None and task.tenant_id != tenant_id:
                return False
            self._tasks.pop(task_id, None)
            return True

    async def list(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Task]:
        async with self._get_lock():
            tasks = list(self._tasks.values())
            if tenant_id is not None:
                tasks = [t for t in tasks if t.tenant_id == tenant_id]
            if status is not None:
                tasks = [t for t in tasks if t.status == status]
            if task_type is not None:
                tasks = [t for t in tasks if t.type == task_type]
            tasks.sort(key=lambda t: (t.priority, t.created_at))
            return tasks[offset : offset + limit]

    async def count(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        async with self._get_lock():
            count = 0
            for task in self._tasks.values():
                if tenant_id is not None and task.tenant_id != tenant_id:
                    continue
                if status is not None and task.status != status:
                    continue
                if task_type is not None and task.type != task_type:
                    continue
                count += 1
            return count

    async def get_next_pending(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        async with self._get_lock():
            pending = [
                t
                for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
                and (task_type is None or t.type == task_type)
                and (tenant_id is None or t.tenant_id == tenant_id)
            ]
            if not pending:
                return None
            pending.sort(key=lambda t: (t.priority, t.created_at))
            return pending[0]

    async def health_check(self) -> bool:
        return True


class RedisTaskStore(TaskStore):
    """Redis-backed task store (placeholder for production)."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._key_prefix = "dvas:task:"

    def _task_key(self, task_id: str) -> str:
        """Generate Redis key for a task."""
        return f"{self._key_prefix}{task_id}"

    def _tenant_key(self, tenant_id: str) -> str:
        """Generate Redis key for tenant task list."""
        return f"{self._key_prefix}tenant:{tenant_id}"

    async def create(self, task: Task) -> Task:
        raise NotImplementedError

    async def get(self, task_id: str, tenant_id: Optional[str] = None) -> Optional[Task]:
        raise NotImplementedError

    async def update(self, task: Task) -> Task:
        raise NotImplementedError

    async def delete(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        raise NotImplementedError

    async def list(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Task]:
        raise NotImplementedError

    async def count(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        raise NotImplementedError

    async def get_next_pending(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError


class PostgresTaskStore(TaskStore):
    """PostgreSQL-backed task store (placeholder for production)."""

    def __init__(self, dsn: str = "postgresql://localhost/dvas") -> None:
        self._dsn = dsn
        self._table_name = "tasks"

    def _row_to_task(self, row) -> Task:
        """Convert a database row to a Task object."""
        if hasattr(row, "_data"):
            data = dict(row._data)
            # Convert datetime objects to ISO strings for from_dict
            for key in ["created_at", "updated_at"]:
                if key in data and hasattr(data[key], "isoformat"):
                    data[key] = data[key].isoformat()
            return Task.from_dict(data)
        return Task.from_dict(dict(row))

    async def create(self, task: Task) -> Task:
        raise NotImplementedError

    async def get(self, task_id: str, tenant_id: Optional[str] = None) -> Optional[Task]:
        raise NotImplementedError

    async def update(self, task: Task) -> Task:
        raise NotImplementedError

    async def delete(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        raise NotImplementedError

    async def list(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Task]:
        raise NotImplementedError

    async def count(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        raise NotImplementedError

    async def get_next_pending(
        self,
        task_type: Optional[TaskType] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Task]:
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError
