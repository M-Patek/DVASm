"""Redis-based task queue and state management."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task status states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class AnnotationTask:
    """Annotation task definition."""

    task_id: str
    video_id: str
    status: TaskStatus = TaskStatus.PENDING
    teacher_model: str = "gpt-5.5"
    num_frames: int = 16
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AnnotationTask":
        data["status"] = TaskStatus(data.get("status", "pending"))
        return cls(**data)


class TaskQueue:
    """Redis-backed task queue."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._local_queue: list = []  # Fallback for non-Redis mode

    def _get_client(self):
        """Get Redis client or raise error."""
        if self.redis is None:
            raise RuntimeError("Redis not configured. Use in-memory mode for local dev.")
        return self.redis

    async def enqueue(self, task: AnnotationTask) -> bool:
        """Add task to queue."""
        if self.redis is None:
            self._local_queue.append(task)
            return True

        try:
            client = self._get_client()
            task_data = json.dumps(task.to_dict())

            # Add to sorted set with priority as score
            await client.zadd(
                "task_queue",
                {task_data: -task.priority}  # Higher priority = lower score
            )

            # Add to task lookup
            await client.set(f"task:{task.task_id}", task_data)

            logger.info("task_enqueued", task_id=task.task_id, priority=task.priority)
            return True

        except Exception as e:
            logger.error("enqueue_failed", task_id=task.task_id, error=str(e))
            return False

    async def dequeue(self) -> Optional[AnnotationTask]:
        """Get highest priority task from queue."""
        if self.redis is None:
            if self._local_queue:
                # Sort by priority
                self._local_queue.sort(key=lambda t: -t.priority)
                return self._local_queue.pop(0)
            return None

        try:
            client = self._get_client()

            # Pop from sorted set (lowest score first)
            result = await client.zpopmin("task_queue", count=1)

            if not result:
                return None

            task_data = json.loads(result[0][0])
            task = AnnotationTask.from_dict(task_data)

            # Update status
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now(timezone.utc).isoformat()
            await client.set(f"task:{task.task_id}", json.dumps(task.to_dict()))

            return task

        except Exception as e:
            logger.error("dequeue_failed", error=str(e))
            return None

    async def update_task(self, task: AnnotationTask) -> bool:
        """Update task status."""
        if self.redis is None:
            return True

        try:
            client = self._get_client()
            await client.set(f"task:{task.task_id}", json.dumps(task.to_dict()))
            return True
        except Exception as e:
            logger.error("update_task_failed", task_id=task.task_id, error=str(e))
            return False

    async def get_task(self, task_id: str) -> Optional[AnnotationTask]:
        """Get task by ID."""
        if self.redis is None:
            for task in self._local_queue:
                if task.task_id == task_id:
                    return task
            return None

        try:
            client = self._get_client()
            data = await client.get(f"task:{task_id}")

            if data:
                return AnnotationTask.from_dict(json.loads(data))
            return None

        except Exception as e:
            logger.error("get_task_failed", task_id=task_id, error=str(e))
            return None

    async def get_queue_length(self) -> int:
        """Get number of tasks in queue."""
        if self.redis is None:
            return len(self._local_queue)

        try:
            client = self._get_client()
            return await client.zcard("task_queue")
        except Exception:
            return 0


class CeleryTaskQueue:
    """Celery-based distributed task queue."""

    def __init__(self, broker_url: str = "redis://localhost:6379/0"):
        self.broker_url = broker_url
        self._celery_app = None

    def get_app(self):
        """Get or create Celery app."""
        if self._celery_app is None:
            from celery import Celery

            self._celery_app = Celery("dvas", broker=self.broker_url)
            self._celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=3600,  # 1 hour timeout
                task_soft_time_limit=3300,  # 55 min soft timeout
            )

        return self._celery_app

    def register_tasks(self):
        """Register Celery tasks."""
        app = self.get_app()

        @app.task(bind=True, max_retries=3)
        def annotate_video_task(self, video_id: str, video_path: str, **kwargs):
            """Celery task for video annotation."""
            from dvas.models.teacher.gpt55 import GPT55Teacher
            from dvas.pipeline.core import AnnotationPipeline

            logger.info("celery_task_started", video_id=video_id)

            try:
                teacher = TeacherModel(model_name=kwargs.get("teacher_model", "gpt-5.5"))
                pipeline = AnnotationPipeline(
                    teacher_model=teacher,
                    num_frames=kwargs.get("num_frames", 16),
                )

                import asyncio

                annotation = asyncio.run(
                    pipeline.annotate_video(
                        video_path=video_path,
                        video_id=video_id,
                    )
                )

                return {
                    "status": "completed",
                    "annotation_id": annotation.id,
                    "video_id": video_id,
                }

            except Exception as e:
                logger.error("celery_task_failed", video_id=video_id, error=str(e))

                # Retry with exponential backoff
                countdown = 2 ** self.request.retries
                raise self.retry(exc=e, countdown=countdown)

        self.annotate_video_task = annotate_video_task
        return annotate_video_task

    def submit_task(self, task: AnnotationTask) -> str:
        """Submit task to Celery."""
        task_def = self.register_tasks()

        result = task_def.delay(
            video_id=task.video_id,
            video_path=task.video_path,
            teacher_model=task.teacher_model,
            num_frames=task.num_frames,
        )

        return result.id


def create_redis_client(redis_url: Optional[str] = None):
    """Create Redis client from URL."""
    try:
        import redis.asyncio as redis

        url = redis_url or "redis://localhost:6379/0"
        return redis.from_url(url, decode_responses=True)

    except ImportError:
        logger.warning("redis_not_installed", message="Install with: pip install redis")
        return None
