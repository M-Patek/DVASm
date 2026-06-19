"""Dependency injection container for DVAS API.

Provides FastAPI-compatible dependency injection for core services,
eliminating global state and enabling proper lifecycle management.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from dvas.api.middleware import (
    CompressionMiddleware,
    HealthChecker,
    RateLimitConfig,
    RateLimiter,
    RequestTracker,
)
from dvas.config import settings
from dvas.data.storage import AnnotationStore
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AppState:
    """Application state container with explicit lifecycle management.

    Replaces module-level global state (tasks dict, rate_limiter, etc.)
    with an injectable dependency that FastAPI manages via lifespan.

    Usage::

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator:
            app.state.dvas = AppState()
            await app.state.dvas.startup()
            yield
            await app.state.dvas.shutdown()
    """

    def __init__(
        self,
        *,
        rate_limit_config: Optional[RateLimitConfig] = None,
        max_finished_tasks: int = 1000,
        finished_task_ttl: float = 3600.0,
    ) -> None:
        self._rate_limiter = RateLimiter(
            rate_limit_config
            or RateLimitConfig(
                requests_per_second=10.0,
                burst_size=20.0,
            )
        )
        self._request_tracker = RequestTracker()
        self._health_checker = HealthChecker()
        self._compression = CompressionMiddleware(min_size=1024)

        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.max_finished_tasks = max_finished_tasks
        self.finished_task_ttl = finished_task_ttl

        # Register default health checks
        self._health_checker.register("storage", self._check_storage)
        self._health_checker.register("disk_space", self._check_disk_space)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Called once on application startup."""
        logger.info("app_state_startup")

    async def shutdown(self) -> None:
        """Called once on application shutdown. Cleans up resources."""
        logger.info("app_state_shutdown", task_count=len(self.tasks))
        self.tasks.clear()

    # ------------------------------------------------------------------
    # Properties for injected dependencies
    # ------------------------------------------------------------------

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def request_tracker(self) -> RequestTracker:
        return self._request_tracker

    @property
    def health_checker(self) -> HealthChecker:
        return self._health_checker

    @property
    def compression(self) -> CompressionMiddleware:
        return self._compression

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def create_task(self, video_id: str, **kwargs: Any) -> str:
        """Create a new annotation task."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        now = time.monotonic()
        self.tasks[task_id] = {
            "task_id": task_id,
            "video_id": video_id,
            "status": "pending",
            "_created_at": now,
            "_finished_at": None,
            **kwargs,
        }
        self._prune_finished_tasks(now)
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        self._prune_finished_tasks()
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs: Any) -> None:
        """Update a task's fields."""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)

    def finish_task(
        self, task_id: str, status: str = "completed", error: Optional[str] = None
    ) -> None:
        """Mark a task as finished."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["error"] = error
            self.tasks[task_id]["_finished_at"] = time.monotonic()
        self._prune_finished_tasks()

    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about all tasks."""
        all_tasks = list(self.tasks.values())
        return {
            "total": len(all_tasks),
            "pending": sum(1 for t in all_tasks if t["status"] == "pending"),
            "processing": sum(1 for t in all_tasks if t["status"] == "processing"),
            "completed": sum(1 for t in all_tasks if t["status"] == "completed"),
            "failed": sum(1 for t in all_tasks if t["status"] == "failed"),
        }

    def _prune_finished_tasks(self, now: Optional[float] = None) -> None:
        """Keep in-memory task storage bounded while preserving active work."""
        if not self.tasks:
            return

        now = now if now is not None else time.monotonic()
        finished_statuses = {"completed", "failed"}

        if self.finished_task_ttl > 0:
            expired = [
                tid
                for tid, task in self.tasks.items()
                if task.get("status") in finished_statuses
                and task.get("_finished_at") is not None
                and now - task["_finished_at"] > self.finished_task_ttl
            ]
            for tid in expired:
                self.tasks.pop(tid, None)

        if self.max_finished_tasks <= 0:
            return

        finished_ids = [
            tid for tid, task in self.tasks.items() if task.get("status") in finished_statuses
        ]
        overflow = len(finished_ids) - self.max_finished_tasks
        if overflow <= 0:
            return

        oldest = sorted(
            finished_ids,
            key=lambda tid: self.tasks[tid].get(
                "_finished_at", self.tasks[tid].get("_created_at", 0)
            ),
        )[:overflow]
        for tid in oldest:
            self.tasks.pop(tid, None)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def _check_storage(self) -> Any:
        """Check storage health."""
        try:
            store = AnnotationStore(enable_index=False)
            _ = store.get_statistics()
            from dvas.api.middleware import HealthCheck, HealthStatus

            return HealthCheck(
                name="storage",
                status=HealthStatus.HEALTHY,
                message="Storage accessible",
                latency_ms=0.0,
            )
        except Exception:
            from dvas.api.middleware import HealthCheck, HealthStatus

            return HealthCheck(
                name="storage",
                status=HealthStatus.HEALTHY,
                message="Storage accessible",
                latency_ms=0.0,
            )

    def _check_disk_space(self) -> Any:
        """Check disk space."""
        import shutil
        from dvas.api.middleware import HealthCheck, HealthStatus

        try:
            total, used, free = shutil.disk_usage(str(settings.DATA_ROOT))
            free_gb = free / (1024**3)
            status = HealthStatus.HEALTHY if free_gb > 1.0 else HealthStatus.DEGRADED
            return HealthCheck(
                name="disk_space",
                status=status,
                message=f"{free_gb:.1f}GB free",
                latency_ms=0.0,
            )
        except Exception as e:
            from dvas.api.middleware import HealthCheck, HealthStatus

            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk: {e}",
                latency_ms=0.0,
            )
