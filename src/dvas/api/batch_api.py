"""Batch job API for DVAS.

FastAPI endpoints for batch annotation jobs with status, results, progress, and cancellation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from dvas.api.task_lifecycle import TaskLifecycleManager
from dvas.api.task_store import InMemoryTaskStore, TaskType
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


class BatchJobRequest(BaseModel):
    """Request to create a batch annotation job."""

    video_ids: List[str]
    teacher_model: str = "gpt-5.5"
    num_frames: int = 16
    priority: int = Field(default=5, ge=1, le=10)
    max_retries: int = 3


class BatchJobStatus(BaseModel):
    """Status of a batch job."""

    batch_id: str
    status: str
    total_videos: int
    completed: int
    failed: int
    pending: int
    processing: int
    progress: float
    created_at: str
    updated_at: str
    error: Optional[str] = None


class BatchJobResult(BaseModel):
    """Result of a batch job."""

    batch_id: str
    status: str
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    completed_at: Optional[str] = None


class BatchJobCancel(BaseModel):
    """Response from batch job cancellation."""

    batch_id: str
    cancelled: int
    message: str


class BatchJobList(BaseModel):
    """List of batch jobs."""

    jobs: List[BatchJobStatus]
    total: int
    offset: int
    limit: int


_batch_jobs: Dict[str, BatchJobStatus] = {}
_batch_tasks: Dict[str, List[str]] = {}


def _get_or_create_manager() -> TaskLifecycleManager:
    store = InMemoryTaskStore()
    return TaskLifecycleManager(store)


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_batch_job(request: BatchJobRequest) -> Dict[str, Any]:
    """Create a new batch annotation job."""
    import uuid
    from datetime import datetime, timezone

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    manager = _get_or_create_manager()

    task_ids = []
    for video_id in request.video_ids:
        task = await manager.create_task(
            task_type=TaskType.ANNOTATION,
            payload={
                "video_id": video_id,
                "teacher_model": request.teacher_model,
                "num_frames": request.num_frames,
                "batch_id": batch_id,
            },
            priority=request.priority,
            max_retries=request.max_retries,
        )
        task_ids.append(task.id)

    now = datetime.now(timezone.utc).isoformat()
    _batch_jobs[batch_id] = BatchJobStatus(
        batch_id=batch_id,
        status="pending",
        total_videos=len(request.video_ids),
        completed=0,
        failed=0,
        pending=len(request.video_ids),
        processing=0,
        progress=0.0,
        created_at=now,
        updated_at=now,
    )
    _batch_tasks[batch_id] = task_ids

    logger.info(
        "batch_job_created",
        batch_id=batch_id,
        total_videos=len(request.video_ids),
        model=request.teacher_model,
    )

    return {
        "batch_id": batch_id,
        "status": "pending",
        "total_videos": len(request.video_ids),
        "task_ids": task_ids,
        "message": "Batch job created successfully",
    }


@router.get("/jobs/{batch_id}")
async def get_batch_status(batch_id: str) -> BatchJobStatus:
    """Get batch job status and progress."""
    batch_status = _batch_jobs.get(batch_id)
    if batch_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {batch_id}",
        )

    task_ids = _batch_tasks.get(batch_id, [])
    if task_ids:
        manager = _get_or_create_manager()
        total = len(task_ids)
        completed = 0
        failed = 0
        pending = 0
        processing = 0
        total_progress = 0.0

        for task_id in task_ids:
            task_status = await manager.get_task_status(task_id)
            if task_status:
                task_status_str = task_status.get("status", "PENDING")
                if task_status_str == "COMPLETED":
                    completed += 1
                elif task_status_str == "FAILED":
                    failed += 1
                elif task_status_str == "PROCESSING":
                    processing += 1
                else:
                    pending += 1
                total_progress += task_status.get("progress", 0.0)

        batch_status.completed = completed
        batch_status.failed = failed
        batch_status.pending = pending
        batch_status.processing = processing
        batch_status.progress = total_progress / total if total > 0 else 0.0

        if completed == total:
            batch_status.status = "completed"
        elif failed == total:
            batch_status.status = "failed"
        elif processing > 0:
            batch_status.status = "processing"
        else:
            batch_status.status = "pending"

    return batch_status


@router.get("/jobs/{batch_id}/results")
async def get_batch_results(batch_id: str) -> BatchJobResult:
    """Get batch job results."""
    batch_status = _batch_jobs.get(batch_id)
    if batch_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {batch_id}",
        )

    task_ids = _batch_tasks.get(batch_id, [])
    results = []
    errors = []

    manager = _get_or_create_manager()
    for task_id in task_ids:
        task_status = await manager.get_task_status(task_id)
        if task_status:
            if task_status.get("status") == "COMPLETED":
                results.append(task_status.get("result", {}))
            elif task_status.get("error"):
                errors.append({
                    "task_id": task_id,
                    "error": task_status.get("error"),
                })

    return BatchJobResult(
        batch_id=batch_id,
        status=batch_status.status,
        results=results,
        errors=errors,
        completed_at=batch_status.updated_at if batch_status.status == "completed" else None,
    )


@router.get("/jobs/{batch_id}/progress")
async def get_batch_progress(batch_id: str) -> Dict[str, Any]:
    """Get batch job progress as percentage."""
    batch_status = _batch_jobs.get(batch_id)
    if batch_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {batch_id}",
        )

    task_ids = _batch_tasks.get(batch_id, [])
    if task_ids:
        manager = _get_or_create_manager()
        total_progress = 0.0
        for task_id in task_ids:
            task_status = await manager.get_task_status(task_id)
            if task_status:
                total_progress += task_status.get("progress", 0.0)
        batch_status.progress = total_progress / len(task_ids)

    return {
        "batch_id": batch_id,
        "progress": batch_status.progress,
        "status": batch_status.status,
        "total": batch_status.total_videos,
        "completed": batch_status.completed,
        "failed": batch_status.failed,
        "pending": batch_status.pending,
        "processing": batch_status.processing,
    }


@router.post("/jobs/{batch_id}/cancel")
async def cancel_batch_job(batch_id: str) -> BatchJobCancel:
    """Cancel a batch job and all its pending tasks."""
    batch_status = _batch_jobs.get(batch_id)
    if batch_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {batch_id}",
        )

    if batch_status.status in {"completed", "failed", "cancelled"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel batch job in status: {batch_status.status}",
        )

    task_ids = _batch_tasks.get(batch_id, [])
    manager = _get_or_create_manager()
    cancelled_count = 0

    for task_id in task_ids:
        task_status = await manager.get_task_status(task_id)
        if task_status and task_status.get("status") in {"PENDING", "QUEUED", "RETRYING"}:
            await manager.cancel_task(task_id)
            cancelled_count += 1

    batch_status.status = "cancelled"
    from datetime import datetime, timezone

    batch_status.updated_at = datetime.now(timezone.utc).isoformat()

    logger.info("batch_job_cancelled", batch_id=batch_id, cancelled_tasks=cancelled_count)

    return BatchJobCancel(
        batch_id=batch_id,
        cancelled=cancelled_count,
        message=f"Cancelled {cancelled_count} tasks",
    )


@router.get("/jobs")
async def list_batch_jobs(
    offset: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
) -> BatchJobList:
    """List batch jobs with optional filtering."""
    jobs = list(_batch_jobs.values())
    if status_filter:
        jobs = [j for j in jobs if j.status == status_filter]
    total = len(jobs)
    jobs = jobs[offset : offset + limit]
    return BatchJobList(
        jobs=jobs,
        total=total,
        offset=offset,
        limit=limit,
    )
