"""FastAPI REST API for DVAS - production ready.

Features:
- API versioning (/api/v1/)
- API key authentication
- Rate limiting with token bucket
- Request/response compression
- Health checks (liveness/readiness)
- Structured logging
"""

import asyncio
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dvas.api.auth import get_auth_status, require_auth, require_auth_strict
from dvas.api.middleware import (
    APIVersion,
    CompressionMiddleware,
    HealthChecker,
    HealthStatus,
    RateLimitConfig,
    RateLimiter,
    RequestTracker,
    api_error,
    api_response,
)
from dvas.config import settings
from dvas.data.storage import AnnotationStore
from dvas.export.adapters import export_annotations as export_annotations_to_file
from dvas.models.teacher import TeacherModel
from dvas.pipeline.core import AnnotationPipeline
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# In-memory task storage (replace with Redis in production)
tasks: Dict[str, Dict] = {}
MAX_FINISHED_TASKS = 1000
FINISHED_TASK_TTL_SECONDS = 3600.0
rate_limiter = RateLimiter(RateLimitConfig(requests_per_second=10.0, burst_size=20.0))
request_tracker = RequestTracker()
health_checker = HealthChecker()
compression = CompressionMiddleware(min_size=1024)


def _prune_finished_tasks(now: Optional[float] = None) -> None:
    """Keep in-memory task storage bounded while preserving active work."""
    if not tasks:
        return

    now = now if now is not None else time.monotonic()
    finished_statuses = {"completed", "failed"}

    if FINISHED_TASK_TTL_SECONDS > 0:
        expired_task_ids = [
            task_id
            for task_id, task in tasks.items()
            if task.get("status") in finished_statuses
            and task.get("_finished_at") is not None
            and now - task["_finished_at"] > FINISHED_TASK_TTL_SECONDS
        ]
        for task_id in expired_task_ids:
            tasks.pop(task_id, None)

    if MAX_FINISHED_TASKS <= 0:
        return

    finished_task_ids = [
        task_id for task_id, task in tasks.items() if task.get("status") in finished_statuses
    ]
    overflow = len(finished_task_ids) - MAX_FINISHED_TASKS
    if overflow <= 0:
        return

    oldest_finished_ids = sorted(
        finished_task_ids,
        key=lambda task_id: tasks[task_id].get("_finished_at", tasks[task_id].get("_created_at", 0)),
    )[:overflow]
    for task_id in oldest_finished_ids:
        tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


def check_storage():
    """Check storage health."""
    try:
        store = AnnotationStore(enable_index=False)
        _stats = store.get_statistics()
        return HealthChecker._run_check.__self__  # type: ignore
    except Exception:
        pass

    return type(
        "HealthCheck",
        (),
        {
            "name": "storage",
            "status": HealthStatus.HEALTHY,
            "message": "Storage accessible",
            "latency_ms": 0.0,
        },
    )()


def check_disk_space():
    """Check disk space."""
    import shutil

    try:
        total, used, free = shutil.disk_usage(str(settings.DATA_ROOT))
        free_gb = free / (1024**3)
        status = HealthStatus.HEALTHY if free_gb > 1.0 else HealthStatus.DEGRADED

        return type(
            "HealthCheck",
            (),
            {
                "name": "disk_space",
                "status": status,
                "message": f"{free_gb:.1f}GB free",
                "latency_ms": 0.0,
            },
        )()
    except Exception as e:
        return type(
            "HealthCheck",
            (),
            {
                "name": "disk_space",
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check disk: {e}",
                "latency_ms": 0.0,
            },
        )()


# Register health checks
health_checker.register("storage", lambda: check_storage())
health_checker.register("disk_space", lambda: check_disk_space())


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VideoUploadResponse(BaseModel):
    """Response from video upload."""

    video_id: str
    filename: str
    status: str
    message: str


class AnnotationTaskRequest(BaseModel):
    """Request to start annotation task."""

    video_id: str
    teacher_model: str = "gpt-5.5"
    num_frames: int = 16
    priority: int = Field(default=5, ge=1, le=10)


class AnnotationTaskResponse(BaseModel):
    """Response from annotation task creation."""

    task_id: str
    video_id: str
    status: str
    message: str


class AnnotationResult(BaseModel):
    """Annotation result response."""

    task_id: str
    video_id: str
    status: str
    annotation: Optional[Dict] = None
    error: Optional[str] = None


class ExportRequest(BaseModel):
    """Export request."""

    video_ids: Optional[List[str]] = None
    format: str = "llava"
    source: str = "gold"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info("api_starting", version="0.2.0")

    # Ensure data directories exist
    for name, path in settings.data_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info("directory_ready", name=name, path=str(path))

    yield

    # Shutdown
    logger.info("api_shutting_down")


app = FastAPI(
    title="DVAS API",
    description="Distilled Video Annotation Specialist - Video annotation API",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Track requests with timing and rate limiting."""
    # Skip tracking for health checks
    if request.url.path in ("/health", "/ready", "/api/v1/health", "/api/v1/ready"):
        return await call_next(request)

    # Rate limiting
    client_ip = request.headers.get(
        "X-Forwarded-For", request.client.host if request.client else "unknown"
    )
    if not rate_limiter.allow_request(client_ip, request.url.path):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=api_error(
                "Rate limit exceeded",
                error_code="RATE_LIMITED",
                status_code=429,
            ),
        )

    rate_limiter.consume(client_ip)

    # Track request
    request_id = request_tracker.start_request(request.method, request.url.path)

    # Process request
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    duration = (asyncio.get_event_loop().time() - start_time) * 1000

    # End tracking
    request_tracker.end_request(request_id, response.status_code)

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(duration, 2))

    return response


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Liveness probe - is the process running?"""
    return health_checker.liveness()


@app.get("/ready")
async def readiness_check() -> Dict:
    """Readiness probe - is the service ready to accept traffic?"""
    return health_checker.readiness()


# ---------------------------------------------------------------------------
# API v1 endpoints
# ---------------------------------------------------------------------------

API_PREFIX = APIVersion.get_path_prefix()


@app.get(f"{API_PREFIX}/auth/status")
async def auth_status() -> Dict:
    """Get authentication status."""
    return api_response(
        data=get_auth_status(),
        message="Authentication status",
    )


@app.post(
    f"{API_PREFIX}/videos/upload",
    response_model=VideoUploadResponse,
    dependencies=[require_auth],
)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
) -> VideoUploadResponse:
    """Upload a video file for annotation."""
    video_id = f"vid_{uuid.uuid4().hex[:12]}"

    # Validate file type
    allowed_types = {
        "video/mp4": ".mp4",
        "video/avi": ".avi",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
    }

    if file.content_type not in allowed_types:
        logger.warning(
            "invalid_file_type",
            video_id=video_id,
            content_type=file.content_type,
            filename=file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Allowed: {list(allowed_types.keys())}",
        )

    # Save file asynchronously
    upload_dir = settings.data_paths["raw"] / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = allowed_types.get(file.content_type, ".mp4")
    file_path = upload_dir / f"{video_id}{ext}"

    try:
        # Use aiofiles for non-blocking I/O
        async with aiofiles.open(file_path, "wb") as f:
            # Read in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                await f.write(chunk)

        file_size = file_path.stat().st_size

        logger.info(
            "video_uploaded",
            video_id=video_id,
            filename=file.filename,
            size_bytes=file_size,
        )

        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            status="success",
            message="Video uploaded successfully",
        )

    except Exception as e:
        logger.error(
            "upload_failed",
            video_id=video_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}",
        )


@app.post(
    f"{API_PREFIX}/annotations/tasks",
    response_model=AnnotationTaskResponse,
    dependencies=[require_auth],
)
async def create_annotation_task(
    request: Request,
    task_request: AnnotationTaskRequest,
) -> AnnotationTaskResponse:
    """Create a new annotation task."""
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # Check if video exists
    upload_dir = settings.data_paths["raw"] / "uploads"
    video_files = list(upload_dir.glob(f"{task_request.video_id}*"))

    if not video_files:
        logger.warning(
            "video_not_found",
            video_id=task_request.video_id,
            task_id=task_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {task_request.video_id}",
        )

    # Create task
    now = time.monotonic()
    tasks[task_id] = {
        "task_id": task_id,
        "video_id": task_request.video_id,
        "status": "pending",
        "teacher_model": task_request.teacher_model,
        "num_frames": task_request.num_frames,
        "priority": task_request.priority,
        "result": None,
        "error": None,
        "_created_at": now,
        "_finished_at": None,
    }
    _prune_finished_tasks(now)

    # Start annotation in background
    asyncio.create_task(run_annotation_task(task_id, task_request))

    logger.info(
        "task_created",
        task_id=task_id,
        video_id=task_request.video_id,
        model=task_request.teacher_model,
    )

    return AnnotationTaskResponse(
        task_id=task_id,
        video_id=task_request.video_id,
        status="pending",
        message="Annotation task created and queued",
    )


async def run_annotation_task(task_id: str, request: AnnotationTaskRequest) -> None:
    """Run annotation task in background."""
    task = tasks[task_id]
    task["status"] = "processing"

    logger.info(
        "task_processing",
        task_id=task_id,
        video_id=request.video_id,
    )

    try:
        # Find video file
        upload_dir = settings.data_paths["raw"] / "uploads"
        video_files = list(upload_dir.glob(f"{request.video_id}*"))

        if not video_files:
            raise FileNotFoundError(f"Video not found: {request.video_id}")

        video_path = video_files[0]

        # Initialize pipeline
        teacher = TeacherModel(model_name=request.teacher_model)
        pipeline = AnnotationPipeline(
            teacher_model=teacher,
            num_frames=request.num_frames,
        )

        # Run annotation
        annotation = await pipeline.annotate_video(
            video_path=video_path,
            video_id=request.video_id,
        )

        # Update task
        task["status"] = "completed"
        task["result"] = annotation.model_dump()
        task["_finished_at"] = time.monotonic()

        logger.info(
            "task_completed",
            task_id=task_id,
            video_id=request.video_id,
            annotation_id=annotation.id,
        )

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        task["_finished_at"] = time.monotonic()

        logger.error(
            "task_failed",
            task_id=task_id,
            video_id=request.video_id,
            error=str(e),
        )
    finally:
        _prune_finished_tasks()


@app.get(
    f"{API_PREFIX}/annotations/tasks/{{task_id}}",
    response_model=AnnotationResult,
    dependencies=[require_auth],
)
async def get_task_status(task_id: str) -> AnnotationResult:
    """Get annotation task status and result."""
    _prune_finished_tasks()
    if task_id not in tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    task = tasks[task_id]

    return AnnotationResult(
        task_id=task_id,
        video_id=task["video_id"],
        status=task["status"],
        annotation=task.get("result"),
        error=task.get("error"),
    )


@app.get(
    f"{API_PREFIX}/annotations/{{video_id}}",
    dependencies=[require_auth],
)
async def get_annotation(video_id: str) -> Dict:
    """Get annotation for a video."""
    store = AnnotationStore()

    # Try to load from any source
    for source in ["gold", "model", "reviewed"]:
        annotation = store.load(video_id, source=source)
        if annotation:
            logger.info(
                "annotation_found",
                video_id=video_id,
                source=source,
            )
            return api_response(
                data=annotation.model_dump(),
                message="Annotation found",
            )

    logger.warning(
        "annotation_not_found",
        video_id=video_id,
    )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Annotation not found for video: {video_id}",
    )


@app.post(
    f"{API_PREFIX}/export",
    dependencies=[require_auth],
)
async def export_annotations(request: ExportRequest) -> FileResponse:
    """Export annotations to file."""
    store = AnnotationStore()

    # Load annotations
    if request.video_ids:
        annotations = []
        seen_annotation_ids = set()
        for vid in request.video_ids:
            candidates = [
                ann
                for ann in (
                    store.load(vid, source=request.source),
                    store.load(f"{vid}_annotated", source=request.source),
                )
                if ann is not None
            ]
            candidates.extend(store.load_all(source=request.source, video_id=vid))

            for ann in candidates:
                if ann.id not in seen_annotation_ids:
                    annotations.append(ann)
                    seen_annotation_ids.add(ann.id)
    else:
        annotations = list(store.load_all(source=request.source))

    if not annotations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotations found for export",
        )

    # Export to temp file
    output_path = Path(tempfile.gettempdir()) / f"export_{uuid.uuid4().hex[:8]}.jsonl"

    try:
        count = export_annotations_to_file(
            annotations=annotations,
            output_path=output_path,
            format=request.format,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    logger.info(
        "export_completed",
        count=count,
        format=request.format,
        source=request.source,
    )

    return FileResponse(
        path=output_path,
        filename=f"dvas_export_{request.format}_{count}items.jsonl",
        media_type="application/jsonlines+json",
    )


@app.get(
    f"{API_PREFIX}/stats",
    dependencies=[require_auth_strict],  # Admin endpoint
)
async def get_statistics() -> Dict:
    """Get storage and request statistics."""
    _prune_finished_tasks()
    store = AnnotationStore()
    stats = store.get_statistics()

    # Add task statistics
    all_tasks = list(tasks.values())
    stats["tasks"] = {
        "total": len(all_tasks),
        "pending": sum(1 for t in all_tasks if t["status"] == "pending"),
        "processing": sum(1 for t in all_tasks if t["status"] == "processing"),
        "completed": sum(1 for t in all_tasks if t["status"] == "completed"),
        "failed": sum(1 for t in all_tasks if t["status"] == "failed"),
    }

    # Add request statistics
    stats["requests"] = request_tracker.get_stats()

    # Add rate limiter statistics
    stats["rate_limiter"] = rate_limiter.get_stats()

    return api_response(
        data=stats,
        message="Statistics retrieved",
    )


@app.get(
    f"{API_PREFIX}/search",
    dependencies=[require_auth],
)
async def search_annotations(
    q: str,
    limit: int = 100,
) -> Dict:
    """Full-text search annotations."""
    store = AnnotationStore(enable_index=True)
    results = store.search(q, limit=limit)

    return api_response(
        data={
            "query": q,
            "results": results,
            "count": len(results),
        },
        message=f"Found {len(results)} results",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Run API server."""
    import uvicorn

    uvicorn.run(
        "dvas.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
