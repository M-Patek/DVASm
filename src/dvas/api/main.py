"""FastAPI REST API for DVAS - optimized with async."""

import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dvas.config import settings
from dvas.data.schemas import Annotation
from dvas.data.storage import AnnotationStore
from dvas.models.teacher.gpt4v import GPT4VTeacher
from dvas.pipeline.core import AnnotationPipeline
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


# Pydantic models for API
class VideoUploadResponse(BaseModel):
    """Response from video upload."""

    video_id: str
    filename: str
    status: str
    message: str


class AnnotationTaskRequest(BaseModel):
    """Request to start annotation task."""

    video_id: str
    teacher_model: str = "gpt-4o"
    num_frames: int = 16
    priority: int = Field(default=5, ge=1, le=10)


class AnnotationTaskResponse(BaseModel):
    """Response from annotation task creation."""

    task_id: str
    video_id: str
    status: str  # pending, processing, completed, failed
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

    video_ids: Optional[List[str]] = None  # None = all
    format: str = "llava"  # llava, openai, sharegpt
    source: str = "gold"  # gold, model, reviewed


# In-memory task storage (replace with Redis in production)
tasks: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info("api_starting", version="0.1.0")

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
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "dvas-api",
    }


@app.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(
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

        logger.info(
            "video_uploaded",
            video_id=video_id,
            filename=file.filename,
            size_bytes=file_path.stat().st_size,
        )

        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            status="success",
            message=f"Video uploaded successfully",
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


@app.post("/annotations/tasks", response_model=AnnotationTaskResponse)
async def create_annotation_task(
    request: AnnotationTaskRequest,
) -> AnnotationTaskResponse:
    """Create a new annotation task."""
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # Check if video exists
    upload_dir = settings.data_paths["raw"] / "uploads"
    video_files = list(upload_dir.glob(f"{request.video_id}*"))

    if not video_files:
        logger.warning(
            "video_not_found",
            video_id=request.video_id,
            task_id=task_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {request.video_id}",
        )

    # Create task
    tasks[task_id] = {
        "task_id": task_id,
        "video_id": request.video_id,
        "status": "pending",
        "teacher_model": request.teacher_model,
        "num_frames": request.num_frames,
        "priority": request.priority,
        "result": None,
        "error": None,
    }

    # Start annotation in background
    asyncio.create_task(run_annotation_task(task_id, request))

    logger.info(
        "task_created",
        task_id=task_id,
        video_id=request.video_id,
        model=request.teacher_model,
    )

    return AnnotationTaskResponse(
        task_id=task_id,
        video_id=request.video_id,
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
        teacher = GPT4VTeacher(model_name=request.teacher_model)
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

        logger.info(
            "task_completed",
            task_id=task_id,
            video_id=request.video_id,
            annotation_id=annotation.id,
        )

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)

        logger.error(
            "task_failed",
            task_id=task_id,
            video_id=request.video_id,
            error=str(e),
        )


@app.get("/annotations/tasks/{task_id}", response_model=AnnotationResult)
async def get_task_status(task_id: str) -> AnnotationResult:
    """Get annotation task status and result."""
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


@app.get("/annotations/{video_id}")
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
            return annotation.model_dump()

    logger.warning(
        "annotation_not_found",
        video_id=video_id,
    )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Annotation not found for video: {video_id}",
    )


@app.post("/export")
async def export_annotations(request: ExportRequest) -> FileResponse:
    """Export annotations to file."""
    store = AnnotationStore()

    # Load annotations
    if request.video_ids:
        annotations = []
        for vid in request.video_ids:
            ann = store.load(vid, source=request.source)
            if ann:
                annotations.append(ann)
    else:
        annotations = store.load_all(source=request.source)

    if not annotations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotations found for export",
        )

    # Export to temp file
    output_path = Path(tempfile.gettempdir()) / f"export_{uuid.uuid4().hex[:8]}.jsonl"

    count = store.export_to_jsonl(
        output_path=output_path,
        source=request.source,
        format=request.format,
    )

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


@app.get("/stats")
async def get_statistics() -> Dict:
    """Get storage statistics."""
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

    return stats


# CLI entry point
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
