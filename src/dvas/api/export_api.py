"""Export job API for DVAS.

FastAPI endpoints for export job creation, status, format selection, and download.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


class ExportJobCreate(BaseModel):
    """Request to create an export job."""

    video_ids: Optional[List[str]] = None
    format: str = "llava"
    source: str = "model"
    include_metadata: bool = True


class ExportJobStatus(BaseModel):
    """Export job status."""

    job_id: str
    status: str
    format: str
    total_videos: int
    exported_count: int
    progress: float
    output_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ExportFormatInfo(BaseModel):
    """Export format information."""

    name: str
    description: str
    content_type: str
    file_extension: str


_export_jobs: Dict[str, Dict[str, Any]] = {}

SUPPORTED_FORMATS = {
    "llava": {"name": "llava", "description": "LLaVA training format", "content_type": "application/jsonlines+json", "file_extension": ".jsonl"},
    "openai": {"name": "openai", "description": "OpenAI fine-tuning format", "content_type": "application/jsonlines+json", "file_extension": ".jsonl"},
}


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_export_job(request: ExportJobCreate) -> Dict[str, Any]:
    """Create a new export job."""
    if request.format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {request.format}",
        )

    import uuid
    from datetime import datetime, timezone

    job_id = f"export_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    job = {
        "job_id": job_id,
        "status": "pending",
        "format": request.format,
        "source": request.source,
        "video_ids": request.video_ids,
        "total_videos": len(request.video_ids) if request.video_ids else 0,
        "exported_count": 0,
        "progress": 0.0,
        "output_path": None,
        "file_size_bytes": None,
        "created_at": now,
        "completed_at": None,
        "error": None,
    }
    _export_jobs[job_id] = job

    logger.info("export_job_created", job_id=job_id, format=request.format)

    return {
        "job_id": job_id,
        "status": "pending",
        "total_videos": job["total_videos"],
        "message": "Export job created",
    }


@router.get("/jobs/{job_id}")
async def get_export_status(job_id: str) -> Dict[str, Any]:
    """Get export job status and progress."""
    job = _export_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export job not found: {job_id}",
        )
    return job


@router.get("/formats")
async def list_export_formats() -> Dict[str, Any]:
    """List supported export formats."""
    return {
        "formats": SUPPORTED_FORMATS,
    }
