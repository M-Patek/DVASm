"""Dataset API for DVAS.

FastAPI endpoints for dataset CRUD operations, listing with filtering, and statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetCreate(BaseModel):
    """Request to create a dataset."""

    name: str
    description: Optional[str] = None
    source: str = "model"
    tags: List[str] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    """Dataset response."""

    id: str
    name: str
    description: Optional[str] = None
    source: str
    tags: List[str]
    video_count: int
    annotation_count: int
    total_duration_seconds: float
    created_at: str
    updated_at: str


class DatasetUpdate(BaseModel):
    """Request to update a dataset."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class DatasetStatistics(BaseModel):
    """Dataset statistics."""

    dataset_id: str
    name: str
    total_annotations: int
    total_videos: int
    total_duration_seconds: float
    average_quality_score: Optional[float] = None
    quality_distribution: Dict[str, int]
    action_verb_counts: Dict[str, int]
    object_counts: Dict[str, int]
    source_breakdown: Dict[str, int]


class DatasetList(BaseModel):
    """List of datasets."""

    datasets: List[DatasetResponse]
    total: int
    offset: int
    limit: int


_datasets: Dict[str, Dict[str, Any]] = {}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(request: DatasetCreate) -> DatasetResponse:
    """Create a new dataset."""
    import uuid
    from datetime import datetime, timezone

    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    dataset = {
        "id": dataset_id,
        "name": request.name,
        "description": request.description,
        "source": request.source,
        "tags": request.tags,
        "video_count": 0,
        "annotation_count": 0,
        "total_duration_seconds": 0.0,
        "created_at": now,
        "updated_at": now,
    }
    _datasets[dataset_id] = dataset

    logger.info("dataset_created", dataset_id=dataset_id, name=request.name)
    return DatasetResponse(**dataset)


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str) -> DatasetResponse:
    """Get a dataset by ID."""
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}",
        )
    return DatasetResponse(**dataset)


@router.put("/{dataset_id}")
async def update_dataset(dataset_id: str, request: DatasetUpdate) -> DatasetResponse:
    """Update a dataset."""
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}",
        )

    if request.name is not None:
        dataset["name"] = request.name
    if request.description is not None:
        dataset["description"] = request.description
    if request.tags is not None:
        dataset["tags"] = request.tags

    from datetime import datetime, timezone

    dataset["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("dataset_updated", dataset_id=dataset_id)
    return DatasetResponse(**dataset)


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Dict[str, str]:
    """Delete a dataset."""
    if dataset_id not in _datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}",
        )

    del _datasets[dataset_id]
    logger.info("dataset_deleted", dataset_id=dataset_id)
    return {"message": f"Dataset {dataset_id} deleted"}


@router.get("")
async def list_datasets(
    source: Optional[str] = None,
    tag: Optional[str] = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
) -> DatasetList:
    """List datasets with filtering."""
    datasets = list(_datasets.values())

    if source:
        datasets = [d for d in datasets if d["source"] == source]
    if tag:
        datasets = [d for d in datasets if tag in d.get("tags", [])]

    total = len(datasets)
    datasets = datasets[offset : offset + limit]

    return DatasetList(
        datasets=[DatasetResponse(**d) for d in datasets],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str) -> DatasetStatistics:
    """Get statistics for a dataset."""
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}",
        )

    return DatasetStatistics(
        dataset_id=dataset_id,
        name=dataset["name"],
        total_annotations=0,
        total_videos=0,
        total_duration_seconds=0.0,
        average_quality_score=None,
        quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
        action_verb_counts={},
        object_counts={},
        source_breakdown={dataset["source"]: 0},
    )


@router.get("/{dataset_id}/annotations")
async def get_dataset_annotations(
    dataset_id: str,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
) -> Dict[str, Any]:
    """Get annotations in a dataset."""
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}",
        )

    return {
        "dataset_id": dataset_id,
        "total": 0,
        "offset": offset,
        "limit": limit,
        "annotations": [],
    }
