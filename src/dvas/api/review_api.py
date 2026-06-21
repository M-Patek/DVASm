"""Annotation review API for DVAS.

FastAPI endpoints for review queue, approve/reject workflow, and review assignment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/review", tags=["review"])


class ReviewQueueItem(BaseModel):
    """Item in the review queue."""

    annotation_id: str
    video_id: str
    quality_score: Optional[float] = None
    status: str
    assigned_to: Optional[str] = None
    submitted_by: Optional[str] = None
    created_at: str
    priority: int = 5


class ReviewDecision(BaseModel):
    """Review decision request."""

    annotation_id: str
    decision: str
    reviewer_id: str
    comments: Optional[str] = None
    corrected_segments: Optional[List[Dict[str, Any]]] = None


class ReviewAssignment(BaseModel):
    """Review assignment request."""

    annotation_id: str
    reviewer_id: str


class ReviewQueueResponse(BaseModel):
    """Review queue response."""

    items: List[ReviewQueueItem]
    total: int
    offset: int
    limit: int


class ReviewStats(BaseModel):
    """Review statistics."""

    total_pending: int
    total_approved: int
    total_rejected: int
    total_unassigned: int
    average_review_time_seconds: Optional[float] = None
    reviewer_breakdown: Dict[str, Dict[str, int]]


_review_items: Dict[str, Dict[str, Any]] = {}


@router.get("/queue")
async def get_review_queue(
    status_filter: Optional[str] = None,
    assigned_to: Optional[str] = None,
    unassigned_only: bool = False,
    offset: int = 0,
    limit: int = Query(default=50, le=500),
) -> ReviewQueueResponse:
    """Get review queue with optional filtering."""
    items = list(_review_items.values())

    if status_filter:
        items = [i for i in items if i["status"] == status_filter]
    if assigned_to:
        items = [i for i in items if i.get("assigned_to") == assigned_to]
    if unassigned_only:
        items = [i for i in items if i.get("assigned_to") is None]

    items.sort(key=lambda i: (i.get("priority", 5), i.get("created_at", "")))

    total = len(items)
    items = items[offset : offset + limit]

    return ReviewQueueResponse(
        items=[ReviewQueueItem(**i) for i in items],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/queue")
async def add_to_review_queue(annotation_id: str, priority: int = 5) -> Dict[str, Any]:
    """Add an annotation to the review queue."""
    # Check if annotation exists (placeholder - would query storage in production)
    # For now, we accept any annotation_id but validate it's not empty
    if not annotation_id or annotation_id.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="annotation_id is required",
        )

    if annotation_id in _review_items:
        return {
            "message": "Annotation already in review queue",
            "annotation_id": annotation_id,
            "status": _review_items[annotation_id]["status"],
        }

    from datetime import datetime, timezone

    item = {
        "annotation_id": annotation_id,
        "video_id": annotation_id,
        "quality_score": None,
        "status": "pending_review",
        "assigned_to": None,
        "submitted_by": "system",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": priority,
    }
    _review_items[annotation_id] = item

    logger.info("review_queue_added", annotation_id=annotation_id, priority=priority)
    return {
        "message": "Annotation added to review queue",
        "annotation_id": annotation_id,
        "status": "pending_review",
    }


@router.post("/assign")
async def assign_review(request: ReviewAssignment) -> Dict[str, Any]:
    """Assign an annotation to a reviewer."""
    item = _review_items.get(request.annotation_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation not in review queue: {request.annotation_id}",
        )

    if item.get("assigned_to") and item["assigned_to"] != request.reviewer_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Already assigned to: {item['assigned_to']}",
        )

    item["assigned_to"] = request.reviewer_id
    item["status"] = "assigned"

    logger.info(
        "review_assigned",
        annotation_id=request.annotation_id,
        reviewer_id=request.reviewer_id,
    )

    return {
        "message": "Review assigned",
        "annotation_id": request.annotation_id,
        "reviewer_id": request.reviewer_id,
    }


@router.post("/decision")
async def submit_review_decision(request: ReviewDecision) -> Dict[str, Any]:
    """Submit a review decision (approve or reject)."""
    item = _review_items.get(request.annotation_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation not in review queue: {request.annotation_id}",
        )

    if item.get("assigned_to") != request.reviewer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Annotation not assigned to this reviewer",
        )

    if request.decision not in {"approve", "reject"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decision must be 'approve' or 'reject'",
        )

    from datetime import datetime, timezone

    item["status"] = "approved" if request.decision == "approve" else "rejected"
    item["reviewer_id"] = request.reviewer_id
    item["comments"] = request.comments
    item["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "review_decision_submitted",
        annotation_id=request.annotation_id,
        decision=request.decision,
        reviewer_id=request.reviewer_id,
    )

    return {
        "message": f"Review {request.decision}d",
        "annotation_id": request.annotation_id,
        "decision": request.decision,
        "reviewer_id": request.reviewer_id,
    }


@router.get("/stats")
async def get_review_statistics() -> ReviewStats:
    """Get review queue statistics."""
    items = list(_review_items.values())

    total_pending = sum(1 for i in items if i["status"] == "pending_review")
    total_approved = sum(1 for i in items if i["status"] == "approved")
    total_rejected = sum(1 for i in items if i["status"] == "rejected")
    total_unassigned = sum(1 for i in items if i.get("assigned_to") is None)

    reviewer_breakdown: Dict[str, Dict[str, int]] = {}
    for i in items:
        reviewer = i.get("reviewer_id")
        if reviewer:
            if reviewer not in reviewer_breakdown:
                reviewer_breakdown[reviewer] = {"approved": 0, "rejected": 0}
            if i["status"] == "approved":
                reviewer_breakdown[reviewer]["approved"] += 1
            elif i["status"] == "rejected":
                reviewer_breakdown[reviewer]["rejected"] += 1

    return ReviewStats(
        total_pending=total_pending,
        total_approved=total_approved,
        total_rejected=total_rejected,
        total_unassigned=total_unassigned,
        average_review_time_seconds=None,
        reviewer_breakdown=reviewer_breakdown,
    )


@router.get("/{annotation_id}")
async def get_review_item(annotation_id: str) -> Dict[str, Any]:
    """Get review details for an annotation."""
    item = _review_items.get(annotation_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation not in review queue: {annotation_id}",
        )

    return {
        "review": item,
        "annotation": None,
    }
