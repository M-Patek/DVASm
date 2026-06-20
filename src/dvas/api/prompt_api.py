"""Prompt registry API for DVAS.

FastAPI endpoints for prompt CRUD and versioning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptVersion(BaseModel):
    """Prompt version information."""

    version: str
    content: str
    created_at: str
    created_by: Optional[str] = None
    changelog: Optional[str] = None
    is_active: bool = False


class PromptCreate(BaseModel):
    """Request to create a prompt."""

    name: str
    description: Optional[str] = None
    content: str
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=list)
    created_by: Optional[str] = None


class PromptResponse(BaseModel):
    """Prompt response."""

    id: str
    name: str
    description: Optional[str]
    category: str
    tags: List[str]
    variables: List[str]
    versions: List[PromptVersion]
    current_version: str
    created_at: str
    updated_at: str
    created_by: Optional[str]


class PromptUpdate(BaseModel):
    """Request to update a prompt."""

    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class PromptList(BaseModel):
    """List of prompts."""

    prompts: List[PromptResponse]
    total: int
    offset: int
    limit: int


_prompts: Dict[str, Dict[str, Any]] = {}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_prompt(request: PromptCreate) -> PromptResponse:
    """Create a new prompt in the registry."""
    import uuid
    from datetime import datetime, timezone

    prompt_id = f"prompt_{request.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc).isoformat()

    version = PromptVersion(
        version="1.0.0",
        content=request.content,
        created_at=now,
        created_by=request.created_by,
        is_active=True,
    )

    prompt_data = {
        "id": prompt_id,
        "name": request.name,
        "description": request.description,
        "category": request.category,
        "tags": request.tags,
        "variables": request.variables,
        "versions": [version.model_dump()],
        "current_version": "1.0.0",
        "created_at": now,
        "updated_at": now,
        "created_by": request.created_by,
    }
    _prompts[prompt_id] = prompt_data

    logger.info("prompt_created", prompt_id=prompt_id, name=request.name)
    return PromptResponse(**prompt_data)


@router.get("/{prompt_id}")
async def get_prompt(prompt_id: str) -> PromptResponse:
    """Get a prompt by ID."""
    prompt = _prompts.get(prompt_id)
    if prompt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt not found: {prompt_id}",
        )
    return PromptResponse(**prompt)


@router.get("/{prompt_id}/content")
async def get_prompt_content(prompt_id: str, version: Optional[str] = None) -> Dict[str, str]:
    """Get prompt content by ID and optional version."""
    prompt = _prompts.get(prompt_id)
    if prompt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt not found: {prompt_id}",
        )

    target_version = version or prompt.get("current_version", "1.0.0")
    for v in prompt.get("versions", []):
        if v["version"] == target_version:
            return {
                "prompt_id": prompt_id,
                "version": target_version,
                "content": v["content"],
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Version {target_version} not found for prompt {prompt_id}",
    )


@router.put("/{prompt_id}")
async def update_prompt(prompt_id: str, request: PromptUpdate) -> PromptResponse:
    """Update a prompt's metadata."""
    prompt = _prompts.get(prompt_id)
    if prompt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt not found: {prompt_id}",
        )

    if request.name is not None:
        prompt["name"] = request.name
    if request.description is not None:
        prompt["description"] = request.description
    if request.category is not None:
        prompt["category"] = request.category
    if request.tags is not None:
        prompt["tags"] = request.tags

    from datetime import datetime, timezone

    prompt["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("prompt_updated", prompt_id=prompt_id)
    return PromptResponse(**prompt)


@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str) -> Dict[str, str]:
    """Delete a prompt."""
    if prompt_id not in _prompts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt not found: {prompt_id}",
        )

    del _prompts[prompt_id]
    logger.info("prompt_deleted", prompt_id=prompt_id)
    return {"message": f"Prompt {prompt_id} deleted"}


@router.get("")
async def list_prompts(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
) -> PromptList:
    """List prompts with filtering."""
    prompts = list(_prompts.values())

    if category:
        prompts = [p for p in prompts if p["category"] == category]
    if tag:
        prompts = [p for p in prompts if tag in p.get("tags", [])]

    total = len(prompts)
    prompts = prompts[offset : offset + limit]

    return PromptList(
        prompts=[PromptResponse(**p) for p in prompts],
        total=total,
        offset=offset,
        limit=limit,
    )
