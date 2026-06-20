"""Model registry API for DVAS.

FastAPI endpoints for model listing, registration, version tracking, and capability queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class ModelCapability(BaseModel):
    """Model capability description."""

    name: str
    description: str
    supported: bool


class ModelVersion(BaseModel):
    """Model version information."""

    version: str
    release_date: str
    changelog: Optional[str] = None
    is_latest: bool = False
    is_deprecated: bool = False
    parameters: Optional[int] = None


class ModelRegistration(BaseModel):
    """Model registration request."""

    name: str
    provider: str
    model_id: str
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    versions: List[ModelVersion] = Field(default_factory=list)
    pricing_input_per_1m: Optional[float] = None
    pricing_output_per_1m: Optional[float] = None
    max_context_length: Optional[int] = None
    supports_vision: bool = False
    supports_video: bool = False


class ModelResponse(BaseModel):
    """Model response."""

    id: str
    name: str
    provider: str
    model_id: str
    description: Optional[str]
    capabilities: List[str]
    versions: List[ModelVersion]
    pricing_input_per_1m: Optional[float]
    pricing_output_per_1m: Optional[float]
    max_context_length: Optional[int]
    supports_vision: bool
    supports_video: bool
    registered_at: str
    updated_at: str


class ModelList(BaseModel):
    """List of models."""

    models: List[ModelResponse]
    total: int
    offset: int
    limit: int


class CapabilityQuery(BaseModel):
    """Capability query request."""

    capabilities: List[str]
    require_all: bool = True


_registered_models: Dict[str, Dict[str, Any]] = {}

DEFAULT_MODELS = [
    {
        "id": "model_gpt4o",
        "name": "GPT-4o",
        "provider": "openai",
        "model_id": "gpt-4o",
        "description": "OpenAI GPT-4o multimodal model",
        "capabilities": ["vision", "text_generation", "function_calling"],
        "versions": [
            {"version": "2024-08-06", "release_date": "2024-08-06", "is_latest": True},
        ],
        "pricing_input_per_1m": 2.50,
        "pricing_output_per_1m": 10.00,
        "max_context_length": 128000,
        "supports_vision": True,
        "supports_video": False,
        "registered_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    },
]

for model_data in DEFAULT_MODELS:
    _registered_models[model_data["id"]] = model_data


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_model(request: ModelRegistration) -> ModelResponse:
    """Register a new model in the registry."""
    import uuid
    from datetime import datetime, timezone

    model_id = f"model_{request.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc).isoformat()

    model_data = {
        "id": model_id,
        "name": request.name,
        "provider": request.provider,
        "model_id": request.model_id,
        "description": request.description,
        "capabilities": request.capabilities,
        "versions": [v.model_dump() for v in request.versions],
        "pricing_input_per_1m": request.pricing_input_per_1m,
        "pricing_output_per_1m": request.pricing_output_per_1m,
        "max_context_length": request.max_context_length,
        "supports_vision": request.supports_vision,
        "supports_video": request.supports_video,
        "registered_at": now,
        "updated_at": now,
    }
    _registered_models[model_id] = model_data

    logger.info("model_registered", model_id=model_id, name=request.name)
    return ModelResponse(**model_data)


@router.get("/{model_id}")
async def get_model(model_id: str) -> ModelResponse:
    """Get a model by ID."""
    model = _registered_models.get(model_id)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    return ModelResponse(**model)


@router.get("")
async def list_models(
    provider: Optional[str] = None,
    supports_vision: Optional[bool] = None,
    supports_video: Optional[bool] = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
) -> ModelList:
    """List registered models with filtering."""
    models = list(_registered_models.values())

    if provider:
        models = [m for m in models if m["provider"] == provider]
    if supports_vision is not None:
        models = [m for m in models if m["supports_vision"] == supports_vision]
    if supports_video is not None:
        models = [m for m in models if m["supports_video"] == supports_video]

    total = len(models)
    models = models[offset : offset + limit]

    return ModelList(
        models=[ModelResponse(**m) for m in models],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/query")
async def query_models_by_capability(request: CapabilityQuery) -> ModelList:
    """Query models by capabilities."""
    models = list(_registered_models.values())
    matching = []

    for model in models:
        model_caps = set(model.get("capabilities", []))
        required_caps = set(request.capabilities)

        if request.require_all:
            if required_caps.issubset(model_caps):
                matching.append(model)
        else:
            if required_caps & model_caps:
                matching.append(model)

    return ModelList(
        models=[ModelResponse(**m) for m in matching],
        total=len(matching),
        offset=0,
        limit=len(matching),
    )


@router.delete("/{model_id}")
async def unregister_model(model_id: str) -> Dict[str, str]:
    """Unregister a model."""
    if model_id not in _registered_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    del _registered_models[model_id]
    logger.info("model_unregistered", model_id=model_id)
    return {"message": f"Model {model_id} unregistered"}
