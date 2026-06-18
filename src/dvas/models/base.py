"""Unified model interface for teacher and student models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ModelType(str, Enum):
    """Type of model used for generation."""

    TEACHER_GPT55 = "gpt-5.5"
    TEACHER_CLAUDE = "claude"
    TEACHER_TOGETHER = "together"
    STUDENT_LOCAL = "student-local"
    STUDENT_EDGE = "student-edge"
    MOCK = "mock"

    # Backwards compatibility
    TEACHER_GPT4V = "gpt-5.5"



class GenerationStatus(str, Enum):
    """Status of a generation request."""

    SUCCESS = "success"
    FAILURE = "failure"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class GenerationResult:
    """Standardized result from any model (teacher or student).

    All models must return this structure, regardless of internal format.
    """

    text: str = ""  # Primary generated text
    structured_data: Optional[Dict[str, Any]] = None  # Parsed structured output
    model_type: ModelType = ModelType.MOCK
    model_version: str = "unknown"
    status: GenerationStatus = GenerationStatus.SUCCESS
    confidence: float = 1.0  # Model's confidence in the result
    latency_ms: float = 0.0  # Generation latency
    token_usage: Dict[str, int] = field(default_factory=dict)  # input/output tokens
    cost_usd: float = 0.0  # Estimated API cost
    error_message: Optional[str] = None  # Populated on failure
    fallback_from: Optional[ModelType] = None  # If this was a fallback, original model
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra model-specific data

    def is_success(self) -> bool:
        """Check if generation was successful."""
        return self.status == GenerationStatus.SUCCESS

    def is_failure(self) -> bool:
        """Check if generation failed."""
        return self.status == GenerationStatus.FAILURE

    def is_timeout(self) -> bool:
        """Check if generation timed out."""
        return self.status == GenerationStatus.TIMEOUT

    def is_rate_limited(self) -> bool:
        """Check if generation was rate limited."""
        return self.status == GenerationStatus.RATE_LIMITED

    def is_recoverable(self) -> bool:
        """Check if error is recoverable (timeout or rate limit)."""
        return self.status in (GenerationStatus.TIMEOUT, GenerationStatus.RATE_LIMITED)

    def is_fallback(self) -> bool:
        """Check if this result came from a fallback model."""
        return self.fallback_from is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "structured_data": self.structured_data,
            "model_type": self.model_type.value,
            "model_version": self.model_version,
            "status": self.status.value,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "error_message": self.error_message,
            "fallback_from": self.fallback_from.value if self.fallback_from else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            structured_data=data.get("structured_data"),
            model_type=ModelType(data.get("model_type", "mock")),
            model_version=data.get("model_version", "unknown"),
            status=GenerationStatus(data.get("status", "success")),
            confidence=data.get("confidence", 1.0),
            latency_ms=data.get("latency_ms", 0.0),
            token_usage=data.get("token_usage", {}),
            cost_usd=data.get("cost_usd", 0.0),
            error_message=data.get("error_message"),
            fallback_from=ModelType(data["fallback_from"]) if data.get("fallback_from") else None,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def failure(
        cls,
        error_message: str,
        model_type: ModelType = ModelType.MOCK,
        model_version: str = "unknown",
    ) -> "GenerationResult":
        """Create a failure result."""
        return cls(
            status=GenerationStatus.FAILURE,
            error_message=error_message,
            model_type=model_type,
            model_version=model_version,
            confidence=0.0,
        )

    @classmethod
    def fallback(
        cls,
        text: str,
        fallback_from: ModelType,
        model_type: ModelType = ModelType.TEACHER_GPT4V,
    ) -> "GenerationResult":
        """Create a fallback result."""
        return cls(
            text=text,
            model_type=model_type,
            fallback_from=fallback_from,
            status=GenerationStatus.FALLBACK,
        )


class UnifiedModel(ABC):
    """Unified interface for all models (teacher and student).

    All implementations must return GenerationResult.
    No Dict[str, Any] allowed at interface boundary.
    """

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version string."""
        pass

    @abstractmethod
    async def generate(
        self,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs
    ) -> GenerationResult:
        """Generate annotation.

        Args:
            frames: Pre-extracted frames as numpy arrays
            video_path: Path to video file (if frames not provided)
            prompt: Custom prompt override
            task: Task type (caption, fine_grained, etc.)
            **kwargs: Model-specific parameters

        Returns:
            GenerationResult with standardized output
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """Batch generation.

        Args:
            items: List of dicts with 'frames' and optional 'prompt'
            **kwargs: Model-specific parameters

        Returns:
            List of GenerationResult, one per item
        """
        pass

    def supports(self, capability: str) -> bool:
        """Check if model supports a capability.

        Capabilities: video, frames, text, multimodal
        """
        return capability in self._capabilities()

    @abstractmethod
    def _capabilities(self) -> List[str]:
        """Return list of supported capabilities."""
        pass

    def estimate_cost(
        self,
        num_frames: int = 16,
        prompt_length: int = 500,
    ) -> float:
        """Estimate cost for a generation request."""
        return 0.0  # Override in concrete classes
