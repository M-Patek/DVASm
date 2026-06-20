"""Provider and model registry for teacher models.

This module provides centralized registry for:
- Provider configurations
- Model frame limits
- Model metadata
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Provider(str, Enum):
    """Supported API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a teacher model.

    Attributes:
        name: Model identifier (e.g., "gpt-5.5")
        provider: API provider
        max_frames: Maximum number of frames the model can process
        max_tokens: Maximum output tokens
        context_window: Total context window size
        features: Set of supported features
        metadata: Additional model metadata
    """

    name: str
    provider: Provider
    max_frames: int = 16
    max_tokens: int = 2048
    context_window: int = 8192
    features: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Model registry with specifications
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # OpenAI models
    "gpt-5.5": ModelSpec(
        name="gpt-5.5",
        provider=Provider.OPENAI,
        max_frames=32,
        max_tokens=4096,
        context_window=128000,
        features={"vision", "json_mode", "function_calling", "streaming"},
        metadata={
            "release_date": "2025-01",
            "recommended_for": ["general", "complex_reasoning"],
        },
    ),
    "gpt-5": ModelSpec(
        name="gpt-5",
        provider=Provider.OPENAI,
        max_frames=32,
        max_tokens=4096,
        context_window=128000,
        features={"vision", "json_mode", "function_calling", "streaming"},
        metadata={
            "release_date": "2025-01",
            "recommended_for": ["general"],
        },
    ),
    "o3": ModelSpec(
        name="o3",
        provider=Provider.OPENAI,
        max_frames=32,
        max_tokens=4096,
        context_window=200000,
        features={"vision", "json_mode", "function_calling", "streaming", "reasoning"},
        metadata={
            "release_date": "2025-01",
            "recommended_for": ["complex_reasoning", "math"],
        },
    ),
    "o1": ModelSpec(
        name="o1",
        provider=Provider.OPENAI,
        max_frames=32,
        max_tokens=4096,
        context_window=200000,
        features={"vision", "reasoning"},
        metadata={
            "release_date": "2024-12",
            "recommended_for": ["complex_reasoning"],
        },
    ),
    # Anthropic models
    "claude-opus-4-8": ModelSpec(
        name="claude-opus-4-8",
        provider=Provider.ANTHROPIC,
        max_frames=20,
        max_tokens=4096,
        context_window=200000,
        features={"vision", "json_mode", "streaming", "tool_use"},
        metadata={
            "release_date": "2025-06",
            "recommended_for": ["complex_reasoning", "long_context"],
        },
    ),
    "claude-sonnet-4-6": ModelSpec(
        name="claude-sonnet-4-6",
        provider=Provider.ANTHROPIC,
        max_frames=20,
        max_tokens=4096,
        context_window=200000,
        features={"vision", "json_mode", "streaming", "tool_use"},
        metadata={
            "release_date": "2025-05",
            "recommended_for": ["general", "balanced"],
        },
    ),
    # Together AI models
    "meta-llama/Llama-3.2-90B-Vision-Instruct": ModelSpec(
        name="meta-llama/Llama-3.2-90B-Vision-Instruct",
        provider=Provider.TOGETHER,
        max_frames=16,
        max_tokens=4096,
        context_window=128000,
        features={"vision", "streaming"},
        metadata={
            "release_date": "2024-09",
            "recommended_for": ["cost_effective"],
        },
    ),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelSpec(
        name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider=Provider.TOGETHER,
        max_frames=16,
        max_tokens=4096,
        context_window=128000,
        features={"vision", "streaming"},
        metadata={
            "release_date": "2025-04",
            "recommended_for": ["cost_effective", "general"],
        },
    ),
    "Qwen/Qwen2.5-VL-72B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-VL-72B-Instruct",
        provider=Provider.TOGETHER,
        max_frames=16,
        max_tokens=4096,
        context_window=32768,
        features={"vision", "streaming"},
        metadata={
            "release_date": "2025-01",
            "recommended_for": ["multilingual", "cost_effective"],
        },
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelSpec(
        name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        provider=Provider.TOGETHER,
        max_frames=16,
        max_tokens=4096,
        context_window=128000,
        features={"vision", "streaming"},
        metadata={
            "release_date": "2025-04",
            "recommended_for": ["cost_effective"],
        },
    ),
}


# Provider patterns for auto-detection
PROVIDER_PATTERNS: Dict[Provider, List[str]] = {
    Provider.OPENAI: ["gpt-", "o1", "o3"],
    Provider.ANTHROPIC: ["claude-"],
    Provider.TOGETHER: ["meta-llama", "qwen", "mixtral", "nousresearch", "Qwen/"],
}


class ModelRegistry:
    """Central registry for teacher models."""

    def __init__(self):
        self._models = dict(MODEL_REGISTRY)
        self._patterns = dict(PROVIDER_PATTERNS)

    def get_model(self, name: str) -> Optional[ModelSpec]:
        """Get model specification by name."""
        return self._models.get(name)

    def list_models(
        self,
        provider: Optional[Provider] = None,
        feature: Optional[str] = None,
    ) -> List[ModelSpec]:
        """List all models, optionally filtered by provider or feature."""
        models = list(self._models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if feature:
            models = [m for m in models if feature in m.features]

        return models

    def detect_provider(self, model_name: str) -> Provider:
        """Detect provider from model name."""
        model_lower = model_name.lower()

        for provider, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.lower() in model_lower:
                    return provider

        # Default to OpenAI for unknown models
        return Provider.OPENAI

    def get_max_frames(self, model_name: str) -> int:
        """Get maximum frames for a model."""
        spec = self._models.get(model_name)
        if spec:
            return spec.max_frames

        # Default based on provider
        provider = self.detect_provider(model_name)
        defaults = {
            Provider.OPENAI: 32,
            Provider.ANTHROPIC: 20,
            Provider.TOGETHER: 16,
        }
        return defaults.get(provider, 16)

    def supports_feature(self, model_name: str, feature: str) -> bool:
        """Check if a model supports a specific feature."""
        spec = self._models.get(model_name)
        if spec:
            return feature in spec.features
        return False

    def register_model(self, spec: ModelSpec) -> None:
        """Register a new model specification."""
        self._models[spec.name] = spec

    def get_provider_models(self, provider: Provider) -> List[str]:
        """Get all model names for a provider."""
        return [m.name for m in self._models.values() if m.provider == provider]

    def get_recommended_models(self, task: str) -> List[ModelSpec]:
        """Get models recommended for a specific task."""
        return [m for m in self._models.values() if task in m.metadata.get("recommended_for", [])]


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
