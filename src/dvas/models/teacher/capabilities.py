"""Capability registry for teacher models.

Provides capability detection and provider capability contracts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Capability(str, Enum):
    """Model capabilities."""

    VISION = "vision"
    VIDEO = "video"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    BATCH_PROCESSING = "batch_processing"
    SYSTEM_PROMPT = "system_prompt"
    MULTILINGUAL = "multilingual"


class OutputFormat(str, Enum):
    """Supported output formats."""

    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Capabilities supported by a provider.

    Attributes:
        provider: Provider name
        capabilities: Set of supported capabilities
        output_formats: Supported output formats
        max_batch_size: Maximum batch size (0 if not supported)
        rate_limit_requests_per_minute: Rate limit for requests
        rate_limit_tokens_per_minute: Rate limit for tokens
        structured_output_schema: JSON schema support level
        features: Additional provider-specific features
    """

    provider: str
    capabilities: Set[Capability] = field(default_factory=set)
    output_formats: Set[OutputFormat] = field(default_factory=lambda: {OutputFormat.TEXT})
    max_batch_size: int = 0
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000
    structured_output_schema: str = "none"  # none, partial, full
    features: Dict[str, Any] = field(default_factory=dict)


# Provider capability definitions
PROVIDER_CAPABILITIES: Dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(
        provider="openai",
        capabilities={
            Capability.VISION,
            Capability.VIDEO,
            Capability.STREAMING,
            Capability.JSON_MODE,
            Capability.FUNCTION_CALLING,
            Capability.STRUCTURED_OUTPUT,
            Capability.SYSTEM_PROMPT,
        },
        output_formats={OutputFormat.TEXT, OutputFormat.JSON, OutputFormat.STRUCTURED},
        max_batch_size=100,
        rate_limit_requests_per_minute=500,
        rate_limit_tokens_per_minute=2000000,
        structured_output_schema="full",
        features={
            "json_schema_validation": True,
            "strict_mode": True,
            "parallel_tool_calls": True,
        },
    ),
    "anthropic": ProviderCapabilities(
        provider="anthropic",
        capabilities={
            Capability.VISION,
            Capability.VIDEO,
            Capability.STREAMING,
            Capability.TOOL_USE,
            Capability.REASONING,
            Capability.SYSTEM_PROMPT,
            Capability.MULTILINGUAL,
        },
        output_formats={OutputFormat.TEXT, OutputFormat.JSON},
        max_batch_size=0,  # No native batch API
        rate_limit_requests_per_minute=100,
        rate_limit_tokens_per_minute=1000000,
        structured_output_schema="partial",  # Via tool use
        features={
            "citations": True,
            "computer_use": True,
            "extended_thinking": True,
        },
    ),
    "together": ProviderCapabilities(
        provider="together",
        capabilities={
            Capability.VISION,
            Capability.VIDEO,
            Capability.STREAMING,
            Capability.SYSTEM_PROMPT,
        },
        output_formats={OutputFormat.TEXT, OutputFormat.JSON},
        max_batch_size=0,
        rate_limit_requests_per_minute= 100,
        rate_limit_tokens_per_minute=500000,
        structured_output_schema="none",
        features={
            "custom_endpoints": True,
            "fine_tuning": True,
        },
    ),
}


@dataclass
class ModelCapabilities:
    """Capabilities for a specific model instance."""

    model_name: str
    provider: str
    capabilities: Set[Capability] = field(default_factory=set)
    output_formats: Set[OutputFormat] = field(default_factory=lambda: {OutputFormat.TEXT})
    max_frames: int = 16
    max_tokens: int = 2048
    supports_structured_output: bool = False


class CapabilityRegistry:
    """Registry for provider and model capabilities."""

    def __init__(self):
        self._provider_caps = dict(PROVIDER_CAPABILITIES)
        self._model_caps: Dict[str, ModelCapabilities] = {}

    def get_provider_capabilities(self, provider: str) -> Optional[ProviderCapabilities]:
        """Get capabilities for a provider."""
        return self._provider_caps.get(provider.lower())

    def get_model_capabilities(self, model_name: str) -> Optional[ModelCapabilities]:
        """Get capabilities for a specific model."""
        return self._model_caps.get(model_name)

    def register_model_capabilities(self, caps: ModelCapabilities) -> None:
        """Register capabilities for a model."""
        self._model_caps[caps.model_name] = caps

    def supports_capability(self, provider: str, capability: Capability) -> bool:
        """Check if a provider supports a capability."""
        caps = self._provider_caps.get(provider.lower())
        if caps:
            return capability in caps.capabilities
        return False

    def supports_output_format(self, provider: str, format: OutputFormat) -> bool:
        """Check if a provider supports an output format."""
        caps = self._provider_caps.get(provider.lower())
        if caps:
            return format in caps.output_formats
        return False

    def can_use_structured_output(self, provider: str, model_name: str) -> bool:
        """Check if structured output is supported."""
        provider_caps = self._provider_caps.get(provider.lower())
        if not provider_caps:
            return False

        if provider_caps.structured_output_schema == "full":
            return True

        if provider_caps.structured_output_schema == "partial":
            # Check model-specific capability
            model_caps = self._model_caps.get(model_name)
            if model_caps:
                return model_caps.supports_structured_output
            # Default to True for partial support
            return True

        return False

    def get_rate_limits(self, provider: str) -> Dict[str, int]:
        """Get rate limits for a provider."""
        caps = self._provider_caps.get(provider.lower())
        if caps:
            return {
                "requests_per_minute": caps.rate_limit_requests_per_minute,
                "tokens_per_minute": caps.rate_limit_tokens_per_minute,
            }
        return {"requests_per_minute": 60, "tokens_per_minute": 100000}

    def list_providers_with_capability(self, capability: Capability) -> List[str]:
        """List all providers that support a capability."""
        return [
            name for name, caps in self._provider_caps.items()
            if capability in caps.capabilities
        ]

    def get_batch_size_limit(self, provider: str) -> int:
        """Get maximum batch size for a provider."""
        caps = self._provider_caps.get(provider.lower())
        if caps:
            return caps.max_batch_size
        return 0

    def infer_model_capabilities(
        self,
        model_name: str,
        provider: str,
        max_frames: int = 16,
    ) -> ModelCapabilities:
        """Infer capabilities for a model based on provider defaults."""
        provider_caps = self._provider_caps.get(provider.lower())

        if provider_caps:
            caps = ModelCapabilities(
                model_name=model_name,
                provider=provider,
                capabilities=set(provider_caps.capabilities),
                output_formats=set(provider_caps.output_formats),
                max_frames=max_frames,
                supports_structured_output=provider_caps.structured_output_schema in ("full", "partial"),
            )
        else:
            caps = ModelCapabilities(
                model_name=model_name,
                provider=provider,
                max_frames=max_frames,
            )

        self._model_caps[model_name] = caps
        return caps


# Global registry instance
_capability_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry instance."""
    global _capability_registry
    if _capability_registry is None:
        _capability_registry = CapabilityRegistry()
    return _capability_registry


def reset_capability_registry() -> None:
    """Reset the global capability registry (useful for testing)."""
    global _capability_registry
    _capability_registry = None
