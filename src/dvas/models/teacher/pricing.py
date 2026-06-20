"""Pricing registry for teacher models.

Tracks per-model pricing for input/output tokens and images.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PricingInfo:
    """Pricing information for a model.

    All prices are in USD per unit.

    Attributes:
        model_name: Model identifier
        input_token_price: Price per 1K input tokens
        output_token_price: Price per 1K output tokens
        image_price: Price per image
        cached_token_price: Price per 1K cached tokens (if supported)
        batch_discount: Discount factor for batch requests (e.g., 0.5 for 50% off)
        currency: Currency code (default USD)
    """

    model_name: str
    input_token_price: float
    output_token_price: float
    image_price: float = 0.0
    cached_token_price: Optional[float] = None
    batch_discount: float = 1.0
    currency: str = "USD"

    def calculate_cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_images: int = 0,
        cached_tokens: int = 0,
        is_batch: bool = False,
    ) -> float:
        """Calculate total cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_images: Number of images/frames
            cached_tokens: Number of cached tokens
            is_batch: Whether this is a batch request

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * self.input_token_price
        output_cost = (output_tokens / 1000) * self.output_token_price
        image_cost = num_images * self.image_price

        cached_cost = 0.0
        if self.cached_token_price is not None and cached_tokens > 0:
            cached_cost = (cached_tokens / 1000) * self.cached_token_price

        total = input_cost + output_cost + image_cost + cached_cost

        if is_batch and self.batch_discount < 1.0:
            total *= self.batch_discount

        return total


# Pricing data (as of June 2026 - approximate values)
# Prices are per 1K tokens unless otherwise noted
MODEL_PRICING: Dict[str, PricingInfo] = {
    # OpenAI models
    "gpt-5.5": PricingInfo(
        model_name="gpt-5.5",
        input_token_price=0.015,
        output_token_price=0.060,
        image_price=0.005,
        cached_token_price=0.0075,
        batch_discount=0.5,
    ),
    "gpt-5": PricingInfo(
        model_name="gpt-5",
        input_token_price=0.005,
        output_token_price=0.015,
        image_price=0.005,
        cached_token_price=0.0025,
        batch_discount=0.5,
    ),
    "o3": PricingInfo(
        model_name="o3",
        input_token_price=0.010,
        output_token_price=0.040,
        image_price=0.005,
        batch_discount=0.5,
    ),
    "o1": PricingInfo(
        model_name="o1",
        input_token_price=0.015,
        output_token_price=0.060,
        image_price=0.005,
        batch_discount=0.5,
    ),

    # Anthropic models
    "claude-opus-4-8": PricingInfo(
        model_name="claude-opus-4-8",
        input_token_price=0.015,
        output_token_price=0.075,
        image_price=0.003,
        cached_token_price=0.001875,
    ),
    "claude-sonnet-4-6": PricingInfo(
        model_name="claude-sonnet-4-6",
        input_token_price=0.003,
        output_token_price=0.015,
        image_price=0.003,
        cached_token_price=0.000375,
    ),

    # Together AI models (approximate)
    "meta-llama/Llama-3.2-90B-Vision-Instruct": PricingInfo(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
        input_token_price=0.001,
        output_token_price=0.001,
        image_price=0.001,
    ),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": PricingInfo(
        model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        input_token_price=0.0008,
        output_token_price=0.0008,
        image_price=0.0008,
    ),
    "Qwen/Qwen2.5-VL-72B-Instruct": PricingInfo(
        model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        input_token_price=0.0015,
        output_token_price=0.0015,
        image_price=0.0015,
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": PricingInfo(
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        input_token_price=0.0006,
        output_token_price=0.0006,
        image_price=0.0006,
    ),
}


class PricingRegistry:
    """Registry for model pricing information."""

    def __init__(self):
        self._pricing = dict(MODEL_PRICING)
        self._provider_defaults: Dict[str, PricingInfo] = {
            "openai": PricingInfo(
                model_name="openai-default",
                input_token_price=0.010,
                output_token_price=0.030,
                image_price=0.005,
            ),
            "anthropic": PricingInfo(
                model_name="anthropic-default",
                input_token_price=0.005,
                output_token_price=0.015,
                image_price=0.003,
            ),
            "together": PricingInfo(
                model_name="together-default",
                input_token_price=0.001,
                output_token_price=0.001,
                image_price=0.001,
            ),
        }

    def get_pricing(self, model_name: str) -> Optional[PricingInfo]:
        """Get pricing information for a model."""
        return self._pricing.get(model_name)

    def get_pricing_or_default(self, model_name: str, provider: str) -> PricingInfo:
        """Get pricing for a model, falling back to provider default."""
        pricing = self._pricing.get(model_name)
        if pricing:
            return pricing

        default = self._provider_defaults.get(provider.lower())
        if default:
            return default

        # Ultimate fallback
        return PricingInfo(
            model_name="default",
            input_token_price=0.010,
            output_token_price=0.030,
            image_price=0.005,
        )

    def register_pricing(self, pricing: PricingInfo) -> None:
        """Register pricing for a model."""
        self._pricing[pricing.model_name] = pricing

    def set_provider_default(self, provider: str, pricing: PricingInfo) -> None:
        """Set default pricing for a provider."""
        self._provider_defaults[provider.lower()] = pricing

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_images: int = 0,
        is_batch: bool = False,
    ) -> float:
        """Estimate cost for a request."""
        pricing = self._pricing.get(model_name)
        if not pricing:
            return 0.0

        return pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_images=num_images,
            is_batch=is_batch,
        )

    def estimate_frame_analysis_cost(
        self,
        model_name: str,
        num_frames: int,
        avg_prompt_tokens: int = 500,
        expected_output_tokens: int = 300,
    ) -> float:
        """Estimate cost for frame analysis.

        Args:
            model_name: Model to use
            num_frames: Number of frames
            avg_prompt_tokens: Average prompt tokens
            expected_output_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = self._pricing.get(model_name)
        if not pricing:
            return 0.0

        # Rough estimate: each image adds ~600 tokens
        image_token_estimate = num_frames * 600
        total_input_tokens = avg_prompt_tokens + image_token_estimate

        return pricing.calculate_cost(
            input_tokens=total_input_tokens,
            output_tokens=expected_output_tokens,
            num_images=num_frames,
        )

    def compare_costs(
        self,
        model_names: List[str],
        input_tokens: int = 1000,
        output_tokens: int = 500,
        num_images: int = 16,
    ) -> Dict[str, float]:
        """Compare costs across multiple models."""
        results = {}
        for name in model_names:
            pricing = self._pricing.get(name)
            if pricing:
                results[name] = pricing.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    num_images=num_images,
                )
            else:
                results[name] = -1.0  # Unknown
        return results

    def list_models(self) -> List[str]:
        """List all models with pricing information."""
        return list(self._pricing.keys())

    def get_cheapest_for_task(
        self,
        num_images: int,
        input_tokens: int = 1000,
        output_tokens: int = 500,
    ) -> Optional[str]:
        """Find the cheapest model for a given task."""
        costs = self.compare_costs(
            self.list_models(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_images=num_images,
        )

        valid_costs = {k: v for k, v in costs.items() if v >= 0}
        if not valid_costs:
            return None

        return min(valid_costs.items(), key=lambda x: x[1])[0]


# Global registry instance
_pricing_registry: Optional[PricingRegistry] = None


def get_pricing_registry() -> PricingRegistry:
    """Get the global pricing registry instance."""
    global _pricing_registry
    if _pricing_registry is None:
        _pricing_registry = PricingRegistry()
    return _pricing_registry


def reset_pricing_registry() -> None:
    """Reset the global pricing registry (useful for testing)."""
    global _pricing_registry
    _pricing_registry = None
