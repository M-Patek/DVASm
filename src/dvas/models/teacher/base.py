"""Base class for teacher models."""

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from PIL import Image

from dvas.config.prompts import PromptManager
from dvas.core.concurrency import FrameEncoderPool
from dvas.models.base import GenerationResult, GenerationStatus, ModelType, UnifiedModel


class TeacherModel(UnifiedModel):
    """Unified teacher model for generating gold-standard annotations.

    Automatically selects the appropriate API based on model_name:
    - GPT-5.5 models: Uses OpenAI API
    - Claude models: Uses Anthropic API
    - Together models: Uses Together AI API

    Example:
        teacher = TeacherModel(model_name="gpt-5.5")  # OpenAI
        teacher = TeacherModel(model_name="claude-opus-4-8")  # Anthropic
        teacher = TeacherModel(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct")  # Together
    """

    # Model name patterns for auto-detection
    OPENAI_PATTERNS = ["gpt-", "o1", "o3"]
    ANTHROPIC_PATTERNS = ["claude-"]
    TOGETHER_PATTERNS = ["meta-llama", "qwen", "mixtral", "nousresearch"]

    # Default model configurations
    DEFAULT_MODELS = {
        "gpt-5.5": {"provider": "openai", "max_frames": 32},
        "claude-opus-4-8": {"provider": "anthropic", "max_frames": 20},
        "claude-sonnet-4-6": {"provider": "anthropic", "max_frames": 20},
        "meta-llama/Llama-3.2-90B-Vision-Instruct": {"provider": "together", "max_frames": 16},
    }

    def __init__(
        self,
        model_name: str = "gpt-5.5",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs: Any,
    ):
        """Initialize teacher model.

        Args:
            model_name: Name of the model to use (e.g., "gpt-5.5", "claude-opus-4-8")
            api_key: Optional API key (defaults to settings)
            max_concurrent: Maximum concurrent API requests
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.config = kwargs
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._encoder_pool: Optional[FrameEncoderPool] = None
        self.__http_client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Lazy imports to avoid loading unused SDKs
        self._openai_client = None
        self._anthropic_client = None

        # Auto-detect provider and set max_frames
        self._provider = self._detect_provider(model_name)
        self.max_frames = self._get_max_frames(model_name)

    def _detect_provider(self, model_name: str) -> str:
        """Detect API provider from model name."""
        model_lower = model_name.lower()

        for pattern in self.OPENAI_PATTERNS:
            if pattern in model_lower:
                return "openai"

        for pattern in self.ANTHROPIC_PATTERNS:
            if pattern in model_lower:
                return "anthropic"

        for pattern in self.TOGETHER_PATTERNS:
            if pattern in model_lower:
                return "together"

        # Default to OpenAI for unknown models
        return "openai"

    def _get_max_frames(self, model_name: str) -> int:
        """Get max frames for the model."""
        if model_name in self.DEFAULT_MODELS:
            return self.DEFAULT_MODELS[model_name]["max_frames"]

        # Default based on provider
        provider_frames = {
            "openai": 32,
            "anthropic": 20,
            "together": 16,
        }
        return provider_frames.get(self._provider, 16)

    @property
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        provider_map = {
            "openai": ModelType.TEACHER_GPT55,
            "anthropic": ModelType.TEACHER_CLAUDE,
            "together": ModelType.TEACHER_TOGETHER,
        }
        return provider_map.get(self._provider, ModelType.TEACHER_GPT55)

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        return self.model_name

    @property
    def encoder_pool(self) -> FrameEncoderPool:
        """Lazy-initialize frame encoder pool."""
        if self._encoder_pool is None:
            max_workers = self.config.get("encoder_workers", 4)
            self._encoder_pool = FrameEncoderPool(max_workers=max_workers)
        return self._encoder_pool

    @property
    def _http_client(self) -> httpx.AsyncClient:
        if self.__http_client is None:
            self.__http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                    keepalive_expiry=30.0,
                ),
                http2=True,  # 启用HTTP/2多路复用
                timeout=httpx.Timeout(connect=10.0, read=120.0, pool=5.0),
                headers={
                    "Connection": "keep-alive",
                    "Keep-Alive": "timeout=30, max=100",
                },
            )
        return self.__http_client

    @_http_client.setter
    def _http_client(self, value: Optional[httpx.AsyncClient]) -> None:
        self.__http_client = value

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import AsyncOpenAI
            from dvas.config import settings

            self._openai_client = AsyncOpenAI(
                api_key=self._api_key or settings.OPENAI_API_KEY,
                http_client=self._http_client,
            )
        return self._openai_client

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic
            from dvas.config import settings

            self._anthropic_client = AsyncAnthropic(
                api_key=self._api_key or settings.ANTHROPIC_API_KEY,
                http_client=self._http_client,
            )
        return self._anthropic_client

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    @semaphore.setter
    def semaphore(self, value: asyncio.Semaphore) -> None:
        self._semaphore = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close HTTP clients and free resources."""
        if self.__http_client is not None:
            await self.__http_client.aclose()

    def _capabilities(self) -> List[str]:
        """Return list of supported capabilities."""
        return ["video", "frames", "text", "multimodal"]

    def estimate_cost(
        self,
        num_frames: int = 16,
        prompt_length: int = 500,
    ) -> float:
        """Estimate cost for a generation request."""
        provider_costs = {
            "openai": {"image": 0.005, "token": 0.005},
            "anthropic": {"image": 0.003, "token": 0.008},
            "together": {"image": 0.001, "token": 0.001},
        }

        costs = provider_costs.get(self._provider, provider_costs["openai"])
        image_cost = num_frames * costs["image"]
        token_cost = (prompt_length / 1000) * costs["token"]
        return image_cost + token_cost

    def _encode_image(self, image: np.ndarray, format: str = "JPEG") -> str:
        """Encode numpy image to base64 string."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image

        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def _encode_frames(self, frames: List[np.ndarray]) -> List[str]:
        """Encode multiple frames to base64 strings using async pool."""
        return await self.encoder_pool.encode_frames(frames, convert_bgr_to_rgb=True)

    def _encode_frames_sync(self, frames: List[np.ndarray]) -> List[str]:
        """Synchronous fallback for frame encoding."""
        if len(frames) <= 4:
            return [self._encode_image(frame) for frame in frames]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(frames))) as executor:
            return list(executor.map(self._encode_image, frames))

    def _get_default_prompt(self, task: str = "caption") -> str:
        """Get default prompt for a task."""
        return PromptManager.get_prompt(task)

    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        temperature: float = 0.2,
        **kwargs,
    ) -> GenerationResult:
        """Generate annotation using the configured model."""
        if frames is None and video_path is None:
            return GenerationResult.failure(
                error_message="Must provide either video_path or frames",
                model_type=self.model_type,
                model_version=self.model_version,
            )

        if frames is None:
            return GenerationResult.failure(
                error_message="Direct video path not supported. Extract frames first.",
                model_type=self.model_type,
                model_version=self.model_version,
            )

        start_time = time.perf_counter()

        try:
            system_prompt = prompt or self._get_default_prompt(task)

            # Sample frames if too many
            if len(frames) > self.max_frames:
                indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
                frames = [frames[i] for i in indices]

            # Encode frames
            encoded_frames = await self._encode_frames(frames)

            # Route to appropriate provider
            if self._provider == "openai":
                return await self._annotate_openai(
                    encoded_frames, system_prompt, temperature, start_time, **kwargs
                )
            elif self._provider == "anthropic":
                return await self._annotate_anthropic(
                    encoded_frames, system_prompt, temperature, start_time, **kwargs
                )
            elif self._provider == "together":
                return await self._annotate_together(
                    encoded_frames, system_prompt, temperature, start_time, **kwargs
                )
            else:
                return GenerationResult.failure(
                    error_message=f"Unknown provider: {self._provider}",
                    model_type=self.model_type,
                    model_version=self.model_version,
                )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                text="",
                model_type=self.model_type,
                model_version=self.model_name,
                status=GenerationStatus.FAILURE,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def _annotate_openai(
        self,
        encoded_frames: List[str],
        system_prompt: str,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> GenerationResult:
        """Annotate using OpenAI API."""

        client = self._get_openai_client()

        content = [{"type": "text", "text": system_prompt}]
        for encoded in encoded_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "low" if len(encoded_frames) > 8 else "high",
                    },
                }
            )

        async with self.semaphore:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video understanding AI specializing in robotic manipulation and egocentric vision.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=2048,
                **kwargs,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        usage = response.usage.model_dump() if response.usage else {}
        token_usage = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }

        return GenerationResult(
            text=response.choices[0].message.content or "",
            model_type=self.model_type,
            model_version=self.model_name,
            status=GenerationStatus.SUCCESS,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost_usd=self.estimate_cost(num_frames=len(encoded_frames)),
            metadata={"finish_reason": response.choices[0].finish_reason},
        )

    async def _annotate_anthropic(
        self,
        encoded_frames: List[str],
        system_prompt: str,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> GenerationResult:
        """Annotate using Anthropic API."""

        client = self._get_anthropic_client()

        content = []
        for encoded in encoded_frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded,
                    },
                }
            )
        content.append({"type": "text", "text": system_prompt})

        async with self.semaphore:
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=temperature,
                system="You are an expert video understanding AI specializing in robotic manipulation and egocentric vision.",
                messages=[{"role": "user", "content": content}],
                **kwargs,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        usage = response.usage
        token_usage = {
            "input": getattr(usage, "input_tokens", 0),
            "output": getattr(usage, "output_tokens", 0),
        }

        return GenerationResult(
            text=response.content[0].text if response.content else "",
            model_type=self.model_type,
            model_version=self.model_name,
            status=GenerationStatus.SUCCESS,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost_usd=self.estimate_cost(num_frames=len(encoded_frames)),
            metadata={"stop_reason": response.stop_reason},
        )

    async def _annotate_together(
        self,
        encoded_frames: List[str],
        system_prompt: str,
        temperature: float,
        start_time: float,
        **kwargs,
    ) -> GenerationResult:
        """Annotate using Together AI API."""
        from openai import AsyncOpenAI
        from dvas.config import settings

        # Use existing client if available, otherwise create new one
        if self._openai_client is not None:
            client = self._openai_client
        else:
            client = AsyncOpenAI(
                api_key=settings.TOGETHER_API_KEY,
                base_url="https://api.together.xyz/v1",
                http_client=self._http_client,
            )

        content = [{"type": "text", "text": system_prompt}]
        for encoded in encoded_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
                }
            )

        async with self.semaphore:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video understanding AI specializing in robotic manipulation and egocentric vision.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=2048,
                **kwargs,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        usage = response.usage.model_dump() if response.usage else {}
        token_usage = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }

        return GenerationResult(
            text=response.choices[0].message.content or "",
            model_type=self.model_type,
            model_version=self.model_name,
            status=GenerationStatus.SUCCESS,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost_usd=self.estimate_cost(num_frames=len(encoded_frames)),
            metadata={"finish_reason": response.choices[0].finish_reason},
        )

    async def annotate_batch(
        self, items: List[Dict[str, Any]], max_concurrent: Optional[int] = None, **kwargs
    ) -> List[GenerationResult]:
        """Batch annotation with concurrency control."""
        sem = asyncio.Semaphore(max_concurrent or self._max_concurrent)

        async def _annotate_one(item: Dict[str, Any]) -> GenerationResult:
            async with sem:
                return await self.annotate(
                    frames=item["frames"], prompt=item.get("prompt"), **kwargs
                )

        tasks = [_annotate_one(item) for item in items]
        return await asyncio.gather(*tasks)

    async def generate(
        self,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """UnifiedModel.generate implementation - delegates to annotate."""
        return await self.annotate(
            video_path=video_path, frames=frames, prompt=prompt, task=task, **kwargs
        )

    async def generate_batch(self, items: List[Dict[str, Any]], **kwargs) -> List[GenerationResult]:
        """UnifiedModel.generate_batch implementation - delegates to annotate_batch."""
        return await self.annotate_batch(items, **kwargs)
