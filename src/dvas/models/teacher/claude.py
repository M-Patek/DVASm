"""Claude teacher model implementation with connection pooling."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from dvas.config import settings
from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.teacher.base import TeacherModel


class ClaudeTeacher(TeacherModel):
    """Claude 3 (Sonnet/Opus) wrapper for video annotation with connection pooling."""

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_frames = 20  # Claude has lower frame limit
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self.__http_client: Optional[httpx.AsyncClient] = None
        self._client: Optional[AsyncAnthropic] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        return ModelType.TEACHER_CLAUDE

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        return self.model_name

    @property
    def _http_client(self) -> httpx.AsyncClient:
        if self.__http_client is None:
            self.__http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self.__http_client

    @_http_client.setter
    def _http_client(self, value: Optional[httpx.AsyncClient]) -> None:
        self.__http_client = value

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self._api_key or settings.ANTHROPIC_API_KEY,
                http_client=self._http_client,
            )
        return self._client

    @client.setter
    def client(self, value: AsyncAnthropic) -> None:
        self._client = value

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
        """Close the HTTP client and free resources."""
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
        # Claude: ~$0.003 per image, ~$0.008 per 1K tokens
        image_cost = num_frames * 0.003
        token_cost = (prompt_length / 1000) * 0.008
        return image_cost + token_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        temperature: float = 0.2,
        **kwargs
    ) -> GenerationResult:
        """Generate annotation using Claude."""
        if frames is None:
            return GenerationResult.failure(
                error_message="Claude requires pre-extracted frames",
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

            # Build content
            content = []
            for encoded in encoded_frames:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded,
                    }
                })
            content.append({"type": "text", "text": system_prompt})

            async with self.semaphore:
                response = await self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=temperature,
                    system="You are an expert video understanding AI specializing in robotic manipulation and egocentric vision.",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    **kwargs
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
                metadata={
                    "stop_reason": response.stop_reason,
                },
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

    async def annotate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """Batch annotation with concurrency control."""
        tasks = [
            self.annotate(
                frames=item["frames"],
                prompt=item.get("prompt"),
                **kwargs
            )
            for item in items
        ]
        return await asyncio.gather(*tasks)

    async def generate(
        self,
        frames: Optional[List[np.ndarray]] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs
    ) -> GenerationResult:
        """UnifiedModel.generate implementation - delegates to annotate."""
        return await self.annotate(
            video_path=video_path,
            frames=frames,
            prompt=prompt,
            task=task,
            **kwargs
        )

    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """UnifiedModel.generate_batch implementation - delegates to annotate_batch."""
        return await self.annotate_batch(items, **kwargs)
