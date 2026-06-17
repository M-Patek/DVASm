"""GPT-4V/GPT-4o teacher model implementation with connection pooling."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from dvas.config import settings
from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.teacher.base import TeacherModel


class GPT4VTeacher(TeacherModel):
    """GPT-4V/4o wrapper for video annotation with connection pooling."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_frames = 32  # GPT-4o can handle up to ~32 frames
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client: Optional[AsyncOpenAI] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def model_type(self) -> ModelType:
        """Return the model type identifier."""
        return ModelType.TEACHER_GPT4V

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
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key or settings.OPENAI_API_KEY,
                http_client=self._http_client,
            )
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
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
        # GPT-4o: ~$0.005 per image, ~$0.005 per 1K tokens
        image_cost = num_frames * 0.005
        token_cost = (prompt_length / 1000) * 0.005
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
        temperature: float = 0.2,
        **kwargs
    ) -> GenerationResult:
        """Generate annotation using GPT-4V/o."""
        if frames is None and video_path is None:
            return GenerationResult.failure(
                error_message="Must provide either video_path or frames",
                model_type=self.model_type,
                model_version=self.model_version,
            )

        start_time = time.perf_counter()

        try:
            # Use provided prompt or default
            system_prompt = prompt or self._get_default_prompt("fine_grained")

            # Prepare image content
            if frames is not None:
                # Sample frames if too many
                if len(frames) > self.max_frames:
                    indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
                    frames = [frames[i] for i in indices]

                # Encode frames to base64
                encoded_frames = self._encode_frames(frames)

                # Build content with images
                content = [{"type": "text", "text": system_prompt}]
                for encoded in encoded_frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": "low" if len(encoded_frames) > 8 else "high"
                        }
                    })
            else:
                # For video paths, we'd need to extract frames first
                # This is handled by the caller
                return GenerationResult.failure(
                    error_message="Direct video path not supported. Extract frames first.",
                    model_type=self.model_type,
                    model_version=self.model_version,
                )

            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert video understanding AI specializing in robotic manipulation and egocentric vision."
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    **kwargs
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
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
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
        max_concurrent: Optional[int] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """Batch annotation with concurrency control.

        Args:
            items: List of items with 'frames' and optional 'prompt'
            max_concurrent: Override default concurrency limit
            **kwargs: Additional arguments passed to annotate()
        Returns:
            List of GenerationResult in same order as items
        """
        sem = asyncio.Semaphore(max_concurrent or self._max_concurrent)

        async def _annotate_one(item: Dict[str, Any]) -> GenerationResult:
            async with sem:
                return await self.annotate(
                    frames=item["frames"],
                    prompt=item.get("prompt"),
                    **kwargs
                )

        tasks = [_annotate_one(item) for item in items]
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
            **kwargs
        )

    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """UnifiedModel.generate_batch implementation - delegates to annotate_batch."""
        return await self.annotate_batch(items, **kwargs)


# Alias for GPT-4o
GPT4VisionTeacher = GPT4VTeacher
