"""Together.ai API wrapper for open-source vision models with connection pooling."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from dvas.config import settings
from dvas.models.teacher.base import TeacherModel


class TogetherTeacher(TeacherModel):
    """
    Together.ai wrapper for open-source vision models.
    Useful for Qwen2-VL, Llama-3-Vision, etc. via API.
    """

    # Model mapping
    MODELS = {
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "llama-3-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    }

    def __init__(
        self,
        model_name: str = "qwen2-vl-7b",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_frames = 8  # Lower limit for most models
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self.__http_client: Optional[httpx.AsyncClient] = None
        self._client: Optional[AsyncOpenAI] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.together_model = self.MODELS.get(model_name, model_name)

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
                api_key=self._api_key or settings.TOGETHER_API_KEY,
                base_url="https://api.together.xyz/v1",
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
    ) -> Dict[str, Any]:
        """Generate annotation using Together AI."""
        if frames is None:
            raise ValueError("Together API requires pre-extracted frames")

        system_prompt = prompt or self._get_default_prompt("fine_grained")

        # Sample frames if too many
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Encode frames
        encoded_frames = self._encode_frames(frames)

        # Build content
        content = [{"type": "text", "text": system_prompt}]
        for encoded in encoded_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                }
            })

        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.together_model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=temperature,
                max_tokens=2048,
                **kwargs
            )

        return {
            "text": response.choices[0].message.content,
            "model": self.together_model,
            "usage": response.usage.model_dump() if response.usage else {},
            "finish_reason": response.choices[0].finish_reason,
        }

    async def annotate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch annotation with concurrency control."""
        tasks = [
            self.annotate(
                frames=item["frames"],
                prompt=item.get("prompt"),
                **kwargs
            )
            for item in items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
