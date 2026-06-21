"""OpenAI Batch API client for cost-efficient bulk processing.

Provides OpenAIBatchAPI for submitting large batches of requests
via the OpenAI Batch API, which offers 50% cost savings.
Falls back to concurrent single requests on failure.

Usage::

    from dvas.models.teacher.batch_api import OpenAIBatchAPI, BatchRequest

    batch_api = OpenAIBatchAPI(api_key="sk-...")

    # Create batch requests
    requests = [
        BatchRequest(
            custom_id=f"req-{i}",
            messages=[{"role": "user", "content": "Describe this video"}],
        )
        for i in range(100)
    ]

    # Submit and wait for results
    results = batch_api.submit_and_wait(requests)

    # Or submit and poll separately
    batch_id = batch_api.submit_batch(requests)
    status = batch_api.poll_status(batch_id)
    if status.is_complete:
        results = batch_api.download_results(batch_id)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Batch API max file size: 100MB
MAX_BATCH_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
# Batch API max requests per file
MAX_BATCH_REQUESTS = 50_000


@dataclass
class BatchRequest:
    """Single request for batch processing.

    Attributes:
        custom_id: Unique identifier for this request
        messages: List of message dicts for the chat completion
        model: Model name (defaults to gpt-4o)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        extra_body: Additional parameters
    """

    custom_id: str
    messages: List[Dict[str, Any]]
    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to OpenAI batch API request format."""
        body = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.extra_body:
            body.update(self.extra_body)

        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }


@dataclass
class BatchResult:
    """Result from a batch request.

    Attributes:
        custom_id: The custom_id from the request
        status: "completed", "failed", or "pending"
        text: The generated text (if completed)
        error: Error message (if failed)
        token_usage: Token usage stats
        raw_response: Full API response
    """

    custom_id: str
    status: str
    text: str = ""
    error: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchStatus:
    """Status of a batch job.

    Attributes:
        batch_id: The batch ID
        status: "validating", "in_progress", "finalizing", "completed", "failed", "expired", "cancelled"
        request_counts: Counts of requests in each state
        is_complete: Whether the batch is finished
        output_file_id: ID of the output file (when complete)
        error_file_id: ID of the error file (when complete)
    """

    batch_id: str
    status: str
    request_counts: Dict[str, int] = field(default_factory=dict)
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "expired", "cancelled")

    @property
    def is_success(self) -> bool:
        return self.status == "completed"


class OpenAIBatchAPI:
    """Client for OpenAI Batch API.

    Provides methods to submit, monitor, and retrieve results
    from batch processing jobs.

    Attributes:
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        poll_interval: Seconds between status polls
        max_poll_time: Maximum seconds to wait for completion
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        poll_interval: int = 30,
        max_poll_time: int = 24 * 60 * 60,  # 24 hours
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def _post(self, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Make a POST request to the OpenAI API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        response = await client.post(url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request to the OpenAI API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

    def _build_jsonl(self, requests: List[BatchRequest]) -> str:
        """Build JSONL string from batch requests.

        Args:
            requests: List of batch requests

        Returns:
            JSONL string (one JSON object per line)
        """
        lines = []
        for req in requests:
            lines.append(json.dumps(req.to_api_format()))
        return "\n".join(lines) + "\n"

    async def submit_batch(self, requests: List[BatchRequest]) -> str:
        """Submit a batch of requests to the OpenAI Batch API.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Batch ID for tracking

        Raises:
            ValueError: If requests list is empty or too large
        """
        if not requests:
            raise ValueError("Requests list cannot be empty")

        if len(requests) > MAX_BATCH_REQUESTS:
            raise ValueError(f"Maximum {MAX_BATCH_REQUESTS} requests per batch")

        # Build JSONL file
        jsonl_content = self._build_jsonl(requests)
        file_size = len(jsonl_content.encode("utf-8"))

        if file_size > MAX_BATCH_FILE_SIZE:
            raise ValueError(
                f"Batch file too large: {file_size / (1024 * 1024):.1f}MB "
                f"(max {MAX_BATCH_FILE_SIZE / (1024 * 1024):.0f}MB)"
            )

        # Upload file
        client = await self._get_client()
        files = {
            "file": ("batch_requests.jsonl", jsonl_content, "application/jsonl"),
            "purpose": (None, "batch"),
        }

        upload_response = await client.post(
            "https://api.openai.com/v1/files",
            files=files,
        )
        upload_response.raise_for_status()
        file_id = upload_response.json()["id"]

        logger.info(
            "Batch file uploaded",
            file_id=file_id,
            requests=len(requests),
            size_mb=round(file_size / (1024 * 1024), 2),
        )

        # Create batch
        batch_response = await self._post(
            "/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )

        batch_id = batch_response["id"]
        logger.info("Batch submitted", batch_id=batch_id, file_id=file_id)

        return batch_id

    async def poll_status(self, batch_id: str) -> BatchStatus:
        """Poll the status of a batch job.

        Args:
            batch_id: The batch ID

        Returns:
            BatchStatus with current state
        """
        data = await self._get(f"/batches/{batch_id}")

        return BatchStatus(
            batch_id=data["id"],
            status=data["status"],
            request_counts=data.get("request_counts", {}),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
        )

    async def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: Optional[int] = None,
        max_poll_time: Optional[int] = None,
    ) -> BatchStatus:
        """Wait for a batch job to complete.

        Args:
            batch_id: The batch ID
            poll_interval: Seconds between polls (default: self.poll_interval)
            max_poll_time: Maximum seconds to wait (default: self.max_poll_time)

        Returns:
            Final BatchStatus

        Raises:
            TimeoutError: If batch doesn't complete within max_poll_time
        """
        interval = poll_interval or self.poll_interval
        max_time = max_poll_time or self.max_poll_time
        start_time = time.time()

        while True:
            status = await self.poll_status(batch_id)

            if status.is_complete:
                logger.info(
                    "Batch completed",
                    batch_id=batch_id,
                    status=status.status,
                    request_counts=status.request_counts,
                )
                return status

            elapsed = time.time() - start_time
            if elapsed > max_time:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {max_time}s"
                )

            logger.debug(
                "Batch still in progress",
                batch_id=batch_id,
                status=status.status,
                elapsed_seconds=int(elapsed),
            )
            await asyncio.sleep(interval)

    async def download_results(self, batch_id: str) -> List[BatchResult]:
        """Download and parse results from a completed batch.

        Args:
            batch_id: The batch ID

        Returns:
            List of BatchResult objects
        """
        status = await self.poll_status(batch_id)

        if not status.is_complete:
            raise ValueError(f"Batch {batch_id} is not complete (status: {status.status})")

        if not status.output_file_id:
            logger.warning("No output file for batch", batch_id=batch_id)
            return []

        # Download output file
        client = await self._get_client()
        response = await client.get(
            f"https://api.openai.com/v1/files/{status.output_file_id}/content"
        )
        response.raise_for_status()

        # Parse results
        results = []
        for line in response.text.strip().split("\n"):
            if not line:
                continue

            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            response_body = data.get("response", {})

            if "error" in data:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        status="failed",
                        error=str(data["error"]),
                    )
                )
                continue

            # Extract text from response
            choices = response_body.get("body", {}).get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
            else:
                text = ""

            usage = response_body.get("body", {}).get("usage", {})

            results.append(
                BatchResult(
                    custom_id=custom_id,
                    status="completed",
                    text=text,
                    token_usage=usage,
                    raw_response=response_body,
                )
            )

        logger.info(
            "Batch results downloaded",
            batch_id=batch_id,
            results=len(results),
        )

        return results

    async def submit_and_wait(
        self,
        requests: List[BatchRequest],
        poll_interval: Optional[int] = None,
        max_poll_time: Optional[int] = None,
    ) -> List[BatchResult]:
        """Submit a batch and wait for completion.

        Args:
            requests: List of BatchRequest objects
            poll_interval: Seconds between polls
            max_poll_time: Maximum seconds to wait

        Returns:
            List of BatchResult objects
        """
        batch_id = await self.submit_batch(requests)
        status = await self.wait_for_completion(batch_id, poll_interval, max_poll_time)

        if status.is_success:
            return await self.download_results(batch_id)
        else:
            logger.error("Batch failed", batch_id=batch_id, status=status.status)
            raise RuntimeError(f"Batch {batch_id} failed with status: {status.status}")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Import asyncio here to avoid circular import issues
import asyncio


def create_batch_request(
    text: str,
    custom_id: Optional[str] = None,
    model: str = "gpt-4o",
    system_prompt: Optional[str] = None,
) -> BatchRequest:
    """Create a BatchRequest from a text prompt.

    Args:
        text: The user prompt text
        custom_id: Optional custom ID (auto-generated if not provided)
        model: Model to use
        system_prompt: Optional system prompt

    Returns:
        BatchRequest ready for submission
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    return BatchRequest(
        custom_id=custom_id or str(uuid.uuid4()),
        messages=messages,
        model=model,
    )
