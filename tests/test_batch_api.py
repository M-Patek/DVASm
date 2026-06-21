"""Tests for OpenAI Batch API client."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.teacher.batch_api import (
    MAX_BATCH_REQUESTS,
    BatchRequest,
    BatchStatus,
    OpenAIBatchAPI,
    create_batch_request,
)


class TestBatchRequest:
    """Test BatchRequest dataclass."""

    def test_to_api_format(self):
        req = BatchRequest(
            custom_id="req-1",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
            max_tokens=100,
            temperature=0.5,
        )
        api_format = req.to_api_format()

        assert api_format["custom_id"] == "req-1"
        assert api_format["method"] == "POST"
        assert api_format["url"] == "/v1/chat/completions"
        assert api_format["body"]["model"] == "gpt-4o"
        assert api_format["body"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert api_format["body"]["max_tokens"] == 100
        assert api_format["body"]["temperature"] == 0.5

    def test_extra_body(self):
        req = BatchRequest(
            custom_id="req-2",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"top_p": 0.9},
        )
        api_format = req.to_api_format()
        assert api_format["body"]["top_p"] == 0.9


class TestBatchStatus:
    """Test BatchStatus dataclass."""

    def test_is_complete(self):
        assert BatchStatus("id", "completed").is_complete is True
        assert BatchStatus("id", "failed").is_complete is True
        assert BatchStatus("id", "expired").is_complete is True
        assert BatchStatus("id", "in_progress").is_complete is False

    def test_is_success(self):
        assert BatchStatus("id", "completed").is_success is True
        assert BatchStatus("id", "failed").is_success is False


class TestOpenAIBatchAPIInit:
    """Test OpenAIBatchAPI initialization."""

    def test_init(self):
        api = OpenAIBatchAPI(api_key="test-key")
        assert api.api_key == "test-key"
        assert api.poll_interval == 30
        assert api.max_poll_time == 24 * 60 * 60

    def test_init_custom_params(self):
        api = OpenAIBatchAPI(
            api_key="test-key",
            poll_interval=10,
            max_poll_time=3600,
        )
        assert api.poll_interval == 10
        assert api.max_poll_time == 3600


class TestOpenAIBatchAPIBuildJsonl:
    """Test JSONL building."""

    def test_build_jsonl(self):
        api = OpenAIBatchAPI(api_key="test-key")
        requests = [
            BatchRequest(custom_id="req-1", messages=[{"role": "user", "content": "Hello"}]),
            BatchRequest(custom_id="req-2", messages=[{"role": "user", "content": "World"}]),
        ]
        jsonl = api._build_jsonl(requests)

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        data = json.loads(lines[0])
        assert data["custom_id"] == "req-1"
        assert data["body"]["messages"][0]["content"] == "Hello"


class TestOpenAIBatchAPISubmit:
    """Test batch submission."""

    @pytest.mark.asyncio
    async def test_submit_batch_success(self):
        api = OpenAIBatchAPI(api_key="test-key")

        # Mock file upload response
        mock_upload_response = MagicMock()
        mock_upload_response.json.return_value = {"id": "file-123"}
        mock_upload_response.raise_for_status = MagicMock()

        # Mock batch create response
        mock_batch_response = MagicMock()
        mock_batch_response.json.return_value = {"id": "batch-456"}
        mock_batch_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_upload_response, mock_batch_response]

        with patch.object(api, "_get_client", return_value=mock_client):
            requests = [
                BatchRequest(custom_id="req-1", messages=[{"role": "user", "content": "Hello"}]),
            ]
            batch_id = await api.submit_batch(requests)
            assert batch_id == "batch-456"

    @pytest.mark.asyncio
    async def test_submit_batch_empty_requests(self):
        api = OpenAIBatchAPI(api_key="test-key")
        with pytest.raises(ValueError, match="empty"):
            await api.submit_batch([])

    def test_submit_batch_too_many_requests(self):
        api = OpenAIBatchAPI(api_key="test-key")
        requests = [
            BatchRequest(custom_id=f"req-{i}", messages=[])
            for i in range(MAX_BATCH_REQUESTS + 1)
        ]
        with pytest.raises(ValueError, match="Maximum"):
            asyncio.run(api.submit_batch(requests))


class TestOpenAIBatchAPIPoll:
    """Test batch status polling."""

    @pytest.mark.asyncio
    async def test_poll_status(self):
        api = OpenAIBatchAPI(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "batch-123",
            "status": "in_progress",
            "request_counts": {"completed": 50, "total": 100},
            "output_file_id": "file-out",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(api, "_get_client", return_value=mock_client):
            status = await api.poll_status("batch-123")
            assert status.batch_id == "batch-123"
            assert status.status == "in_progress"
            assert status.request_counts["completed"] == 50
            assert status.output_file_id == "file-out"

    @pytest.mark.asyncio
    async def test_wait_for_completion(self):
        api = OpenAIBatchAPI(api_key="test-key", poll_interval=0.1)

        # First poll: in_progress, Second poll: completed
        mock_responses = [
            {
                "id": "batch-123",
                "status": "in_progress",
                "request_counts": {},
            },
            {
                "id": "batch-123",
                "status": "completed",
                "request_counts": {"completed": 10},
            },
        ]

        call_count = 0

        async def mock_get(endpoint):
            nonlocal call_count
            response = MagicMock()
            response.json.return_value = mock_responses[min(call_count, len(mock_responses) - 1)]
            response.raise_for_status = MagicMock()
            call_count += 1
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        with patch.object(api, "_get_client", return_value=mock_client):
            status = await api.wait_for_completion("batch-123", poll_interval=0.01, max_poll_time=5)
            assert status.is_complete is True
            assert status.status == "completed"

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        api = OpenAIBatchAPI(api_key="test-key", poll_interval=0.01)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "batch-123",
            "status": "in_progress",
            "request_counts": {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(api, "_get_client", return_value=mock_client):
            with pytest.raises(TimeoutError):
                await api.wait_for_completion("batch-123", poll_interval=0.01, max_poll_time=0.05)


class TestOpenAIBatchAPIDownload:
    """Test result downloading."""

    @pytest.mark.asyncio
    async def test_download_results(self):
        api = OpenAIBatchAPI(api_key="test-key")

        # Mock status response
        mock_status_response = MagicMock()
        mock_status_response.json.return_value = {
            "id": "batch-123",
            "status": "completed",
            "request_counts": {"completed": 1},
            "output_file_id": "file-out-123",
        }
        mock_status_response.raise_for_status = MagicMock()

        # Mock download response
        mock_download_response = MagicMock()
        mock_download_response.text = json.dumps({
            "custom_id": "req-1",
            "response": {
                "body": {
                    "choices": [{"message": {"content": "Hello, world!"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            }
        })
        mock_download_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.side_effect = [mock_status_response, mock_download_response]

        with patch.object(api, "_get_client", return_value=mock_client):
            results = await api.download_results("batch-123")
            assert len(results) == 1
            assert results[0].custom_id == "req-1"
            assert results[0].text == "Hello, world!"
            assert results[0].token_usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_download_results_not_complete(self):
        api = OpenAIBatchAPI(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "batch-123",
            "status": "in_progress",
            "request_counts": {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(api, "_get_client", return_value=mock_client):
            with pytest.raises(ValueError, match="not complete"):
                await api.download_results("batch-123")


class TestCreateBatchRequest:
    """Test convenience function."""

    def test_create_simple(self):
        req = create_batch_request("Hello, world!")
        assert req.messages == [{"role": "user", "content": "Hello, world!"}]
        assert req.model == "gpt-4o"
        assert req.custom_id is not None

    def test_create_with_system_prompt(self):
        req = create_batch_request(
            "Hello!",
            custom_id="my-id",
            system_prompt="You are a helpful assistant.",
        )
        assert req.custom_id == "my-id"
        assert len(req.messages) == 2
        assert req.messages[0]["role"] == "system"
        assert req.messages[0]["content"] == "You are a helpful assistant."
        assert req.messages[1]["role"] == "user"

    def test_create_with_model(self):
        req = create_batch_request("Hello!", model="gpt-4")
        assert req.model == "gpt-4"
