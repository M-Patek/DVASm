"""Tests for teacher models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


class TestTeacherModelBase:
    """Test base teacher model functionality."""

    def test_encode_image(self):
        """Test image encoding."""
        from dvas.models.teacher.base import TeacherModel

        class MockTeacher(TeacherModel):
            async def annotate(self, **kwargs):
                return {}

            async def annotate_batch(self, **kwargs):
                return []

        teacher = MockTeacher("test-model")
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded = teacher._encode_image(image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_encode_frames(self):
        """Test batch frame encoding."""
        from dvas.models.teacher.base import TeacherModel

        class MockTeacher(TeacherModel):
            async def annotate(self, **kwargs):
                return {}

            async def annotate_batch(self, **kwargs):
                return []

        teacher = MockTeacher("test-model")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        encoded = teacher._encode_frames(frames)
        assert len(encoded) == 3
        assert all(isinstance(e, str) for e in encoded)

    def test_get_default_prompt(self):
        """Test prompt template retrieval."""
        from dvas.models.teacher.base import TeacherModel

        class MockTeacher(TeacherModel):
            async def annotate(self, **kwargs):
                return {}

            async def annotate_batch(self, **kwargs):
                return []

        teacher = MockTeacher("test-model")
        prompt = teacher._get_default_prompt("caption")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test fine_grained prompt
        fg_prompt = teacher._get_default_prompt("fine_grained")
        assert "robotic manipulation" in fg_prompt.lower()


class TestGPT4VTeacher:
    """Test GPT-4V teacher."""

    @pytest.mark.asyncio
    async def test_annotate_with_mock(self):
        """Test annotation with mocked API."""
        from dvas.models.teacher.gpt4v import GPT4VTeacher

        teacher = GPT4VTeacher(api_key="test-key")

        # Mock the client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test annotation"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 50}

        teacher.client.chat.completions.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames, prompt="Describe this video")

        assert result["text"] == "Test annotation"
        assert result["model"] == "gpt-4o"
        assert "usage" in result

        await teacher.close()

    def test_max_frames(self):
        """Test max frames configuration."""
        from dvas.models.teacher.gpt4v import GPT4VTeacher

        teacher = GPT4VTeacher()
        assert teacher.max_frames == 32


class TestClaudeTeacher:
    """Test Claude teacher."""

    @pytest.mark.asyncio
    async def test_annotate_with_mock(self):
        """Test annotation with mocked API."""
        from dvas.models.teacher.claude import ClaudeTeacher

        teacher = ClaudeTeacher(api_key="test-key")

        # Mock the client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test annotation from Claude"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.stop_reason = "end_turn"

        teacher.client.messages.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames, prompt="Describe this video")

        assert result["text"] == "Test annotation from Claude"
        assert result["model"] == "claude-3-sonnet-20240229"

        await teacher.close()

    def test_max_frames(self):
        """Test max frames configuration."""
        from dvas.models.teacher.claude import ClaudeTeacher

        teacher = ClaudeTeacher()
        assert teacher.max_frames == 20


class TestTogetherTeacher:
    """Test Together.ai teacher."""

    @pytest.mark.asyncio
    async def test_annotate_with_mock(self):
        """Test annotation with mocked API."""
        from dvas.models.teacher.together import TogetherTeacher

        teacher = TogetherTeacher(api_key="test-key")

        # Mock the client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test annotation from Together"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 50}

        teacher.client.chat.completions.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames, prompt="Describe this video")

        assert result["text"] == "Test annotation from Together"
        assert "Qwen" in result["model"]

        await teacher.close()

    def test_model_mapping(self):
        """Test model name mapping."""
        from dvas.models.teacher.together import TogetherTeacher

        teacher = TogetherTeacher(model_name="qwen2-vl-7b")
        assert teacher.together_model == "Qwen/Qwen2-VL-7B-Instruct"

        teacher2 = TogetherTeacher(model_name="llama-3-vision")
        assert teacher2.together_model == "meta-llama/Llama-3.2-11B-Vision-Instruct"


class TestConnectionPooling:
    """Test HTTP connection pooling."""

    def test_gpt4v_has_http_client(self):
        """Test GPT-4V has HTTP client."""
        from dvas.models.teacher.gpt4v import GPT4VTeacher

        teacher = GPT4VTeacher()
        assert teacher._http_client is not None

    def test_claude_has_http_client(self):
        """Test Claude has HTTP client."""
        from dvas.models.teacher.claude import ClaudeTeacher

        teacher = ClaudeTeacher()
        assert teacher._http_client is not None

    def test_together_has_http_client(self):
        """Test Together has HTTP client."""
        from dvas.models.teacher.together import TogetherTeacher

        teacher = TogetherTeacher()
        assert teacher._http_client is not None
