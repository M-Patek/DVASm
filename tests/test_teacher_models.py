"""Tests for teacher models returning GenerationResult."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.teacher.base import TeacherModel


class TestTeacherModel:
    """Test TeacherModel returns GenerationResult."""

    @pytest.fixture
    def mock_teacher(self):
        """Create a TeacherModel with mocked OpenAI client."""
        from unittest.mock import AsyncMock

        teacher = TeacherModel(model_name="gpt-5.5", api_key="test-key")

        # Create mock client
        mock_client = AsyncMock()
        teacher._openai_client = mock_client
        yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_teacher):
        """Test annotate returns GenerationResult."""
        teacher, mock_client = mock_teacher

        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test annotation"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames)

        assert isinstance(result, GenerationResult)
        assert result.text == "Test annotation"
        assert result.model_type == ModelType.TEACHER_GPT55
        assert result.model_version == "gpt-5.5"
        assert result.status == GenerationStatus.SUCCESS
        assert result.is_success() is True
        assert result.latency_ms >= 0
        assert result.token_usage == {"input": 100, "output": 50}
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_annotate_no_frames_error(self, mock_teacher):
        """Test annotate with no frames returns failure."""
        teacher, _ = mock_teacher
        result = await teacher.annotate(frames=None)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert result.is_failure() is True
        assert "Must provide either video_path or frames" in result.error_message

    @pytest.mark.asyncio
    async def test_annotate_api_error(self, mock_teacher):
        """Test annotate handles API errors gracefully."""
        teacher, mock_client = mock_teacher
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert "API Error" in result.error_message
        assert result.latency_ms >= 0

    def test_model_type(self, mock_teacher):
        """Test model_type property."""
        teacher, _ = mock_teacher
        assert teacher.model_type == ModelType.TEACHER_GPT55

    def test_model_version(self, mock_teacher):
        """Test model_version property."""
        teacher, _ = mock_teacher
        assert teacher.model_version == "gpt-5.5"

    def test_estimate_cost(self, mock_teacher):
        """Test cost estimation."""
        teacher, _ = mock_teacher
        cost = teacher.estimate_cost(num_frames=16, prompt_length=500)
        assert cost > 0

    def test_capabilities(self, mock_teacher):
        """Test supported capabilities."""
        teacher, _ = mock_teacher
        assert teacher.supports("video") is True
        assert teacher.supports("frames") is True
        assert teacher.supports("text") is True
        assert teacher.supports("multimodal") is True
        assert teacher.supports("audio") is False

    def test_max_frames(self, mock_teacher):
        """Test max frames configuration."""
        teacher, _ = mock_teacher
        assert teacher.max_frames == 32


class TestTeacherModelClaude:
    """Test TeacherModel with Claude backend."""

    @pytest.fixture
    def mock_claude_teacher(self):
        """Create a TeacherModel with mocked Anthropic client."""
        from unittest.mock import AsyncMock

        teacher = TeacherModel(model_name="claude-opus-4-8", api_key="test-key")

        # Create mock client
        mock_client = AsyncMock()
        teacher._anthropic_client = mock_client
        yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_claude_teacher):
        """Test annotate returns GenerationResult with Claude."""
        teacher, mock_client = mock_claude_teacher

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Claude annotation"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames)

        assert isinstance(result, GenerationResult)
        assert result.text == "Claude annotation"
        assert result.model_type == ModelType.TEACHER_CLAUDE
        assert result.status == GenerationStatus.SUCCESS

    def test_max_frames_claude(self, mock_claude_teacher):
        """Test max frames for Claude models."""
        teacher, _ = mock_claude_teacher
        assert teacher.max_frames == 20


class TestTeacherModelTogether:
    """Test TeacherModel with Together AI backend."""

    @pytest.fixture
    def mock_together_teacher(self):
        """Create a TeacherModel with mocked Together client."""
        from unittest.mock import AsyncMock

        teacher = TeacherModel(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct")

        # Create mock client (Together uses OpenAI client)
        mock_client = AsyncMock()
        teacher._openai_client = mock_client
        yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_together_teacher):
        """Test annotate returns GenerationResult with Together."""
        teacher, mock_client = mock_together_teacher

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Together annotation"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 50,
            "completion_tokens": 25,
        }

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames)

        assert isinstance(result, GenerationResult)
        assert result.text == "Together annotation"
        assert result.model_type == ModelType.TEACHER_TOGETHER
        assert result.status == GenerationStatus.SUCCESS

    def test_max_frames_together(self, mock_together_teacher):
        """Test max frames for Together models."""
        teacher, _ = mock_together_teacher
        assert teacher.max_frames == 16


class TestTeacherModelAutoDetection:
    """Test provider auto-detection."""

    def test_detect_openai(self):
        teacher = TeacherModel(model_name="gpt-5.5")
        assert teacher._provider == "openai"
        assert teacher.model_type == ModelType.TEACHER_GPT55

    def test_detect_claude(self):
        teacher = TeacherModel(model_name="claude-opus-4-8")
        assert teacher._provider == "anthropic"
        assert teacher.model_type == ModelType.TEACHER_CLAUDE

    def test_detect_together(self):
        teacher = TeacherModel(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct")
        assert teacher._provider == "together"
        assert teacher.model_type == ModelType.TEACHER_TOGETHER

    def test_detect_qwen_together(self):
        teacher = TeacherModel(model_name="Qwen/Qwen2-VL-7B-Instruct")
        assert teacher._provider == "together"

    def test_default_to_openai(self):
        # Unknown models default to OpenAI
        teacher = TeacherModel(model_name="some-unknown-model")
        assert teacher._provider == "openai"


class TestTeacherModelBaseFeatures:
    """Test TeacherModel base class features."""

    def test_teacher_is_unified_model(self):
        """Test TeacherModel extends UnifiedModel."""
        from dvas.models.base import UnifiedModel
        assert issubclass(TeacherModel, UnifiedModel)

    def test_teacher_has_model_name(self):
        """Test TeacherModel stores model_name."""
        teacher = TeacherModel(model_name="gpt-5.5")
        assert teacher.model_name == "gpt-5.5"
        assert teacher.model_version == "gpt-5.5"

    def test_encode_image(self):
        """Test image encoding."""
        teacher = TeacherModel(model_name="gpt-5.5")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded = teacher._encode_image(image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    @pytest.mark.asyncio
    async def test_encode_frames(self):
        """Test batch frame encoding."""
        teacher = TeacherModel(model_name="gpt-5.5")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        encoded = await teacher._encode_frames(frames)
        assert len(encoded) == 3
        assert all(isinstance(e, str) for e in encoded)

    def test_get_default_prompt(self):
        """Test prompt template retrieval."""
        teacher = TeacherModel(model_name="gpt-5.5")
        prompt = teacher._get_default_prompt("caption")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        fg_prompt = teacher._get_default_prompt("fine_grained")
        assert "robotic manipulation" in fg_prompt.lower()
