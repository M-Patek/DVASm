"""Tests for teacher models (GPT4V, Claude, Together) returning GenerationResult."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from dvas.models.base import GenerationResult, GenerationStatus, ModelType


class TestGPT4VTeacher:
    """Test GPT4VTeacher returns GenerationResult."""

    @pytest.fixture
    def mock_gpt4v(self):
        """Create a GPT4VTeacher with mocked client."""
        with patch("dvas.models.teacher.gpt4v.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            from dvas.models.teacher.gpt4v import GPT4VTeacher
            teacher = GPT4VTeacher(api_key="test-key")
            teacher._client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_gpt4v):
        """Test annotate returns GenerationResult."""
        teacher, mock_client = mock_gpt4v

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
        assert result.model_type == ModelType.TEACHER_GPT4V
        assert result.model_version == "gpt-4o"
        assert result.status == GenerationStatus.SUCCESS
        assert result.is_success() is True
        assert result.latency_ms >= 0
        assert result.token_usage == {"input": 100, "output": 50}
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_annotate_no_frames_error(self, mock_gpt4v):
        """Test annotate with no frames returns failure."""
        teacher, _ = mock_gpt4v
        result = await teacher.annotate(frames=None)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert result.is_failure() is True
        assert "Must provide either video_path or frames" in result.error_message

    @pytest.mark.asyncio
    async def test_annotate_api_error(self, mock_gpt4v):
        """Test annotate handles API errors gracefully."""
        teacher, mock_client = mock_gpt4v
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = await teacher.annotate(frames=frames)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert "API Error" in result.error_message
        assert result.latency_ms >= 0

    def test_model_type(self, mock_gpt4v):
        """Test model_type property."""
        teacher, _ = mock_gpt4v
        assert teacher.model_type == ModelType.TEACHER_GPT4V

    def test_model_version(self, mock_gpt4v):
        """Test model_version property."""
        teacher, _ = mock_gpt4v
        assert teacher.model_version == "gpt-4o"

    def test_estimate_cost(self, mock_gpt4v):
        """Test cost estimation."""
        teacher, _ = mock_gpt4v
        cost = teacher.estimate_cost(num_frames=16, prompt_length=500)
        assert cost > 0

    def test_capabilities(self, mock_gpt4v):
        """Test supported capabilities."""
        teacher, _ = mock_gpt4v
        assert teacher.supports("video") is True
        assert teacher.supports("frames") is True
        assert teacher.supports("text") is True
        assert teacher.supports("multimodal") is True
        assert teacher.supports("audio") is False

    def test_max_frames(self, mock_gpt4v):
        """Test max frames configuration."""
        teacher, _ = mock_gpt4v
        assert teacher.max_frames == 32


class TestClaudeTeacher:
    """Test ClaudeTeacher returns GenerationResult."""

    @pytest.fixture
    def mock_claude(self):
        """Create a ClaudeTeacher with mocked client."""
        with patch("dvas.models.teacher.claude.AsyncAnthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            from dvas.models.teacher.claude import ClaudeTeacher
            teacher = ClaudeTeacher(api_key="test-key")
            teacher._client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_claude):
        """Test annotate returns GenerationResult."""
        teacher, mock_client = mock_claude

        # Mock the API response
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

    @pytest.mark.asyncio
    async def test_annotate_no_frames_error(self, mock_claude):
        """Test annotate with no frames returns failure."""
        teacher, _ = mock_claude
        result = await teacher.annotate(frames=None)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert "Claude requires pre-extracted frames" in result.error_message

    def test_model_type(self, mock_claude):
        """Test model_type property."""
        teacher, _ = mock_claude
        assert teacher.model_type == ModelType.TEACHER_CLAUDE

    def test_max_frames(self, mock_claude):
        """Test max frames configuration."""
        teacher, _ = mock_claude
        assert teacher.max_frames == 20


class TestTogetherTeacher:
    """Test TogetherTeacher returns GenerationResult."""

    @pytest.fixture
    def mock_together(self):
        """Create a TogetherTeacher with mocked client."""
        with patch("dvas.models.teacher.together.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            from dvas.models.teacher.together import TogetherTeacher
            teacher = TogetherTeacher(api_key="test-key")
            teacher._client = mock_client
            yield teacher, mock_client

    @pytest.mark.asyncio
    async def test_annotate_returns_generation_result(self, mock_together):
        """Test annotate returns GenerationResult."""
        teacher, mock_client = mock_together

        # Mock the API response
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

    @pytest.mark.asyncio
    async def test_annotate_no_frames_error(self, mock_together):
        """Test annotate with no frames returns failure."""
        teacher, _ = mock_together
        result = await teacher.annotate(frames=None)

        assert isinstance(result, GenerationResult)
        assert result.status == GenerationStatus.FAILURE
        assert "Together API requires pre-extracted frames" in result.error_message

    def test_model_type(self, mock_together):
        """Test model_type property."""
        teacher, _ = mock_together
        assert teacher.model_type == ModelType.TEACHER_TOGETHER

    def test_model_mapping(self, mock_together):
        """Test model name mapping."""
        teacher, _ = mock_together
        assert "qwen" in teacher.together_model.lower()


class TestTeacherModelBase:
    """Test TeacherModel base class."""

    def test_teacher_is_unified_model(self):
        """Test TeacherModel extends UnifiedModel."""
        from dvas.models.teacher.base import TeacherModel
        from dvas.models.base import UnifiedModel

        assert issubclass(TeacherModel, UnifiedModel)

    def test_teacher_has_model_name(self):
        """Test TeacherModel stores model_name."""
        from dvas.models.teacher.base import TeacherModel

        class DummyTeacher(TeacherModel):
            @property
            def model_type(self):
                return ModelType.MOCK

            async def annotate(self, **kwargs):
                return GenerationResult()

            async def annotate_batch(self, items, **kwargs):
                return []

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
                return []

        teacher = DummyTeacher(model_name="test-model")
        assert teacher.model_name == "test-model"
        assert teacher.model_version == "test-model"

    def test_encode_image(self):
        """Test image encoding."""
        from dvas.models.teacher.base import TeacherModel

        class MockTeacher(TeacherModel):
            @property
            def model_type(self):
                return ModelType.MOCK

            async def annotate(self, **kwargs):
                return GenerationResult()

            async def annotate_batch(self, items, **kwargs):
                return []

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
                return []

        teacher = MockTeacher("test-model")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded = teacher._encode_image(image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_encode_frames(self):
        """Test batch frame encoding."""
        from dvas.models.teacher.base import TeacherModel

        class MockTeacher(TeacherModel):
            @property
            def model_type(self):
                return ModelType.MOCK

            async def annotate(self, **kwargs):
                return GenerationResult()

            async def annotate_batch(self, items, **kwargs):
                return []

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
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
            @property
            def model_type(self):
                return ModelType.MOCK

            async def annotate(self, **kwargs):
                return GenerationResult()

            async def annotate_batch(self, items, **kwargs):
                return []

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
                return []

        teacher = MockTeacher("test-model")
        prompt = teacher._get_default_prompt("caption")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        fg_prompt = teacher._get_default_prompt("fine_grained")
        assert "robotic manipulation" in fg_prompt.lower()
