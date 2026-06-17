"""Tests for models/base.py - GenerationResult and UnifiedModel."""

import pytest
import numpy as np

from dvas.models.base import (
    GenerationResult,
    GenerationStatus,
    ModelType,
    UnifiedModel,
)


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_default_creation(self):
        """Test GenerationResult with default values."""
        result = GenerationResult()
        assert result.text == ""
        assert result.structured_data is None
        assert result.model_type == ModelType.MOCK
        assert result.model_version == "unknown"
        assert result.status == GenerationStatus.SUCCESS
        assert result.confidence == 1.0
        assert result.latency_ms == 0.0
        assert result.token_usage == {}
        assert result.cost_usd == 0.0
        assert result.error_message is None
        assert result.fallback_from is None
        assert result.metadata == {}

    def test_full_creation(self):
        """Test GenerationResult with all fields."""
        result = GenerationResult(
            text="A person cooking",
            structured_data={"scene": "kitchen"},
            model_type=ModelType.TEACHER_GPT4V,
            model_version="gpt-4o",
            status=GenerationStatus.SUCCESS,
            confidence=0.95,
            latency_ms=1234.5,
            token_usage={"input": 100, "output": 50},
            cost_usd=0.05,
            error_message=None,
            fallback_from=None,
            metadata={"finish_reason": "stop"},
        )
        assert result.text == "A person cooking"
        assert result.model_type == ModelType.TEACHER_GPT4V
        assert result.model_version == "gpt-4o"
        assert result.confidence == 0.95
        assert result.latency_ms == 1234.5

    def test_is_success(self):
        """Test is_success method."""
        success = GenerationResult(status=GenerationStatus.SUCCESS)
        failure = GenerationResult(status=GenerationStatus.FAILURE)
        fallback = GenerationResult(
            status=GenerationStatus.FALLBACK,
            fallback_from=ModelType.TEACHER_GPT4V,
        )

        assert success.is_success() is True
        assert failure.is_success() is False
        assert fallback.is_success() is False

    def test_is_failure(self):
        """Test is_failure method."""
        success = GenerationResult(status=GenerationStatus.SUCCESS)
        failure = GenerationResult(status=GenerationStatus.FAILURE)
        timeout = GenerationResult(status=GenerationStatus.TIMEOUT)

        assert success.is_failure() is False
        assert failure.is_failure() is True
        # TIMEOUT is its own status, not FAILURE
        assert timeout.is_failure() is False

    def test_is_fallback(self):
        """Test is_fallback method."""
        normal = GenerationResult()
        fallback = GenerationResult(
            fallback_from=ModelType.TEACHER_GPT4V,
        )

        assert normal.is_fallback() is False
        assert fallback.is_fallback() is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = GenerationResult(
            text="test",
            model_type=ModelType.TEACHER_CLAUDE,
            status=GenerationStatus.SUCCESS,
            fallback_from=ModelType.TEACHER_GPT4V,
        )
        d = result.to_dict()
        assert d["text"] == "test"
        assert d["model_type"] == "claude"
        assert d["status"] == "success"
        assert d["fallback_from"] == "gpt-4v"
        assert "structured_data" in d

    def test_to_dict_no_fallback(self):
        """Test to_dict when fallback_from is None."""
        result = GenerationResult()
        d = result.to_dict()
        assert d["fallback_from"] is None

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "text": "test",
            "structured_data": None,
            "model_type": "gpt-4v",
            "model_version": "gpt-4o",
            "status": "success",
            "confidence": 0.9,
            "latency_ms": 100.0,
            "token_usage": {},
            "cost_usd": 0.0,
            "error_message": None,
            "fallback_from": None,
            "metadata": {},
        }
        result = GenerationResult.from_dict(data)
        assert result.text == "test"
        assert result.model_type == ModelType.TEACHER_GPT4V
        assert result.status == GenerationStatus.SUCCESS
        assert result.confidence == 0.9

    def test_from_dict_with_fallback(self):
        """Test from_dict with fallback_from set."""
        data = {
            "text": "fallback text",
            "model_type": "claude",
            "status": "fallback",
            "fallback_from": "gpt-4v",
        }
        result = GenerationResult.from_dict(data)
        assert result.fallback_from == ModelType.TEACHER_GPT4V
        assert result.status == GenerationStatus.FALLBACK

    def test_failure_factory(self):
        """Test GenerationResult.failure factory method."""
        result = GenerationResult.failure(
            error_message="API timeout",
            model_type=ModelType.TEACHER_GPT4V,
            model_version="gpt-4o",
        )
        assert result.status == GenerationStatus.FAILURE
        assert result.error_message == "API timeout"
        assert result.model_type == ModelType.TEACHER_GPT4V
        assert result.model_version == "gpt-4o"
        assert result.confidence == 0.0
        assert result.text == ""

    def test_fallback_factory(self):
        """Test GenerationResult.fallback factory method."""
        result = GenerationResult.fallback(
            text="fallback result",
            fallback_from=ModelType.TEACHER_GPT4V,
            model_type=ModelType.TEACHER_CLAUDE,
        )
        assert result.status == GenerationStatus.FALLBACK
        assert result.text == "fallback result"
        assert result.fallback_from == ModelType.TEACHER_GPT4V
        assert result.model_type == ModelType.TEACHER_CLAUDE

    def test_roundtrip_serialization(self):
        """Test roundtrip to_dict -> from_dict."""
        original = GenerationResult(
            text="test text",
            structured_data={"key": "value"},
            model_type=ModelType.TEACHER_TOGETHER,
            model_version="qwen2-vl-7b",
            status=GenerationStatus.SUCCESS,
            confidence=0.85,
            latency_ms=500.0,
            token_usage={"input": 10, "output": 20},
            cost_usd=0.01,
            metadata={"foo": "bar"},
        )
        d = original.to_dict()
        restored = GenerationResult.from_dict(d)
        assert restored.text == original.text
        assert restored.model_type == original.model_type
        assert restored.confidence == original.confidence
        assert restored.token_usage == original.token_usage


class TestModelType:
    """Test ModelType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ModelType.TEACHER_GPT4V.value == "gpt-4v"
        assert ModelType.TEACHER_CLAUDE.value == "claude"
        assert ModelType.TEACHER_TOGETHER.value == "together"
        assert ModelType.STUDENT_LOCAL.value == "student-local"
        assert ModelType.STUDENT_EDGE.value == "student-edge"
        assert ModelType.MOCK.value == "mock"

    def test_enum_from_string(self):
        """Test creating ModelType from string."""
        assert ModelType("gpt-4v") == ModelType.TEACHER_GPT4V
        assert ModelType("mock") == ModelType.MOCK


class TestGenerationStatus:
    """Test GenerationStatus enum."""

    def test_enum_values(self):
        """Test all enum values."""
        assert GenerationStatus.SUCCESS.value == "success"
        assert GenerationStatus.FAILURE.value == "failure"
        assert GenerationStatus.FALLBACK.value == "fallback"
        assert GenerationStatus.TIMEOUT.value == "timeout"
        assert GenerationStatus.RATE_LIMITED.value == "rate_limited"


class TestUnifiedModel:
    """Test UnifiedModel abstract base class."""

    def test_unified_model_is_abstract(self):
        """Test that UnifiedModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            UnifiedModel()

    def test_unified_model_requires_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteModel(UnifiedModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_supports_method(self):
        """Test the supports capability check."""
        class MockModel(UnifiedModel):
            @property
            def model_type(self):
                return ModelType.MOCK

            @property
            def model_version(self):
                return "mock-v1"

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
                return ["video", "frames", "text"]

        model = MockModel()
        assert model.supports("video") is True
        assert model.supports("frames") is True
        assert model.supports("text") is True
        assert model.supports("multimodal") is False

    def test_estimate_cost_default(self):
        """Test default estimate_cost returns 0.0."""
        class MockModel(UnifiedModel):
            @property
            def model_type(self):
                return ModelType.MOCK

            @property
            def model_version(self):
                return "mock-v1"

            async def generate(self, **kwargs):
                return GenerationResult()

            async def generate_batch(self, items, **kwargs):
                return []

            def _capabilities(self):
                return []

        model = MockModel()
        assert model.estimate_cost() == 0.0
        assert model.estimate_cost(num_frames=32, prompt_length=1000) == 0.0
