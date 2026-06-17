"""Tests for student inference engine."""

import sys
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import pytest

from dvas.models.base import GenerationResult, GenerationStatus, ModelType

# Patch the dpo_trainer module to avoid datasets dependency
sys.modules['dvas.models.student.dpo_trainer'] = MagicMock()
sys.modules['dvas.models.student.sft_trainer'] = MagicMock()

from dvas.models.student.inference import (
    StudentInferenceEngine,
    StudentTeacherBridge,
    batch_inference,
)


class TestStudentInferenceEngine:
    """Test StudentInferenceEngine implements UnifiedModel."""

    def test_model_type(self):
        """Test model_type property."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            assert engine.model_type == ModelType.STUDENT_LOCAL

    def test_model_version(self):
        """Test model_version property."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path/to/model")
            assert engine.model_version == "model"

    def test_estimate_cost(self):
        """Test cost estimation (should be 0 for local inference)."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            assert engine.estimate_cost() == 0.0
            assert engine.estimate_cost(num_frames=32, prompt_length=1000) == 0.0

    def test_capabilities(self):
        """Test supported capabilities."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            assert engine.supports("video") is True
            assert engine.supports("frames") is True
            assert engine.supports("text") is True
            assert engine.supports("multimodal") is True
            assert engine.supports("audio") is False


class TestStudentTeacherBridge:
    """Test StudentTeacherBridge fallback behavior."""

    @pytest.mark.asyncio
    async def test_bridge_returns_generation_result(self):
        """Test bridge returns GenerationResult."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            bridge = StudentTeacherBridge(engine, fallback_to_teacher=False)

            # Mock the student's generate method as async
            async def mock_generate(**kwargs):
                return GenerationResult(
                    text="student result",
                    model_type=ModelType.STUDENT_LOCAL,
                    status=GenerationStatus.SUCCESS,
                    confidence=0.8,
                )
            engine.generate = mock_generate

            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
            result = await bridge.annotate(frames=frames)

            assert isinstance(result, GenerationResult)
            assert result.text == "student result"
            assert result.model_type == ModelType.STUDENT_LOCAL

    @pytest.mark.asyncio
    async def test_bridge_low_confidence_fallback(self):
        """Test bridge falls back to teacher on low confidence."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            bridge = StudentTeacherBridge(
                engine,
                fallback_to_teacher=True,
                confidence_threshold=0.9,
            )

            # Mock low confidence result
            async def mock_generate(**kwargs):
                return GenerationResult(
                    text="uncertain",
                    model_type=ModelType.STUDENT_LOCAL,
                    status=GenerationStatus.SUCCESS,
                    confidence=0.5,
                )
            engine.generate = mock_generate

            # Mock fallback teacher
            with patch("dvas.models.teacher.gpt4v.GPT4VTeacher") as MockTeacher:
                mock_teacher = MagicMock()
                mock_teacher.model_type = ModelType.TEACHER_GPT4V
                mock_teacher.model_version = "gpt-4o"

                async def mock_annotate(**kwargs):
                    return GenerationResult(
                        text="teacher result",
                        model_type=ModelType.TEACHER_GPT4V,
                        status=GenerationStatus.SUCCESS,
                    )
                mock_teacher.annotate = mock_annotate
                MockTeacher.return_value = mock_teacher

                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
                result = await bridge.annotate(frames=frames)

                assert isinstance(result, GenerationResult)
                assert result.text == "teacher result"
                assert result.model_type == ModelType.TEACHER_GPT4V

    @pytest.mark.asyncio
    async def test_bridge_no_fallback_on_failure(self):
        """Test bridge returns failure when fallback disabled."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            bridge = StudentTeacherBridge(
                engine,
                fallback_to_teacher=False,
                confidence_threshold=0.9,
            )

            # Mock low confidence result
            async def mock_generate(**kwargs):
                return GenerationResult(
                    text="uncertain",
                    model_type=ModelType.STUDENT_LOCAL,
                    status=GenerationStatus.SUCCESS,
                    confidence=0.5,
                )
            engine.generate = mock_generate

            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
            result = await bridge.annotate(frames=frames)

            assert isinstance(result, GenerationResult)
            assert result.status == GenerationStatus.FAILURE
            assert "fallback disabled" in result.error_message.lower()

    def test_bridge_model_type(self):
        """Test bridge model_type."""
        with patch.object(StudentInferenceEngine, '_load_model'):
            engine = StudentInferenceEngine("/fake/path")
            bridge = StudentTeacherBridge(engine)
            assert bridge.model_type == ModelType.STUDENT_EDGE


class TestBatchInference:
    """Test batch_inference function."""

    def test_batch_inference_signature(self):
        """Test batch_inference function exists and has correct signature."""
        import inspect
        sig = inspect.signature(batch_inference)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "video_paths" in params
        assert "prompts" in params
        assert "batch_size" in params
        assert "max_new_tokens" in params
