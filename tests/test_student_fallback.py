"""Tests for low-confidence fallback."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.student.fallback import (
    AdaptiveFallback,
    FallbackStats,
    LowConfidenceFallback,
    create_fallback_router,
)


class TestFallbackStats:
    """Test FallbackStats dataclass."""

    def test_initial_state(self):
        """Test initial state of stats."""
        stats = FallbackStats()
        assert stats.total_requests == 0
        assert stats.student_success == 0
        assert stats.fallback_rate == 0.0
        assert stats.student_success_rate == 0.0

    def test_rates_calculation(self):
        """Test rate calculations."""
        stats = FallbackStats(
            total_requests=100,
            student_success=70,
            student_low_confidence=20,
            student_failed=10,
            fallback_success=25,
            fallback_failed=5,
        )

        assert stats.fallback_rate == 0.3  # 30/100
        assert stats.student_success_rate == 0.7  # 70/100
        assert stats.fallback_success_rate == 25 / 30

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = FallbackStats(total_requests=10, student_success=7)
        data = stats.to_dict()

        assert data["total_requests"] == 10
        assert data["student_success"] == 7
        assert "fallback_rate" in data
        assert "student_success_rate" in data


class TestLowConfidenceFallback:
    """Test LowConfidenceFallback."""

    @pytest.fixture
    def mock_student(self):
        """Create mock student engine."""
        student = MagicMock()
        student.model_type = ModelType.STUDENT_LOCAL
        student.model_version = "test-v1"
        return student

    @pytest.fixture
    def mock_teacher(self):
        """Create mock teacher."""
        teacher = MagicMock()
        teacher.model_type = ModelType.TEACHER_CLAUDE
        teacher.model_version = "claude-3-5"
        return teacher

    @pytest.mark.asyncio
    async def test_high_confidence_uses_student(self, mock_student):
        """Test high confidence prediction uses student."""
        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            confidence_threshold=0.7,
        )

        # Mock successful student prediction
        mock_student.generate = AsyncMock(
            return_value=GenerationResult(
                text="student prediction",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,
            )
        )

        result = await fallback.generate(video_path=Path("test.mp4"))

        assert result.text == "student prediction"
        assert result.model_type == ModelType.STUDENT_LOCAL
        assert fallback.stats.student_success == 1
        assert fallback.stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_low_confidence_fallback(self, mock_student, mock_teacher):
        """Test low confidence triggers fallback."""
        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            teacher_model=mock_teacher,
            confidence_threshold=0.8,
            fallback_to_teacher=True,
        )

        # Mock low confidence student prediction
        mock_student.generate = AsyncMock(
            return_value=GenerationResult(
                text="uncertain prediction",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,  # Below threshold
            )
        )

        # Mock successful teacher prediction
        mock_teacher.annotate = AsyncMock(
            return_value=GenerationResult(
                text="teacher prediction",
                model_type=ModelType.TEACHER_CLAUDE,
                status=GenerationStatus.SUCCESS,
                confidence=0.95,
            )
        )

        result = await fallback.generate(video_path=Path("test.mp4"))

        assert result.text == "teacher prediction"
        assert result.metadata["fallback"] is True
        assert result.metadata["fallback_reason"] == "low_confidence"
        assert fallback.stats.fallback_success == 1

    @pytest.mark.asyncio
    async def test_student_failure_fallback(self, mock_student, mock_teacher):
        """Test student failure triggers fallback."""
        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            teacher_model=mock_teacher,
            fallback_to_teacher=True,
        )

        # Mock student failure
        mock_student.generate = AsyncMock(side_effect=Exception("CUDA out of memory"))

        # Mock successful teacher
        mock_teacher.annotate = AsyncMock(
            return_value=GenerationResult(
                text="teacher prediction",
                model_type=ModelType.TEACHER_CLAUDE,
                status=GenerationStatus.SUCCESS,
            )
        )

        result = await fallback.generate(video_path=Path("test.mp4"))

        assert result.text == "teacher prediction"
        assert fallback.stats.student_failed == 1

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, mock_student):
        """Test fallback disabled returns failure."""
        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            fallback_to_teacher=False,
            confidence_threshold=0.8,
        )

        # Mock low confidence prediction
        mock_student.generate = AsyncMock(
            return_value=GenerationResult(
                text="uncertain",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,
            )
        )

        result = await fallback.generate(video_path=Path("test.mp4"))

        assert result.status == GenerationStatus.FAILURE
        assert "fallback disabled" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_calibrated_confidence(self, mock_student):
        """Test calibrated confidence threshold."""
        from dvas.models.student.calibration import ConfidenceCalibrator

        calibrator = MagicMock(spec=ConfidenceCalibrator)
        calibrator.calibrate = MagicMock(
            return_value=GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.6,  # Calibrated lower
            )
        )

        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            use_calibrated_confidence=True,
            calibrator=calibrator,
            confidence_threshold=0.7,
        )

        # Mock high raw confidence, but calibration reduces it
        mock_student.generate = AsyncMock(
            return_value=GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,  # Raw confidence
            )
        )

        # Calibrated confidence (0.6) < threshold (0.7) should fallback
        # But we'll mock calibrator to return success
        calibrator.calibrate.return_value = GenerationResult(
            text="calibrated",
            model_type=ModelType.STUDENT_LOCAL,
            status=GenerationStatus.SUCCESS,
            confidence=0.8,  # Above threshold after calibration
        )

        await fallback.generate(video_path=Path("test.mp4"))

        calibrator.calibrate.assert_called_once()

    def test_get_and_reset_stats(self, mock_student):
        """Test stats management."""
        fallback = LowConfidenceFallback(student_engine=mock_student)

        # Initial state
        stats = fallback.get_stats()
        assert stats.total_requests == 0

        # Reset
        stats.total_requests = 10
        fallback.reset_stats()
        stats = fallback.get_stats()
        assert stats.total_requests == 0

    def test_update_threshold(self, mock_student):
        """Test threshold update."""
        fallback = LowConfidenceFallback(
            student_engine=mock_student,
            confidence_threshold=0.7,
        )

        fallback.update_threshold(0.85)
        assert fallback.confidence_threshold == 0.85

        # Clamping
        fallback.update_threshold(1.5)
        assert fallback.confidence_threshold == 1.0

        fallback.update_threshold(-0.5)
        assert fallback.confidence_threshold == 0.0


class TestAdaptiveFallback:
    """Test AdaptiveFallback."""

    @pytest.fixture
    def mock_student(self):
        """Create mock student."""
        student = MagicMock()
        student.generate = AsyncMock(
            return_value=GenerationResult(
                text="student",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.75,
            )
        )
        return student

    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment(self, mock_student):
        """Test threshold adapts based on performance."""
        fallback = AdaptiveFallback(
            student_engine=mock_student,
            initial_threshold=0.5,
            target_fallback_rate=0.2,
            adjustment_rate=0.05,
        )

        # Run some predictions to trigger adaptation
        for _ in range(25):
            await fallback.generate(video_path=Path("test.mp4"))

        # Threshold should have been adjusted
        assert fallback.confidence_threshold != 0.5

    @pytest.mark.asyncio
    async def test_target_accuracy_tracking(self, mock_student):
        """Test adaptation for target accuracy."""
        # Make student fail sometimes
        responses = [
            GenerationResult(
                text="success",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS if i % 2 == 0 else GenerationStatus.FAILURE,
                confidence=0.6,
            )
            for i in range(25)
        ]
        mock_student.generate = AsyncMock(side_effect=responses)

        fallback = AdaptiveFallback(
            student_engine=mock_student,
            initial_threshold=0.5,
            target_accuracy=0.9,
            adjustment_rate=0.05,
        )

        # Run predictions
        for _ in range(25):
            await fallback.generate(video_path=Path("test.mp4"))

        # Recent results should be tracked
        assert len(fallback._recent_results) > 0


class TestCreateFallbackRouter:
    """Test factory function."""

    @patch("dvas.models.student.fallback.StudentInferenceEngine")
    def test_create_basic_fallback(self, mock_engine_class):
        """Test creating basic fallback router."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        router = create_fallback_router(
            student_path="/fake/model",
            confidence_threshold=0.7,
            adaptive=False,
        )

        assert isinstance(router, LowConfidenceFallback)
        assert router.confidence_threshold == 0.7
        assert not isinstance(router, AdaptiveFallback)

    @patch("dvas.models.student.fallback.StudentInferenceEngine")
    def test_create_adaptive_fallback(self, mock_engine_class):
        """Test creating adaptive fallback router."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        router = create_fallback_router(
            student_path="/fake/model",
            confidence_threshold=0.7,
            adaptive=True,
            target_fallback_rate=0.2,
        )

        assert isinstance(router, AdaptiveFallback)
        assert router.target_fallback_rate == 0.2
