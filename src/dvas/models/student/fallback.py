"""Low-confidence fallback to teacher model.

Routes requests to teacher model when student confidence is below threshold,
enabling graceful degradation and cost-quality tradeoffs.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.student.calibration import ConfidenceCalibrator
from dvas.models.student.inference import StudentInferenceEngine
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FallbackStats:
    """Statistics for fallback behavior."""

    total_requests: int = 0
    student_success: int = 0
    student_low_confidence: int = 0
    student_failed: int = 0
    fallback_success: int = 0
    fallback_failed: int = 0

    @property
    def fallback_rate(self) -> float:
        """Percentage of requests that fell back to teacher."""
        if self.total_requests == 0:
            return 0.0
        return (self.student_low_confidence + self.student_failed) / self.total_requests

    @property
    def student_success_rate(self) -> float:
        """Percentage of requests handled successfully by student."""
        if self.total_requests == 0:
            return 0.0
        return self.student_success / self.total_requests

    @property
    def fallback_success_rate(self) -> float:
        """Percentage of fallbacks that succeeded."""
        total_fallbacks = self.student_low_confidence + self.student_failed
        if total_fallbacks == 0:
            return 0.0
        return self.fallback_success / total_fallbacks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "student_success": self.student_success,
            "student_low_confidence": self.student_low_confidence,
            "student_failed": self.student_failed,
            "fallback_success": self.fallback_success,
            "fallback_failed": self.fallback_failed,
            "fallback_rate": self.fallback_rate,
            "student_success_rate": self.student_success_rate,
            "fallback_success_rate": self.fallback_success_rate,
        }


class LowConfidenceFallback:
    """Fallback router for low-confidence student predictions.

    Automatically routes to teacher model when:
    1. Student confidence is below threshold
    2. Student inference fails
    3. Student prediction is empty/invalid

    Supports calibrated confidence thresholds for better accuracy.
    """

    def __init__(
        self,
        student_engine: StudentInferenceEngine,
        teacher_model=None,  # Lazy loaded if not provided
        confidence_threshold: float = 0.7,
        fallback_to_teacher: bool = True,
        use_calibrated_confidence: bool = False,
        calibrator: Optional[ConfidenceCalibrator] = None,
        max_retries: int = 1,
    ):
        self.student = student_engine
        self._teacher = teacher_model
        self.confidence_threshold = confidence_threshold
        self.fallback_to_teacher = fallback_to_teacher
        self.use_calibrated_confidence = use_calibrated_confidence
        self.calibrator = calibrator
        self.max_retries = max_retries

        self.stats = FallbackStats()

    @property
    def teacher(self):
        """Lazy load teacher model."""
        if self._teacher is None:
            from dvas.models.teacher import TeacherModel

            self._teacher = TeacherModel()
            logger.info("Lazy-loaded teacher model for fallback")
        return self._teacher

    async def generate(
        self,
        frames: Optional[List] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """Generate with automatic fallback on low confidence.

        Args:
            frames: Video frames
            video_path: Path to video file
            prompt: Generation prompt
            task: Task type
            **kwargs: Additional arguments

        Returns:
            GenerationResult (from student or teacher)
        """
        self.stats.total_requests += 1

        # Try student model first
        student_result = None
        try:
            student_result = await self.student.generate(
                frames=frames,
                video_path=video_path,
                prompt=prompt,
                task=task,
                **kwargs,
            )
        except Exception as e:
            logger.error("Student inference failed", error=str(e))
            self.stats.student_failed += 1
            # Fall through to fallback

        # Check if we should use student result
        if student_result and student_result.is_success():
            # Apply calibration if enabled
            confidence = student_result.confidence
            if self.use_calibrated_confidence and self.calibrator:
                calibrated = self.calibrator.calibrate(student_result)
                confidence = calibrated.confidence

            # Check confidence threshold
            if confidence >= self.confidence_threshold:
                self.stats.student_success += 1
                logger.debug(
                    "Using student prediction",
                    confidence=confidence,
                    threshold=self.confidence_threshold,
                )
                return student_result
            else:
                self.stats.student_low_confidence += 1
                logger.info(
                    "Student confidence below threshold, falling back",
                    confidence=confidence,
                    threshold=self.confidence_threshold,
                )
        elif student_result:
            self.stats.student_failed += 1
            logger.warning(
                "Student inference unsuccessful, falling back",
                status=student_result.status,
            )

        # Fallback to teacher if enabled
        if not self.fallback_to_teacher:
            return GenerationResult.failure(
                error_message="Student failed, fallback disabled",
                model_type=ModelType.STUDENT_LOCAL,
            )

        # Fallback to teacher
        try:
            teacher_result = await self.teacher.annotate(
                video_path=video_path,
                frames=frames,
                prompt=prompt,
                task=task,
                **kwargs,
            )

            if teacher_result.is_success():
                self.stats.fallback_success += 1
                logger.info("Fallback to teacher succeeded")

                # Mark as fallback result
                return GenerationResult(
                    text=teacher_result.text,
                    model_type=teacher_result.model_type,
                    model_version=teacher_result.model_version,
                    status=GenerationStatus.SUCCESS,
                    confidence=teacher_result.confidence,
                    latency_ms=teacher_result.latency_ms,
                    cost_usd=teacher_result.cost_usd,
                    metadata={
                        **(teacher_result.metadata or {}),
                        "fallback": True,
                        "fallback_reason": "low_confidence" if student_result else "student_failed",
                        "student_confidence": student_result.confidence if student_result else None,
                    },
                )
            else:
                self.stats.fallback_failed += 1
                logger.error("Fallback to teacher failed")
                return teacher_result

        except Exception as e:
            self.stats.fallback_failed += 1
            logger.error("Fallback inference failed", error=str(e))

            # Return student result if we have it, otherwise failure
            if student_result:
                return student_result

            return GenerationResult.failure(
                error_message=f"Both student and teacher failed: {e}",
                model_type=ModelType.STUDENT_LOCAL,
            )

    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs,
    ) -> List[GenerationResult]:
        """Batch generation with fallback.

        Args:
            items: List of generation items
            **kwargs: Additional arguments

        Returns:
            List of GenerationResults
        """
        tasks = [
            self.generate(
                frames=item.get("frames"),
                video_path=item.get("video_path"),
                prompt=item.get("prompt"),
                task=item.get("task", "fine_grained"),
                **kwargs,
            )
            for item in items
        ]

        return await asyncio.gather(*tasks)

    def get_stats(self) -> FallbackStats:
        """Get current fallback statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = FallbackStats()

    def update_threshold(self, new_threshold: float) -> None:
        """Update confidence threshold.

        Args:
            new_threshold: New confidence threshold (0-1)
        """
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(
            "Updated confidence threshold",
            old=old_threshold,
            new=self.confidence_threshold,
        )


class AdaptiveFallback(LowConfidenceFallback):
    """Adaptive fallback that adjusts threshold based on performance.

    Automatically adjusts confidence threshold to meet target
    accuracy or cost constraints.
    """

    def __init__(
        self,
        student_engine: StudentInferenceEngine,
        teacher_model=None,
        initial_threshold: float = 0.7,
        target_accuracy: Optional[float] = None,
        target_fallback_rate: Optional[float] = None,
        adjustment_rate: float = 0.05,
    ):
        super().__init__(
            student_engine=student_engine,
            teacher_model=teacher_model,
            confidence_threshold=initial_threshold,
        )
        self.target_accuracy = target_accuracy
        self.target_fallback_rate = target_fallback_rate
        self.adjustment_rate = adjustment_rate

        self._recent_results: List[Dict] = []
        self._window_size = 100

    async def generate(
        self,
        frames: Optional[List] = None,
        video_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> GenerationResult:
        """Generate with adaptive threshold."""
        result = await super().generate(
            frames=frames,
            video_path=video_path,
            prompt=prompt,
            task=task,
            **kwargs,
        )

        # Track result for adaptation
        self._recent_results.append({
            "used_fallback": result.metadata.get("fallback", False),
            "success": result.is_success(),
        })

        # Keep window size bounded
        if len(self._recent_results) > self._window_size:
            self._recent_results = self._recent_results[-self._window_size:]

        # Adapt threshold periodically
        if len(self._recent_results) % 20 == 0:
            self._adapt_threshold()

        return result

    def _adapt_threshold(self) -> None:
        """Adapt threshold based on recent performance."""
        if len(self._recent_results) < 20:
            return

        recent = self._recent_results[-50:]
        fallback_rate = sum(r["used_fallback"] for r in recent) / len(recent)
        success_rate = sum(r["success"] for r in recent) / len(recent)

        old_threshold = self.confidence_threshold

        # Adjust based on targets
        if self.target_fallback_rate is not None:
            if fallback_rate > self.target_fallback_rate:
                # Too many fallbacks, raise threshold (be more confident)
                self.confidence_threshold = min(
                    1.0,
                    self.confidence_threshold + self.adjustment_rate
                )
            elif fallback_rate < self.target_fallback_rate * 0.8:
                # Too few fallbacks, lower threshold
                self.confidence_threshold = max(
                    0.0,
                    self.confidence_threshold - self.adjustment_rate
                )

        if self.target_accuracy is not None:
            if success_rate < self.target_accuracy:
                # Accuracy too low, raise threshold
                self.confidence_threshold = min(
                    1.0,
                    self.confidence_threshold + self.adjustment_rate
                )

        if self.confidence_threshold != old_threshold:
            logger.info(
                "Adapted confidence threshold",
                old=old_threshold,
                new=self.confidence_threshold,
                fallback_rate=fallback_rate,
                success_rate=success_rate,
            )


def create_fallback_router(
    student_path: Union[str, Path],
    teacher_model=None,
    confidence_threshold: float = 0.7,
    adaptive: bool = False,
    **kwargs,
) -> Union[LowConfidenceFallback, AdaptiveFallback]:
    """Factory function to create fallback router.

    Args:
        student_path: Path to student model
        teacher_model: Optional teacher model instance
        confidence_threshold: Confidence threshold for fallback
        adaptive: Whether to use adaptive threshold
        **kwargs: Additional arguments for adaptive fallback

    Returns:
        Configured fallback router
    """
    student_engine = StudentInferenceEngine(student_path)

    if adaptive:
        return AdaptiveFallback(
            student_engine=student_engine,
            teacher_model=teacher_model,
            initial_threshold=confidence_threshold,
            **kwargs,
        )
    else:
        return LowConfidenceFallback(
            student_engine=student_engine,
            teacher_model=teacher_model,
            confidence_threshold=confidence_threshold,
        )
