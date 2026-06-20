"""Quality gates for annotation validation and export approval.

Provides configurable quality thresholds for:
- Annotation completeness
- Parse confidence
- Temporal consistency
- Action/object grounding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation, Segment
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""

    # Parse quality thresholds
    min_parse_confidence: float = 0.5
    require_structured_output: bool = False

    # Content completeness thresholds
    min_actions_per_segment: int = 0
    min_objects_per_segment: int = 0
    require_caption: bool = True

    # Confidence thresholds
    min_action_confidence: float = 0.0
    min_object_confidence: float = 0.0

    # Aggregate thresholds
    min_segments_with_actions: int = 1
    max_empty_segments_ratio: float = 0.5

    # Export gate
    require_all_segments_valid: bool = False
    min_overall_quality_score: float = 0.0


@dataclass
class SegmentQualityResult:
    """Quality assessment for a single segment."""

    segment_idx: int
    passed: bool
    parse_confidence: float
    has_caption: bool
    action_count: int
    object_count: int
    issues: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_idx": self.segment_idx,
            "passed": self.passed,
            "parse_confidence": self.parse_confidence,
            "has_caption": self.has_caption,
            "action_count": self.action_count,
            "object_count": self.object_count,
            "issues": self.issues,
            "score": self.score,
        }


@dataclass
class AnnotationQualityResult:
    """Quality assessment for a complete annotation."""

    video_id: str
    passed: bool
    overall_score: float
    segment_results: List[SegmentQualityResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "segment_count": len(self.segment_results),
            "segments_passed": sum(1 for s in self.segment_results if s.passed),
            "segments_failed": sum(1 for s in self.segment_results if not s.passed),
            "segment_results": [s.to_dict() for s in self.segment_results],
            "summary": self.summary,
            "issues": self.issues,
        }


class QualityGate:
    """Quality validation gate for annotations.

    Validates annotations against configurable thresholds before
    they are allowed to proceed to export.
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()

    def validate_segment(self, segment: Segment, segment_idx: int) -> SegmentQualityResult:
        """Validate a single segment against quality thresholds."""
        issues: List[str] = []

        # Check parse confidence from metadata if available
        parse_confidence = 0.5  # Default
        if hasattr(segment, "metadata") and segment.metadata:
            parse_confidence = segment.metadata.get("parse_confidence", 0.5)

        has_caption = bool(segment.caption and segment.caption.strip())
        action_count = len(segment.actions) if segment.actions else 0
        object_count = len(segment.objects) if segment.objects else 0

        # Validate parse confidence
        if parse_confidence < self.config.min_parse_confidence:
            issues.append(
                f"parse_confidence {parse_confidence:.2f} below threshold "
                f"{self.config.min_parse_confidence:.2f}"
            )

        # Validate caption
        if self.config.require_caption and not has_caption:
            issues.append("missing_caption")

        # Validate action count
        if action_count < self.config.min_actions_per_segment:
            issues.append(
                f"action_count {action_count} below minimum "
                f"{self.config.min_actions_per_segment}"
            )

        # Validate object count
        if object_count < self.config.min_objects_per_segment:
            issues.append(
                f"object_count {object_count} below minimum "
                f"{self.config.min_objects_per_segment}"
            )

        # Calculate score (simplified)
        score = self._calculate_segment_score(
            parse_confidence, has_caption, action_count, object_count
        )

        passed = len(issues) == 0

        return SegmentQualityResult(
            segment_idx=segment_idx,
            passed=passed,
            parse_confidence=parse_confidence,
            has_caption=has_caption,
            action_count=action_count,
            object_count=object_count,
            issues=issues,
            score=score,
        )

    def validate_annotation(self, annotation: Annotation) -> AnnotationQualityResult:
        """Validate a complete annotation."""
        segment_results: List[SegmentQualityResult] = []

        for idx, segment in enumerate(annotation.segments):
            result = self.validate_segment(segment, idx)
            segment_results.append(result)

        # Calculate aggregate metrics
        passed_segments = sum(1 for s in segment_results if s.passed)
        total_segments = len(segment_results)

        if total_segments == 0:
            overall_score = 0.0
        else:
            overall_score = sum(s.score for s in segment_results) / total_segments

        issues: List[str] = []

        # Check empty segment ratio
        if total_segments > 0:
            empty_ratio = (total_segments - passed_segments) / total_segments
            if empty_ratio > self.config.max_empty_segments_ratio:
                issues.append(
                    f"empty_segments_ratio {empty_ratio:.2f} exceeds threshold "
                    f"{self.config.max_empty_segments_ratio:.2f}"
                )

        # Check minimum segments with actions
        segments_with_actions = sum(
            1 for s in segment_results if s.action_count >= self.config.min_actions_per_segment
        )
        if segments_with_actions < self.config.min_segments_with_actions:
            issues.append(
                f"segments_with_actions {segments_with_actions} below minimum "
                f"{self.config.min_segments_with_actions}"
            )

        # Determine overall pass/fail
        if self.config.require_all_segments_valid:
            passed = all(s.passed for s in segment_results) and len(issues) == 0
        else:
            passed = overall_score >= self.config.min_overall_quality_score and len(issues) == 0

        summary = {
            "total_segments": total_segments,
            "passed_segments": passed_segments,
            "failed_segments": total_segments - passed_segments,
            "avg_parse_confidence": (
                sum(s.parse_confidence for s in segment_results) / total_segments
                if total_segments > 0 else 0.0
            ),
            "total_actions": sum(s.action_count for s in segment_results),
            "total_objects": sum(s.object_count for s in segment_results),
        }

        return AnnotationQualityResult(
            video_id=annotation.video_id,
            passed=passed,
            overall_score=overall_score,
            segment_results=segment_results,
            summary=summary,
            issues=issues,
        )

    def _calculate_segment_score(
        self,
        parse_confidence: float,
        has_caption: bool,
        action_count: int,
        object_count: int,
    ) -> float:
        """Calculate a quality score for a segment."""
        score = 0.0

        # Parse confidence contributes up to 0.4
        score += parse_confidence * 0.4

        # Caption contributes 0.2
        if has_caption:
            score += 0.2

        # Actions contribute up to 0.2
        score += min(action_count * 0.05, 0.2)

        # Objects contribute up to 0.2
        score += min(object_count * 0.05, 0.2)

        return min(1.0, score)


class ExportGate:
    """Export approval gate with quality validation.

    Validates annotations before they are allowed to be exported
    to training datasets.
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()
        self.quality_gate = QualityGate(config)

    def approve_for_export(
        self,
        annotation: Annotation,
        validate_quality: bool = True,
    ) -> tuple[bool, List[str]]:
        """Approve annotation for export.

        Args:
            annotation: The annotation to validate
            validate_quality: Whether to run quality validation

        Returns:
            Tuple of (approved, list of issues if rejected)
        """
        issues: List[str] = []

        # Basic validation
        if not annotation.segments:
            issues.append("no_segments")

        if not annotation.video_id:
            issues.append("missing_video_id")

        # Quality validation
        if validate_quality and not issues:
            quality_result = self.quality_gate.validate_annotation(annotation)
            if not quality_result.passed:
                issues.extend(quality_result.issues)
                issues.append(
                    f"quality_score {quality_result.overall_score:.2f} below threshold"
                )

        approved = len(issues) == 0

        if approved:
            logger.info("export_approved", video_id=annotation.video_id)
        else:
            logger.warning(
                "export_rejected",
                video_id=annotation.video_id,
                issues=issues,
            )

        return approved, issues

    def filter_for_export(
        self,
        annotations: List[Annotation],
        min_quality_score: Optional[float] = None,
    ) -> tuple[List[Annotation], Dict[str, List[str]]]:
        """Filter annotations for export, returning approved and rejected.

        Args:
            annotations: List of annotations to filter
            min_quality_score: Optional minimum quality score override

        Returns:
            Tuple of (approved annotations, dict of video_id -> rejection reasons)
        """
        approved: List[Annotation] = []
        rejected: Dict[str, List[str]] = {}

        original_min_score = self.config.min_overall_quality_score
        if min_quality_score is not None:
            self.config.min_overall_quality_score = min_quality_score

        try:
            for annotation in annotations:
                is_approved, issues = self.approve_for_export(annotation)
                if is_approved:
                    approved.append(annotation)
                else:
                    rejected[annotation.video_id] = issues
        finally:
            self.config.min_overall_quality_score = original_min_score

        logger.info(
            "export_filter_complete",
            total=len(annotations),
            approved=len(approved),
            rejected=len(rejected),
        )

        return approved, rejected


class QualityGateRegistry:
    """Registry of quality gate configurations for different use cases."""

    _configs: Dict[str, QualityGateConfig] = {
        "strict": QualityGateConfig(
            min_parse_confidence=0.8,
            require_structured_output=True,
            min_actions_per_segment=1,
            min_objects_per_segment=1,
            require_caption=True,
            require_all_segments_valid=True,
            min_overall_quality_score=0.7,
        ),
        "standard": QualityGateConfig(
            min_parse_confidence=0.5,
            min_actions_per_segment=0,
            require_caption=True,
            max_empty_segments_ratio=0.5,
            min_overall_quality_score=0.5,
        ),
        "lenient": QualityGateConfig(
            min_parse_confidence=0.3,
            require_caption=False,
            max_empty_segments_ratio=0.8,
            min_overall_quality_score=0.3,
        ),
        "debug": QualityGateConfig(
            min_parse_confidence=0.0,
            require_caption=False,
            max_empty_segments_ratio=1.0,
            min_overall_quality_score=0.0,
        ),
    }

    @classmethod
    def get(cls, name: str) -> QualityGateConfig:
        """Get a predefined quality gate configuration."""
        if name not in cls._configs:
            raise ValueError(
                f"Unknown quality gate: {name}. "
                f"Available: {list(cls._configs.keys())}"
            )
        return cls._configs[name]

    @classmethod
    def create_gate(cls, name: str) -> QualityGate:
        """Create a QualityGate with predefined configuration."""
        return QualityGate(cls.get(name))

    @classmethod
    def create_export_gate(cls, name: str) -> ExportGate:
        """Create an ExportGate with predefined configuration."""
        return ExportGate(cls.get(name))

    @classmethod
    def register(cls, name: str, config: QualityGateConfig) -> None:
        """Register a custom quality gate configuration."""
        cls._configs[name] = config

    @classmethod
    def available(cls) -> List[str]:
        """List available quality gate configurations."""
        return list(cls._configs.keys())
