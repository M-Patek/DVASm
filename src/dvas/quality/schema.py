"""Quality schema and score definitions for annotation quality assessment.

Defines quality dimensions and scoring structures for comprehensive
annotation quality analysis and feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class QualityDimension(str, Enum):
    """Quality dimensions for annotation assessment."""

    FACTUALITY = "factuality"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    OBJECT_GROUNDING = "object_grounding"
    ACTION_GROUNDING = "action_grounding"
    AFFORDANCE = "affordance"
    ROBOTIC_USEFULNESS = "robotic_usefulness"
    LANGUAGE_CLARITY = "language_clarity"
    PARSE_CONFIDENCE = "parse_confidence"
    REVIEWER_CONFIDENCE = "reviewer_confidence"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float = 1.0  # Confidence in the score itself
    weight: float = 1.0  # Weight for aggregation
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate score ranges."""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.weight = max(0.0, self.weight)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "weight": self.weight,
            "details": self.details,
            "issues": self.issues,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionScore":
        """Create from dictionary."""
        return cls(
            dimension=QualityDimension(data["dimension"]),
            score=data["score"],
            confidence=data.get("confidence", 1.0),
            weight=data.get("weight", 1.0),
            details=data.get("details", {}),
            issues=data.get("issues", []),
        )


@dataclass
class QualityScores:
    """Complete quality scores for an annotation.

    Contains all quality dimensions and aggregate scores.
    """

    annotation_id: str
    video_id: str

    # Individual dimension scores
    factuality_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.FACTUALITY, score=0.0
        )
    )
    temporal_consistency_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.TEMPORAL_CONSISTENCY, score=0.0
        )
    )
    object_grounding_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.OBJECT_GROUNDING, score=0.0
        )
    )
    action_grounding_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.ACTION_GROUNDING, score=0.0
        )
    )
    affordance_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.AFFORDANCE, score=0.0
        )
    )
    robotic_usefulness_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.ROBOTIC_USEFULNESS, score=0.0
        )
    )
    language_clarity_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.LANGUAGE_CLARITY, score=0.0
        )
    )
    parse_confidence_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.PARSE_CONFIDENCE, score=0.0
        )
    )
    reviewer_confidence_score: DimensionScore = field(
        default_factory=lambda: DimensionScore(
            dimension=QualityDimension.REVIEWER_CONFIDENCE, score=0.0
        )
    )

    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)
    computed_by: str = "automatic"  # "automatic", "llm_judge", "human"
    version: str = "1.0"

    # Aggregate scores
    overall_score: float = 0.0
    weighted_score: float = 0.0

    def __post_init__(self):
        """Compute aggregate scores after initialization."""
        self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        """Compute overall and weighted aggregate scores."""
        scores = self.all_scores

        if not scores:
            self.overall_score = 0.0
            self.weighted_score = 0.0
            return

        # Simple average
        self.overall_score = sum(s.score for s in scores) / len(scores)

        # Weighted average
        total_weight = sum(s.weight for s in scores)
        if total_weight > 0:
            self.weighted_score = sum(
                s.score * s.weight for s in scores
            ) / total_weight
        else:
            self.weighted_score = self.overall_score

    @property
    def all_scores(self) -> List[DimensionScore]:
        """Get all dimension scores as a list."""
        return [
            self.factuality_score,
            self.temporal_consistency_score,
            self.object_grounding_score,
            self.action_grounding_score,
            self.affordance_score,
            self.robotic_usefulness_score,
            self.language_clarity_score,
            self.parse_confidence_score,
            self.reviewer_confidence_score,
        ]

    @property
    def failed_dimensions(self) -> List[QualityDimension]:
        """Get list of dimensions that failed (score < 0.5)."""
        return [s.dimension for s in self.all_scores if s.score < 0.5]

    @property
    def all_issues(self) -> List[str]:
        """Get all issues from all dimensions."""
        issues = []
        for score in self.all_scores:
            issues.extend(score.issues)
        return issues

    def get_score(self, dimension: QualityDimension) -> DimensionScore:
        """Get score for a specific dimension."""
        score_map = {
            QualityDimension.FACTUALITY: self.factuality_score,
            QualityDimension.TEMPORAL_CONSISTENCY: self.temporal_consistency_score,
            QualityDimension.OBJECT_GROUNDING: self.object_grounding_score,
            QualityDimension.ACTION_GROUNDING: self.action_grounding_score,
            QualityDimension.AFFORDANCE: self.affordance_score,
            QualityDimension.ROBOTIC_USEFULNESS: self.robotic_usefulness_score,
            QualityDimension.LANGUAGE_CLARITY: self.language_clarity_score,
            QualityDimension.PARSE_CONFIDENCE: self.parse_confidence_score,
            QualityDimension.REVIEWER_CONFIDENCE: self.reviewer_confidence_score,
        }
        return score_map.get(dimension, self.factuality_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "dimensions": {s.dimension.value: s.to_dict() for s in self.all_scores},
            "overall_score": self.overall_score,
            "weighted_score": self.weighted_score,
            "computed_at": self.computed_at.isoformat(),
            "computed_by": self.computed_by,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityScores":
        """Create from dictionary."""
        dimensions_data = data.get("dimensions", {})

        def get_dim_score(dim: QualityDimension) -> DimensionScore:
            if dim.value in dimensions_data:
                return DimensionScore.from_dict(dimensions_data[dim.value])
            return DimensionScore(dimension=dim, score=0.0)

        return cls(
            annotation_id=data["annotation_id"],
            video_id=data["video_id"],
            factuality_score=get_dim_score(QualityDimension.FACTUALITY),
            temporal_consistency_score=get_dim_score(QualityDimension.TEMPORAL_CONSISTENCY),
            object_grounding_score=get_dim_score(QualityDimension.OBJECT_GROUNDING),
            action_grounding_score=get_dim_score(QualityDimension.ACTION_GROUNDING),
            affordance_score=get_dim_score(QualityDimension.AFFORDANCE),
            robotic_usefulness_score=get_dim_score(QualityDimension.ROBOTIC_USEFULNESS),
            language_clarity_score=get_dim_score(QualityDimension.LANGUAGE_CLARITY),
            parse_confidence_score=get_dim_score(QualityDimension.PARSE_CONFIDENCE),
            reviewer_confidence_score=get_dim_score(QualityDimension.REVIEWER_CONFIDENCE),
            computed_at=datetime.fromisoformat(data["computed_at"]),
            computed_by=data.get("computed_by", "automatic"),
            version=data.get("version", "1.0"),
        )


@dataclass
class QualityThresholds:
    """Configurable thresholds for quality dimensions.

    Used to determine pass/fail for each quality dimension.
    """

    factuality_min: float = 0.7
    temporal_consistency_min: float = 0.6
    object_grounding_min: float = 0.6
    action_grounding_min: float = 0.6
    affordance_min: float = 0.5
    robotic_usefulness_min: float = 0.5
    language_clarity_min: float = 0.7
    parse_confidence_min: float = 0.5
    reviewer_confidence_min: float = 0.8

    overall_min: float = 0.6
    weighted_min: float = 0.6

    # Maximum allowed failed dimensions
    max_failed_dimensions: int = 2

    def check_score(self, scores: QualityScores) -> tuple[bool, List[str]]:
        """Check if scores meet all thresholds.

        Returns:
            Tuple of (passed, list of failure reasons)
        """
        failures = []

        checks = [
            (scores.factuality_score.score, self.factuality_min, "factuality"),
            (
                scores.temporal_consistency_score.score,
                self.temporal_consistency_min,
                "temporal_consistency",
            ),
            (
                scores.object_grounding_score.score,
                self.object_grounding_min,
                "object_grounding",
            ),
            (
                scores.action_grounding_score.score,
                self.action_grounding_min,
                "action_grounding",
            ),
            (scores.affordance_score.score, self.affordance_min, "affordance"),
            (
                scores.robotic_usefulness_score.score,
                self.robotic_usefulness_min,
                "robotic_usefulness",
            ),
            (
                scores.language_clarity_score.score,
                self.language_clarity_min,
                "language_clarity",
            ),
            (
                scores.parse_confidence_score.score,
                self.parse_confidence_min,
                "parse_confidence",
            ),
            (
                scores.reviewer_confidence_score.score,
                self.reviewer_confidence_min,
                "reviewer_confidence",
            ),
        ]

        for score, threshold, name in checks:
            if score < threshold:
                failures.append(f"{name}_below_threshold ({score:.2f} < {threshold:.2f})")

        if scores.overall_score < self.overall_min:
            failures.append(
                f"overall_score_below_threshold ({scores.overall_score:.2f} < {self.overall_min:.2f})"
            )

        if scores.weighted_score < self.weighted_min:
            failures.append(
                f"weighted_score_below_threshold ({scores.weighted_score:.2f} < {self.weighted_min:.2f})"
            )

        if len(scores.failed_dimensions) > self.max_failed_dimensions:
            failures.append(
                f"too_many_failed_dimensions ({len(scores.failed_dimensions)} > {self.max_failed_dimensions})"
            )

        return len(failures) == 0, failures


class QualityProfile(Enum):
    """Predefined quality profiles."""

    STRICT = QualityThresholds(
        factuality_min=0.8,
        temporal_consistency_min=0.8,
        object_grounding_min=0.8,
        action_grounding_min=0.8,
        affordance_min=0.7,
        robotic_usefulness_min=0.7,
        language_clarity_min=0.8,
        parse_confidence_min=0.7,
        reviewer_confidence_min=0.9,
        overall_min=0.75,
        weighted_min=0.75,
        max_failed_dimensions=1,
    )

    STANDARD = QualityThresholds(
        factuality_min=0.7,
        temporal_consistency_min=0.6,
        object_grounding_min=0.6,
        action_grounding_min=0.6,
        affordance_min=0.5,
        robotic_usefulness_min=0.5,
        language_clarity_min=0.7,
        parse_confidence_min=0.5,
        reviewer_confidence_min=0.8,
        overall_min=0.6,
        weighted_min=0.6,
        max_failed_dimensions=2,
    )

    LENIENT = QualityThresholds(
        factuality_min=0.5,
        temporal_consistency_min=0.5,
        object_grounding_min=0.5,
        action_grounding_min=0.5,
        affordance_min=0.4,
        robotic_usefulness_min=0.4,
        language_clarity_min=0.5,
        parse_confidence_min=0.3,
        reviewer_confidence_min=0.6,
        overall_min=0.5,
        weighted_min=0.5,
        max_failed_dimensions=3,
    )

    ROBOTICS = QualityThresholds(
        factuality_min=0.7,
        temporal_consistency_min=0.6,
        object_grounding_min=0.7,
        action_grounding_min=0.7,
        affordance_min=0.7,
        robotic_usefulness_min=0.8,
        language_clarity_min=0.6,
        parse_confidence_min=0.5,
        reviewer_confidence_min=0.8,
        overall_min=0.65,
        weighted_min=0.65,
        max_failed_dimensions=2,
    )
