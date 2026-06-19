"""Dataset acceptance criteria for export quality control.

Defines acceptance thresholds and gates for filtering annotations
before they are exported to training datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dvas.data.schemas import Annotation
from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityProfile,
    QualityScores,
    QualityThresholds,
)
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AcceptanceLevel(str, Enum):
    """Acceptance levels for dataset export."""

    GOLD = "gold"  # Highest quality, for final training
    SILVER = "silver"  # Good quality, for training with weighting
    BRONZE = "bronze"  # Acceptable quality, for pre-training
    REJECT = "reject"  # Below acceptable quality


@dataclass
class AcceptanceCriteria:
    """Criteria for accepting annotations into a dataset.

    Defines thresholds for quality dimensions and overall scores
    required for an annotation to be accepted at a given level.
    """

    name: str
    description: str = ""

    # Quality thresholds
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # Minimum scores per dimension (override thresholds)
    min_dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Overall score requirements
    min_overall_score: float = 0.6
    min_weighted_score: float = 0.6

    # Required dimensions (must pass these)
    required_dimensions: List[QualityDimension] = field(default_factory=list)

    # Maximum allowed issues
    max_issues: int = 10

    # Dimension weights for weighted score calculation
    dimension_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default required dimensions if not specified."""
        if not self.required_dimensions:
            self.required_dimensions = [
                QualityDimension.FACTUALITY,
                QualityDimension.LANGUAGE_CLARITY,
            ]

        # Set default weights if not specified
        if not self.dimension_weights:
            self.dimension_weights = {
                QualityDimension.FACTUALITY.value: 1.5,
                QualityDimension.TEMPORAL_CONSISTENCY.value: 1.0,
                QualityDimension.OBJECT_GROUNDING.value: 1.0,
                QualityDimension.ACTION_GROUNDING.value: 1.0,
                QualityDimension.AFFORDANCE.value: 0.8,
                QualityDimension.ROBOTIC_USEFULNESS.value: 1.2,
                QualityDimension.LANGUAGE_CLARITY.value: 1.0,
                QualityDimension.PARSE_CONFIDENCE.value: 0.8,
                QualityDimension.REVIEWER_CONFIDENCE.value: 1.0,
            }

    def evaluate(self, scores: QualityScores) -> tuple[AcceptanceLevel, List[str]]:
        """Evaluate quality scores against acceptance criteria.

        Args:
            scores: Quality scores to evaluate

        Returns:
            Tuple of (acceptance level, list of failure reasons)
        """
        failures = []

        # Check overall scores
        if scores.overall_score < self.min_overall_score:
            failures.append(
                f"overall_score {scores.overall_score:.2f} < {self.min_overall_score:.2f}"
            )

        if scores.weighted_score < self.min_weighted_score:
            failures.append(
                f"weighted_score {scores.weighted_score:.2f} < {self.min_weighted_score:.2f}"
            )

        # Check required dimensions
        for dim in self.required_dimensions:
            dim_score = scores.get_score(dim)
            min_score = self.min_dimension_scores.get(
                dim.value, self.thresholds.check_score(scores)[1]
            )

            # Get threshold for this dimension
            threshold_map = {
                QualityDimension.FACTUALITY: self.thresholds.factuality_min,
                QualityDimension.TEMPORAL_CONSISTENCY: self.thresholds.temporal_consistency_min,
                QualityDimension.OBJECT_GROUNDING: self.thresholds.object_grounding_min,
                QualityDimension.ACTION_GROUNDING: self.thresholds.action_grounding_min,
                QualityDimension.AFFORDANCE: self.thresholds.affordance_min,
                QualityDimension.ROBOTIC_USEFULNESS: self.thresholds.robotic_usefulness_min,
                QualityDimension.LANGUAGE_CLARITY: self.thresholds.language_clarity_min,
                QualityDimension.PARSE_CONFIDENCE: self.thresholds.parse_confidence_min,
                QualityDimension.REVIEWER_CONFIDENCE: self.thresholds.reviewer_confidence_min,
            }
            threshold = threshold_map.get(dim, 0.5)

            if dim_score.score < threshold:
                failures.append(
                    f"{dim.value}_score {dim_score.score:.2f} < {threshold:.2f}"
                )

        # Check issue count
        if len(scores.all_issues) > self.max_issues:
            failures.append(f"too_many_issues ({len(scores.all_issues)} > {self.max_issues})")

        # Determine acceptance level
        if not failures:
            # Check if it's gold quality
            if scores.overall_score >= 0.85 and scores.weighted_score >= 0.85:
                return AcceptanceLevel.GOLD, []
            elif scores.overall_score >= 0.7:
                return AcceptanceLevel.SILVER, []
            else:
                return AcceptanceLevel.BRONZE, []
        elif scores.overall_score >= 0.4:
            return AcceptanceLevel.BRONZE, failures
        else:
            return AcceptanceLevel.REJECT, failures

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "min_overall_score": self.min_overall_score,
            "min_weighted_score": self.min_weighted_score,
            "required_dimensions": [d.value for d in self.required_dimensions],
            "max_issues": self.max_issues,
            "dimension_weights": self.dimension_weights,
        }


class AcceptanceGate:
    """Gate for filtering annotations based on acceptance criteria.

    Validates annotations before they are allowed to be exported
    to training datasets at different quality levels.
    """

    def __init__(self, criteria: Optional[AcceptanceCriteria] = None):
        """Initialize acceptance gate.

        Args:
            criteria: Acceptance criteria to use
        """
        self.criteria = criteria or AcceptanceCriteria(
            name="standard",
            description="Standard acceptance criteria",
        )

    def check(
        self,
        annotation: Annotation,
        quality_scores: QualityScores,
    ) -> tuple[bool, AcceptanceLevel, List[str]]:
        """Check if annotation passes acceptance criteria.

        Args:
            annotation: Annotation to check
            quality_scores: Quality scores for the annotation

        Returns:
            Tuple of (passed, acceptance level, failure reasons)
        """
        level, failures = self.criteria.evaluate(quality_scores)
        passed = level != AcceptanceLevel.REJECT

        if passed:
            logger.debug(
                "annotation_accepted",
                annotation_id=annotation.id,
                level=level.value,
                overall_score=quality_scores.overall_score,
            )
        else:
            logger.debug(
                "annotation_rejected",
                annotation_id=annotation.id,
                failures=failures,
                overall_score=quality_scores.overall_score,
            )

        return passed, level, failures

    def filter_annotations(
        self,
        annotations: List[Annotation],
        scores_map: Dict[str, QualityScores],
        min_level: AcceptanceLevel = AcceptanceLevel.BRONZE,
    ) -> Dict[AcceptanceLevel, List[Annotation]]:
        """Filter annotations by acceptance level.

        Args:
            annotations: List of annotations to filter
            scores_map: Map of annotation_id to QualityScores
            min_level: Minimum acceptance level to include

        Returns:
            Dictionary mapping acceptance level to list of annotations
        """
        results: Dict[AcceptanceLevel, List[Annotation]] = {
            AcceptanceLevel.GOLD: [],
            AcceptanceLevel.SILVER: [],
            AcceptanceLevel.BRONZE: [],
            AcceptanceLevel.REJECT: [],
        }

        level_order = {
            AcceptanceLevel.GOLD: 3,
            AcceptanceLevel.SILVER: 2,
            AcceptanceLevel.BRONZE: 1,
            AcceptanceLevel.REJECT: 0,
        }
        min_order = level_order.get(min_level, 1)

        for annotation in annotations:
            scores = scores_map.get(annotation.id)
            if not scores:
                logger.warning("no_scores_for_annotation", annotation_id=annotation.id)
                results[AcceptanceLevel.REJECT].append(annotation)
                continue

            level, _ = self.criteria.evaluate(scores)
            results[level].append(annotation)

        # Filter by minimum level
        filtered: Dict[AcceptanceLevel, List[Annotation]] = {}
        for level, anns in results.items():
            if level_order.get(level, 0) >= min_order:
                filtered[level] = anns

        logger.info(
            "annotations_filtered",
            total=len(annotations),
            gold=len(results[AcceptanceLevel.GOLD]),
            silver=len(results[AcceptanceLevel.SILVER]),
            bronze=len(results[AcceptanceLevel.BRONZE]),
            rejected=len(results[AcceptanceLevel.REJECT]),
        )

        return filtered

    def get_acceptance_stats(
        self,
        annotations: List[Annotation],
        scores_map: Dict[str, QualityScores],
    ) -> Dict[str, Any]:
        """Get statistics about acceptance rates.

        Args:
            annotations: List of annotations
            scores_map: Map of annotation_id to QualityScores

        Returns:
            Statistics dictionary
        """
        results = self.filter_annotations(annotations, scores_map)

        total = len(annotations)
        gold = len(results.get(AcceptanceLevel.GOLD, []))
        silver = len(results.get(AcceptanceLevel.SILVER, []))
        bronze = len(results.get(AcceptanceLevel.BRONZE, []))
        rejected = total - gold - silver - bronze

        return {
            "total": total,
            "by_level": {
                "gold": gold,
                "silver": silver,
                "bronze": bronze,
                "rejected": rejected,
            },
            "rates": {
                "gold": gold / total if total > 0 else 0,
                "silver": silver / total if total > 0 else 0,
                "bronze": bronze / total if total > 0 else 0,
                "rejected": rejected / total if total > 0 else 0,
                "accepted": (gold + silver + bronze) / total if total > 0 else 0,
            },
        }


class AcceptanceCriteriaRegistry:
    """Registry of predefined acceptance criteria."""

    _criteria: Dict[str, AcceptanceCriteria] = {
        "strict": AcceptanceCriteria(
            name="strict",
            description="Strict criteria for gold-standard datasets",
            thresholds=QualityProfile.STRICT.value,
            min_overall_score=0.75,
            min_weighted_score=0.75,
            required_dimensions=[
                QualityDimension.FACTUALITY,
                QualityDimension.TEMPORAL_CONSISTENCY,
                QualityDimension.LANGUAGE_CLARITY,
                QualityDimension.OBJECT_GROUNDING,
            ],
            max_issues=3,
        ),
        "standard": AcceptanceCriteria(
            name="standard",
            description="Standard criteria for general training",
            thresholds=QualityProfile.STANDARD.value,
            min_overall_score=0.6,
            min_weighted_score=0.6,
            required_dimensions=[
                QualityDimension.FACTUALITY,
                QualityDimension.LANGUAGE_CLARITY,
            ],
            max_issues=5,
        ),
        "lenient": AcceptanceCriteria(
            name="lenient",
            description="Lenient criteria for pre-training",
            thresholds=QualityProfile.LENIENT.value,
            min_overall_score=0.5,
            min_weighted_score=0.5,
            required_dimensions=[QualityDimension.FACTUALITY],
            max_issues=10,
        ),
        "robotics": AcceptanceCriteria(
            name="robotics",
            description="Criteria optimized for robotics training",
            thresholds=QualityProfile.ROBOTICS.value,
            min_overall_score=0.65,
            min_weighted_score=0.65,
            required_dimensions=[
                QualityDimension.FACTUALITY,
                QualityDimension.ACTION_GROUNDING,
                QualityDimension.ROBOTIC_USEFULNESS,
            ],
            max_issues=5,
            dimension_weights={
                QualityDimension.FACTUALITY.value: 1.2,
                QualityDimension.ACTION_GROUNDING.value: 1.5,
                QualityDimension.ROBOTIC_USEFULNESS.value: 1.5,
                QualityDimension.AFFORDANCE.value: 1.2,
                QualityDimension.OBJECT_GROUNDING.value: 1.0,
                QualityDimension.TEMPORAL_CONSISTENCY.value: 0.8,
                QualityDimension.LANGUAGE_CLARITY.value: 0.8,
                QualityDimension.PARSE_CONFIDENCE.value: 0.8,
                QualityDimension.REVIEWER_CONFIDENCE.value: 1.0,
            },
        ),
    }

    @classmethod
    def get(cls, name: str) -> AcceptanceCriteria:
        """Get predefined acceptance criteria.

        Args:
            name: Name of the criteria

        Returns:
            AcceptanceCriteria

        Raises:
            ValueError: If criteria not found
        """
        if name not in cls._criteria:
            raise ValueError(
                f"Unknown acceptance criteria: {name}. "
                f"Available: {list(cls._criteria.keys())}"
            )
        return cls._criteria[name]

    @classmethod
    def create_gate(cls, name: str) -> AcceptanceGate:
        """Create an AcceptanceGate with predefined criteria.

        Args:
            name: Name of the criteria

        Returns:
            AcceptanceGate
        """
        return AcceptanceGate(cls.get(name))

    @classmethod
    def register(cls, name: str, criteria: AcceptanceCriteria) -> None:
        """Register custom acceptance criteria.

        Args:
            name: Name for the criteria
            criteria: AcceptanceCriteria to register
        """
        cls._criteria[name] = criteria

    @classmethod
    def available(cls) -> List[str]:
        """List available acceptance criteria names."""
        return list(cls._criteria.keys())

    @classmethod
    def get_all(cls) -> Dict[str, AcceptanceCriteria]:
        """Get all registered criteria."""
        return cls._criteria.copy()
