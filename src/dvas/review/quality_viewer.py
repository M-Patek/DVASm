"""Quality score viewer for displaying quality metrics.

Provides per-dimension score breakdown and quality trend
visualization data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.quality.schema import QualityDimension, QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DimensionBreakdown:
    """Breakdown of a single quality dimension."""

    dimension: QualityDimension
    score: float
    confidence: float
    weight: float
    status: str  # "pass", "fail", "warning"
    threshold: float = 0.5
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "weight": self.weight,
            "status": self.status,
            "threshold": self.threshold,
            "issues": self.issues,
        }


@dataclass
class QualityTrendPoint:
    """A single point in a quality trend."""

    timestamp: str
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "dimension_scores": self.dimension_scores,
        }


class QualityScoreViewer:
    """Viewer for displaying quality metrics and trends.

    Provides per-dimension score breakdown and quality trend
    visualization data for annotations.
    """

    def __init__(self):
        self._scores: Dict[str, QualityScores] = {}
        self._history: Dict[str, List[QualityScores]] = {}

    def add_scores(self, scores: QualityScores) -> None:
        """Add quality scores for viewing.

        Args:
            scores: QualityScores to add
        """
        self._scores[scores.annotation_id] = scores

        if scores.annotation_id not in self._history:
            self._history[scores.annotation_id] = []
        self._history[scores.annotation_id].append(scores)

        logger.info(
            "quality_scores_added",
            annotation_id=scores.annotation_id,
            overall_score=scores.overall_score,
        )

    def get_dimension_breakdown(self, annotation_id: str) -> List[DimensionBreakdown]:
        """Get per-dimension score breakdown for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            List of DimensionBreakdown for each dimension
        """
        scores = self._scores.get(annotation_id)
        if not scores:
            return []

        breakdowns = []
        for dim_score in scores.all_scores:
            status = "pass" if dim_score.score >= 0.5 else "fail"
            if dim_score.score >= 0.5 and dim_score.score < 0.7:
                status = "warning"

            breakdowns.append(
                DimensionBreakdown(
                    dimension=dim_score.dimension,
                    score=dim_score.score,
                    confidence=dim_score.confidence,
                    weight=dim_score.weight,
                    status=status,
                    threshold=0.5,
                    issues=dim_score.issues,
                )
            )

        return breakdowns

    def get_overall_score(self, annotation_id: str) -> Optional[float]:
        """Get the overall quality score for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            Overall score or None if not found
        """
        scores = self._scores.get(annotation_id)
        return scores.overall_score if scores else None

    def get_score_summary(self, annotation_id: str) -> Dict[str, Any]:
        """Get a summary of quality scores for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            Dict with score summary
        """
        scores = self._scores.get(annotation_id)
        if not scores:
            return {}

        breakdown = self.get_dimension_breakdown(annotation_id)
        passing = sum(1 for b in breakdown if b.status == "pass")
        failing = sum(1 for b in breakdown if b.status == "fail")
        warning = sum(1 for b in breakdown if b.status == "warning")

        return {
            "annotation_id": annotation_id,
            "overall_score": round(scores.overall_score, 4),
            "weighted_score": round(scores.weighted_score, 4),
            "dimensions_total": len(breakdown),
            "dimensions_passing": passing,
            "dimensions_failing": failing,
            "dimensions_warning": warning,
            "computed_by": scores.computed_by,
            "computed_at": scores.computed_at.isoformat(),
        }

    def get_trend_data(
        self,
        annotation_id: str,
        max_points: int = 50,
    ) -> List[QualityTrendPoint]:
        """Get quality trend data for visualization.

        Args:
            annotation_id: ID of the annotation
            max_points: Maximum number of data points to return

        Returns:
            List of QualityTrendPoint
        """
        history = self._history.get(annotation_id, [])
        if not history:
            return []

        # Sort by computed_at
        sorted_history = sorted(history, key=lambda s: s.computed_at)

        # Sample if too many points
        if len(sorted_history) > max_points:
            step = len(sorted_history) // max_points
            sorted_history = sorted_history[::step][:max_points]

        trend_points = []
        for scores in sorted_history:
            dim_scores = {}
            for dim_score in scores.all_scores:
                dim_scores[dim_score.dimension.value] = round(dim_score.score, 4)

            trend_points.append(
                QualityTrendPoint(
                    timestamp=scores.computed_at.isoformat(),
                    overall_score=scores.overall_score,
                    dimension_scores=dim_scores,
                )
            )

        return trend_points

    def get_dimension_comparison(
        self,
        annotation_ids: List[str],
    ) -> Dict[str, Any]:
        """Compare quality dimensions across multiple annotations.

        Args:
            annotation_ids: List of annotation IDs to compare

        Returns:
            Dict with dimension comparison data
        """
        result: Dict[str, Any] = {
            "annotations": [],
            "dimensions": [],
            "data": {},
        }

        for ann_id in annotation_ids:
            scores = self._scores.get(ann_id)
            if not scores:
                continue

            result["annotations"].append(ann_id)
            result["data"][ann_id] = {}

            for dim_score in scores.all_scores:
                dim_name = dim_score.dimension.value
                if dim_name not in result["dimensions"]:
                    result["dimensions"].append(dim_name)
                result["data"][ann_id][dim_name] = round(dim_score.score, 4)

        return result

    def get_lowest_dimensions(self, annotation_id: str, n: int = 3) -> List[DimensionBreakdown]:
        """Get the N lowest-scoring dimensions for an annotation.

        Args:
            annotation_id: ID of the annotation
            n: Number of dimensions to return

        Returns:
            List of lowest-scoring DimensionBreakdown
        """
        breakdown = self.get_dimension_breakdown(annotation_id)
        sorted_breakdown = sorted(breakdown, key=lambda b: b.score)
        return sorted_breakdown[:n]

    def get_passing_rate(self, annotation_id: str, threshold: float = 0.5) -> float:
        """Get the percentage of dimensions passing a threshold.

        Args:
            annotation_id: ID of the annotation
            threshold: Score threshold for passing

        Returns:
            Passing rate (0.0 to 1.0)
        """
        breakdown = self.get_dimension_breakdown(annotation_id)
        if not breakdown:
            return 0.0

        passing = sum(1 for b in breakdown if b.score >= threshold)
        return passing / len(breakdown)
