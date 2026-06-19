"""Quality trend dashboard data structures and metrics.

Tracks quality metrics over time and provides rollup data
for dashboard consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.quality.schema import QualityDimension, QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class TimePeriod(str, Enum):
    """Time periods for trend aggregation."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class DimensionTrend:
    """Trend data for a single quality dimension."""

    dimension: QualityDimension
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    trend_direction: str = "stable"  # "improving", "declining", "stable"

    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a data point."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.utcnow())
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Compute statistics from values."""
        if not self.values:
            return

        import statistics

        self.mean = statistics.mean(self.values)
        if len(self.values) > 1:
            try:
                self.std = statistics.stdev(self.values)
            except statistics.StatisticsError:
                self.std = 0.0
        self.min_value = min(self.values)
        self.max_value = max(self.values)

        # Compute trend direction
        if len(self.values) >= 3:
            # Simple linear trend: compare first half to second half
            mid = len(self.values) // 2
            first_half = statistics.mean(self.values[:mid])
            second_half = statistics.mean(self.values[mid:])

            diff = second_half - first_half
            threshold = 0.05  # 5% change threshold

            if diff > threshold:
                self.trend_direction = "improving"
            elif diff < -threshold:
                self.trend_direction = "declining"
            else:
                self.trend_direction = "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "values": self.values,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "statistics": {
                "mean": self.mean,
                "std": self.std,
                "min": self.min_value,
                "max": self.max_value,
                "trend_direction": self.trend_direction,
            },
        }


@dataclass
class QualitySnapshot:
    """Quality metrics snapshot at a point in time."""

    timestamp: datetime
    period: TimePeriod

    # Aggregate metrics
    total_annotations: int = 0
    avg_overall_score: float = 0.0
    pass_rate: float = 0.0

    # Dimension averages
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Distribution
    score_distribution: Dict[str, int] = field(default_factory=dict)

    # Issues
    top_issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "period": self.period.value,
            "total_annotations": self.total_annotations,
            "avg_overall_score": self.avg_overall_score,
            "pass_rate": self.pass_rate,
            "dimension_scores": self.dimension_scores,
            "score_distribution": self.score_distribution,
            "top_issues": self.top_issues,
        }


@dataclass
class DatasetQualityRollup:
    """Quality rollup for a specific dataset."""

    dataset_id: str
    dataset_name: str

    # Time range
    start_time: datetime
    end_time: datetime

    # Metrics
    total_annotations: int = 0
    avg_quality_score: float = 0.0
    pass_rate: float = 0.0

    # By dimension
    dimension_averages: Dict[str, float] = field(default_factory=dict)

    # Trends
    score_trend: str = "stable"
    trend_change_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "metrics": {
                "total_annotations": self.total_annotations,
                "avg_quality_score": self.avg_quality_score,
                "pass_rate": self.pass_rate,
            },
            "dimension_averages": self.dimension_averages,
            "trends": {
                "direction": self.score_trend,
                "change_percent": self.trend_change_percent,
            },
        }


@dataclass
class ModelQualityRollup:
    """Quality rollup for a specific model."""

    model_id: str
    model_name: str
    model_type: str  # "teacher", "student"

    # Time range
    start_time: datetime
    end_time: datetime

    # Metrics
    total_annotations: int = 0
    avg_quality_score: float = 0.0
    pass_rate: float = 0.0

    # Dimension performance
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Comparison to baseline
    vs_baseline_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "metrics": {
                "total_annotations": self.total_annotations,
                "avg_quality_score": self.avg_quality_score,
                "pass_rate": self.pass_rate,
            },
            "dimension_scores": self.dimension_scores,
            "vs_baseline_percent": self.vs_baseline_percent,
        }


class QualityTrendDashboard:
    """Dashboard for tracking quality trends over time.

    Aggregates quality metrics by time period, dataset, and model,
    providing data for visualization and analysis.
    """

    def __init__(self):
        """Initialize the trend dashboard."""
        self._snapshots: List[QualitySnapshot] = []
        self._dimension_trends: Dict[QualityDimension, DimensionTrend] = {
            dim: DimensionTrend(dimension=dim) for dim in QualityDimension
        }
        self._dataset_rollups: Dict[str, DatasetQualityRollup] = {}
        self._model_rollups: Dict[str, ModelQualityRollup] = {}

    def add_scores(
        self,
        scores: QualityScores,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add quality scores to the dashboard.

        Args:
            scores: Quality scores to add
            dataset_id: Optional dataset identifier
            model_id: Optional model identifier
            timestamp: Optional timestamp (defaults to now)
        """
        ts = timestamp or datetime.utcnow()

        # Update dimension trends
        for dim_score in scores.all_scores:
            self._dimension_trends[dim_score.dimension].add_point(
                dim_score.score, ts
            )

        logger.debug(
            "added_scores_to_dashboard",
            annotation_id=scores.annotation_id,
            dataset_id=dataset_id,
            model_id=model_id,
        )

    def add_batch(
        self,
        scores_list: List[QualityScores],
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """Add multiple quality scores."""
        for scores in scores_list:
            self.add_scores(scores, dataset_id, model_id)

    def create_snapshot(
        self,
        period: TimePeriod,
        scores_list: Optional[List[QualityScores]] = None,
        timestamp: Optional[datetime] = None,
    ) -> QualitySnapshot:
        """Create a quality snapshot.

        Args:
            period: Time period for the snapshot
            scores_list: Optional list of scores to include
            timestamp: Optional timestamp

        Returns:
            QualitySnapshot
        """
        ts = timestamp or datetime.utcnow()

        if scores_list is None:
            # Use trend data
            scores_list = []

        snapshot = QualitySnapshot(
            timestamp=ts,
            period=period,
            total_annotations=len(scores_list),
        )

        if scores_list:
            # Compute averages
            snapshot.avg_overall_score = sum(
                s.overall_score for s in scores_list
            ) / len(scores_list)

            # Pass rate (assuming 0.6 threshold)
            passed = sum(1 for s in scores_list if s.overall_score >= 0.6)
            snapshot.pass_rate = passed / len(scores_list)

            # Dimension averages
            for dim in QualityDimension:
                dim_scores = [s.get_score(dim).score for s in scores_list]
                snapshot.dimension_scores[dim.value] = sum(dim_scores) / len(dim_scores)

            # Score distribution
            buckets = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
            snapshot.score_distribution = {b: 0 for b in buckets}
            for s in scores_list:
                score = s.overall_score
                if score < 0.2:
                    snapshot.score_distribution["0.0-0.2"] += 1
                elif score < 0.4:
                    snapshot.score_distribution["0.2-0.4"] += 1
                elif score < 0.6:
                    snapshot.score_distribution["0.4-0.6"] += 1
                elif score < 0.8:
                    snapshot.score_distribution["0.6-0.8"] += 1
                else:
                    snapshot.score_distribution["0.8-1.0"] += 1

            # Top issues
            issue_counts: Dict[str, int] = {}
            for s in scores_list:
                for issue in s.all_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

            snapshot.top_issues = [
                {"issue": issue, "count": count}
                for issue, count in sorted(
                    issue_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

        self._snapshots.append(snapshot)
        return snapshot

    def create_dataset_rollup(
        self,
        dataset_id: str,
        dataset_name: str,
        scores_list: List[QualityScores],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> DatasetQualityRollup:
        """Create a quality rollup for a dataset.

        Args:
            dataset_id: Dataset identifier
            dataset_name: Human-readable dataset name
            scores_list: List of quality scores
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DatasetQualityRollup
        """
        now = datetime.utcnow()
        rollup = DatasetQualityRollup(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            start_time=start_time or now - timedelta(days=7),
            end_time=end_time or now,
            total_annotations=len(scores_list),
        )

        if scores_list:
            rollup.avg_quality_score = sum(
                s.overall_score for s in scores_list
            ) / len(scores_list)

            passed = sum(1 for s in scores_list if s.overall_score >= 0.6)
            rollup.pass_rate = passed / len(scores_list)

            # Dimension averages
            for dim in QualityDimension:
                dim_scores = [s.get_score(dim).score for s in scores_list]
                rollup.dimension_averages[dim.value] = sum(dim_scores) / len(dim_scores)

        self._dataset_rollups[dataset_id] = rollup
        return rollup

    def create_model_rollup(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        scores_list: List[QualityScores],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ModelQualityRollup:
        """Create a quality rollup for a model.

        Args:
            model_id: Model identifier
            model_name: Human-readable model name
            model_type: "teacher" or "student"
            scores_list: List of quality scores
            start_time: Start of time range
            end_time: End of time range

        Returns:
            ModelQualityRollup
        """
        now = datetime.utcnow()
        rollup = ModelQualityRollup(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            start_time=start_time or now - timedelta(days=7),
            end_time=end_time or now,
            total_annotations=len(scores_list),
        )

        if scores_list:
            rollup.avg_quality_score = sum(
                s.overall_score for s in scores_list
            ) / len(scores_list)

            passed = sum(1 for s in scores_list if s.overall_score >= 0.6)
            rollup.pass_rate = passed / len(scores_list)

            # Dimension scores
            for dim in QualityDimension:
                dim_scores = [s.get_score(dim).score for s in scores_list]
                rollup.dimension_scores[dim.value] = sum(dim_scores) / len(dim_scores)

        self._model_rollups[model_id] = rollup
        return rollup

    def get_dimension_trend(self, dimension: QualityDimension) -> DimensionTrend:
        """Get trend data for a specific dimension."""
        return self._dimension_trends[dimension]

    def get_all_dimension_trends(self) -> Dict[QualityDimension, DimensionTrend]:
        """Get all dimension trends."""
        return self._dimension_trends

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export all dashboard data for visualization.

        Returns:
            Dictionary with all dashboard data
        """
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "dimension_trends": {
                dim.value: trend.to_dict()
                for dim, trend in self._dimension_trends.items()
            },
            "snapshots": [s.to_dict() for s in self._snapshots],
            "dataset_rollups": {
                k: v.to_dict() for k, v in self._dataset_rollups.items()
            },
            "model_rollups": {
                k: v.to_dict() for k, v in self._model_rollups.items()
            },
        }

    def export_json(self, filepath: str) -> None:
        """Export dashboard data to JSON file."""
        import json

        data = self.export_dashboard_data()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("dashboard_data_exported", filepath=filepath)
