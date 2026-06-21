"""Tests for trend dashboard."""

from datetime import datetime, timedelta

from dvas.quality.schema import DimensionScore, QualityDimension, QualityScores
from dvas.quality.trend_dashboard import (
    DatasetQualityRollup,
    DimensionTrend,
    ModelQualityRollup,
    QualitySnapshot,
    QualityTrendDashboard,
    TimePeriod,
)


class TestTimePeriod:
    """Test TimePeriod enum."""

    def test_period_values(self):
        """Test period values."""
        assert TimePeriod.HOUR.value == "hour"
        assert TimePeriod.DAY.value == "day"
        assert TimePeriod.WEEK.value == "week"
        assert TimePeriod.MONTH.value == "month"


class TestDimensionTrend:
    """Test DimensionTrend."""

    def test_basic_creation(self):
        """Test creating dimension trend."""
        trend = DimensionTrend(dimension=QualityDimension.FACTUALITY)
        assert trend.dimension == QualityDimension.FACTUALITY
        assert trend.values == []

    def test_add_point(self):
        """Test adding data points."""
        trend = DimensionTrend(dimension=QualityDimension.FACTUALITY)
        trend.add_point(0.8)
        trend.add_point(0.85)

        assert len(trend.values) == 2
        assert trend.mean == 0.825

    def test_trend_direction_improving(self):
        """Test detecting improving trend."""
        trend = DimensionTrend(dimension=QualityDimension.FACTUALITY)
        # Add increasing values
        for i in range(6):
            trend.add_point(0.5 + i * 0.1)

        assert trend.trend_direction == "improving"

    def test_trend_direction_declining(self):
        """Test detecting declining trend."""
        trend = DimensionTrend(dimension=QualityDimension.FACTUALITY)
        # Add decreasing values
        for i in range(6):
            trend.add_point(0.9 - i * 0.1)

        assert trend.trend_direction == "declining"

    def test_to_dict(self):
        """Test serialization."""
        trend = DimensionTrend(dimension=QualityDimension.FACTUALITY)
        trend.add_point(0.8)
        data = trend.to_dict()
        assert data["dimension"] == "factuality"
        assert "statistics" in data


class TestQualitySnapshot:
    """Test QualitySnapshot."""

    def test_basic_creation(self):
        """Test creating snapshot."""
        snapshot = QualitySnapshot(
            timestamp=datetime.utcnow(),
            period=TimePeriod.DAY,
        )
        assert snapshot.period == TimePeriod.DAY

    def test_to_dict(self):
        """Test serialization."""
        snapshot = QualitySnapshot(
            timestamp=datetime.utcnow(),
            period=TimePeriod.DAY,
            total_annotations=100,
            avg_overall_score=0.75,
        )
        data = snapshot.to_dict()
        assert data["total_annotations"] == 100
        assert data["avg_overall_score"] == 0.75


class TestDatasetQualityRollup:
    """Test DatasetQualityRollup."""

    def test_basic_creation(self):
        """Test creating dataset rollup."""
        now = datetime.utcnow()
        rollup = DatasetQualityRollup(
            dataset_id="ds_001",
            dataset_name="Training Set A",
            start_time=now - timedelta(days=7),
            end_time=now,
        )
        assert rollup.dataset_id == "ds_001"
        assert rollup.dataset_name == "Training Set A"

    def test_to_dict(self):
        """Test serialization."""
        now = datetime.utcnow()
        rollup = DatasetQualityRollup(
            dataset_id="ds_001",
            dataset_name="Training Set A",
            start_time=now - timedelta(days=7),
            end_time=now,
            total_annotations=1000,
            avg_quality_score=0.8,
            pass_rate=0.9,
        )
        data = rollup.to_dict()
        assert data["dataset_id"] == "ds_001"
        assert data["metrics"]["total_annotations"] == 1000


class TestModelQualityRollup:
    """Test ModelQualityRollup."""

    def test_basic_creation(self):
        """Test creating model rollup."""
        now = datetime.utcnow()
        rollup = ModelQualityRollup(
            model_id="model_001",
            model_name="GPT-5.5",
            model_type="teacher",
            start_time=now - timedelta(days=7),
            end_time=now,
        )
        assert rollup.model_id == "model_001"
        assert rollup.model_type == "teacher"

    def test_to_dict(self):
        """Test serialization."""
        now = datetime.utcnow()
        rollup = ModelQualityRollup(
            model_id="model_001",
            model_name="GPT-5.5",
            model_type="teacher",
            start_time=now - timedelta(days=7),
            end_time=now,
            total_annotations=500,
            avg_quality_score=0.85,
        )
        data = rollup.to_dict()
        assert data["model_name"] == "GPT-5.5"
        assert data["metrics"]["avg_quality_score"] == 0.85


class TestQualityTrendDashboard:
    """Test QualityTrendDashboard."""

    def test_basic_creation(self):
        """Test creating dashboard."""
        dashboard = QualityTrendDashboard()
        assert len(dashboard._dimension_trends) == len(QualityDimension)

    def test_add_scores(self):
        """Test adding scores."""
        dashboard = QualityTrendDashboard()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.8),
        )
        dashboard.add_scores(scores)

        trend = dashboard.get_dimension_trend(QualityDimension.FACTUALITY)
        assert len(trend.values) == 1
        assert trend.values[0] == 0.8

    def test_add_batch(self):
        """Test adding batch of scores."""
        dashboard = QualityTrendDashboard()
        scores_list = [
            QualityScores(
                annotation_id=f"ann_{i:03d}",
                video_id=f"vid_{i:03d}",
                factuality_score=DimensionScore(
                    dimension=QualityDimension.FACTUALITY, score=0.7 + i * 0.05
                ),
            )
            for i in range(5)
        ]
        dashboard.add_batch(scores_list)

        trend = dashboard.get_dimension_trend(QualityDimension.FACTUALITY)
        assert len(trend.values) == 5

    def test_create_snapshot(self):
        """Test creating snapshot."""
        dashboard = QualityTrendDashboard()
        scores_list = [
            QualityScores(
                annotation_id=f"ann_{i:03d}",
                video_id=f"vid_{i:03d}",
                factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.75),
            )
            for i in range(10)
        ]
        snapshot = dashboard.create_snapshot(TimePeriod.DAY, scores_list)

        assert snapshot.total_annotations == 10
        assert snapshot.avg_overall_score > 0
        assert len(snapshot.score_distribution) == 5  # 5 buckets

    def test_create_dataset_rollup(self):
        """Test creating dataset rollup."""
        dashboard = QualityTrendDashboard()
        scores_list = [
            QualityScores(
                annotation_id=f"ann_{i:03d}",
                video_id=f"vid_{i:03d}",
                factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.8),
            )
            for i in range(100)
        ]
        rollup = dashboard.create_dataset_rollup(
            dataset_id="ds_001",
            dataset_name="Test Dataset",
            scores_list=scores_list,
        )

        assert rollup.total_annotations == 100
        assert rollup.avg_quality_score > 0

    def test_create_model_rollup(self):
        """Test creating model rollup."""
        dashboard = QualityTrendDashboard()
        scores_list = [
            QualityScores(
                annotation_id=f"ann_{i:03d}",
                video_id=f"vid_{i:03d}",
                factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.85),
            )
            for i in range(50)
        ]
        rollup = dashboard.create_model_rollup(
            model_id="model_001",
            model_name="Test Model",
            model_type="teacher",
            scores_list=scores_list,
        )

        assert rollup.total_annotations == 50
        assert rollup.model_type == "teacher"

    def test_export_dashboard_data(self):
        """Test exporting dashboard data."""
        dashboard = QualityTrendDashboard()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        dashboard.add_scores(scores)

        data = dashboard.export_dashboard_data()
        assert "generated_at" in data
        assert "dimension_trends" in data
        assert "snapshots" in data
