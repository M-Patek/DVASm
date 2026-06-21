"""Tests for acceptance criteria."""

import pytest

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.quality.acceptance import (
    AcceptanceCriteria,
    AcceptanceCriteriaRegistry,
    AcceptanceGate,
    AcceptanceLevel,
)
from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityScores,
)


@pytest.fixture
def sample_annotation():
    """Create a sample annotation."""
    return Annotation(
        id="ann_001",
        video_id="vid_001",
        video_path="/path/to/video.mp4",
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=60.0,
            total_frames=1800,
        ),
        segments=[],
    )


@pytest.fixture
def high_quality_scores():
    """Create high quality scores."""
    return QualityScores(
        annotation_id="ann_001",
        video_id="vid_001",
        factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.9),
        temporal_consistency_score=DimensionScore(
            dimension=QualityDimension.TEMPORAL_CONSISTENCY, score=0.9
        ),
        object_grounding_score=DimensionScore(
            dimension=QualityDimension.OBJECT_GROUNDING, score=0.9
        ),
        action_grounding_score=DimensionScore(
            dimension=QualityDimension.ACTION_GROUNDING, score=0.9
        ),
        affordance_score=DimensionScore(dimension=QualityDimension.AFFORDANCE, score=0.9),
        robotic_usefulness_score=DimensionScore(
            dimension=QualityDimension.ROBOTIC_USEFULNESS, score=0.9
        ),
        language_clarity_score=DimensionScore(
            dimension=QualityDimension.LANGUAGE_CLARITY, score=0.9
        ),
        parse_confidence_score=DimensionScore(
            dimension=QualityDimension.PARSE_CONFIDENCE, score=0.9
        ),
        reviewer_confidence_score=DimensionScore(
            dimension=QualityDimension.REVIEWER_CONFIDENCE, score=0.9
        ),
    )


@pytest.fixture
def low_quality_scores():
    """Create low quality scores."""
    return QualityScores(
        annotation_id="ann_001",
        video_id="vid_001",
        factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.3),
        temporal_consistency_score=DimensionScore(
            dimension=QualityDimension.TEMPORAL_CONSISTENCY, score=0.3
        ),
        object_grounding_score=DimensionScore(
            dimension=QualityDimension.OBJECT_GROUNDING, score=0.3
        ),
        action_grounding_score=DimensionScore(
            dimension=QualityDimension.ACTION_GROUNDING, score=0.3
        ),
        affordance_score=DimensionScore(dimension=QualityDimension.AFFORDANCE, score=0.3),
        robotic_usefulness_score=DimensionScore(
            dimension=QualityDimension.ROBOTIC_USEFULNESS, score=0.3
        ),
        language_clarity_score=DimensionScore(
            dimension=QualityDimension.LANGUAGE_CLARITY, score=0.3
        ),
        parse_confidence_score=DimensionScore(
            dimension=QualityDimension.PARSE_CONFIDENCE, score=0.3
        ),
        reviewer_confidence_score=DimensionScore(
            dimension=QualityDimension.REVIEWER_CONFIDENCE, score=0.3
        ),
    )


class TestAcceptanceLevel:
    """Test AcceptanceLevel enum."""

    def test_level_values(self):
        """Test level values."""
        assert AcceptanceLevel.GOLD.value == "gold"
        assert AcceptanceLevel.SILVER.value == "silver"
        assert AcceptanceLevel.BRONZE.value == "bronze"
        assert AcceptanceLevel.REJECT.value == "reject"


class TestAcceptanceCriteria:
    """Test AcceptanceCriteria."""

    def test_basic_creation(self):
        """Test creating criteria."""
        criteria = AcceptanceCriteria(
            name="test",
            description="Test criteria",
        )
        assert criteria.name == "test"
        assert criteria.min_overall_score == 0.6

    def test_default_required_dimensions(self):
        """Test default required dimensions."""
        criteria = AcceptanceCriteria(name="test")
        assert QualityDimension.FACTUALITY in criteria.required_dimensions
        assert QualityDimension.LANGUAGE_CLARITY in criteria.required_dimensions

    def test_evaluate_gold_level(self, high_quality_scores):
        """Test evaluation resulting in gold level."""
        criteria = AcceptanceCriteria(name="test")
        level, failures = criteria.evaluate(high_quality_scores)
        assert level == AcceptanceLevel.GOLD
        assert len(failures) == 0

    def test_evaluate_silver_level(self):
        """Test evaluation resulting in silver level."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        # Set some dimensions to good but not all excellent
        for dim in QualityDimension:
            score = DimensionScore(dimension=dim, score=0.75)
            if dim == QualityDimension.FACTUALITY:
                scores.factuality_score = score
            elif dim == QualityDimension.TEMPORAL_CONSISTENCY:
                scores.temporal_consistency_score = score
            elif dim == QualityDimension.OBJECT_GROUNDING:
                scores.object_grounding_score = score
            elif dim == QualityDimension.ACTION_GROUNDING:
                scores.action_grounding_score = score
            elif dim == QualityDimension.AFFORDANCE:
                scores.affordance_score = score
            elif dim == QualityDimension.ROBOTIC_USEFULNESS:
                scores.robotic_usefulness_score = score
            elif dim == QualityDimension.LANGUAGE_CLARITY:
                scores.language_clarity_score = score
            elif dim == QualityDimension.PARSE_CONFIDENCE:
                scores.parse_confidence_score = score
            elif dim == QualityDimension.REVIEWER_CONFIDENCE:
                scores.reviewer_confidence_score = score

        scores._compute_aggregates()
        criteria = AcceptanceCriteria(name="test")
        level, failures = criteria.evaluate(scores)
        assert level == AcceptanceLevel.SILVER

    def test_evaluate_reject(self, low_quality_scores):
        """Test evaluation resulting in rejection."""
        criteria = AcceptanceCriteria(name="test")
        level, failures = criteria.evaluate(low_quality_scores)
        assert level == AcceptanceLevel.REJECT
        assert len(failures) > 0

    def test_evaluate_required_dimension_failure(self, high_quality_scores):
        """Test failure when required dimension is too low."""
        criteria = AcceptanceCriteria(name="test")
        # Lower factuality score - also need to lower overall to trigger reject
        high_quality_scores.factuality_score = DimensionScore(
            dimension=QualityDimension.FACTUALITY, score=0.3
        )
        # Lower most dimensions to get overall below threshold
        for dim in [
            QualityDimension.TEMPORAL_CONSISTENCY,
            QualityDimension.OBJECT_GROUNDING,
            QualityDimension.ACTION_GROUNDING,
            QualityDimension.AFFORDANCE,
            QualityDimension.ROBOTIC_USEFULNESS,
            QualityDimension.LANGUAGE_CLARITY,
            QualityDimension.PARSE_CONFIDENCE,
            QualityDimension.REVIEWER_CONFIDENCE,
        ]:
            score = DimensionScore(dimension=dim, score=0.3)
            if dim == QualityDimension.TEMPORAL_CONSISTENCY:
                high_quality_scores.temporal_consistency_score = score
            elif dim == QualityDimension.OBJECT_GROUNDING:
                high_quality_scores.object_grounding_score = score
            elif dim == QualityDimension.ACTION_GROUNDING:
                high_quality_scores.action_grounding_score = score
            elif dim == QualityDimension.AFFORDANCE:
                high_quality_scores.affordance_score = score
            elif dim == QualityDimension.ROBOTIC_USEFULNESS:
                high_quality_scores.robotic_usefulness_score = score
            elif dim == QualityDimension.LANGUAGE_CLARITY:
                high_quality_scores.language_clarity_score = score
            elif dim == QualityDimension.PARSE_CONFIDENCE:
                high_quality_scores.parse_confidence_score = score
            elif dim == QualityDimension.REVIEWER_CONFIDENCE:
                high_quality_scores.reviewer_confidence_score = score

        high_quality_scores._compute_aggregates()

        level, failures = criteria.evaluate(high_quality_scores)
        assert level == AcceptanceLevel.REJECT
        assert any("factuality" in f for f in failures)

    def test_evaluate_too_many_issues(self):
        """Test rejection due to too many issues."""
        criteria = AcceptanceCriteria(name="test", max_issues=2)
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY,
                score=0.9,
                issues=["issue1", "issue2", "issue3"],
            ),
        )
        level, failures = criteria.evaluate(scores)
        assert level == AcceptanceLevel.REJECT

    def test_to_dict(self):
        """Test serialization."""
        criteria = AcceptanceCriteria(
            name="test",
            description="Test criteria",
            min_overall_score=0.7,
        )
        data = criteria.to_dict()
        assert data["name"] == "test"
        assert data["min_overall_score"] == 0.7


class TestAcceptanceGate:
    """Test AcceptanceGate."""

    def test_basic_creation(self):
        """Test creating gate."""
        gate = AcceptanceGate()
        assert gate.criteria.name == "standard"

    def test_check_pass(self, sample_annotation, high_quality_scores):
        """Test passing check."""
        gate = AcceptanceGate()
        passed, level, failures = gate.check(sample_annotation, high_quality_scores)
        assert passed is True
        assert level in [AcceptanceLevel.GOLD, AcceptanceLevel.SILVER]
        assert len(failures) == 0

    def test_check_fail(self, sample_annotation, low_quality_scores):
        """Test failing check."""
        gate = AcceptanceGate()
        passed, level, failures = gate.check(sample_annotation, low_quality_scores)
        assert passed is False
        assert level == AcceptanceLevel.REJECT
        assert len(failures) > 0

    def test_filter_annotations(self, sample_annotation, high_quality_scores):
        """Test filtering annotations."""
        gate = AcceptanceGate()
        annotations = [sample_annotation]
        scores_map = {"ann_001": high_quality_scores}

        results = gate.filter_annotations(annotations, scores_map)
        assert AcceptanceLevel.GOLD in results or AcceptanceLevel.SILVER in results

    def test_get_acceptance_stats(self, sample_annotation, high_quality_scores):
        """Test getting acceptance stats."""
        gate = AcceptanceGate()
        annotations = [sample_annotation]
        scores_map = {"ann_001": high_quality_scores}

        stats = gate.get_acceptance_stats(annotations, scores_map)
        assert stats["total"] == 1
        assert "rates" in stats
        assert stats["rates"]["accepted"] > 0


class TestAcceptanceCriteriaRegistry:
    """Test AcceptanceCriteriaRegistry."""

    def test_get_strict(self):
        """Test getting strict criteria."""
        criteria = AcceptanceCriteriaRegistry.get("strict")
        assert criteria.name == "strict"
        assert criteria.min_overall_score == 0.75

    def test_get_standard(self):
        """Test getting standard criteria."""
        criteria = AcceptanceCriteriaRegistry.get("standard")
        assert criteria.name == "standard"

    def test_get_lenient(self):
        """Test getting lenient criteria."""
        criteria = AcceptanceCriteriaRegistry.get("lenient")
        assert criteria.name == "lenient"

    def test_get_robotics(self):
        """Test getting robotics criteria."""
        criteria = AcceptanceCriteriaRegistry.get("robotics")
        assert criteria.name == "robotics"
        # Robotics places higher emphasis on robotic_usefulness
        assert criteria.dimension_weights[QualityDimension.ROBOTIC_USEFULNESS.value] == 1.5

    def test_get_unknown_raises(self):
        """Test getting unknown criteria raises error."""
        with pytest.raises(ValueError, match="Unknown acceptance criteria"):
            AcceptanceCriteriaRegistry.get("unknown")

    def test_create_gate(self):
        """Test creating gate from registry."""
        gate = AcceptanceCriteriaRegistry.create_gate("strict")
        assert gate.criteria.name == "strict"

    def test_available(self):
        """Test listing available criteria."""
        available = AcceptanceCriteriaRegistry.available()
        assert "strict" in available
        assert "standard" in available
        assert "lenient" in available
        assert "robotics" in available

    def test_register_custom(self):
        """Test registering custom criteria."""
        custom = AcceptanceCriteria(name="custom", min_overall_score=0.9)
        AcceptanceCriteriaRegistry.register("custom", custom)

        retrieved = AcceptanceCriteriaRegistry.get("custom")
        assert retrieved.name == "custom"
        assert retrieved.min_overall_score == 0.9
