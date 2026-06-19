"""Tests for quality schema and scores."""

import pytest
from datetime import datetime

from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityProfile,
    QualityScores,
    QualityThresholds,
)


class TestQualityDimension:
    """Test QualityDimension enum."""

    def test_dimension_values(self):
        """Test that all dimensions have correct values."""
        assert QualityDimension.FACTUALITY.value == "factuality"
        assert QualityDimension.TEMPORAL_CONSISTENCY.value == "temporal_consistency"
        assert QualityDimension.OBJECT_GROUNDING.value == "object_grounding"
        assert QualityDimension.ACTION_GROUNDING.value == "action_grounding"
        assert QualityDimension.AFFORDANCE.value == "affordance"
        assert QualityDimension.ROBOTIC_USEFULNESS.value == "robotic_usefulness"
        assert QualityDimension.LANGUAGE_CLARITY.value == "language_clarity"
        assert QualityDimension.PARSE_CONFIDENCE.value == "parse_confidence"
        assert QualityDimension.REVIEWER_CONFIDENCE.value == "reviewer_confidence"


class TestDimensionScore:
    """Test DimensionScore dataclass."""

    def test_basic_creation(self):
        """Test creating a dimension score."""
        score = DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=0.85,
            confidence=0.9,
            weight=1.5,
        )
        assert score.dimension == QualityDimension.FACTUALITY
        assert score.score == 0.85
        assert score.confidence == 0.9
        assert score.weight == 1.5

    def test_score_clamping(self):
        """Test that scores are clamped to [0, 1]."""
        score = DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=1.5,
        )
        assert score.score == 1.0

        score = DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=-0.5,
        )
        assert score.score == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        score = DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=0.85,
            confidence=0.9,
            details={"reason": "test"},
            issues=["issue1"],
        )
        data = score.to_dict()
        assert data["dimension"] == "factuality"
        assert data["score"] == 0.85
        assert data["confidence"] == 0.9
        assert data["details"] == {"reason": "test"}
        assert data["issues"] == ["issue1"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "dimension": "factuality",
            "score": 0.85,
            "confidence": 0.9,
            "weight": 1.5,
            "details": {},
            "issues": [],
        }
        score = DimensionScore.from_dict(data)
        assert score.dimension == QualityDimension.FACTUALITY
        assert score.score == 0.85


class TestQualityScores:
    """Test QualityScores dataclass."""

    def test_basic_creation(self):
        """Test creating quality scores."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        assert scores.annotation_id == "ann_001"
        assert scores.video_id == "vid_001"
        assert scores.overall_score >= 0.0
        assert scores.weighted_score >= 0.0

    def test_aggregate_scores_computed(self):
        """Test that aggregate scores are computed."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        # Set all dimensions to known values
        for dim in QualityDimension:
            score = DimensionScore(dimension=dim, score=0.85)
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
        # Should compute average of all 9 dimensions
        assert scores.overall_score == pytest.approx(0.85, abs=0.01)

    def test_failed_dimensions(self):
        """Test detecting failed dimensions."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY, score=0.4  # Below 0.5
            ),
            language_clarity_score=DimensionScore(
                dimension=QualityDimension.LANGUAGE_CLARITY, score=0.9
            ),
        )
        failed = scores.failed_dimensions
        assert QualityDimension.FACTUALITY in failed
        assert QualityDimension.LANGUAGE_CLARITY not in failed

    def test_all_issues(self):
        """Test collecting all issues."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY,
                score=0.8,
                issues=["issue1", "issue2"],
            ),
            language_clarity_score=DimensionScore(
                dimension=QualityDimension.LANGUAGE_CLARITY,
                score=0.9,
                issues=["issue3"],
            ),
        )
        issues = scores.all_issues
        assert "issue1" in issues
        assert "issue2" in issues
        assert "issue3" in issues

    def test_get_score(self):
        """Test getting score for specific dimension."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY, score=0.8
            ),
        )
        score = scores.get_score(QualityDimension.FACTUALITY)
        assert score.score == 0.8

    def test_to_dict(self):
        """Test serialization."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY, score=0.8
            ),
        )
        data = scores.to_dict()
        assert data["annotation_id"] == "ann_001"
        assert data["video_id"] == "vid_001"
        assert "dimensions" in data
        assert "overall_score" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "annotation_id": "ann_001",
            "video_id": "vid_001",
            "dimensions": {
                "factuality": {
                    "dimension": "factuality",
                    "score": 0.8,
                    "confidence": 1.0,
                    "weight": 1.0,
                    "details": {},
                    "issues": [],
                }
            },
            "overall_score": 0.8,
            "weighted_score": 0.8,
            "computed_at": datetime.utcnow().isoformat(),
            "computed_by": "test",
            "version": "1.0",
        }
        scores = QualityScores.from_dict(data)
        assert scores.annotation_id == "ann_001"
        assert scores.factuality_score.score == 0.8


class TestQualityThresholds:
    """Test QualityThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = QualityThresholds()
        assert thresholds.factuality_min == 0.7
        assert thresholds.overall_min == 0.6

    def test_check_score_pass(self):
        """Test checking scores that pass."""
        thresholds = QualityThresholds()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        # Set all scores above thresholds
        for dim in QualityDimension:
            score = DimensionScore(dimension=dim, score=0.9)
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
        passed, failures = thresholds.check_score(scores)
        assert passed is True
        assert len(failures) == 0

    def test_check_score_fail(self):
        """Test checking scores that fail."""
        thresholds = QualityThresholds()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
            factuality_score=DimensionScore(
                dimension=QualityDimension.FACTUALITY, score=0.3  # Below threshold
            ),
        )
        scores._compute_aggregates()
        passed, failures = thresholds.check_score(scores)
        assert passed is False
        assert len(failures) > 0


class TestQualityProfile:
    """Test QualityProfile presets."""

    def test_strict_profile(self):
        """Test strict quality profile."""
        profile = QualityProfile.STRICT.value
        assert profile.factuality_min == 0.8
        assert profile.max_failed_dimensions == 1

    def test_standard_profile(self):
        """Test standard quality profile."""
        profile = QualityProfile.STANDARD.value
        assert profile.factuality_min == 0.7
        assert profile.max_failed_dimensions == 2

    def test_lenient_profile(self):
        """Test lenient quality profile."""
        profile = QualityProfile.LENIENT.value
        assert profile.factuality_min == 0.5
        assert profile.max_failed_dimensions == 3

    def test_robotics_profile(self):
        """Test robotics quality profile."""
        profile = QualityProfile.ROBOTICS.value
        assert profile.robotic_usefulness_min == 0.8
        assert profile.action_grounding_min == 0.7
