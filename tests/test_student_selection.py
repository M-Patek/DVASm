"""Tests for active learning sample selection."""

import numpy as np
import pytest

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.student.selection import (
    ActiveLearningSampler,
    DiversitySampling,
    ExpectedModelChange,
    HybridSelection,
    QueryByCommittee,
    SampleScore,
    UncertaintySampling,
    create_strategy,
)


class TestSampleScore:
    """Test SampleScore dataclass."""

    def test_creation(self):
        """Test basic creation."""
        score = SampleScore(sample_id="test_001", score=0.8)
        assert score.sample_id == "test_001"
        assert score.score == 0.8
        assert score.metadata == {}

    def test_with_metadata(self):
        """Test creation with metadata."""
        score = SampleScore(
            sample_id="test_002",
            score=0.9,
            metadata={"source": "test"},
        )
        assert score.metadata["source"] == "test"


class TestUncertaintySampling:
    """Test UncertaintySampling strategy."""

    def test_least_confidence(self):
        """Test least confidence selection."""
        strategy = UncertaintySampling(method="least_confidence")

        candidates = [
            {"id": "low_conf", "video_id": "vid1"},
            {"id": "high_conf", "video_id": "vid2"},
            {"id": "med_conf", "video_id": "vid3"},
        ]

        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,  # High uncertainty
            ),
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.95,  # Low uncertainty
            ),
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.7,
            ),
        ]

        selected = strategy.select(candidates, predictions, n_select=2)

        assert len(selected) == 2
        # Should select lowest confidence first
        assert selected[0].sample_id == "low_conf"
        assert selected[0].score > selected[1].score  # Higher uncertainty score

    def test_entropy(self):
        """Test entropy-based selection."""
        strategy = UncertaintySampling(method="entropy")

        candidates = [{"id": f"item_{i}"} for i in range(3)]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=c,
            )
            for c in [0.5, 0.9, 0.7]
        ]

        selected = strategy.select(candidates, predictions, n_select=2)

        assert len(selected) == 2
        assert all(s.score >= 0 for s in selected)


class TestDiversitySampling:
    """Test DiversitySampling strategy."""

    def test_no_embeddings_fallback(self):
        """Test behavior when no embeddings available."""
        strategy = DiversitySampling(feature_key="embedding")

        candidates = [{"id": "no_embedding_1"}, {"id": "no_embedding_2"}]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
            )
            for _ in candidates
        ]

        selected = strategy.select(candidates, predictions, n_select=2)

        # Should return empty when no embeddings
        assert selected == []

    def test_diverse_selection(self):
        """Test selection for diversity."""
        strategy = DiversitySampling(feature_key="embedding")

        # Create candidates with random embeddings
        np.random.seed(42)
        candidates = [
            {
                "id": f"item_{i}",
                "embedding": np.random.randn(10).tolist(),
            }
            for i in range(5)
        ]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
            )
            for _ in candidates
        ]

        selected = strategy.select(candidates, predictions, n_select=3)

        assert len(selected) == 3
        # All should have same score for diversity
        assert all(s.score == 1.0 for s in selected)

    def test_greedy_facility_location(self):
        """Test the facility location algorithm."""
        strategy = DiversitySampling()

        # Create embeddings with known structure
        embeddings = np.array([
            [1, 0, 0],  # Close to nothing else
            [0.9, 0.1, 0],  # Close to first
            [0, 1, 0],  # Different cluster
            [0, 0.9, 0.1],  # Close to third
            [0, 0, 1],  # Different cluster
        ])

        selected = strategy._greedy_facility_location(embeddings, n_select=3)

        assert len(selected) == 3
        # Should pick diverse points from different "clusters"
        assert len(set(selected)) == 3


class TestExpectedModelChange:
    """Test ExpectedModelChange strategy."""

    def test_selection_by_complexity(self):
        """Test selection based on complexity."""
        strategy = ExpectedModelChange()

        candidates = [{"id": f"item_{i}"} for i in range(3)]
        predictions = [
            GenerationResult(
                text="short",  # Low complexity
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,  # Low uncertainty
            ),
            GenerationResult(
                text="This is a much longer prediction with many words",  # High complexity
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,  # High uncertainty
            ),
            GenerationResult(
                text="medium length here",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.7,
            ),
        ]

        selected = strategy.select(candidates, predictions, n_select=2)

        assert len(selected) == 2
        # Highest expected change should be the complex+uncertain one
        assert selected[0].sample_id in ["item_0", "item_1"]
        # Check metadata
        assert "text_length" in selected[0].metadata
        assert "complexity" in selected[0].metadata


class TestQueryByCommittee:
    """Test QueryByCommittee strategy."""

    def test_vote_entropy(self):
        """Test vote entropy disagreement."""
        strategy = QueryByCommittee(disagreement_metric="vote_entropy")

        candidates = [{"id": f"item_{i}"} for i in range(3)]

        # Committee predictions - high disagreement on first item
        predictions = [
            [  # item_0 - high disagreement (3 different answers)
                GenerationResult(text="answer A", model_type=ModelType.TEACHER_GPT55, status=GenerationStatus.SUCCESS),
                GenerationResult(text="answer B", model_type=ModelType.TEACHER_CLAUDE, status=GenerationStatus.SUCCESS),
                GenerationResult(text="answer C", model_type=ModelType.TEACHER_TOGETHER, status=GenerationStatus.SUCCESS),
            ],
            [  # item_1 - agreement
                GenerationResult(text="same", model_type=ModelType.TEACHER_GPT55, status=GenerationStatus.SUCCESS),
                GenerationResult(text="same", model_type=ModelType.TEACHER_CLAUDE, status=GenerationStatus.SUCCESS),
                GenerationResult(text="same", model_type=ModelType.TEACHER_TOGETHER, status=GenerationStatus.SUCCESS),
            ],
            [  # item_2 - partial agreement
                GenerationResult(text="yes", model_type=ModelType.TEACHER_GPT55, status=GenerationStatus.SUCCESS),
                GenerationResult(text="yes", model_type=ModelType.TEACHER_CLAUDE, status=GenerationStatus.SUCCESS),
                GenerationResult(text="no", model_type=ModelType.TEACHER_TOGETHER, status=GenerationStatus.SUCCESS),
            ],
        ]

        selected = strategy.select(candidates, predictions, n_select=2)

        assert len(selected) == 2
        # Highest disagreement should be item_0
        assert selected[0].sample_id == "item_0"
        assert selected[0].score > selected[1].score

    def test_average_kl(self):
        """Test KL divergence disagreement."""
        strategy = QueryByCommittee(disagreement_metric="average_kl")

        candidates = [{"id": "item_1"}]
        predictions = [[
            GenerationResult(
                text="test",
                model_type=ModelType.TEACHER_GPT55,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,
            ),
            GenerationResult(
                text="test",
                model_type=ModelType.TEACHER_CLAUDE,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,
            ),
        ]]

        selected = strategy.select(candidates, predictions, n_select=1)

        assert len(selected) == 1
        assert selected[0].score >= 0  # Variance should be non-negative


class TestHybridSelection:
    """Test HybridSelection strategy."""

    def test_weighted_combination(self):
        """Test weighted strategy combination."""
        strategies = [
            (UncertaintySampling(), 0.6),
            (DiversitySampling(), 0.4),
        ]
        hybrid = HybridSelection(strategies)

        # Need embeddings for diversity
        candidates = [
            {"id": "item_1", "embedding": [1.0, 0.0]},
            {"id": "item_2", "embedding": [0.0, 1.0]},
        ]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,
            ),
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,
            ),
        ]

        selected = hybrid.select(candidates, predictions, n_select=1)

        assert len(selected) == 1
        # Should have combined score components
        assert "components" in selected[0].metadata


class TestActiveLearningSampler:
    """Test ActiveLearningSampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        strategy = UncertaintySampling()
        sampler = ActiveLearningSampler(strategy, budget=50)

        assert sampler.strategy == strategy
        assert sampler.budget == 50
        assert sampler.total_selected == 0

    def test_select_samples(self):
        """Test sample selection."""
        strategy = UncertaintySampling()
        sampler = ActiveLearningSampler(strategy, budget=10)

        candidates = [{"id": f"item_{i}"} for i in range(20)]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5 + i * 0.02,
            )
            for i in range(20)
        ]

        selected = sampler.select_samples(candidates, predictions, n_select=5)

        assert len(selected) == 5
        assert sampler.total_selected == 5
        assert len(sampler.selection_history) == 1

    def test_default_budget_selection(self):
        """Test using default budget."""
        strategy = UncertaintySampling()
        sampler = ActiveLearningSampler(strategy, budget=5)

        candidates = [{"id": f"item_{i}"} for i in range(10)]
        predictions = [
            GenerationResult(
                text="test",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.5,
            )
            for _ in candidates
        ]

        selected = sampler.select_samples(candidates, predictions)  # No n_select

        assert len(selected) == 5  # Should use budget

    def test_get_statistics(self):
        """Test getting selection statistics."""
        strategy = UncertaintySampling()
        sampler = ActiveLearningSampler(strategy, budget=10)

        # No selections yet
        stats = sampler.get_selection_statistics()
        assert stats["total_selected"] == 0

        # Make some selections
        for _ in range(3):
            candidates = [{"id": f"item_{i}"} for i in range(5)]
            predictions = [
                GenerationResult(
                    text="test",
                    model_type=ModelType.STUDENT_LOCAL,
                    status=GenerationStatus.SUCCESS,
                    confidence=0.5,
                )
                for _ in candidates
            ]
            sampler.select_samples(candidates, predictions, n_select=2)

        stats = sampler.get_selection_statistics()
        assert stats["total_selected"] == 6
        assert stats["iterations"] == 3
        assert stats["avg_per_iteration"] == 2.0


class TestCreateStrategy:
    """Test strategy factory function."""

    def test_create_uncertainty(self):
        """Test creating uncertainty strategy."""
        strategy = create_strategy("uncertainty", method="entropy")
        assert isinstance(strategy, UncertaintySampling)
        assert strategy.method == "entropy"

    def test_create_diversity(self):
        """Test creating diversity strategy."""
        strategy = create_strategy("diversity", feature_key="embed")
        assert isinstance(strategy, DiversitySampling)
        assert strategy.feature_key == "embed"

    def test_create_expected_change(self):
        """Test creating expected change strategy."""
        strategy = create_strategy("expected_change")
        assert isinstance(strategy, ExpectedModelChange)

    def test_create_committee(self):
        """Test creating committee strategy."""
        strategy = create_strategy("committee", disagreement_metric="vote_entropy")
        assert isinstance(strategy, QueryByCommittee)

    def test_create_unknown(self):
        """Test error on unknown strategy."""
        with pytest.raises(ValueError) as exc_info:
            create_strategy("unknown")

        assert "Unknown strategy" in str(exc_info.value)
        assert "uncertainty" in str(exc_info.value)
