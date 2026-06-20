"""Active learning sample selection for student model training.

Implements various selection strategies to identify the most valuable
samples for labeling, reducing annotation cost while maximizing model improvement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dvas.models.base import GenerationResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SampleScore:
    """Score for a single sample."""

    sample_id: str
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SelectionStrategy(ABC):
    """Base class for sample selection strategies."""

    @abstractmethod
    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[GenerationResult],
        n_select: int,
    ) -> List[SampleScore]:
        """Select samples from candidates.

        Args:
            candidates: List of candidate samples (with metadata)
            predictions: Model predictions for candidates
            n_select: Number of samples to select

        Returns:
            List of selected samples with scores
        """
        pass


class UncertaintySampling(SelectionStrategy):
    """Select samples with highest prediction uncertainty.

    Uncertainty can be measured by:
    - Least confidence: 1 - max_prob
    - Margin: difference between top two probabilities
    - Entropy: full distribution entropy
    """

    def __init__(self, method: str = "least_confidence"):
        self.method = method

    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[GenerationResult],
        n_select: int,
    ) -> List[SampleScore]:
        """Select samples with highest uncertainty."""
        scores = []

        for candidate, pred in zip(candidates, predictions):
            # For generation, use confidence as inverse uncertainty
            confidence = pred.confidence if pred.confidence is not None else 0.5

            if self.method == "least_confidence":
                # Higher score = more uncertain
                uncertainty = 1 - confidence
            elif self.method == "entropy":
                # Binary entropy approximation
                p = max(0.001, min(0.999, confidence))
                uncertainty = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            else:
                uncertainty = 1 - confidence

            scores.append(SampleScore(
                sample_id=candidate.get("id", candidate.get("video_id", "unknown")),
                score=uncertainty,
                metadata={
                    "confidence": confidence,
                    "method": self.method,
                },
            ))

        # Sort by uncertainty (descending) and take top n
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:n_select]


class DiversitySampling(SelectionStrategy):
    """Select diverse samples to cover the data distribution.

    Uses clustering or embedding-based diversity to avoid redundant selections.
    """

    def __init__(self, feature_key: str = "embedding"):
        self.feature_key = feature_key

    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[GenerationResult],
        n_select: int,
    ) -> List[SampleScore]:
        """Select diverse samples using greedy facility location."""
        if not candidates:
            return []

        # Extract embeddings if available
        embeddings = []
        valid_indices = []

        for i, candidate in enumerate(candidates):
            if self.feature_key in candidate:
                embeddings.append(candidate[self.feature_key])
                valid_indices.append(i)

        if len(embeddings) < n_select:
            # Fall back to random selection if not enough with embeddings
            logger.warning(
                "Not enough samples with embeddings, using all available",
                available=len(embeddings),
                requested=n_select,
            )
            n_select = len(embeddings)

        if n_select == 0:
            return []

        embeddings = np.array(embeddings)

        # Greedy selection for diversity
        selected_indices = self._greedy_facility_location(embeddings, n_select)

        scores = []
        for idx in selected_indices:
            candidate_idx = valid_indices[idx]
            candidate = candidates[candidate_idx]
            scores.append(SampleScore(
                sample_id=candidate.get("id", candidate.get("video_id", "unknown")),
                score=1.0,  # All selected are equally "good" for diversity
                metadata={
                    "selection_method": "diversity",
                    "embedding_norm": float(np.linalg.norm(embeddings[idx])),
                },
            ))

        return scores

    def _greedy_facility_location(
        self,
        embeddings: np.ndarray,
        n_select: int,
    ) -> List[int]:
        """Greedy facility location for diverse selection."""
        n = len(embeddings)
        selected = []
        remaining = set(range(n))

        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        for _ in range(n_select):
            if not remaining:
                break

            # Find point that maximizes minimum distance to selected
            best_idx = None
            best_score = -1

            for idx in remaining:
                if not selected:
                    # First point: pick one with highest norm (most "distinctive")
                    score = np.linalg.norm(embeddings[idx])
                else:
                    # Score = min distance to any selected point
                    distances = [
                        np.linalg.norm(embeddings[idx] - embeddings[s])
                        for s in selected
                    ]
                    score = min(distances)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


class ExpectedModelChange(SelectionStrategy):
    """Select samples expected to cause largest model update.

    Approximates gradient magnitude as proxy for model change.
    Useful for identifying "hard" examples that drive learning.
    """

    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[GenerationResult],
        n_select: int,
    ) -> List[SampleScore]:
        """Select samples with high expected gradient magnitude."""
        scores = []

        for candidate, pred in zip(candidates, predictions):
            # Approximate gradient magnitude
            # High loss + high confidence in wrong answer = large gradient
            confidence = pred.confidence if pred.confidence is not None else 0.5

            # Use prediction length as proxy for complexity
            text_length = len(pred.text.split()) if pred.text else 0

            # Score: uncertain predictions on longer/complex inputs
            # This is a heuristic approximation
            uncertainty = 1 - confidence
            complexity = np.log1p(text_length)

            expected_change = uncertainty * complexity

            scores.append(SampleScore(
                sample_id=candidate.get("id", candidate.get("video_id", "unknown")),
                score=expected_change,
                metadata={
                    "uncertainty": uncertainty,
                    "complexity": complexity,
                    "text_length": text_length,
                },
            ))

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:n_select]


class QueryByCommittee(SelectionStrategy):
    """Select samples with highest disagreement among ensemble members.

    Requires multiple model predictions per sample.
    """

    def __init__(self, disagreement_metric: str = "vote_entropy"):
        self.disagreement_metric = disagreement_metric

    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[List[GenerationResult]],  # List of predictions per model
        n_select: int,
    ) -> List[SampleScore]:
        """Select samples with highest committee disagreement.

        Args:
            candidates: List of candidate samples
            predictions: List of prediction lists (one per committee member)
            n_select: Number to select
        """
        scores = []

        for i, candidate in enumerate(candidates):
            # Get predictions from all committee members for this sample
            committee_preds = [preds[i] for preds in predictions]

            # Calculate disagreement
            if self.disagreement_metric == "vote_entropy":
                # Treat different texts as different "votes"
                texts = [p.text for p in committee_preds]
                unique_texts = list(set(texts))
                votes = [texts.count(t) for t in unique_texts]
                probs = np.array(votes) / len(votes)
                disagreement = -np.sum(probs * np.log2(probs + 1e-10))

            elif self.disagreement_metric == "average_kl":
                # Average confidence divergence
                confidences = [
                    p.confidence if p.confidence else 0.5
                    for p in committee_preds
                ]
                _ = np.mean(confidences)  # mean_conf calculated for reference
                # Variance as proxy for KL divergence
                disagreement = np.var(confidences)

            else:
                # Default: variance in confidences
                confidences = [
                    p.confidence if p.confidence else 0.5
                    for p in committee_preds
                ]
                disagreement = np.var(confidences)

            scores.append(SampleScore(
                sample_id=candidate.get("id", candidate.get("video_id", "unknown")),
                score=disagreement,
                metadata={
                    "committee_size": len(committee_preds),
                    "disagreement_metric": self.disagreement_metric,
                },
            ))

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:n_select]


class HybridSelection(SelectionStrategy):
    """Combine multiple selection strategies with weighted scoring."""

    def __init__(
        self,
        strategies: List[Tuple[SelectionStrategy, float]],
    ):
        """Initialize with strategies and their weights.

        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies

    def select(
        self,
        candidates: List[Dict[str, Any]],
        predictions: List[GenerationResult],
        n_select: int,
    ) -> List[SampleScore]:
        """Select using weighted combination of strategies."""
        # Collect scores from all strategies
        all_scores: Dict[str, Dict] = {}

        for strategy, weight in self.strategies:
            strategy_scores = strategy.select(candidates, predictions, len(candidates))

            for score in strategy_scores:
                if score.sample_id not in all_scores:
                    all_scores[score.sample_id] = {
                        "total_score": 0.0,
                        "metadata": {"components": {}},
                    }

                all_scores[score.sample_id]["total_score"] += weight * score.score
                all_scores[score.sample_id]["metadata"]["components"][
                    strategy.__class__.__name__
                ] = score.score

        # Convert to SampleScore list
        combined_scores = [
            SampleScore(
                sample_id=sid,
                score=data["total_score"],
                metadata=data["metadata"],
            )
            for sid, data in all_scores.items()
        ]

        # Sort and select top
        combined_scores.sort(key=lambda x: x.score, reverse=True)
        return combined_scores[:n_select]


class ActiveLearningSampler:
    """Main interface for active learning sample selection.

    Manages the selection pipeline and tracks selection history.
    """

    def __init__(
        self,
        strategy: SelectionStrategy,
        budget: int = 100,
        selection_history_path: Optional[Path] = None,
    ):
        self.strategy = strategy
        self.budget = budget
        self.selection_history_path = selection_history_path
        self.selection_history: List[Dict] = []
        self.total_selected = 0

    def select_samples(
        self,
        candidates: List[Dict[str, Any]],
        predictions: Union[List[GenerationResult], List[List[GenerationResult]]],
        n_select: Optional[int] = None,
    ) -> List[SampleScore]:
        """Select samples for labeling.

        Args:
            candidates: Pool of unlabeled candidates
            predictions: Model predictions (single or committee)
            n_select: Number to select (defaults to budget)

        Returns:
            Selected samples with scores
        """
        if n_select is None:
            n_select = self.budget

        n_select = min(n_select, len(candidates))

        logger.info(
            "Selecting samples for active learning",
            pool_size=len(candidates),
            n_select=n_select,
            strategy=self.strategy.__class__.__name__,
        )

        # Run selection
        selected = self.strategy.select(candidates, predictions, n_select)

        # Record in history
        selection_record = {
            "iteration": len(self.selection_history),
            "pool_size": len(candidates),
            "n_selected": len(selected),
            "strategy": self.strategy.__class__.__name__,
            "selected_ids": [s.sample_id for s in selected],
        }
        self.selection_history.append(selection_record)
        self.total_selected += len(selected)

        # Save history if path provided
        if self.selection_history_path:
            self._save_history()

        logger.info(
            "Sample selection complete",
            selected=len(selected),
            total_selected=self.total_selected,
        )

        return selected

    def _save_history(self) -> None:
        """Save selection history to disk."""
        import json

        if self.selection_history_path:
            self.selection_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.selection_history_path, "w", encoding="utf-8") as f:
                json.dump(self.selection_history, f, indent=2)

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about selections made."""
        if not self.selection_history:
            return {"total_selected": 0, "iterations": 0}

        return {
            "total_selected": self.total_selected,
            "iterations": len(self.selection_history),
            "avg_per_iteration": self.total_selected / len(self.selection_history),
            "strategy": self.strategy.__class__.__name__,
        }


def create_strategy(
    strategy_name: str,
    **kwargs,
) -> SelectionStrategy:
    """Factory function to create selection strategies by name.

    Args:
        strategy_name: Name of strategy
        **kwargs: Strategy-specific arguments

    Returns:
        Configured SelectionStrategy
    """
    strategies = {
        "uncertainty": UncertaintySampling,
        "diversity": DiversitySampling,
        "expected_change": ExpectedModelChange,
        "committee": QueryByCommittee,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name](**kwargs)
