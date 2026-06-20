"""Human agreement benchmark.

Measures human-AI agreement metrics including Cohen's kappa,
percentage agreement, and correlation-based measures.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HumanAgreementResult:
    """Result of human-AI agreement analysis.

    Attributes:
        metric_name: Name of agreement metric
        score: Agreement score
        n_samples: Number of samples compared
        details: Additional metric details
    """

    metric_name: str
    score: float
    n_samples: int
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "n_samples": self.n_samples,
            "details": self.details,
        }


class HumanAgreementBenchmark(BaseBenchmark):
    """Human-AI agreement benchmark.

    Measures agreement between human annotators and AI model outputs
    using various statistical metrics.

    Args:
        benchmark_dir: Directory for storing benchmark data
    """

    def __init__(self, benchmark_dir: Union[str, Path]):
        super().__init__(benchmark_dir, "human_agreement")

    def compute_cohens_kappa(
        self,
        human_labels: List[str],
        ai_labels: List[str],
    ) -> float:
        """Compute Cohen's kappa for human-AI agreement.

        Measures agreement between two raters, corrected for chance.

        Args:
            human_labels: Human annotator labels
            ai_labels: AI model labels

        Returns:
            Cohen's kappa (-1 to 1, higher = more agreement)
        """
        if len(human_labels) != len(ai_labels):
            raise ValueError("human_labels and ai_labels must have same length")

        if not human_labels:
            return 0.0

        n = len(human_labels)

        # Observed agreement
        observed_agreement = sum(1 for h, a in zip(human_labels, ai_labels) if h == a) / n

        # Expected agreement by chance
        human_counts = {}
        ai_counts = {}
        for h in human_labels:
            human_counts[h] = human_counts.get(h, 0) + 1
        for a in ai_labels:
            ai_counts[a] = ai_counts.get(a, 0) + 1

        expected_agreement = 0.0
        for label in set(human_labels) | set(ai_labels):
            p_human = human_counts.get(label, 0) / n
            p_ai = ai_counts.get(label, 0) / n
            expected_agreement += p_human * p_ai

        # Cohen's kappa
        if expected_agreement >= 1.0:
            return 0.0

        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return float(np.clip(kappa, -1.0, 1.0))

    def compute_percentage_agreement(
        self,
        human_labels: List[str],
        ai_labels: List[str],
    ) -> float:
        """Compute simple percentage agreement.

        Args:
            human_labels: Human annotator labels
            ai_labels: AI model labels

        Returns:
            Percentage agreement (0-1)
        """
        if len(human_labels) != len(ai_labels):
            raise ValueError("human_labels and ai_labels must have same length")

        if not human_labels:
            return 0.0

        matches = sum(1 for h, a in zip(human_labels, ai_labels) if h == a)
        return matches / len(human_labels)

    def compute_pearson_correlation(
        self,
        human_scores: List[float],
        ai_scores: List[float],
    ) -> float:
        """Compute Pearson correlation between human and AI scores.

        Args:
            human_scores: Human annotator scores
            ai_scores: AI model scores

        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        if len(human_scores) != len(ai_scores):
            raise ValueError("human_scores and ai_scores must have same length")

        if len(human_scores) < 2:
            return 0.0

        human_array = np.array(human_scores)
        ai_array = np.array(ai_scores)

        # Check for constant values
        if np.std(human_array) == 0 or np.std(ai_array) == 0:
            return 0.0

        correlation = np.corrcoef(human_array, ai_array)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def compute_spearman_correlation(
        self,
        human_scores: List[float],
        ai_scores: List[float],
    ) -> float:
        """Compute Spearman rank correlation.

        More robust to outliers than Pearson correlation.

        Args:
            human_scores: Human annotator scores
            ai_scores: AI model scores

        Returns:
            Spearman correlation coefficient (-1 to 1)
        """
        if len(human_scores) != len(ai_scores):
            raise ValueError("human_scores and ai_scores must have same length")

        if len(human_scores) < 2:
            return 0.0

        # Convert to ranks
        human_ranks = np.argsort(np.argsort(human_scores))
        ai_ranks = np.argsort(np.argsort(ai_scores))

        return self.compute_pearson_correlation(human_ranks.tolist(), ai_ranks.tolist())

    def compute_semantic_similarity(
        self,
        human_texts: List[str],
        ai_texts: List[str],
    ) -> float:
        """Compute semantic similarity using word overlap.

        Args:
            human_texts: Human annotations
            ai_texts: AI predictions

        Returns:
            Average semantic similarity (0-1)
        """
        if len(human_texts) != len(ai_texts):
            raise ValueError("human_texts and ai_texts must have same length")

        if not human_texts:
            return 0.0

        similarities = []
        for h, a in zip(human_texts, ai_texts):
            h_words = set(h.lower().split())
            a_words = set(a.lower().split())

            if not h_words or not a_words:
                similarities.append(0.0)
                continue

            intersection = len(h_words & a_words)
            union = len(h_words | a_words)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)

        return float(np.mean(similarities))

    def compute_confusion_matrix(
        self,
        human_labels: List[str],
        ai_labels: List[str],
    ) -> Dict[str, Dict[str, int]]:
        """Compute confusion matrix for human-AI agreement.

        Args:
            human_labels: Human annotator labels
            ai_labels: AI model labels

        Returns:
            Confusion matrix as nested dictionary
        """
        if len(human_labels) != len(ai_labels):
            raise ValueError("human_labels and ai_labels must have same length")

        matrix: Dict[str, Dict[str, int]] = {}

        for h, a in zip(human_labels, ai_labels):
            if h not in matrix:
                matrix[h] = {}
            if a not in matrix[h]:
                matrix[h][a] = 0
            matrix[h][a] += 1

        return matrix

    def compute_per_class_agreement(
        self,
        human_labels: List[str],
        ai_labels: List[str],
    ) -> Dict[str, float]:
        """Compute per-class agreement rates.

        Args:
            human_labels: Human annotator labels
            ai_labels: AI model labels

        Returns:
            Dictionary mapping class to agreement rate
        """
        if len(human_labels) != len(ai_labels):
            raise ValueError("human_labels and ai_labels must have same length")

        class_counts: Dict[str, int] = {}
        class_correct: Dict[str, int] = {}

        for h, a in zip(human_labels, ai_labels):
            if h not in class_counts:
                class_counts[h] = 0
                class_correct[h] = 0
            class_counts[h] += 1
            if h == a:
                class_correct[h] += 1

        return {
            cls: class_correct.get(cls, 0) / count
            for cls, count in class_counts.items()
        }

    def run_benchmark(
        self,
        model_id: str,
        human_labels: List[str],
        ai_labels: List[str],
        human_scores: Optional[List[float]] = None,
        ai_scores: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the full human agreement benchmark.

        Args:
            model_id: Identifier for the model
            human_labels: Human annotator labels
            ai_labels: AI model labels
            human_scores: Optional human scores for correlation
            ai_scores: Optional AI scores for correlation
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with agreement metrics
        """
        if len(human_labels) != len(ai_labels):
            raise ValueError("human_labels and ai_labels must have same length")

        logger.info(
            "Running human agreement benchmark",
            model=model_id,
            n_samples=len(human_labels),
        )

        # Compute metrics
        cohens_kappa = self.compute_cohens_kappa(human_labels, ai_labels)
        percentage_agreement = self.compute_percentage_agreement(human_labels, ai_labels)
        semantic_similarity = self.compute_semantic_similarity(human_labels, ai_labels)

        metrics = {
            "cohens_kappa": cohens_kappa,
            "percentage_agreement": percentage_agreement,
            "semantic_similarity": semantic_similarity,
            "n_samples": len(human_labels),
        }

        # Add correlation metrics if scores provided
        if human_scores and ai_scores and len(human_scores) == len(ai_scores):
            pearson = self.compute_pearson_correlation(human_scores, ai_scores)
            spearman = self.compute_spearman_correlation(human_scores, ai_scores)
            metrics["pearson_correlation"] = pearson
            metrics["spearman_correlation"] = spearman

        # Per-class agreement
        per_class = self.compute_per_class_agreement(human_labels, ai_labels)
        for cls, agreement in per_class.items():
            metrics[f"class_agreement_{cls}"] = agreement

        result = BenchmarkResult(
            benchmark_name="human_agreement",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=ai_labels,
            references=human_labels,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Human agreement benchmark complete", metrics=metrics)
        return result

    def analyze_agreement_by_difficulty(
        self,
        human_labels: List[str],
        ai_labels: List[str],
        difficulties: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze agreement broken down by difficulty level.

        Args:
            human_labels: Human annotator labels
            ai_labels: AI model labels
            difficulties: Difficulty level for each sample

        Returns:
            Dictionary mapping difficulty to agreement metrics
        """
        if not (len(human_labels) == len(ai_labels) == len(difficulties)):
            raise ValueError("All inputs must have same length")

        # Group by difficulty
        by_difficulty: Dict[str, List[Tuple[str, str]]] = {}
        for h, a, d in zip(human_labels, ai_labels, difficulties):
            if d not in by_difficulty:
                by_difficulty[d] = []
            by_difficulty[d].append((h, a))

        results = {}
        for difficulty, pairs in by_difficulty.items():
            h_labels, a_labels = zip(*pairs)
            results[difficulty] = {
                "cohens_kappa": self.compute_cohens_kappa(list(h_labels), list(a_labels)),
                "percentage_agreement": self.compute_percentage_agreement(list(h_labels), list(a_labels)),
                "n_samples": len(pairs),
            }

        return results
