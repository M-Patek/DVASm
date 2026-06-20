"""Annotation consistency benchmark.

Measures inter-annotator consistency and model stability
across multiple runs and annotators.
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
class ConsistencyResult:
    """Consistency measurement result.

    Attributes:
        metric_name: Name of consistency metric
        score: Consistency score (0-1, higher = more consistent)
        n_annotations: Number of annotations compared
        details: Additional metric details
    """

    metric_name: str
    score: float
    n_annotations: int
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "n_annotations": self.n_annotations,
            "details": self.details,
        }


class AnnotationConsistencyBenchmark(BaseBenchmark):
    """Annotation consistency benchmark.

    Measures inter-annotator consistency and model stability
    across multiple runs and annotators.

    Args:
        benchmark_dir: Directory for storing benchmark data
    """

    def __init__(self, benchmark_dir: Union[str, Path]):
        super().__init__(benchmark_dir, "annotation_consistency")

    def compute_fleiss_kappa(
        self,
        annotations: List[List[str]],
        categories: Optional[List[str]] = None,
    ) -> float:
        """Compute Fleiss' kappa for inter-annotator agreement.

        Measures agreement among multiple annotators rating
        the same items into categories.

        Args:
            annotations: List of annotation lists (one per item, each containing annotations from all annotators)
            categories: List of possible categories (auto-detected if None)

        Returns:
            Fleiss' kappa score (-1 to 1, higher = more consistent)
        """
        if not annotations or not annotations[0]:
            return 0.0

        n_items = len(annotations)
        n_annotators = len(annotations[0])

        if n_annotators < 2:
            return 0.0

        # Auto-detect categories
        if categories is None:
            categories = sorted(set(a for item in annotations for a in item))

        n_categories = len(categories)
        if n_categories < 2:
            return 0.0

        # Count annotations per category per item
        category_counts = []
        for item in annotations:
            counts = {cat: 0 for cat in categories}
            for annotation in item:
                if annotation in counts:
                    counts[annotation] += 1
            category_counts.append([counts[cat] for cat in categories])

        category_counts = np.array(category_counts)

        # Compute P (proportion of agreement)
        p = np.sum(category_counts * (category_counts - 1), axis=1)
        p = np.sum(p) / (n_items * n_annotators * (n_annotators - 1))

        # Compute Pe (expected agreement by chance)
        category_proportions = np.sum(category_counts, axis=0) / (n_items * n_annotators)
        pe = np.sum(category_proportions ** 2)

        # Compute kappa
        if pe >= 1.0:
            return 0.0

        kappa = (p - pe) / (1 - pe)
        return float(np.clip(kappa, -1.0, 1.0))

    def compute_krippendorff_alpha(
        self,
        annotations: List[List[str]],
        categories: Optional[List[str]] = None,
    ) -> float:
        """Compute Krippendorff's alpha for inter-annotator agreement.

        More robust than Fleiss' kappa for missing data.

        Args:
            annotations: List of annotation lists
            categories: List of possible categories

        Returns:
            Krippendorff's alpha (0-1, higher = more consistent)
        """
        if not annotations or not annotations[0]:
            return 0.0

        # Simplified implementation - falls back to Fleiss' kappa for nominal data
        return max(0.0, self.compute_fleiss_kappa(annotations, categories))

    def compute_pairwise_iou(
        self,
        annotations_a: List[str],
        annotations_b: List[str],
    ) -> float:
        """Compute pairwise IoU (Intersection over Union) for text annotations.

        Args:
            annotations_a: First set of annotations
            annotations_b: Second set of annotations

        Returns:
            Average IoU score
        """
        if len(annotations_a) != len(annotations_b):
            raise ValueError("Annotation lists must have same length")

        if not annotations_a:
            return 0.0

        ious = []
        for a, b in zip(annotations_a, annotations_b):
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())

            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

        return float(np.mean(ious))

    def compute_stability_score(
        self,
        run1_annotations: List[str],
        run2_annotations: List[str],
    ) -> float:
        """Compute stability score between two model runs.

        Measures how consistent model outputs are across runs.

        Args:
            run1_annotations: Annotations from first run
            run2_annotations: Annotations from second run

        Returns:
            Stability score (0-1, higher = more stable)
        """
        return self.compute_pairwise_iou(run1_annotations, run2_annotations)

    def compute_semantic_consistency(
        self,
        annotations: List[List[str]],
    ) -> float:
        """Compute semantic consistency using pairwise similarity.

        Args:
            annotations: List of annotation lists (one per item, each containing annotations from all annotators)

        Returns:
            Semantic consistency score (0-1)
        """
        if not annotations or not annotations[0]:
            return 0.0

        n_annotators = len(annotations[0])
        if n_annotators < 2:
            return 1.0

        # Extract annotations per annotator
        annotator_annotations: List[List[str]] = [[] for _ in range(n_annotators)]
        for item_annotations in annotations:
            for i, annotation in enumerate(item_annotations):
                annotator_annotations[i].append(annotation)

        # Compute pairwise consistency
        consistencies = []
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                consistency = self.compute_pairwise_iou(
                    annotator_annotations[i],
                    annotator_annotations[j],
                )
                consistencies.append(consistency)

        return float(np.mean(consistencies)) if consistencies else 0.0

    def compute_temporal_consistency(
        self,
        annotations: List[str],
        window_size: int = 3,
    ) -> float:
        """Compute temporal consistency for sequential annotations.

        Measures how consistent annotations are over time.

        Args:
            annotations: Sequential annotations
            window_size: Window size for consistency check

        Returns:
            Temporal consistency score (0-1)
        """
        if len(annotations) < window_size:
            return 1.0

        consistencies = []
        for i in range(len(annotations) - window_size + 1):
            window = annotations[i:i + window_size]
            # Check if all annotations in window are similar
            for j in range(len(window) - 1):
                words_a = set(window[j].lower().split())
                words_b = set(window[j + 1].lower().split())
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                iou = intersection / union if union > 0 else 0.0
                consistencies.append(iou)

        return float(np.mean(consistencies)) if consistencies else 0.0

    def run_benchmark(
        self,
        model_id: str,
        annotations: List[List[str]],
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the full annotation consistency benchmark.

        Args:
            model_id: Identifier for the model/annotator set
            annotations: List of annotation lists (one per item)
            categories: Optional list of possible categories
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with consistency metrics
        """
        if not annotations or not annotations[0]:
            raise ValueError("At least one annotation set required")

        n_annotators = len(annotations[0])
        logger.info(
            "Running annotation consistency benchmark",
            model=model_id,
            n_items=len(annotations),
            n_annotators=n_annotators,
        )

        # Compute metrics
        fleiss_kappa = self.compute_fleiss_kappa(annotations, categories)
        krippendorff_alpha = self.compute_krippendorff_alpha(annotations, categories)
        semantic_consistency = self.compute_semantic_consistency(annotations)

        # Flatten for text-based metrics
        all_annotations = [a for item in annotations for a in item]
        references = all_annotations  # Self-referential

        metrics = {
            "fleiss_kappa": fleiss_kappa,
            "krippendorff_alpha": krippendorff_alpha,
            "semantic_consistency": semantic_consistency,
            "n_items": len(annotations),
            "n_annotators": n_annotators,
        }

        # Add temporal consistency if annotations are sequential
        if len(annotations) >= 3:
            # Use first annotator's annotations for temporal consistency
            temporal = self.compute_temporal_consistency([a[0] for a in annotations])
            metrics["temporal_consistency"] = temporal

        result = BenchmarkResult(
            benchmark_name="annotation_consistency",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=all_annotations,
            references=references,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Annotation consistency benchmark complete", metrics=metrics)
        return result

    def compare_annotators(
        self,
        annotations: List[List[str]],
    ) -> Dict[Tuple[int, int], float]:
        """Compare all pairs of annotators.

        Args:
            annotations: List of annotation lists

        Returns:
            Dictionary mapping (i, j) to pairwise agreement score
        """
        if not annotations or not annotations[0]:
            return {}

        n_annotators = len(annotations[0])
        agreements = {}

        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                annotator_i = [item[i] for item in annotations]
                annotator_j = [item[j] for item in annotations]
                agreement = self.compute_pairwise_iou(annotator_i, annotator_j)
                agreements[(i, j)] = agreement

        return agreements
