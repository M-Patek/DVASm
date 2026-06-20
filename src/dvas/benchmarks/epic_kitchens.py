"""EPIC-KITCHENS benchmark for dataset-specific evaluation.

Evaluates models on EPIC-KITCHENS action recognition tasks,
including verb/noun prediction, action segmentation, and
fine-grained action understanding.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EPICAction:
    """EPIC-KITCHENS action annotation.

    Attributes:
        verb: Action verb (e.g., "cut", "wash")
        noun: Object noun (e.g., "tomato", "knife")
        start_frame: Start frame index
        end_frame: End frame index
        participant_id: Participant identifier
        video_id: Video identifier
    """

    verb: str
    noun: str
    start_frame: int
    end_frame: int
    participant_id: str
    video_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verb": self.verb,
            "noun": self.noun,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "participant_id": self.participant_id,
            "video_id": self.video_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EPICAction":
        return cls(
            verb=data["verb"],
            noun=data["noun"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            participant_id=data["participant_id"],
            video_id=data["video_id"],
        )


class EPIKitchensBenchmark(BaseBenchmark):
    """EPIC-KITCHENS dataset benchmark.

    Evaluates models on EPIC-KITCHENS action recognition,
    including verb accuracy, noun accuracy, and action
    retrieval metrics.

    Args:
        benchmark_dir: Directory for storing benchmark data
        epic_root: Path to EPIC-KITCHENS dataset root
    """

    # EPIC-KITCHENS splits
    SPLITS = {"train", "test", "validation"}

    # Standard verb classes (top 90 verbs)
    VERB_CLASSES = [
        "take",
        "put",
        "cut",
        "wash",
        "open",
        "close",
        "pour",
        "mix",
        "move",
        "turn",
        "remove",
        "add",
        "throw",
        "dry",
        "spread",
        "peel",
        "squeeze",
        "close",
        "open",
        "stir",
    ]

    # Standard noun classes (top 300 nouns)
    NOUN_CLASSES = [
        "knife",
        "pan",
        "bowl",
        "plate",
        "cup",
        "spoon",
        "fork",
        "bottle",
        "container",
        "bag",
        "box",
        "board",
        "towel",
        "onion",
        "tomato",
        "pepper",
        "potato",
        "carrot",
        "lettuce",
    ]

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        epic_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__(benchmark_dir, "epic_kitchens")
        self.epic_root = Path(epic_root) if epic_root else None
        self.verb_classes = set(self.VERB_CLASSES)
        self.noun_classes = set(self.NOUN_CLASSES)

    def load_annotations(self, split: str = "test") -> List[EPICAction]:
        """Load EPIC-KITCHENS annotations for a split.

        Args:
            split: Dataset split (train/test/validation)

        Returns:
            List of EPICAction annotations

        Raises:
            ValueError: If split is not valid
        """
        if split not in self.SPLITS:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.SPLITS}")

        if not self.epic_root or not self.epic_root.exists():
            logger.warning("EPIC-KITCHENS root not found, returning empty annotations")
            return []

        annotations = []
        annotation_file = self.epic_root / f"annotations_{split}.json"

        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("annotations", []):
            try:
                action = EPICAction.from_dict(item)
                annotations.append(action)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid annotation: {e}")
                continue

        return annotations

    def evaluate_verb_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[EPICAction],
    ) -> float:
        """Evaluate verb prediction accuracy.

        Args:
            predictions: List of predicted verbs
            ground_truth: List of ground truth EPICAction objects

        Returns:
            Verb accuracy (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return 0.0

        correct = sum(
            1
            for pred, gt in zip(predictions, ground_truth)
            if pred.strip().lower() == gt.verb.strip().lower()
        )
        return correct / len(predictions)

    def evaluate_noun_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[EPICAction],
    ) -> float:
        """Evaluate noun prediction accuracy.

        Args:
            predictions: List of predicted nouns
            ground_truth: List of ground truth EPICAction objects

        Returns:
            Noun accuracy (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return 0.0

        correct = sum(
            1
            for pred, gt in zip(predictions, ground_truth)
            if pred.strip().lower() == gt.noun.strip().lower()
        )
        return correct / len(predictions)

    def evaluate_action_retrieval(
        self,
        predictions: List[Tuple[str, str]],
        ground_truth: List[EPICAction],
        k: int = 5,
    ) -> Dict[str, float]:
        """Evaluate action retrieval (verb + noun combined).

        Args:
            predictions: List of (verb, noun) tuples
            ground_truth: List of ground truth EPICAction objects
            k: Top-k for retrieval metrics

        Returns:
            Dictionary with recall@k and precision@k
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return {"recall@k": 0.0, "precision@k": 0.0}

        correct = 0
        for (pred_verb, pred_noun), gt in zip(predictions, ground_truth):
            if (
                pred_verb.strip().lower() == gt.verb.strip().lower()
                and pred_noun.strip().lower() == gt.noun.strip().lower()
            ):
                correct += 1

        recall = correct / len(ground_truth) if ground_truth else 0.0
        precision = correct / len(predictions) if predictions else 0.0

        return {
            "recall@k": recall,
            "precision@k": precision,
            "top1_accuracy": correct / len(predictions) if predictions else 0.0,
        }

    def run_benchmark(
        self,
        model_id: str,
        verb_predictions: List[str],
        noun_predictions: List[str],
        ground_truth: List[EPICAction],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the full EPIC-KITCHENS benchmark.

        Args:
            model_id: Identifier for the model
            verb_predictions: Predicted verbs
            noun_predictions: Predicted nouns
            ground_truth: Ground truth EPICAction objects
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with all metrics
        """
        if len(verb_predictions) != len(ground_truth):
            raise ValueError("verb_predictions and ground_truth must have same length")
        if len(noun_predictions) != len(ground_truth):
            raise ValueError("noun_predictions and ground_truth must have same length")

        logger.info(
            "Running EPIC-KITCHENS benchmark",
            model=model_id,
            n_samples=len(ground_truth),
        )

        # Compute metrics
        verb_acc = self.evaluate_verb_accuracy(verb_predictions, ground_truth)
        noun_acc = self.evaluate_noun_accuracy(noun_predictions, ground_truth)

        combined = [(v, n) for v, n in zip(verb_predictions, noun_predictions)]
        retrieval = self.evaluate_action_retrieval(combined, ground_truth)

        metrics = {
            "verb_accuracy": verb_acc,
            "noun_accuracy": noun_acc,
            **retrieval,
        }

        # Text predictions for BLEU/ROUGE
        pred_texts = [f"{v} {n}" for v, n in zip(verb_predictions, noun_predictions)]
        ref_texts = [f"{gt.verb} {gt.noun}" for gt in ground_truth]

        result = BenchmarkResult(
            benchmark_name="epic_kitchens",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=pred_texts,
            references=ref_texts,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("EPIC-KITCHENS benchmark complete", metrics=metrics)
        return result

    def get_top_verbs(self, annotations: List[EPICAction], n: int = 20) -> List[Tuple[str, int]]:
        """Get top n most frequent verbs in annotations.

        Args:
            annotations: List of EPICAction objects
            n: Number of top verbs to return

        Returns:
            List of (verb, count) tuples
        """
        from collections import Counter

        verb_counts = Counter(a.verb for a in annotations)
        return verb_counts.most_common(n)

    def get_top_nouns(self, annotations: List[EPICAction], n: int = 20) -> List[Tuple[str, int]]:
        """Get top n most frequent nouns in annotations.

        Args:
            annotations: List of EPICAction objects
            n: Number of top nouns to return

        Returns:
            List of (noun, count) tuples
        """
        from collections import Counter

        noun_counts = Counter(a.noun for a in annotations)
        return noun_counts.most_common(n)
