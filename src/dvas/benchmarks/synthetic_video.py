"""Synthetic video benchmark for controlled evaluation.

Evaluates models on synthetic video data with known ground truth,
enabling precise measurement of model capabilities under
controlled conditions.
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
class SyntheticVideo:
    """Synthetic video metadata.

    Attributes:
        video_id: Video identifier
        scene_type: Type of synthetic scene
        num_objects: Number of objects in the scene
        actions: List of actions performed
        duration: Video duration in seconds
        complexity: Complexity score (1-10)
    """

    video_id: str
    scene_type: str
    num_objects: int
    actions: List[str]
    duration: float
    complexity: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "scene_type": self.scene_type,
            "num_objects": self.num_objects,
            "actions": self.actions,
            "duration": self.duration,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticVideo":
        return cls(
            video_id=data["video_id"],
            scene_type=data["scene_type"],
            num_objects=data["num_objects"],
            actions=data["actions"],
            duration=data["duration"],
            complexity=data["complexity"],
        )


class SyntheticVideoBenchmark(BaseBenchmark):
    """Synthetic video benchmark.

    Evaluates models on synthetic video data with known ground truth.
    Supports controlled evaluation across different complexity levels
    and scene types.

    Args:
        benchmark_dir: Directory for storing benchmark data
        dataset_root: Path to synthetic dataset root
    """

    # Scene types
    SCENE_TYPES = {
        "simple_interaction",
        "multi_object",
        "occlusion",
        "lighting_variation",
        "camera_motion",
        "temporal_reasoning",
    }

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__(benchmark_dir, "synthetic_video")
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def load_synthetic_videos(self, scene_type: Optional[str] = None) -> List[SyntheticVideo]:
        """Load synthetic video metadata.

        Args:
            scene_type: Optional filter by scene type

        Returns:
            List of SyntheticVideo objects
        """
        if not self.dataset_root or not self.dataset_root.exists():
            logger.warning("Dataset root not found, returning empty videos")
            return []

        metadata_file = self.dataset_root / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return []

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        videos = []
        for item in data.get("videos", []):
            try:
                video = SyntheticVideo.from_dict(item)
                if scene_type is None or video.scene_type == scene_type:
                    videos.append(video)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid video: {e}")
                continue

        return videos

    def evaluate_by_complexity(
        self,
        predictions: List[str],
        ground_truth: List[str],
        videos: List[SyntheticVideo],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance grouped by complexity level.

        Args:
            predictions: Predicted descriptions
            ground_truth: Ground truth descriptions
            videos: SyntheticVideo metadata for each sample

        Returns:
            Dictionary mapping complexity level to metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(videos):
            raise ValueError("predictions, ground_truth, and videos must have same length")

        if not predictions:
            return {}

        # Group by complexity
        by_complexity: Dict[int, List[Tuple[str, str]]] = {}
        for pred, gt, video in zip(predictions, ground_truth, videos):
            if video.complexity not in by_complexity:
                by_complexity[video.complexity] = []
            by_complexity[video.complexity].append((pred, gt))

        results = {}
        for complexity, pairs in by_complexity.items():
            preds, gts = zip(*pairs)
            results[f"complexity_{complexity}"] = {
                "accuracy": self.compute_accuracy(list(preds), list(gts)),
                "bleu": self.compute_bleu(list(preds), list(gts)),
                "count": len(pairs),
            }

        return results

    def evaluate_by_scene_type(
        self,
        predictions: List[str],
        ground_truth: List[str],
        videos: List[SyntheticVideo],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance grouped by scene type.

        Args:
            predictions: Predicted descriptions
            ground_truth: Ground truth descriptions
            videos: SyntheticVideo metadata for each sample

        Returns:
            Dictionary mapping scene type to metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(videos):
            raise ValueError("predictions, ground_truth, and videos must have same length")

        if not predictions:
            return {}

        # Group by scene type
        by_scene: Dict[str, List[Tuple[str, str]]] = {}
        for pred, gt, video in zip(predictions, ground_truth, videos):
            if video.scene_type not in by_scene:
                by_scene[video.scene_type] = []
            by_scene[video.scene_type].append((pred, gt))

        results = {}
        for scene_type, pairs in by_scene.items():
            preds, gts = zip(*pairs)
            results[scene_type] = {
                "accuracy": self.compute_accuracy(list(preds), list(gts)),
                "bleu": self.compute_bleu(list(preds), list(gts)),
                "count": len(pairs),
            }

        return results

    def evaluate_object_count_accuracy(
        self,
        predicted_counts: List[int],
        ground_truth_counts: List[int],
    ) -> Dict[str, float]:
        """Evaluate object count prediction accuracy.

        Args:
            predicted_counts: Predicted object counts
            ground_truth_counts: Ground truth object counts

        Returns:
            Dictionary with count accuracy metrics
        """
        if len(predicted_counts) != len(ground_truth_counts):
            raise ValueError("Predicted and ground truth counts must have same length")

        if not predicted_counts:
            return {"exact_match": 0.0, "mean_absolute_error": 0.0, "within_1": 0.0}

        exact_matches = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if p == g)
        within_1 = sum(1 for p, g in zip(predicted_counts, ground_truth_counts) if abs(p - g) <= 1)
        mae = np.mean([abs(p - g) for p, g in zip(predicted_counts, ground_truth_counts)])

        return {
            "exact_match": exact_matches / len(predicted_counts),
            "within_1": within_1 / len(predicted_counts),
            "mean_absolute_error": float(mae),
        }

    def run_benchmark(
        self,
        model_id: str,
        predictions: List[str],
        ground_truth: List[str],
        videos: List[SyntheticVideo],
        predicted_counts: Optional[List[int]] = None,
        ground_truth_counts: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the full synthetic video benchmark.

        Args:
            model_id: Identifier for the model
            predictions: Predicted descriptions
            ground_truth: Ground truth descriptions
            videos: SyntheticVideo metadata for each sample
            predicted_counts: Optional predicted object counts
            ground_truth_counts: Optional ground truth object counts
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with all metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(videos):
            raise ValueError("predictions, ground_truth, and videos must have same length")

        logger.info(
            "Running synthetic video benchmark",
            model=model_id,
            n_samples=len(predictions),
        )

        # Overall metrics
        overall_metrics = {
            "accuracy": self.compute_accuracy(predictions, ground_truth),
            "bleu": self.compute_bleu(predictions, ground_truth),
            "rouge_l": self.compute_rouge_l(predictions, ground_truth),
        }

        # Complexity breakdown
        complexity_metrics = self.evaluate_by_complexity(predictions, ground_truth, videos)

        # Scene type breakdown
        scene_metrics = self.evaluate_by_scene_type(predictions, ground_truth, videos)

        # Object count metrics
        if predicted_counts and ground_truth_counts:
            count_metrics = self.evaluate_object_count_accuracy(
                predicted_counts, ground_truth_counts
            )
            overall_metrics.update(count_metrics)

        # Combine all metrics
        all_metrics = overall_metrics.copy()
        for key, value in complexity_metrics.items():
            for metric_name, metric_value in value.items():
                if metric_name != "count":
                    all_metrics[f"{key}_{metric_name}"] = metric_value

        for key, value in scene_metrics.items():
            for metric_name, metric_value in value.items():
                if metric_name != "count":
                    all_metrics[f"{key}_{metric_name}"] = metric_value

        result = BenchmarkResult(
            benchmark_name="synthetic_video",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=all_metrics,
            predictions=predictions,
            references=ground_truth,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Synthetic video benchmark complete", metrics=overall_metrics)
        return result
