"""Long-video benchmark for extended sequence evaluation.

Evaluates models on long-duration videos (minutes to hours),
measuring temporal coherence, memory retention, and
long-range dependency understanding.
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
class LongVideoSegment:
    """Segment of a long video with annotations.

    Attributes:
        video_id: Video identifier
        start_time: Start time in seconds
        end_time: End time in seconds
        segment_index: Index of this segment in the video
        total_segments: Total number of segments
        caption: Description of this segment
        key_events: List of key events in this segment
    """

    video_id: str
    start_time: float
    end_time: float
    segment_index: int
    total_segments: int
    caption: str
    key_events: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "segment_index": self.segment_index,
            "total_segments": self.total_segments,
            "caption": self.caption,
            "key_events": self.key_events,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongVideoSegment":
        return cls(
            video_id=data["video_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            segment_index=data["segment_index"],
            total_segments=data["total_segments"],
            caption=data["caption"],
            key_events=data.get("key_events", []),
        )


class LongVideoBenchmark(BaseBenchmark):
    """Long-video benchmark.

    Evaluates models on extended video sequences,
    measuring temporal coherence, memory retention,
    and long-range dependency understanding.

    Args:
        benchmark_dir: Directory for storing benchmark data
        dataset_root: Path to long-video dataset root
    """

    # Duration categories
    DURATION_CATEGORIES = {
        "short": (0, 60),  # 0-60 seconds
        "medium": (60, 300),  # 1-5 minutes
        "long": (300, 1800),  # 5-30 minutes
        "extended": (1800, float("inf")),  # 30+ minutes
    }

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__(benchmark_dir, "long_video")
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def load_segments(self, split: str = "test") -> List[LongVideoSegment]:
        """Load long video segment annotations.

        Args:
            split: Dataset split

        Returns:
            List of LongVideoSegment objects
        """
        if not self.dataset_root or not self.dataset_root.exists():
            logger.warning("Dataset root not found, returning empty segments")
            return []

        segments_file = self.dataset_root / f"segments_{split}.json"
        if not segments_file.exists():
            logger.warning(f"Segments file not found: {segments_file}")
            return []

        with open(segments_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for item in data.get("segments", []):
            try:
                segments.append(LongVideoSegment.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid segment: {e}")
                continue

        return segments

    def evaluate_temporal_coherence(
        self,
        predictions: List[str],
        ground_truth: List[str],
        segments: List[LongVideoSegment],
    ) -> Dict[str, float]:
        """Evaluate temporal coherence across segments.

        Measures consistency between adjacent segment predictions.

        Args:
            predictions: Predicted captions for each segment
            ground_truth: Ground truth captions
            segments: Segment metadata

        Returns:
            Dictionary with coherence metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(segments):
            raise ValueError("predictions, ground_truth, and segments must have same length")

        if len(predictions) < 2:
            return {"temporal_coherence": 0.0, "consistency_score": 0.0}

        # Group by video
        by_video: Dict[str, List[Tuple[int, str, str, LongVideoSegment]]] = {}
        for i, (pred, gt, seg) in enumerate(zip(predictions, ground_truth, segments)):
            if seg.video_id not in by_video:
                by_video[seg.video_id] = []
            by_video[seg.video_id].append((i, pred, gt, seg))

        coherence_scores = []
        consistency_scores = []

        for video_id, video_segments in by_video.items():
            if len(video_segments) < 2:
                continue

            # Sort by segment index
            video_segments.sort(key=lambda x: x[3].segment_index)

            for i in range(len(video_segments) - 1):
                _, pred_curr, gt_curr, seg_curr = video_segments[i]
                _, pred_next, gt_next, seg_next = video_segments[i + 1]

                # Check if predictions are temporally consistent
                # (e.g., no contradictory statements)
                pred_words_curr = set(pred_curr.lower().split())
                pred_words_next = set(pred_next.lower().split())
                overlap = len(pred_words_curr & pred_words_next)
                union = len(pred_words_curr | pred_words_next)
                coherence = overlap / union if union > 0 else 0.0
                coherence_scores.append(coherence)

                # Check if predictions maintain narrative consistency
                gt_words_curr = set(gt_curr.lower().split())
                gt_words_next = set(gt_next.lower().split())
                gt_overlap = len(gt_words_curr & gt_words_next)
                gt_union = len(gt_words_curr | gt_words_next)
                gt_coherence = gt_overlap / gt_union if gt_union > 0 else 0.0

                consistency = 1.0 - abs(coherence - gt_coherence)
                consistency_scores.append(consistency)

        return {
            "temporal_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "consistency_score": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
        }

    def evaluate_event_detection(
        self,
        predicted_events: List[List[str]],
        ground_truth_events: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate key event detection accuracy.

        Args:
            predicted_events: List of predicted key events per segment
            ground_truth_events: List of ground truth key events per segment

        Returns:
            Dictionary with event detection metrics
        """
        if len(predicted_events) != len(ground_truth_events):
            raise ValueError("predicted_events and ground_truth_events must have same length")

        if not predicted_events:
            return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0}

        total_precision = 0.0
        total_recall = 0.0

        for pred_events, gt_events in zip(predicted_events, ground_truth_events):
            pred_set = set(e.lower().strip() for e in pred_events)
            gt_set = set(e.lower().strip() for e in gt_events)

            if pred_set:
                precision = len(pred_set & gt_set) / len(pred_set)
            else:
                precision = 0.0

            if gt_set:
                recall = len(pred_set & gt_set) / len(gt_set)
            else:
                recall = 0.0

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / len(predicted_events)
        avg_recall = total_recall / len(predicted_events)
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        return {
            "event_precision": avg_precision,
            "event_recall": avg_recall,
            "event_f1": f1,
        }

    def evaluate_by_duration(
        self,
        predictions: List[str],
        ground_truth: List[str],
        segments: List[LongVideoSegment],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance grouped by video duration.

        Args:
            predictions: Predicted captions
            ground_truth: Ground truth captions
            segments: Segment metadata

        Returns:
            Dictionary mapping duration category to metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(segments):
            raise ValueError("predictions, ground_truth, and segments must have same length")

        if not predictions:
            return {}

        # Calculate video durations
        video_durations: Dict[str, float] = {}
        for seg in segments:
            if seg.video_id not in video_durations:
                video_durations[seg.video_id] = 0.0
            video_durations[seg.video_id] = max(video_durations[seg.video_id], seg.end_time)

        # Group by duration category
        by_category: Dict[str, List[Tuple[str, str]]] = {}
        for pred, gt, seg in zip(predictions, ground_truth, segments):
            duration = video_durations.get(seg.video_id, 0.0)
            category = self._get_duration_category(duration)
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((pred, gt))

        results = {}
        for category, pairs in by_category.items():
            preds, gts = zip(*pairs)
            results[category] = {
                "accuracy": self.compute_accuracy(list(preds), list(gts)),
                "bleu": self.compute_bleu(list(preds), list(gts)),
                "count": len(pairs),
            }

        return results

    def _get_duration_category(self, duration: float) -> str:
        """Get duration category for a given duration.

        Args:
            duration: Duration in seconds

        Returns:
            Duration category name
        """
        for category, (min_d, max_d) in self.DURATION_CATEGORIES.items():
            if min_d <= duration < max_d:
                return category
        return "extended"

    def run_benchmark(
        self,
        model_id: str,
        predictions: List[str],
        ground_truth: List[str],
        segments: List[LongVideoSegment],
        predicted_events: Optional[List[List[str]]] = None,
        ground_truth_events: Optional[List[List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the full long-video benchmark.

        Args:
            model_id: Identifier for the model
            predictions: Predicted captions for each segment
            ground_truth: Ground truth captions
            segments: Segment metadata
            predicted_events: Optional predicted key events
            ground_truth_events: Optional ground truth key events
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with all metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(segments):
            raise ValueError("predictions, ground_truth, and segments must have same length")

        logger.info(
            "Running long-video benchmark",
            model=model_id,
            n_segments=len(predictions),
        )

        # Overall metrics
        overall_metrics = {
            "accuracy": self.compute_accuracy(predictions, ground_truth),
            "bleu": self.compute_bleu(predictions, ground_truth),
            "rouge_l": self.compute_rouge_l(predictions, ground_truth),
        }

        # Temporal coherence
        coherence = self.evaluate_temporal_coherence(predictions, ground_truth, segments)
        overall_metrics.update(coherence)

        # Event detection
        if predicted_events and ground_truth_events:
            event_metrics = self.evaluate_event_detection(predicted_events, ground_truth_events)
            overall_metrics.update(event_metrics)

        # Duration breakdown
        duration_metrics = self.evaluate_by_duration(predictions, ground_truth, segments)

        # Combine all metrics
        all_metrics = overall_metrics.copy()
        for key, value in duration_metrics.items():
            for metric_name, metric_value in value.items():
                if metric_name != "count":
                    all_metrics[f"{key}_{metric_name}"] = metric_value

        result = BenchmarkResult(
            benchmark_name="long_video",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=all_metrics,
            predictions=predictions,
            references=ground_truth,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Long-video benchmark complete", metrics=overall_metrics)
        return result
