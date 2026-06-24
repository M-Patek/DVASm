"""Data quality analysis and monitoring platform."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from dvas.data.schemas import Annotation
from dvas.data.storage import AnnotationStore
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetQualityMetrics:
    """Quality metrics for a dataset."""

    total_annotations: int
    avg_segments_per_video: float
    avg_caption_length: float
    vocabulary_size: int
    verb_diversity: float
    noun_diversity: float
    action_balance_score: float  # 0-1, higher = more balanced
    temporal_coverage: float  # ratio of annotated time to total
    qaq_pairs_per_segment: float
    missing_fields_rate: float
    outlier_annotations: List[str]  # IDs of potential outliers


@dataclass
class DataDistribution:
    """Distribution analysis results."""

    verb_distribution: Dict[str, int]
    noun_distribution: Dict[str, int]
    duration_distribution: Dict[str, int]  # binned durations
    segment_count_distribution: Dict[int, int]
    quality_score_distribution: Dict[str, int]
    source_distribution: Dict[str, int]


class AnomalyDetector:
    """Detect anomalous annotations using statistical methods."""

    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold

    def detect_outliers(self, annotations: List[Annotation]) -> List[Tuple[str, str]]:
        """Detect outlier annotations with reasons."""
        if not annotations:
            return []

        outliers = []

        # Feature extraction
        features = []
        for ann in annotations:
            feat = {
                "id": ann.id,
                "num_segments": len(ann.segments),
                "total_duration": ann.get_total_duration(),
                "avg_caption_len": np.mean(
                    [len(s.caption) for s in ann.segments] if ann.segments else [0]
                ),
                "num_verbs": len(ann.get_action_verbs()),
                "num_objects": len(ann.get_object_names()),
            }
            features.append(feat)

        # Z-score analysis for each feature
        for key in ["num_segments", "total_duration", "avg_caption_len", "num_verbs"]:
            values = np.array([f[key] for f in features])
            z_scores = np.abs(stats.zscore(values))

            for i, (feat, z_score) in enumerate(zip(features, z_scores)):
                if z_score > self.z_threshold:
                    outliers.append(
                        (feat["id"], f"{key} z-score={z_score:.2f} (threshold={self.z_threshold})")
                    )

        # Remove duplicates
        seen = set()
        unique_outliers = []
        for ann_id, reason in outliers:
            if ann_id not in seen:
                seen.add(ann_id)
                unique_outliers.append((ann_id, reason))

        return unique_outliers

    def detect_duplicates(
        self, annotations: List[Annotation], similarity_threshold: float = 0.9
    ) -> List[Tuple[str, str, float]]:
        """Detect near-duplicate annotations."""
        duplicates = []

        ann_texts = []
        for ann in annotations:
            text = " ".join(s.caption for s in ann.segments)
            ann_texts.append((ann.id, text))

        # Simple Jaccard similarity (can be improved with embeddings)
        for i, (id1, text1) in enumerate(ann_texts):
            set1 = set(text1.lower().split())
            if not set1:
                continue

            for id2, text2 in ann_texts[i + 1 :]:
                set2 = set(text2.lower().split())
                if not set2:
                    continue

                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0

                if jaccard >= similarity_threshold:
                    duplicates.append((id1, id2, jaccard))

        return duplicates


class DataQualityAnalyzer:
    """Analyze dataset quality and distribution."""

    def __init__(self, store: Optional[AnnotationStore] = None):
        self.store = store or AnnotationStore()
        self.anomaly_detector = AnomalyDetector()

    def analyze_dataset(
        self, source: str = "gold", sample_size: Optional[int] = None
    ) -> Tuple[DatasetQualityMetrics, DataDistribution]:
        """Full dataset analysis."""
        logger.info("analyzing_dataset", source=source)

        annotations = list(self.store.load_all(source=source))

        if sample_size and len(annotations) > sample_size:
            import random

            annotations = random.sample(annotations, sample_size)

        if not annotations:
            logger.warning("no_annotations_found", source=source)
            return self._empty_metrics(), self._empty_distribution()

        # Compute metrics
        metrics = self._compute_quality_metrics(annotations)
        distribution = self._compute_distribution(annotations)

        logger.info(
            "analysis_complete",
            total=len(annotations),
            vocabulary=metrics.vocabulary_size,
            outliers=len(metrics.outlier_annotations),
        )

        return metrics, distribution

    def _compute_quality_metrics(self, annotations: List[Annotation]) -> DatasetQualityMetrics:
        """Compute comprehensive quality metrics."""
        total = len(annotations)

        # Segments
        segments_counts = [len(ann.segments) for ann in annotations]
        avg_segments = np.mean(segments_counts)

        # Captions
        caption_lengths = []
        for ann in annotations:
            for seg in ann.segments:
                caption_lengths.append(len(seg.caption))
        avg_caption_len = np.mean(caption_lengths) if caption_lengths else 0

        # Vocabulary
        all_words = set()
        for ann in annotations:
            for seg in ann.segments:
                all_words.update(seg.caption.lower().split())
        vocab_size = len(all_words)

        # Actions
        all_verbs = []
        all_nouns = []
        for ann in annotations:
            for seg in ann.segments:
                for action in seg.actions:
                    all_verbs.append(action.verb)
                    all_nouns.append(action.noun)

        verb_counts = Counter(all_verbs)
        noun_counts = Counter(all_nouns)

        # Diversity (unique actions / total actions)
        verb_diversity = len(verb_counts) / len(all_verbs) if all_verbs else 0
        noun_diversity = len(noun_counts) / len(all_nouns) if all_nouns else 0

        # Balance score (entropy-based)
        if verb_counts:
            probs = np.array(list(verb_counts.values())) / sum(verb_counts.values())
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            balance_score = min(entropy / np.log(len(verb_counts) + 1), 1.0)  # Normalize
        else:
            balance_score = 0

        # Temporal coverage
        total_duration = sum(ann.metadata.duration for ann in annotations if ann.metadata)
        annotated_duration = sum(ann.get_total_duration() for ann in annotations)
        temporal_coverage = annotated_duration / total_duration if total_duration > 0 else 0

        # QA pairs
        qa_counts = [len(seg.qa_pairs) for ann in annotations for seg in ann.segments]
        avg_qa = np.mean(qa_counts) if qa_counts else 0

        # Missing fields
        missing_count = 0
        total_fields = 0
        for ann in annotations:
            total_fields += 1
            if not ann.segments:
                missing_count += 1
            for seg in ann.segments:
                total_fields += 3
                if not seg.caption:
                    missing_count += 1
                if not seg.actions:
                    missing_count += 1
                if not seg.objects:
                    missing_count += 1

        missing_rate = missing_count / total_fields if total_fields > 0 else 0

        # Outliers
        outliers = self.anomaly_detector.detect_outliers(annotations)

        return DatasetQualityMetrics(
            total_annotations=total,
            avg_segments_per_video=float(avg_segments),
            avg_caption_length=float(avg_caption_len),
            vocabulary_size=vocab_size,
            verb_diversity=float(verb_diversity),
            noun_diversity=float(noun_diversity),
            action_balance_score=float(balance_score),
            temporal_coverage=float(temporal_coverage),
            qaq_pairs_per_segment=float(avg_qa),
            missing_fields_rate=float(missing_rate),
            outlier_annotations=[o[0] for o in outliers],
        )

    def _compute_distribution(self, annotations: List[Annotation]) -> DataDistribution:
        """Compute distribution statistics."""
        verbs = Counter()
        nouns = Counter()
        durations = Counter()
        segment_counts = Counter()
        quality_scores = Counter()
        sources = Counter()

        for ann in annotations:
            # Source
            sources[ann.source] += 1

            # Duration bins
            total_dur = ann.get_total_duration()
            if total_dur < 5:
                durations["0-5s"] += 1
            elif total_dur < 15:
                durations["5-15s"] += 1
            elif total_dur < 30:
                durations["15-30s"] += 1
            else:
                durations["30s+"] += 1

            # Segment counts
            segment_counts[len(ann.segments)] += 1

            # Quality scores
            if ann.quality_score is not None:
                bucket = f"{int(ann.quality_score * 10) / 10:.1f}"
                quality_scores[bucket] += 1
            else:
                quality_scores["unknown"] += 1

            # Verbs and nouns
            for verb in ann.get_action_verbs():
                verbs[verb] += 1
            for obj in ann.get_object_names():
                nouns[obj] += 1

        return DataDistribution(
            verb_distribution=dict(verbs.most_common(50)),
            noun_distribution=dict(nouns.most_common(50)),
            duration_distribution=dict(durations),
            segment_count_distribution=dict(segment_counts),
            quality_score_distribution=dict(quality_scores),
            source_distribution=dict(sources),
        )

    def _empty_metrics(self) -> DatasetQualityMetrics:
        return DatasetQualityMetrics(
            total_annotations=0,
            avg_segments_per_video=0,
            avg_caption_length=0,
            vocabulary_size=0,
            verb_diversity=0,
            noun_diversity=0,
            action_balance_score=0,
            temporal_coverage=0,
            qaq_pairs_per_segment=0,
            missing_fields_rate=0,
            outlier_annotations=[],
        )

    def _empty_distribution(self) -> DataDistribution:
        return DataDistribution(
            verb_distribution={},
            noun_distribution={},
            duration_distribution={},
            segment_count_distribution={},
            quality_score_distribution={},
            source_distribution={},
        )

    def generate_quality_report(self, output_path: Path, source: str = "gold") -> Path:
        """Generate comprehensive HTML quality report."""
        metrics, distribution = self.analyze_dataset(source)

        # Detect duplicates
        annotations = list(self.store.load_all(source=source))
        duplicates = self.anomaly_detector.detect_duplicates(annotations)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "metrics": asdict(metrics),
            "distribution": asdict(distribution),
            "duplicates": [{"id1": d[0], "id2": d[1], "similarity": d[2]} for d in duplicates],
            "recommendations": self._generate_recommendations(metrics, distribution),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("quality_report_generated", path=str(output_path))

        return output_path

    def _generate_recommendations(
        self, metrics: DatasetQualityMetrics, distribution: DataDistribution
    ) -> List[Dict]:
        """Generate data quality recommendations."""
        recommendations = []

        if metrics.missing_fields_rate > 0.1:
            recommendations.append(
                {
                    "type": "warning",
                    "issue": "High missing field rate",
                    "value": f"{metrics.missing_fields_rate:.1%}",
                    "action": "Review annotation pipeline for incomplete outputs",
                }
            )

        if metrics.action_balance_score < 0.5:
            recommendations.append(
                {
                    "type": "warning",
                    "issue": "Unbalanced action distribution",
                    "value": f"{metrics.action_balance_score:.2f}",
                    "action": "Collect more diverse training data",
                }
            )

        if len(metrics.outlier_annotations) > metrics.total_annotations * 0.05:
            recommendations.append(
                {
                    "type": "alert",
                    "issue": "High outlier rate",
                    "value": f"{len(metrics.outlier_annotations)} outliers",
                    "action": "Review outlier annotations for quality issues",
                }
            )

        if metrics.vocabulary_size < 100:
            recommendations.append(
                {
                    "type": "info",
                    "issue": "Limited vocabulary diversity",
                    "value": f"{metrics.vocabulary_size} words",
                    "action": "Consider collecting more varied data",
                }
            )

        return recommendations


class DataAugmenter:
    """Augment training data through synthetic generation."""

    def __init__(self):
        self.augmentation_strategies = {
            "paraphrase": self._augment_paraphrase,
            "temporal_shift": self._augment_temporal_shift,
            "object_swap": self._augment_object_swap,
        }

    def augment_annotation(self, annotation: Annotation, strategy: str) -> Annotation:
        """Apply augmentation strategy to annotation."""
        if strategy not in self.augmentation_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.augmentation_strategies[strategy](annotation)

    def _augment_paraphrase(self, annotation: Annotation) -> Annotation:
        """Generate paraphrased captions."""
        # Placeholder: In production, use T5 or GPT for paraphrasing
        import copy

        new_annotation = copy.deepcopy(annotation)
        new_annotation.id = f"{annotation.id}_aug_para"

        for seg in new_annotation.segments:
            # Simple augmentation: swap word order, add synonyms
            words = seg.caption.split()
            if len(words) > 3:
                # Swap first two words
                words[0], words[1] = words[1], words[0]
                seg.caption = " ".join(words)

        return new_annotation

    def _augment_temporal_shift(self, annotation: Annotation) -> Annotation:
        """Create temporally shifted versions."""
        import copy

        new_annotation = copy.deepcopy(annotation)
        new_annotation.id = f"{annotation.id}_aug_time"

        # Shift timestamps
        for seg in new_annotation.segments:
            seg.start_time *= 0.95
            seg.end_time = min(seg.end_time * 1.05, annotation.metadata.duration)

        return new_annotation

    def _augment_object_swap(self, annotation: Annotation) -> Annotation:
        """Swap similar objects in annotations."""
        # Placeholder: Would use semantic similarity
        return annotation
