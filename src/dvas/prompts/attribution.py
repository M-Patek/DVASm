"""Prompt quality attribution tracking.

Tracks which prompt version produced which annotation and attributes
quality scores to individual prompts for performance rollup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation
from dvas.quality.schema import QualityDimension, QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptAttributionRecord:
    """Record of a single annotation attributed to a prompt."""

    annotation_id: str
    prompt_id: str
    prompt_version: str
    video_id: str
    quality_score: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "video_id": self.video_id,
            "quality_score": self.quality_score,
            "dimensions": self.dimensions.copy(),
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PromptPerformanceSummary:
    """Performance summary for a single prompt."""

    prompt_id: str
    prompt_version: str
    total_annotations: int = 0
    avg_quality_score: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    best_annotation_id: Optional[str] = None
    best_score: float = 0.0
    worst_annotation_id: Optional[str] = None
    worst_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "total_annotations": self.total_annotations,
            "avg_quality_score": self.avg_quality_score,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost": self.avg_cost,
            "dimension_scores": self.dimension_scores.copy(),
            "best_annotation_id": self.best_annotation_id,
            "best_score": self.best_score,
            "worst_annotation_id": self.worst_annotation_id,
            "worst_score": self.worst_score,
        }


class PromptAttributionTracker:
    """Tracks which prompts produced which annotations and their quality."""

    def __init__(self) -> None:
        self._records: Dict[str, List[PromptAttributionRecord]] = {}
        self._annotation_to_prompt: Dict[str, str] = {}

    def record_attribution(
        self,
        annotation: Annotation,
        prompt_id: str,
        prompt_version: str,
        quality_scores: Optional[QualityScores] = None,
        latency_ms: float = 0.0,
        cost: float = 0.0,
    ) -> PromptAttributionRecord:
        """Record that a prompt produced an annotation.

        Args:
            annotation: The annotation produced.
            prompt_id: ID of the prompt used.
            prompt_version: Version of the prompt used.
            quality_scores: Optional quality scores for the annotation.
            latency_ms: Latency in milliseconds.
            cost: Cost of generation.

        Returns:
            The attribution record.
        """
        quality_score = quality_scores.overall_score if quality_scores else 0.0

        dimensions: Dict[str, float] = {}
        if quality_scores:
            for dim_score in quality_scores.all_scores:
                dimensions[dim_score.dimension.value] = dim_score.score

        record = PromptAttributionRecord(
            annotation_id=annotation.id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            video_id=annotation.video_id,
            quality_score=quality_score,
            dimensions=dimensions,
            latency_ms=latency_ms,
            cost=cost,
        )

        key = f"{prompt_id}:{prompt_version}"
        if key not in self._records:
            self._records[key] = []
        self._records[key].append(record)
        self._annotation_to_prompt[annotation.id] = key

        logger.info(
            "attribution_recorded",
            annotation_id=annotation.id,
            prompt_id=prompt_id,
            version=prompt_version,
            quality_score=quality_score,
        )

        return record

    def get_records_for_prompt(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
    ) -> List[PromptAttributionRecord]:
        """Get all attribution records for a prompt.

        Args:
            prompt_id: The prompt ID.
            prompt_version: Optional version filter.

        Returns:
            List of attribution records.
        """
        if prompt_version:
            key = f"{prompt_id}:{prompt_version}"
            return self._records.get(key, []).copy()

        records: List[PromptAttributionRecord] = []
        for key, recs in self._records.items():
            if key.startswith(f"{prompt_id}:"):
                records.extend(recs)
        return records

    def get_prompt_for_annotation(self, annotation_id: str) -> Optional[str]:
        """Get the prompt key that produced an annotation.

        Returns:
            Prompt key in format "prompt_id:version" or None.
        """
        return self._annotation_to_prompt.get(annotation_id)

    def compute_performance_summary(
        self,
        prompt_id: str,
        prompt_version: str,
    ) -> PromptPerformanceSummary:
        """Compute performance summary for a prompt version.

        Args:
            prompt_id: The prompt ID.
            prompt_version: The prompt version.

        Returns:
            Performance summary.
        """
        key = f"{prompt_id}:{prompt_version}"
        records = self._records.get(key, [])

        summary = PromptPerformanceSummary(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            total_annotations=len(records),
        )

        if not records:
            return summary

        total_quality = 0.0
        total_latency = 0.0
        total_cost = 0.0
        dim_totals: Dict[str, List[float]] = {}

        for record in records:
            total_quality += record.quality_score
            total_latency += record.latency_ms
            total_cost += record.cost

            # Track best/worst
            if record.quality_score > summary.best_score:
                summary.best_score = record.quality_score
                summary.best_annotation_id = record.annotation_id
            if record.quality_score < summary.worst_score:
                summary.worst_score = record.quality_score
                summary.worst_annotation_id = record.annotation_id

            # Aggregate dimension scores
            for dim_name, score in record.dimensions.items():
                if dim_name not in dim_totals:
                    dim_totals[dim_name] = []
                dim_totals[dim_name].append(score)

        n = len(records)
        summary.avg_quality_score = total_quality / n
        summary.avg_latency_ms = total_latency / n
        summary.avg_cost = total_cost / n
        summary.dimension_scores = {
            name: sum(scores) / len(scores) for name, scores in dim_totals.items()
        }

        return summary

    def compute_rollup(
        self,
        prompt_ids: Optional[List[str]] = None,
    ) -> Dict[str, PromptPerformanceSummary]:
        """Compute performance rollup for multiple prompts.

        Args:
            prompt_ids: Optional list of prompt IDs to include. If None, all prompts.

        Returns:
            Dict mapping prompt keys to performance summaries.
        """
        rollup: Dict[str, PromptPerformanceSummary] = {}

        for key, records in self._records.items():
            if not records:
                continue

            prompt_id = records[0].prompt_id
            if prompt_ids and prompt_id not in prompt_ids:
                continue

            prompt_version = records[0].prompt_version
            summary = self.compute_performance_summary(prompt_id, prompt_version)
            rollup[key] = summary

        return rollup

    def compare_prompts(
        self,
        prompt_id_a: str,
        version_a: str,
        prompt_id_b: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two prompt versions.

        Returns:
            Comparison dictionary with differences.
        """
        summary_a = self.compute_performance_summary(prompt_id_a, version_a)
        summary_b = self.compute_performance_summary(prompt_id_b, version_b)

        return {
            "prompt_a": {
                "id": prompt_id_a,
                "version": version_a,
                "avg_quality": summary_a.avg_quality_score,
                "total_annotations": summary_a.total_annotations,
                "avg_latency_ms": summary_a.avg_latency_ms,
                "avg_cost": summary_a.avg_cost,
            },
            "prompt_b": {
                "id": prompt_id_b,
                "version": version_b,
                "avg_quality": summary_b.avg_quality_score,
                "total_annotations": summary_b.total_annotations,
                "avg_latency_ms": summary_b.avg_latency_ms,
                "avg_cost": summary_b.avg_cost,
            },
            "difference": {
                "quality": summary_b.avg_quality_score - summary_a.avg_quality_score,
                "latency": summary_b.avg_latency_ms - summary_a.avg_latency_ms,
                "cost": summary_b.avg_cost - summary_a.avg_cost,
            },
        }
