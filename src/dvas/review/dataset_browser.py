"""Dataset browser for reviewing annotation datasets.

Provides filtering, sorting, and pagination over annotation collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation
from dvas.quality.schema import QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class SortField(str, Enum):
    """Fields available for sorting."""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    QUALITY_SCORE = "quality_score"
    VIDEO_ID = "video_id"
    SEGMENT_COUNT = "segment_count"


class SortOrder(str, Enum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class DatasetFilter:
    """Filter criteria for dataset browsing."""

    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    status: Optional[str] = None
    source: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_actions: Optional[bool] = None
    has_objects: Optional[bool] = None
    tags: Optional[List[str]] = None
    video_ids: Optional[List[str]] = None
    video_id_prefix: Optional[str] = None
    min_segments: Optional[int] = None
    max_segments: Optional[int] = None

    def matches(
        self, annotation: Annotation, quality_scores: Optional[QualityScores] = None
    ) -> bool:
        """Check if an annotation matches this filter."""
        score = annotation.quality_score
        if quality_scores:
            score = quality_scores.overall_score

        if self.min_quality_score is not None:
            if score is None or score < self.min_quality_score:
                return False

        if self.max_quality_score is not None:
            if score is not None and score > self.max_quality_score:
                return False

        if self.source is not None and annotation.source != self.source:
            return False

        if self.date_from is not None:
            if annotation.created_at < self.date_from:
                return False

        if self.date_to is not None:
            if annotation.created_at > self.date_to:
                return False

        if self.has_actions is not None:
            has_any = any(seg.actions for seg in annotation.segments)
            if has_any != self.has_actions:
                return False

        if self.has_objects is not None:
            has_any = any(seg.objects for seg in annotation.segments)
            if has_any != self.has_objects:
                return False

        if self.tags is not None:
            if not any(t in annotation.tags for t in self.tags):
                return False

        if self.video_ids is not None:
            if annotation.video_id not in self.video_ids:
                return False

        if self.video_id_prefix is not None:
            if not annotation.video_id.startswith(self.video_id_prefix):
                return False

        seg_count = len(annotation.segments)
        if self.min_segments is not None and seg_count < self.min_segments:
            return False
        if self.max_segments is not None and seg_count > self.max_segments:
            return False

        return True


@dataclass
class PaginationResult:
    """Result of a paginated query."""

    items: List[Annotation] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20
    total_pages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [a.model_dump() for a in self.items],
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
        }


@dataclass
class DatasetStatistics:
    """Aggregated statistics for a dataset."""

    total_annotations: int = 0
    avg_quality_score: float = 0.0
    min_quality_score: float = 0.0
    max_quality_score: float = 0.0
    by_source: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    total_segments: int = 0
    total_actions: int = 0
    total_objects: int = 0
    avg_segments_per_annotation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_annotations": self.total_annotations,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "min_quality_score": round(self.min_quality_score, 4),
            "max_quality_score": round(self.max_quality_score, 4),
            "by_source": self.by_source,
            "by_status": self.by_status,
            "total_segments": self.total_segments,
            "total_actions": self.total_actions,
            "total_objects": self.total_objects,
            "avg_segments_per_annotation": round(self.avg_segments_per_annotation, 2),
        }


class DatasetBrowser:
    """Browser for annotation datasets with filtering and pagination."""

    def __init__(self, annotations: Optional[List[Annotation]] = None):
        self._annotations: List[Annotation] = annotations or []
        self._quality_map: Dict[str, QualityScores] = {}

    def add_annotation(
        self, annotation: Annotation, quality: Optional[QualityScores] = None
    ) -> None:
        self._annotations.append(annotation)
        if quality:
            self._quality_map[annotation.id] = quality

    def add_annotations(
        self, annotations: List[Annotation], quality_map: Optional[Dict[str, QualityScores]] = None
    ) -> None:
        for ann in annotations:
            self.add_annotation(ann, quality_map.get(ann.id) if quality_map else None)

    def filter_annotations(
        self, dataset_filter: Optional[DatasetFilter] = None
    ) -> List[Annotation]:
        """Filter annotations by criteria."""
        if dataset_filter is None:
            return list(self._annotations)
        results = []
        for ann in self._annotations:
            quality = self._quality_map.get(ann.id)
            if dataset_filter.matches(ann, quality):
                results.append(ann)
        return results

    def sort_annotations(
        self,
        annotations: List[Annotation],
        sort_field: SortField = SortField.CREATED_AT,
        sort_order: SortOrder = SortOrder.DESC,
    ) -> List[Annotation]:
        """Sort annotations by field."""
        reverse = sort_order == SortOrder.DESC

        def sort_key(ann: Annotation) -> Any:
            if sort_field == SortField.QUALITY_SCORE:
                quality = self._quality_map.get(ann.id)
                return ann.quality_score or (quality.overall_score if quality else 0.0)
            elif sort_field == SortField.CREATED_AT:
                return ann.created_at
            elif sort_field == SortField.VIDEO_ID:
                return ann.video_id
            elif sort_field == SortField.SOURCE:
                return ann.source
            return ann.created_at

        return sorted(annotations, key=sort_key, reverse=reverse)

    def paginate(
        self, annotations: List[Annotation], page: int = 1, page_size: int = 20
    ) -> PaginationResult:
        """Paginate a list of annotations."""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20

        total = len(annotations)
        total_pages = (total + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size

        return PaginationResult(
            items=annotations[start:end],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def browse(
        self,
        dataset_filter: Optional[DatasetFilter] = None,
        sort_field: SortField = SortField.CREATED_AT,
        sort_order: SortOrder = SortOrder.DESC,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginationResult:
        """Browse annotations with filtering, sorting, and pagination."""
        filtered = self.filter_annotations(dataset_filter)
        sorted_results = self.sort_annotations(filtered, sort_field, sort_order)
        return self.paginate(sorted_results, page, page_size)

    def get_statistics(self, dataset_filter: Optional[DatasetFilter] = None) -> DatasetStatistics:
        """Get aggregated statistics for the dataset (or filtered subset)."""
        annotations = self.filter_annotations(dataset_filter)

        if not annotations:
            return DatasetStatistics()

        scores = []
        for ann in annotations:
            quality = self._quality_map.get(ann.id)
            score = ann.quality_score or (quality.overall_score if quality else None)
            if score is not None:
                scores.append(score)

        total_segments = sum(len(ann.segments) for ann in annotations)
        total_actions = sum(len(seg.actions) for ann in annotations for seg in ann.segments)
        total_objects = sum(len(seg.objects) for ann in annotations for seg in ann.segments)

        stats = DatasetStatistics(
            total_annotations=len(annotations),
            avg_quality_score=sum(scores) / len(scores) if scores else 0.0,
            min_quality_score=min(scores) if scores else 0.0,
            max_quality_score=max(scores) if scores else 0.0,
            total_segments=total_segments,
            total_actions=total_actions,
            total_objects=total_objects,
            avg_segments_per_annotation=total_segments / len(annotations) if annotations else 0.0,
        )

        for ann in annotations:
            stats.by_source[ann.source] = stats.by_source.get(ann.source, 0) + 1

        return stats

    def get_annotation_quality(self, annotation_id: str) -> Optional[QualityScores]:
        """Get quality scores for an annotation."""
        return self._quality_map.get(annotation_id)
