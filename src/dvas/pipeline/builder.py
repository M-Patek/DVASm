"""Annotation builder from parsed model responses."""

from typing import Any, Dict, List, Literal, Optional

from dvas.data.schemas import Annotation, Segment, VideoMetadata
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AnnotationBuilder:
    """Builds Annotation objects from parsed segment data.

    Separated from pipeline to allow independent testing and customization.
    """

    def __init__(self, model_version: Optional[str] = None):
        """Initialize builder with model version for tracking.

        Args:
            model_version: Version identifier of the model that generated annotations
        """
        self.model_version = model_version or "unknown"

    def build_annotation(
        self,
        video_id: str,
        video_path: str,
        segments: List[Segment],
        metadata: VideoMetadata,
        source: Literal["teacher", "student", "human", "hybrid"] = "teacher",
    ) -> Annotation:
        """Build a complete Annotation from segments and metadata."""
        return Annotation(
            id=f"{video_id}_annotated",
            video_id=video_id,
            video_path=video_path,
            segments=segments,
            metadata=metadata,
            source=source,
            model_version=self.model_version,
            quality_score=None,
        )

    def build_segment(
        self,
        start_time: float,
        end_time: float,
        response_text: str,
        parsed: Dict[str, Any],
    ) -> Segment:
        """Build a Segment from parsed response data."""
        return Segment(
            start_time=start_time,
            end_time=end_time,
            caption=parsed.get("scene_description", ""),
            caption_dense=response_text,
            qa_pairs=parsed.get("qa_pairs", []),
            objects=parsed.get("objects", []),
            actions=parsed.get("actions", []),
        )

    def build_empty_segment(
        self,
        start_time: float,
        end_time: float,
        reason: str = "",
    ) -> Segment:
        """Build an empty segment when annotation fails."""
        return Segment(
            start_time=start_time,
            end_time=end_time,
            caption=f"[Annotation failed: {reason}]" if reason else "",
        )
