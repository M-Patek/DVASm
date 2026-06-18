"""Annotation builder from parsed model responses."""

from typing import Any, Dict, List, Literal, Optional

from dvas.data.schemas import Action, Annotation, Object, QAPair, Segment, VideoMetadata
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def _convert_to_object(obj_data: Any) -> Optional[Object]:
    """Convert dict or Object to Object."""
    if isinstance(obj_data, Object):
        return obj_data
    if isinstance(obj_data, dict):
        try:
            return Object(**obj_data)
        except Exception as e:
            logger.warning("object_conversion_failed", error=str(e), data=obj_data)
            return None
    return None


def _convert_to_action(action_data: Any) -> Optional[Action]:
    """Convert dict or Action to Action."""
    if isinstance(action_data, Action):
        return action_data
    if isinstance(action_data, dict):
        try:
            return Action(**action_data)
        except Exception as e:
            logger.warning("action_conversion_failed", error=str(e), data=action_data)
            return None
    return None


def _convert_to_qa_pair(qa_data: Any) -> Optional[QAPair]:
    """Convert dict or QAPair to QAPair."""
    if isinstance(qa_data, QAPair):
        return qa_data
    if isinstance(qa_data, dict):
        try:
            return QAPair(**qa_data)
        except Exception as e:
            logger.warning("qa_pair_conversion_failed", error=str(e), data=qa_data)
            return None
    return None


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
        # Convert lists with validation
        raw_objects = parsed.get("objects", [])
        objects = [o for o in (_convert_to_object(obj) for obj in raw_objects) if o is not None]
        if len(objects) != len(raw_objects):
            logger.warning(
                "object_conversion_dropped",
                dropped=len(raw_objects) - len(objects),
                total=len(raw_objects),
            )

        raw_actions = parsed.get("actions", [])
        actions = [a for a in (_convert_to_action(act) for act in raw_actions) if a is not None]
        if len(actions) != len(raw_actions):
            logger.warning(
                "action_conversion_dropped",
                dropped=len(raw_actions) - len(actions),
                total=len(raw_actions),
            )

        raw_qa_pairs = parsed.get("qa_pairs", [])
        qa_pairs = [q for q in (_convert_to_qa_pair(qa) for qa in raw_qa_pairs) if q is not None]
        if len(qa_pairs) != len(raw_qa_pairs):
            logger.warning(
                "qa_pair_conversion_dropped",
                dropped=len(raw_qa_pairs) - len(qa_pairs),
                total=len(raw_qa_pairs),
            )

        return Segment(
            start_time=start_time,
            end_time=end_time,
            caption=parsed.get("scene_description", ""),
            caption_dense=response_text,
            qa_pairs=qa_pairs,
            objects=objects,
            actions=actions,
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
