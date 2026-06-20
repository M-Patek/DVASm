"""Video segment viewer for reviewing annotations.

Provides viewing of video segments with frame-level annotation overlays
and temporal navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dvas.data.schemas import Annotation, Segment
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FrameAnnotation:
    """Annotation overlay for a single frame."""

    frame_idx: int
    timestamp: float
    objects: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "objects": self.objects,
            "actions": self.actions,
            "caption": self.caption,
        }


@dataclass
class SegmentView:
    """View of a segment with frame-level annotations."""

    segment_idx: int
    start_time: float
    end_time: float
    duration: float
    caption: str
    frame_annotations: List[FrameAnnotation] = field(default_factory=list)
    object_count: int = 0
    action_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_idx": self.segment_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "caption": self.caption,
            "frame_annotations": [fa.to_dict() for fa in self.frame_annotations],
            "object_count": self.object_count,
            "action_count": self.action_count,
        }


class SegmentViewer:
    """Viewer for video segments with annotation overlays.

    Provides temporal navigation through segments and frame-level
    annotation display.
    """

    def __init__(self, annotation: Annotation, fps: float = 30.0):
        self.annotation = annotation
        self.fps = fps
        self._current_segment_idx: int = 0
        self._current_frame: int = 0

    @property
    def current_segment(self) -> Optional[Segment]:
        """Get the currently selected segment."""
        if 0 <= self._current_segment_idx < len(self.annotation.segments):
            return self.annotation.segments[self._current_segment_idx]
        return None

    def navigate_to_segment(self, segment_idx: int) -> Optional[SegmentView]:
        """Navigate to a specific segment by index."""
        if not (0 <= segment_idx < len(self.annotation.segments)):
            logger.warning(
                "segment_index_out_of_range",
                segment_idx=segment_idx,
                total_segments=len(self.annotation.segments),
            )
            return None

        self._current_segment_idx = segment_idx
        self._current_frame = 0
        return self.get_segment_view(segment_idx)

    def next_segment(self) -> Optional[SegmentView]:
        """Move to the next segment."""
        return self.navigate_to_segment(self._current_segment_idx + 1)

    def previous_segment(self) -> Optional[SegmentView]:
        """Move to the previous segment."""
        return self.navigate_to_segment(self._current_segment_idx - 1)

    def get_segment_view(self, segment_idx: int) -> Optional[SegmentView]:
        """Get a view of a specific segment."""
        if not (0 <= segment_idx < len(self.annotation.segments)):
            return None

        segment = self.annotation.segments[segment_idx]
        frame_annotations = self._build_frame_annotations(segment)

        return SegmentView(
            segment_idx=segment_idx,
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=segment.duration,
            caption=segment.caption,
            frame_annotations=frame_annotations,
            object_count=len(segment.objects),
            action_count=len(segment.actions),
        )

    def _build_frame_annotations(self, segment: Segment) -> List[FrameAnnotation]:
        """Build frame-level annotations for a segment."""
        frame_annotations: List[FrameAnnotation] = []

        # Use key frames if available, otherwise sample evenly
        if segment.key_frames:
            for frame_idx in segment.key_frames:
                timestamp = frame_idx / self.fps
                fa = FrameAnnotation(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                # Add objects and actions visible at this frame
                fa.objects = [obj.model_dump() for obj in segment.objects]
                fa.actions = [
                    {
                        "verb": a.verb,
                        "noun": a.noun,
                        "hand": a.hand.value if a.hand else None,
                    }
                    for a in segment.actions
                ]
                if segment.caption:
                    fa.caption = segment.caption
                frame_annotations.append(fa)
        else:
            # Sample frames evenly across segment duration
            duration = segment.duration
            num_samples = max(1, min(5, int(duration)))
            for i in range(num_samples):
                timestamp = segment.start_time + (duration * i / max(num_samples - 1, 1))
                frame_idx = int(timestamp * self.fps)
                fa = FrameAnnotation(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                fa.objects = [obj.model_dump() for obj in segment.objects]
                fa.actions = [
                    {
                        "verb": a.verb,
                        "noun": a.noun,
                        "hand": a.hand.value if a.hand else None,
                    }
                    for a in segment.actions
                ]
                if segment.caption:
                    fa.caption = segment.caption
                frame_annotations.append(fa)

        return frame_annotations

    def get_all_segment_views(self) -> List[SegmentView]:
        """Get views for all segments."""
        return [self.get_segment_view(i) for i in range(len(self.annotation.segments))]

    def jump_to_time(self, timestamp: float) -> Optional[Tuple[int, FrameAnnotation]]:
        """Jump to a specific timestamp, returns (segment_idx, frame_annotation)."""
        for seg_idx, segment in enumerate(self.annotation.segments):
            if segment.start_time <= timestamp <= segment.end_time:
                self._current_segment_idx = seg_idx
                # Find closest frame annotation
                frame_annotations = self._build_frame_annotations(segment)
                if not frame_annotations:
                    return seg_idx, None
                closest = min(
                    frame_annotations,
                    key=lambda fa: abs(fa.timestamp - timestamp),
                )
                self._current_frame = closest.frame_idx
                return seg_idx, closest
        return None
