"""Frame strip viewer for keyframe display and comparison.

Provides thumbnail strip visualization with frame selection
and comparison capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dvas.data.schemas import Annotation, Segment
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThumbnailMetadata:
    """Metadata for a thumbnail in a frame strip."""

    frame_idx: int
    timestamp: float
    width: int = 256
    height: int = 144
    format: str = "jpeg"
    source: str = "video"  # "video", "synthetic", "cached"
    cached: bool = False
    cache_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "source": self.source,
            "cached": self.cached,
            "cache_key": self.cache_key,
        }


@dataclass
class FrameStrip:
    """A strip of frames for display."""

    segment_idx: int
    frames: List[ThumbnailMetadata] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    comparison_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_idx": self.segment_idx,
            "frames": [f.to_dict() for f in self.frames],
            "selected_indices": self.selected_indices,
            "comparison_pairs": self.comparison_pairs,
        }


class FrameStripViewer:
    """Viewer for displaying keyframe strips with selection and comparison.

    Generates thumbnail metadata for frame strips and manages
    frame selection and comparison state.
    """

    def __init__(self, annotation: Annotation, fps: float = 30.0):
        self.annotation = annotation
        self.fps = fps
        self._selected_frames: Dict[str, List[int]] = {}  # segment_idx -> frame_indices
        self._comparison_pairs: Dict[str, List[Tuple[int, int]]] = {}

    def generate_strip(
        self,
        segment_idx: int,
        num_frames: int = 5,
        strip_width: int = 256,
        strip_height: int = 144,
    ) -> Optional[FrameStrip]:
        """Generate a frame strip for a segment.

        Args:
            segment_idx: Index of the segment
            num_frames: Number of frames to include in the strip
            strip_width: Width of each thumbnail
            strip_height: Height of each thumbnail

        Returns:
            FrameStrip with thumbnail metadata, or None if segment not found
        """
        if not (0 <= segment_idx < len(self.annotation.segments)):
            return None

        segment = self.annotation.segments[segment_idx]
        frames = self._sample_frames(segment, num_frames, strip_width, strip_height)

        key = str(segment_idx)
        selected = self._selected_frames.get(key, [])
        pairs = self._comparison_pairs.get(key, [])

        return FrameStrip(
            segment_idx=segment_idx,
            frames=frames,
            selected_indices=selected,
            comparison_pairs=pairs,
        )

    def _sample_frames(
        self,
        segment: Segment,
        num_frames: int,
        width: int,
        height: int,
    ) -> List[ThumbnailMetadata]:
        """Sample frames evenly across a segment."""
        duration = segment.duration
        if duration <= 0:
            return []

        frames: List[ThumbnailMetadata] = []
        num_samples = max(1, min(num_frames, int(duration * self.fps)))

        for i in range(num_samples):
            if num_samples == 1:
                timestamp = segment.start_time + duration / 2
            else:
                timestamp = segment.start_time + (duration * i / (num_samples - 1))

            frame_idx = int(timestamp * self.fps)
            frames.append(
                ThumbnailMetadata(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    format="jpeg",
                    source="video",
                )
            )

        return frames

    def select_frame(self, segment_idx: int, frame_idx: int) -> bool:
        """Select a frame for comparison.

        Args:
            segment_idx: Index of the segment
            frame_idx: Frame index to select

        Returns:
            True if selection was successful
        """
        key = str(segment_idx)
        if key not in self._selected_frames:
            self._selected_frames[key] = []

        if frame_idx not in self._selected_frames[key]:
            self._selected_frames[key].append(frame_idx)
            logger.info(
                "frame_selected",
                segment_idx=segment_idx,
                frame_idx=frame_idx,
            )
        return True

    def deselect_frame(self, segment_idx: int, frame_idx: int) -> bool:
        """Deselect a frame.

        Args:
            segment_idx: Index of the segment
            frame_idx: Frame index to deselect

        Returns:
            True if deselection was successful
        """
        key = str(segment_idx)
        if key in self._selected_frames and frame_idx in self._selected_frames[key]:
            self._selected_frames[key].remove(frame_idx)
            # Remove any comparison pairs involving this frame
            if key in self._comparison_pairs:
                self._comparison_pairs[key] = [
                    pair for pair in self._comparison_pairs[key] if frame_idx not in pair
                ]
            logger.info(
                "frame_deselected",
                segment_idx=segment_idx,
                frame_idx=frame_idx,
            )
        return True

    def clear_selection(self, segment_idx: int) -> None:
        """Clear all selections for a segment."""
        key = str(segment_idx)
        self._selected_frames[key] = []
        self._comparison_pairs[key] = []

    def add_comparison_pair(self, segment_idx: int, frame_a: int, frame_b: int) -> bool:
        """Add a comparison pair between two frames.

        Args:
            segment_idx: Index of the segment
            frame_a: First frame index
            frame_b: Second frame index

        Returns:
            True if pair was added
        """
        if frame_a == frame_b:
            return False

        key = str(segment_idx)
        if key not in self._comparison_pairs:
            self._comparison_pairs[key] = []

        pair = (min(frame_a, frame_b), max(frame_a, frame_b))
        if pair not in self._comparison_pairs[key]:
            self._comparison_pairs[key].append(pair)
            # Ensure both frames are selected
            self.select_frame(segment_idx, frame_a)
            self.select_frame(segment_idx, frame_b)
            logger.info(
                "comparison_pair_added",
                segment_idx=segment_idx,
                frame_a=frame_a,
                frame_b=frame_b,
            )
        return True

    def remove_comparison_pair(self, segment_idx: int, frame_a: int, frame_b: int) -> bool:
        """Remove a comparison pair."""
        key = str(segment_idx)
        if key not in self._comparison_pairs:
            return False

        pair = (min(frame_a, frame_b), max(frame_a, frame_b))
        if pair in self._comparison_pairs[key]:
            self._comparison_pairs[key].remove(pair)
            return True
        return False

    def get_comparison_pairs(self, segment_idx: int) -> List[Tuple[int, int]]:
        """Get all comparison pairs for a segment."""
        return self._comparison_pairs.get(str(segment_idx), [])

    def get_selected_frames(self, segment_idx: int) -> List[int]:
        """Get all selected frame indices for a segment."""
        return self._selected_frames.get(str(segment_idx), [])

    def generate_all_strips(
        self,
        num_frames: int = 5,
        strip_width: int = 256,
        strip_height: int = 144,
    ) -> List[FrameStrip]:
        """Generate frame strips for all segments."""
        strips = []
        for i in range(len(self.annotation.segments)):
            strip = self.generate_strip(i, num_frames, strip_width, strip_height)
            if strip:
                strips.append(strip)
        return strips

    def get_thumbnail_metadata(
        self, segment_idx: int, frame_idx: int
    ) -> Optional[ThumbnailMetadata]:
        """Get thumbnail metadata for a specific frame."""
        strip = self.generate_strip(segment_idx, num_frames=1)
        if not strip:
            return None

        for thumb in strip.frames:
            if thumb.frame_idx == frame_idx:
                return thumb
        return None
