"""Video reading with minimal responsibility: open, read, close."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from dvas.data.schemas import VideoMetadata


# Formats that OpenCV's default FFmpeg backend can decode. Listing them
# explicitly lets us fail fast on truly unsupported files (e.g. WMV, RM)
# and gives downstream code a stable, introspectable contract.
SUPPORTED_VIDEO_FORMATS: frozenset[str] = frozenset(
    {
        "mp4",
        "m4v",
        "mov",
        "avi",
        "mkv",
        "webm",
        "flv",
        "3gp",
        "3gpp",
        "ts",
        "mpeg",
        "mpg",
        "ogv",
    }
)


@dataclass
class Frame:
    """Video frame with metadata."""

    idx: int
    timestamp: float
    data: np.ndarray


class VideoReader:
    """Minimal video reader. Only responsibility: open video, yield frames, close.

    No sampling, no scene detection, no motion estimation.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        ext = self.video_path.suffix.lstrip(".").lower()
        if ext and ext not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: '.{ext}'. "
                f"Supported formats: {sorted(SUPPORTED_VIDEO_FORMATS)}"
            )

    def __enter__(self) -> "VideoReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close()

    def _open(self) -> None:
        """Open video capture."""
        if self._cap is not None:
            return
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

    def _close(self) -> None:
        """Release video capture."""
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata (lazy-loaded, cached)."""
        if self._metadata is None:
            need_close = self._cap is None
            self._open()

            fps = self._cap.get(cv2.CAP_PROP_FPS)
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Extract codec info
            codec_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            codec = (
                chr(codec_int & 0xFF)
                + chr((codec_int >> 8) & 0xFF)
                + chr((codec_int >> 16) & 0xFF)
                + chr((codec_int >> 24) & 0xFF)
            )

            self._metadata = VideoMetadata(
                fps=fps,
                resolution=[width, height],
                duration=duration,
                total_frames=total_frames,
                codec=codec,
            )

            if need_close:
                self._close()

        return self._metadata

    def read_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> Iterator[Frame]:
        """Read frames sequentially.

        Args:
            start_frame: First frame index to read
            end_frame: Last frame index (exclusive), None for all
            step: Read every Nth frame
        Returns:
            Iterator of Frame objects
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Use 'with' statement.")

        meta = self.metadata
        end = end_frame or meta.total_frames
        step = max(1, step)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current = start_frame
        while current < end:
            # When step > 1, seek directly to target frame instead of reading each one
            if step > 1 and current > start_frame:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, current)

            ret, frame_data = self._cap.read()
            if not ret:
                break

            yield Frame(
                idx=current,
                timestamp=current / meta.fps,
                data=frame_data,
            )
            current += step
