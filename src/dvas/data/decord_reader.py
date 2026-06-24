"""Hardware-accelerated video reader using decord.

Provides DecordVideoReader with GPU decoding support via NVDEC.
Falls back to CPU decoding when GPU is unavailable.

Usage::

    from dvas.data.decord_reader import DecordVideoReader

    # CPU decoding
    reader = DecordVideoReader("video.mp4", ctx="cpu")

    # GPU decoding (requires CUDA)
    reader = DecordVideoReader("video.mp4", ctx="cuda:0")

    # Read specific frames (O(1) random access)
    frames = reader.read_frames(start_frame=0, end_frame=100, step=2)

    # Batch read (most efficient)
    frames = reader.get_batch([0, 10, 20, 30])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import numpy as np

from dvas.data.schemas import VideoMetadata
from dvas.data.video_reader import Frame, SUPPORTED_VIDEO_FORMATS
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import decord to avoid hard dependency
try:
    import decord as dec

    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    dec = None  # type: ignore


def _get_decord_ctx(ctx: Union[str, int]):
    """Convert context string to decord context object.

    Args:
        ctx: "cpu", "cuda", "cuda:0", or device index

    Returns:
        decord.Context object
    """
    if not _DECORD_AVAILABLE:
        raise ImportError("decord is not installed. Install with: pip install decord")

    if isinstance(ctx, int):
        return dec.cpu(ctx) if ctx == 0 else dec.gpu(ctx)

    if ctx == "cpu":
        return dec.cpu(0)

    if ctx.startswith("cuda:"):
        device_id = int(ctx.split(":")[1])
        return dec.gpu(device_id)

    if ctx == "cuda":
        return dec.gpu(0)

    return dec.cpu(0)


def get_optimal_video_context(prefer_gpu: bool = True) -> str:
    """自动检测最佳解码上下文。

    按照以下优先级选择解码设备:
    1. PyTorch CUDA (如果可用)
    2. Decord原生GPU支持
    3. CPU解码 (fallback)

    Args:
        prefer_gpu: 是否优先使用GPU解码

    Returns:
        最优解码上下文字符串 ("cuda:N" 或 "cpu")
    """
    if not _DECORD_AVAILABLE or not prefer_gpu:
        return "cpu"

    # 优先检测PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            return f"cuda:{device_id}"
    except ImportError:
        pass

    # 直接检测decord GPU支持
    try:
        dec.gpu(0)
        return "cuda:0"
    except Exception:
        pass

    return "cpu"


class DecordVideoReader:
    """Hardware-accelerated video reader using decord.

    Supports GPU decoding via NVDEC for up to 50x speedup over OpenCV.
    Provides O(1) random access to any frame via get_batch().

    Attributes:
        video_path: Path to the video file
        ctx: Decoding context ("cpu" or "cuda:N")
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        ctx: Union[str, int] = "cpu",
    ):
        if not _DECORD_AVAILABLE:
            raise ImportError(
                "decord is required for DecordVideoReader. Install with: pip install decord"
            )

        self.video_path = Path(video_path)
        self.ctx_str = ctx if isinstance(ctx, str) else f"cpu:{ctx}"
        self._vr: Optional[Any] = None
        self._metadata: Optional[VideoMetadata] = None

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        ext = self.video_path.suffix.lstrip(".").lower()
        if ext and ext not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: '.{ext}'. "
                f"Supported formats: {sorted(SUPPORTED_VIDEO_FORMATS)}"
            )

    def __enter__(self) -> "DecordVideoReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close()

    def _open(self) -> None:
        """Open the video reader."""
        if self._vr is not None:
            return

        decord_ctx = _get_decord_ctx(self.ctx_str)
        self._vr = dec.VideoReader(str(self.video_path), ctx=decord_ctx)
        logger.debug(
            "Opened video with decord",
            path=str(self.video_path),
            ctx=self.ctx_str,
            total_frames=len(self._vr),
        )

    def _close(self) -> None:
        """Close the video reader."""
        if self._vr is not None:
            self._vr = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        if self._metadata is None:
            need_close = self._vr is None
            self._open()

            assert self._vr is not None
            # decord provides: width, height, duration, avg_fps, frame_cnt
            _ = self._vr.get_avg_fps()  # noqa: F841
            total_frames = len(self._vr)

            # Get first frame to determine dimensions
            first_frame = self._vr[0].asnumpy()
            height, width = first_frame.shape[:2]

            # Estimate duration from frame count and fps
            fps = self._vr.get_avg_fps()
            duration = total_frames / fps if fps > 0 else 0

            self._metadata = VideoMetadata(
                fps=fps,
                resolution=[width, height],
                duration=duration,
                total_frames=total_frames,
                codec="unknown",  # decord doesn't expose codec easily
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

        Yields:
            Frame objects
        """
        if self._vr is None:
            self._open()

        assert self._vr is not None
        total_frames = len(self._vr)
        end = end_frame or total_frames
        step = max(1, step)

        # Build list of frame indices
        indices = list(range(start_frame, min(end, total_frames), step))

        if not indices:
            return

        # Use get_batch for efficiency
        batch = self.get_batch(indices)
        yield from batch

    def get_batch(self, indices: List[int]) -> List[Frame]:
        """Read a batch of frames by index (O(1) random access).

        This is the most efficient way to read frames with decord,
        as it uses hardware-accelerated batch decoding.

        Args:
            indices: List of frame indices to read

        Returns:
            List of Frame objects in the same order as indices
        """
        if self._vr is None:
            self._open()

        assert self._vr is not None
        total_frames = len(self._vr)
        fps = self._vr.get_avg_fps()

        # Validate indices
        valid_indices = [i for i in indices if 0 <= i < total_frames]
        if len(valid_indices) != len(indices):
            invalid = set(indices) - set(valid_indices)
            logger.warning(
                "Some frame indices are out of range and will be skipped",
                invalid=sorted(invalid),
                total_frames=total_frames,
            )

        if not valid_indices:
            return []

        # decord.get_batch returns frames in the order of indices
        frames_data = self._vr.get_batch(valid_indices).asnumpy()

        frames = []
        for idx, frame_data in zip(valid_indices, frames_data):
            frames.append(
                Frame(
                    idx=idx,
                    timestamp=idx / fps if fps > 0 else 0,
                    data=frame_data,
                )
            )

        return frames

    def get_frame(self, index: int) -> Optional[Frame]:
        """Read a single frame by index.

        Args:
            index: Frame index

        Returns:
            Frame object or None if out of range
        """
        if self._vr is None:
            self._open()

        assert self._vr is not None
        total_frames = len(self._vr)

        if not (0 <= index < total_frames):
            return None

        frame_data = self._vr[index].asnumpy()
        fps = self._vr.get_avg_fps()

        return Frame(
            idx=index,
            timestamp=index / fps if fps > 0 else 0,
            data=frame_data,
        )

    def get_keyframes(self, max_frames: int = 100) -> List[int]:
        """Get keyframe indices using scene detection.

        Uses frame difference to identify scene changes,
        then samples uniformly within scenes.

        Args:
            max_frames: Maximum number of keyframes to return

        Returns:
            List of keyframe indices
        """
        if self._vr is None:
            self._open()

        assert self._vr is not None
        total_frames = len(self._vr)

        if total_frames <= max_frames:
            return list(range(total_frames))

        # Sample frames uniformly for scene detection
        sample_indices = np.linspace(
            0, total_frames - 1, min(max_frames * 2, total_frames), dtype=int
        )
        frames = self.get_batch(sample_indices.tolist())

        # Compute frame differences
        scene_changes = [0]  # First frame is always a keyframe
        prev_gray = None

        for i, frame in enumerate(frames):
            gray = (
                np.mean(frame.data, axis=2).astype(np.uint8)
                if len(frame.data.shape) == 3
                else frame.data
            )

            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                if diff > 30:  # Threshold for scene change
                    scene_changes.append(sample_indices[i])

            prev_gray = gray

        # If too few scenes, add uniform samples
        if len(scene_changes) < max_frames:
            step = total_frames // max_frames
            extra = list(range(0, total_frames, step))[:max_frames]
            scene_changes = sorted(set(scene_changes + extra))

        return scene_changes[:max_frames]


def create_video_reader(
    video_path: Union[str, Path],
    use_decord: bool = True,
    ctx: Union[str, int] = "cpu",
) -> Union["DecordVideoReader", Any]:
    """Factory function to create the best available video reader.

    Automatically selects DecordVideoReader if decord is installed,
    otherwise falls back to the standard VideoReader.

    Args:
        video_path: Path to the video file
        use_decord: Whether to try decord first
        ctx: Decoding context ("cpu" or "cuda:N")

    Returns:
        VideoReader instance
    """
    from dvas.data.video_reader import VideoReader

    if use_decord and _DECORD_AVAILABLE:
        try:
            return DecordVideoReader(video_path, ctx=ctx)
        except Exception as e:
            logger.warning(
                "Failed to create DecordVideoReader, falling back to VideoReader",
                error=str(e),
                path=str(video_path),
            )
            return VideoReader(video_path)

    return VideoReader(video_path)
