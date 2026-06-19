"""Frame sampling strategies for video processing."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple

import numpy as np

from dvas.data.video_reader import Frame, VideoReader


class SamplerConfig:
    """Configuration for frame sampling."""

    def __init__(
        self,
        num_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.resize = resize


class FrameSampler(ABC):
    """Abstract base for frame sampling strategies."""

    @abstractmethod
    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample frames from video according to strategy."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Sampler name for logging."""
        pass

    def _resize_frame(self, frame: Frame) -> Frame:
        """Resize frame if resize config is set."""
        if self.config.resize:
            import cv2

            resized = cv2.resize(frame.data, self.config.resize)
            return Frame(idx=frame.idx, timestamp=frame.timestamp, data=resized)
        return frame

    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()


class UniformSampler(FrameSampler):
    """Uniform sampling: evenly spaced frames across time range."""

    @property
    def name(self) -> str:
        return "uniform"

    def __init__(self, config: Optional[SamplerConfig] = None):
        super().__init__(config)

    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample frames uniformly across time range."""
        meta = reader.metadata
        fps = meta.fps

        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        if self.config.num_frames is None:
            # No limit: return all frames in range
            for frame in reader.read_frames(start_frame, end_frame):
                yield self._resize_frame(frame)
            return

        # Calculate step to achieve desired num_frames
        total_frames_in_range = end_frame - start_frame
        step = max(1, total_frames_in_range // self.config.num_frames)

        yielded = 0
        for frame in reader.read_frames(start_frame, end_frame, step):
            if yielded >= self.config.num_frames:
                break
            yield self._resize_frame(frame)
            yielded += 1


class TemporalSampler(FrameSampler):
    """Sample frames at specific temporal positions."""

    @property
    def name(self) -> str:
        return "temporal"

    def __init__(
        self,
        timestamps: List[float],
        config: Optional[SamplerConfig] = None,
    ):
        super().__init__(config)
        self.timestamps = timestamps

    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample frames at specified timestamps."""
        meta = reader.metadata
        fps = meta.fps

        for ts in self.timestamps:
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue

            frame_idx = int(ts * fps)
            for frame in reader.read_frames(frame_idx, frame_idx + 1):
                yield self._resize_frame(frame)
                break


class KeyFrameSampler(FrameSampler):
    """Sample key frames based on visual importance."""

    @property
    def name(self) -> str:
        return "keyframe"

    def __init__(
        self,
        num_frames: int = 8,
        config: Optional[SamplerConfig] = None,
    ):
        super().__init__(config)
        self.num_frames = num_frames

    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample key frames using motion-based importance."""
        import cv2

        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        # Use a min-heap to track top-N frames by motion score
        # This avoids storing all frames in memory
        import heapq

        heap: List[Tuple[float, int, Optional[np.ndarray]]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        for frame in reader.read_frames(start_frame, end_frame):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Simple frame difference as motion proxy
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(diff.mean())
            else:
                motion = 0.0

            # Store in min-heap, keeping only top num_frames
            # Use negative motion for max-heap behavior
            if len(heap) < self.num_frames:
                heapq.heappush(heap, (motion, frame_idx, frame.data))
            elif motion > heap[0][0]:
                heapq.heapreplace(heap, (motion, frame_idx, frame.data))

            prev_gray = gray
            frame_idx += 1

        if not heap:
            return

        # Sort by original frame index to preserve temporal order
        heap.sort(key=lambda x: x[1])

        for motion, idx, data in heap:
            yield self._resize_frame(
                Frame(idx=start_frame + idx, timestamp=(start_frame + idx) / fps, data=data)
            )


class FrameSamplerRegistry:
    """Registry of available sampling strategies."""

    _strategies: dict = {
        "uniform": UniformSampler,
        "temporal": TemporalSampler,
        "keyframe": KeyFrameSampler,
    }

    @classmethod
    def get(cls, name: str) -> type:
        """Get sampler class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, config: Optional[SamplerConfig] = None, **kwargs) -> FrameSampler:
        """Create sampler instance by name."""
        sampler_cls = cls.get(name)
        return sampler_cls(config=config, **kwargs)

    @classmethod
    def register(cls, name: str, sampler_cls: type) -> None:
        """Register a new sampling strategy."""
        cls._strategies[name] = sampler_cls

    @classmethod
    def available(cls) -> List[str]:
        """List available sampler names."""
        return list(cls._strategies.keys())
