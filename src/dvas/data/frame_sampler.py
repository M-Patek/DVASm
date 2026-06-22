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
        use_hierarchical: bool = True,  # 启用分层采样优化
        motion_threshold: float = 30.0,  # 场景变化阈值
    ):
        super().__init__(config)
        self.num_frames = num_frames
        self.use_hierarchical = use_hierarchical
        self.motion_threshold = motion_threshold

    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample key frames using motion-based importance with hierarchical optimization."""
        if self.use_hierarchical:
            yield from self._sample_hierarchical(reader, start_time, end_time)
        else:
            yield from self._sample_simple(reader, start_time, end_time)

    def _sample_hierarchical(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """分层采样: 先粗粒度检测，再细粒度精选 (O(n/10)复杂度)."""
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        total_frames = end_frame - start_frame

        # 快速路径: 视频很短，直接全量扫描
        if total_frames <= self.num_frames * 2:
            yield from self._sample_simple(reader, start_time, end_time)
            return

        # Phase 1: 粗粒度扫描 (Coarse sampling)
        coarse_factor = 10  # 每10帧采样1帧
        coarse_step = max(1, total_frames // (self.num_frames * coarse_factor))
        coarse_indices = list(range(start_frame, end_frame, coarse_step))

        # 使用get_batch批量读取提高效率
        coarse_frames = []
        if hasattr(reader, 'get_batch'):
            coarse_frames = reader.get_batch(coarse_indices)
        else:
            for idx in coarse_indices:
                frame = reader.get_frame(idx)
                if frame:
                    coarse_frames.append(frame)

        if not coarse_frames:
            return

        # 计算粗粒度运动分数，检测关键帧区域
        scene_changes = [0]  # 第一帧始终是关键时刻
        prev_gray = None

        for i, frame in enumerate(coarse_frames):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # 快速差异计算
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = float(diff.mean())

                if mean_diff > self.motion_threshold:
                    scene_changes.append(i)

            prev_gray = gray

        # Phase 2: 细粒度精选 (Fine sampling)
        # 在场景变化附近进行细粒度采样
        fine_regions = []
        for change_idx in scene_changes[:self.num_frames]:
            # 扩展区域边界
            region_start = max(0, change_idx - 2)
            region_end = min(len(coarse_frames), change_idx + 3)
            fine_regions.append((region_start, region_end))

        # 并行处理细粒度区域
        def process_region(region):
            start, end = region
            frames = coarse_frames[start:end]
            if not frames:
                return []

            # 选择区域内运动最大的帧
            max_motion = 0.0
            best_frame = frames[0]

            prev = cv2.cvtColor(frames[0].data, cv2.COLOR_BGR2GRAY)
            for frame in frames[1:]:
                gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev)
                motion = float(diff.mean())

                if motion > max_motion:
                    max_motion = motion
                    best_frame = frame

                prev = gray

            return [(max_motion, best_frame)]

        # 使用线程池并行处理区域
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_region, r) for r in fine_regions]
            for f in futures:
                results.extend(f.result())

        if not results:
            # 如果找不到关键帧，使用均匀采样
            yield from UniformSampler(
                SamplerConfig(num_frames=self.num_frames, resize=self.config.resize)
            ).sample(reader, start_time, end_time)
            return

        # 按运动分数排序，选择top-K
        results.sort(key=lambda x: x[0], reverse=True)
        selected = results[:self.num_frames]

        # 保持时间顺序输出
        selected_frames = [f for _, f in selected]
        selected_frames.sort(key=lambda f: f.idx)

        for frame in selected_frames:
            yield self._resize_frame(frame)

    def _sample_simple(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """简单的关键帧采样 (原始实现)."""
        import cv2

        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        import heapq

        heap: List[Tuple[float, int, Optional[np.ndarray]]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        for frame in reader.read_frames(start_frame, end_frame):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(diff.mean())
            else:
                motion = 0.0

            if len(heap) < self.num_frames:
                heapq.heappush(heap, (motion, frame_idx, frame.data))
            elif motion > heap[0][0]:
                heapq.heapreplace(heap, (motion, frame_idx, frame.data))

            prev_gray = gray
            frame_idx += 1

        if not heap:
            return

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
