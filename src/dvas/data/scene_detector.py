"""Scene boundary detection strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from dvas.data.video_reader import VideoReader


@dataclass
class SceneBoundary:
    """A detected scene boundary."""

    start_time: float
    end_time: float
    confidence: float = 1.0  # Detection confidence
    method: str = ""  # Detection method used


class SceneDetector(ABC):
    """Abstract base for scene boundary detection strategies."""

    @abstractmethod
    def detect(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[SceneBoundary]:
        """Detect scene boundaries in video."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name for logging."""
        pass


class HistogramSceneDetector(SceneDetector):
    """Scene detection using histogram comparison."""

    @property
    def name(self) -> str:
        return "histogram"

    def __init__(
        self,
        threshold: float = 30.0,
        min_duration: float = 1.0,
        max_scenes: int = 50,
        sample_rate: int = 2,
    ):
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_scenes = max_scenes
        self.sample_rate = sample_rate

    def detect(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[SceneBoundary]:
        """Detect scene changes using histogram comparison."""
        meta = reader.metadata
        duration = (end_time or meta.duration) - (start_time or 0)
        start_frame = int((start_time or 0) * meta.fps)
        end_frame = int((end_time or meta.duration) * meta.fps)
        end_frame = min(end_frame, meta.total_frames)

        # Adaptive sampling
        total_samples = min(1000, int(duration * self.sample_rate))
        sample_step = max(1, (end_frame - start_frame) // total_samples)

        scenes = []
        current_start = start_time or 0.0
        prev_hist: Optional[np.ndarray] = None

        for frame in reader.read_frames(start_frame, end_frame, sample_step):
            hist = cv2.calcHist([frame.data], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                if diff > self.threshold:
                    if frame.timestamp - current_start >= self.min_duration:
                        scenes.append(
                            SceneBoundary(
                                start_time=current_start,
                                end_time=frame.timestamp,
                                confidence=min(diff / self.threshold, 1.0),
                                method=self.name,
                            )
                        )
                        current_start = frame.timestamp

                        if len(scenes) >= self.max_scenes:
                            break

            prev_hist = hist

        # Add final scene
        final_end = end_time or meta.duration
        if final_end - current_start >= self.min_duration:
            scenes.append(
                SceneBoundary(
                    start_time=current_start,
                    end_time=final_end,
                    confidence=1.0,
                    method=self.name,
                )
            )

        return (
            scenes
            if scenes
            else [SceneBoundary(start_time=start_time or 0.0, end_time=final_end, method=self.name)]
        )


class OpticalFlowSceneDetector(SceneDetector):
    """Scene detection using optical flow magnitude."""

    @property
    def name(self) -> str:
        return "optical_flow"

    def __init__(
        self,
        threshold: float = 0.5,
        min_duration: float = 1.0,
        max_scenes: int = 50,
    ):
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_scenes = max_scenes

    def detect(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[SceneBoundary]:
        """Detect scene changes using optical flow."""
        meta = reader.metadata
        start_frame = int((start_time or 0) * meta.fps)
        end_frame = int((end_time or meta.duration) * meta.fps)
        end_frame = min(end_frame, meta.total_frames)

        scenes = []
        current_start = start_time or 0.0
        prev_gray: Optional[np.ndarray] = None

        for frame in reader.read_frames(start_frame, end_frame):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                avg_motion = magnitude.mean()

                # High motion often indicates scene change
                if avg_motion > self.threshold:
                    if frame.timestamp - current_start >= self.min_duration:
                        scenes.append(
                            SceneBoundary(
                                start_time=current_start,
                                end_time=frame.timestamp,
                                confidence=min(avg_motion / self.threshold, 1.0),
                                method=self.name,
                            )
                        )
                        current_start = frame.timestamp

                        if len(scenes) >= self.max_scenes:
                            break

            prev_gray = gray

        final_end = end_time or meta.duration
        if final_end - current_start >= self.min_duration:
            scenes.append(
                SceneBoundary(
                    start_time=current_start,
                    end_time=final_end,
                    confidence=1.0,
                    method=self.name,
                )
            )

        return (
            scenes
            if scenes
            else [SceneBoundary(start_time=start_time or 0.0, end_time=final_end, method=self.name)]
        )


class SceneDetectorRegistry:
    """Registry of available scene detection strategies."""

    _strategies: dict = {
        "histogram": HistogramSceneDetector,
        "optical_flow": OpticalFlowSceneDetector,
    }

    @classmethod
    def get(cls, name: str) -> type:
        """Get detector class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown detector: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> SceneDetector:
        """Create detector instance by name."""
        detector_cls = cls.get(name)
        return detector_cls(**kwargs)

    @classmethod
    def register(cls, name: str, detector_cls: type) -> None:
        """Register a new detection strategy."""
        cls._strategies[name] = detector_cls

    @classmethod
    def available(cls) -> List[str]:
        """List available detector names."""
        return list(cls._strategies.keys())
