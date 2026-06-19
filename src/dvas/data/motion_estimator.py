"""Motion estimation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from dvas.data.video_reader import VideoReader


@dataclass
class MotionResult:
    """Result of motion estimation."""

    score: float  # 0-1 normalized motion score
    raw_value: float  # Raw motion metric
    method: str  # Estimation method
    confidence: float = 1.0


class MotionEstimator(ABC):
    """Abstract base for motion estimation strategies."""

    @abstractmethod
    def estimate(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_frames: int = 10,
    ) -> MotionResult:
        """Estimate motion intensity in video segment."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Estimator name for logging."""
        pass


class OpticalFlowEstimator(MotionEstimator):
    """Motion estimation using sparse optical flow."""

    @property
    def name(self) -> str:
        return "optical_flow"

    def estimate(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_frames: int = 10,
    ) -> MotionResult:
        """Estimate motion using optical flow."""
        meta = reader.metadata
        start_frame = int((start_time or 0) * meta.fps)
        end_frame = int((end_time or meta.duration) * meta.fps)
        end_frame = min(end_frame, meta.total_frames)

        total_frames = end_frame - start_frame
        if total_frames <= 0:
            return MotionResult(score=0.0, raw_value=0.0, method=self.name)

        step = max(1, total_frames // sample_frames)
        motions = []
        prev_gray: Optional[np.ndarray] = None

        for frame in reader.read_frames(start_frame, end_frame, step):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Detect good features to track for optical flow
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
                )
                if prev_pts is not None:
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray,
                        prev_pts,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                    )
                    if flow[0] is not None and flow[1] is not None:
                        # Calculate magnitude of flow vectors
                        mag = np.linalg.norm(flow[0] - prev_pts, axis=2).mean()
                        motions.append(mag)

            prev_gray = gray

        if not motions:
            return MotionResult(score=0.0, raw_value=0.0, method=self.name)

        raw_value = np.mean(motions)
        # Normalize to 0-1 (empirical scaling)
        score = min(1.0, raw_value / 10.0)

        return MotionResult(
            score=score,
            raw_value=raw_value,
            method=self.name,
            confidence=0.8 if len(motions) > 2 else 0.5,
        )


class FrameDifferenceEstimator(MotionEstimator):
    """Motion estimation using simple frame differencing."""

    @property
    def name(self) -> str:
        return "frame_difference"

    def estimate(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_frames: int = 10,
    ) -> MotionResult:
        """Estimate motion using frame differencing."""
        meta = reader.metadata
        start_frame = int((start_time or 0) * meta.fps)
        end_frame = int((end_time or meta.duration) * meta.fps)
        end_frame = min(end_frame, meta.total_frames)

        total_frames = end_frame - start_frame
        if total_frames <= 0:
            return MotionResult(score=0.0, raw_value=0.0, method=self.name)

        step = max(1, total_frames // sample_frames)
        diffs = []
        prev_gray: Optional[np.ndarray] = None

        for frame in reader.read_frames(start_frame, end_frame, step):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(diff.mean())

            prev_gray = gray

        if not diffs:
            return MotionResult(score=0.0, raw_value=0.0, method=self.name)

        raw_value = np.mean(diffs)
        # Normalize (empirical: 50 is high motion)
        score = min(1.0, raw_value / 50.0)

        return MotionResult(
            score=score,
            raw_value=raw_value,
            method=self.name,
            confidence=0.7 if len(diffs) > 2 else 0.4,
        )


class MotionEstimatorRegistry:
    """Registry of available motion estimation strategies."""

    _strategies: dict = {
        "optical_flow": OpticalFlowEstimator,
        "frame_difference": FrameDifferenceEstimator,
    }

    @classmethod
    def get(cls, name: str) -> type:
        """Get estimator class by name."""
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown estimator: {name}. Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> MotionEstimator:
        """Create estimator instance by name."""
        estimator_cls = cls.get(name)
        return estimator_cls(**kwargs)

    @classmethod
    def register(cls, name: str, estimator_cls: type) -> None:
        """Register a new estimation strategy."""
        cls._strategies[name] = estimator_cls

    @classmethod
    def available(cls) -> List[str]:
        """List available estimator names."""
        return list(cls._strategies.keys())
