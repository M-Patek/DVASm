"""Frame importance metrics for DVAS video processing.

Provides various algorithms for scoring frame importance including
motion, edge density, color variance, and histogram entropy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class FrameImportanceMetric(ABC):
    """Abstract base for frame importance scoring."""

    @abstractmethod
    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate importance score for a frame."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class MotionImportance(FrameImportanceMetric):
    """Score frames by motion magnitude (optical flow)."""

    @property
    def name(self) -> str:
        return "motion"

    def __init__(self, prev_frame: Optional[np.ndarray] = None) -> None:
        self.prev_frame = prev_frame

    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return 0.0

        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        score = float(magnitude.mean())

        self.prev_frame = frame.copy()
        return score


class EdgeImportance(FrameImportanceMetric):
    """Score frames by edge density (structural complexity)."""

    @property
    def name(self) -> str:
        return "edge"

    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.sum()) / (edges.size * 255.0)


class ColorVarianceImportance(FrameImportanceMetric):
    """Score frames by color variance (visual richness)."""

    @property
    def name(self) -> str:
        return "color_variance"

    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        variance = float(np.std(lab, axis=(0, 1)).mean())
        # Normalize to 0-1 range (typical LAB std is 0-100)
        return min(variance / 50.0, 1.0)


class HistogramEntropyImportance(FrameImportanceMetric):
    """Score frames by histogram entropy (information content)."""

    @property
    def name(self) -> str:
        return "entropy"

    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy / 8.0  # Normalize (max entropy for 8-bit is 8)


class CompositeImportance(FrameImportanceMetric):
    """Combine multiple importance metrics with weights."""

    @property
    def name(self) -> str:
        return "composite"

    def __init__(self, metrics: Optional[List[tuple]] = None) -> None:
        self.metrics = metrics or [
            (MotionImportance(), 0.3),
            (EdgeImportance(), 0.2),
            (ColorVarianceImportance(), 0.25),
            (HistogramEntropyImportance(), 0.25),
        ]

    def score(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> float:
        total = 0.0
        for metric, weight in self.metrics:
            total += metric.score(frame, context) * weight
        return total


__all__ = [
    "FrameImportanceMetric",
    "MotionImportance",
    "EdgeImportance",
    "ColorVarianceImportance",
    "HistogramEntropyImportance",
    "CompositeImportance",
]
