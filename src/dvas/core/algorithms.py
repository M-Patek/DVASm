"""Advanced algorithms and data structures for DVAS video processing.

Optimizations:
- Adaptive sampling based on content complexity
- Keyframe extraction using multiple importance metrics
- Video summarization with shot boundary detection
- Semantic-aware processing
- Efficient data structures for video analysis
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np

from dvas.data.video_reader import Frame, VideoReader
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(order=True)
class ScoredFrame:
    """Frame with an importance score for priority queue operations."""

    score: float = field(compare=True)
    frame_idx: int = field(compare=False)
    timestamp: float = field(compare=False)
    data: np.ndarray = field(compare=False, repr=False)


class MinMaxHeap:
    """Min-max heap for efficient median and percentile queries.

    Maintains two heaps: max-heap for lower half, min-heap for upper half.
    Supports O(log n) insertion and O(1) median query.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        self._min_heap: List[Tuple[float, int, Any]] = []  # upper half
        self._max_heap: List[Tuple[float, int, Any]] = []  # lower half (negated)
        self._capacity = capacity
        self._counter = 0  # For stable ordering

    def push(self, score: float, value: Any) -> None:
        """Add a scored item."""
        entry = (score, self._counter, value)
        self._counter += 1

        if not self._max_heap or score <= -self._max_heap[0][0]:
            heapq.heappush(self._max_heap, (-score, self._counter, value))
        else:
            heapq.heappush(self._min_heap, entry)

        self._rebalance()

        # Evict if over capacity (remove from appropriate heap)
        if self._capacity and len(self) > self._capacity:
            self._evict()

    def _rebalance(self) -> None:
        """Rebalance heaps to maintain size invariant."""
        # max_heap can have at most 1 more element than min_heap
        if len(self._max_heap) > len(self._min_heap) + 1:
            _, _, val = heapq.heappop(self._max_heap)
            heapq.heappush(
                self._min_heap,
                (-self._max_heap[-1][0] if self._max_heap else 0, self._counter, val),
            )
        elif len(self._min_heap) > len(self._max_heap):
            _, _, val = heapq.heappop(self._min_heap)
            heapq.heappush(
                self._max_heap, (-self._min_heap[0][0] if self._min_heap else 0, self._counter, val)
            )

    def _evict(self) -> None:
        """Remove lowest scoring item when over capacity."""
        if self._max_heap:
            heapq.heappop(self._max_heap)
        self._rebalance()

    def median(self) -> float:
        """Get median score."""
        if not self._max_heap:
            return 0.0
        if len(self._max_heap) > len(self._min_heap):
            return -self._max_heap[0][0]
        return (-self._max_heap[0][0] + self._min_heap[0][0]) / 2.0

    def percentile(self, p: float) -> float:
        """Get approximate percentile (simplified)."""
        if not self._max_heap:
            return 0.0
        if p <= 50:
            idx = int(len(self._max_heap) * (p / 50.0))
            idx = min(idx, len(self._max_heap) - 1)
            return sorted(-x[0] for x in self._max_heap)[idx]
        else:
            idx = int(len(self._min_heap) * ((p - 50) / 50.0))
            idx = min(idx, len(self._min_heap) - 1)
            return sorted(x[0] for x in self._min_heap)[idx]

    def __len__(self) -> int:
        return len(self._max_heap) + len(self._min_heap)

    def __bool__(self) -> bool:
        return len(self) > 0


class SlidingWindowBuffer:
    """Efficient sliding window for streaming frame analysis.

    Uses circular buffer to avoid memory reallocation.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._buffer: List[Optional[np.ndarray]] = [None] * size
        self._timestamps: List[float] = [0.0] * size
        self._indices: List[int] = [0] * size
        self._head = 0
        self._count = 0

    def push(self, frame: np.ndarray, timestamp: float, idx: int) -> None:
        """Add frame to buffer."""
        pos = self._head % self.size
        self._buffer[pos] = frame
        self._timestamps[pos] = timestamp
        self._indices[pos] = idx
        self._head += 1
        self._count = min(self._count + 1, self.size)

    def get_window(self) -> List[Tuple[np.ndarray, float, int]]:
        """Get current window contents in chronological order."""
        result = []
        for i in range(self._count):
            pos = (self._head - self._count + i) % self.size
            if self._buffer[pos] is not None:
                result.append((self._buffer[pos], self._timestamps[pos], self._indices[pos]))
        return result

    def is_full(self) -> bool:
        return self._count >= self.size

    def clear(self) -> None:
        self._head = 0
        self._count = 0
        self._buffer = [None] * self.size


# ---------------------------------------------------------------------------
# Importance Metrics
# ---------------------------------------------------------------------------


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

    def __init__(self, metrics: Optional[List[Tuple[FrameImportanceMetric, float]]] = None) -> None:
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


# ---------------------------------------------------------------------------
# Adaptive Sampling
# ---------------------------------------------------------------------------


class AdaptiveSampler:
    """Adaptive frame sampling based on content complexity.

    Allocates more frames to complex regions and fewer to static regions.
    """

    def __init__(
        self,
        num_frames: int = 16,
        metric: Optional[FrameImportanceMetric] = None,
        min_frames_per_scene: int = 2,
        window_size: int = 5,
    ) -> None:
        self.num_frames = num_frames
        self.metric = metric or CompositeImportance()
        self.min_frames_per_scene = min_frames_per_scene
        self.window_size = window_size

    def sample(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Frame]:
        """Sample frames adaptively based on content complexity.

        Algorithm:
        1. Score all frames using importance metric
        2. Use min-max heap to track top frames
        3. Ensure temporal coverage (min frames per scene)
        4. Return selected frames in temporal order
        """
        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        total_frames = end_frame - start_frame
        if total_frames <= self.num_frames:
            # Return all frames if fewer than requested
            for frame in reader.read_frames(start_frame, end_frame):
                yield frame
            return

        # Phase 1: Score frames using sliding window for efficiency
        logger.info(
            "adaptive_sampling_start", total_frames=total_frames, target_frames=self.num_frames
        )

        heap = MinMaxHeap(capacity=self.num_frames * 2)  # Oversample then refine
        window = SlidingWindowBuffer(self.window_size)

        frame_idx = 0
        for frame in reader.read_frames(start_frame, end_frame):
            # Calculate importance score
            score = self.metric.score(frame.data)

            # Store in heap
            heap.push(score, (frame_idx, frame.timestamp, frame.data))

            # Update sliding window
            window.push(frame.data, frame.timestamp, frame_idx)
            frame_idx += 1

        if not heap:
            logger.warning("adaptive_sampling_no_frames")
            return

        # Phase 2: Extract top frames and ensure temporal coverage
        # Get all scored frames from heap
        all_frames: List[Tuple[float, int, float, np.ndarray]] = []

        # Collect from both heaps
        for score, _, (idx, ts, data) in heap._max_heap:
            all_frames.append((-score, idx, ts, data))
        for score, _, (idx, ts, data) in heap._min_heap:
            all_frames.append((score, idx, ts, data))

        # Sort by score descending
        all_frames.sort(key=lambda x: x[0], reverse=True)

        # Select top frames with temporal constraint
        selected: List[Tuple[int, float, np.ndarray]] = []
        selected_indices: Set[int] = set()

        # Minimum temporal spacing (in frames)
        min_spacing = max(1, total_frames // (self.num_frames * 2))

        for score, idx, ts, data in all_frames:
            if len(selected) >= self.num_frames:
                break

            # Check temporal spacing
            too_close = any(abs(idx - s_idx) < min_spacing for s_idx in selected_indices)
            if too_close and len(selected) < self.num_frames // 2:
                # Still have room, skip this one
                continue

            selected.append((idx, ts, data))
            selected_indices.add(idx)

        # Phase 3: Sort by time and yield
        selected.sort(key=lambda x: x[0])

        logger.info(
            "adaptive_sampling_complete",
            selected_frames=len(selected),
            avg_score=sum(f[0] for f in all_frames[: len(selected)]) / len(selected)
            if selected
            else 0,
        )

        for idx, ts, data in selected:
            yield Frame(idx=start_frame + idx, timestamp=ts, data=data)


# ---------------------------------------------------------------------------
# Keyframe Extraction
# ---------------------------------------------------------------------------


class KeyframeExtractor:
    """Extract representative keyframes from video using multiple strategies.

    Supports:
    - Uniform temporal distribution
    - Motion-based selection
    - Entropy-based selection
    - Scene-boundary-aware selection
    """

    def __init__(
        self,
        num_keyframes: int = 8,
        strategy: str = "composite",
        min_scene_duration: float = 1.0,
    ) -> None:
        self.num_keyframes = num_keyframes
        self.strategy = strategy
        self.min_scene_duration = min_scene_duration

    def extract(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Frame]:
        """Extract keyframes from video.

        Returns list of Frame objects sorted by timestamp.
        """
        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        duration = (end_time or meta.duration) - (start_time or 0)
        total_frames = end_frame - start_frame

        if total_frames <= self.num_keyframes:
            return list(reader.read_frames(start_frame, end_frame))

        if self.strategy == "uniform":
            return self._extract_uniform(reader, start_frame, end_frame, total_frames)
        elif self.strategy == "motion":
            return self._extract_motion(reader, start_frame, end_frame)
        elif self.strategy == "entropy":
            return self._extract_entropy(reader, start_frame, end_frame)
        elif self.strategy == "composite":
            return self._extract_composite(reader, start_frame, end_frame, duration)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _extract_uniform(
        self, reader: VideoReader, start_frame: int, end_frame: int, total: int
    ) -> List[Frame]:
        """Extract uniformly spaced keyframes."""
        step = max(1, total // self.num_keyframes)
        frames = []
        for i, frame in enumerate(reader.read_frames(start_frame, end_frame, step)):
            if i >= self.num_keyframes:
                break
            frames.append(frame)
        return frames

    def _extract_motion(self, reader: VideoReader, start_frame: int, end_frame: int) -> List[Frame]:
        """Extract keyframes with highest motion."""
        metric = MotionImportance()
        heap: List[Tuple[float, int, Frame]] = []

        for frame in reader.read_frames(start_frame, end_frame):
            score = metric.score(frame.data)
            entry = (score, frame.idx, frame)
            if len(heap) < self.num_keyframes:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

        # Sort by frame index
        return sorted([entry[2] for entry in heap], key=lambda f: f.idx)

    def _extract_entropy(
        self, reader: VideoReader, start_frame: int, end_frame: int
    ) -> List[Frame]:
        """Extract keyframes with highest entropy."""
        metric = HistogramEntropyImportance()
        heap: List[Tuple[float, int, Frame]] = []

        for frame in reader.read_frames(start_frame, end_frame):
            score = metric.score(frame.data)
            entry = (score, frame.idx, frame)
            if len(heap) < self.num_keyframes:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

        return sorted([entry[2] for entry in heap], key=lambda f: f.idx)

    def _extract_composite(
        self, reader: VideoReader, start_frame: int, end_frame: int, duration: float
    ) -> List[Frame]:
        """Extract keyframes using composite strategy with temporal coverage.

        Ensures at least one keyframe per scene-sized chunk.
        """
        num_scenes = max(1, int(duration / self.min_scene_duration))
        frames_per_scene = max(1, self.num_keyframes // num_scenes)

        # First pass: detect rough scene boundaries using simple histogram
        scene_boundaries = self._detect_scene_boundaries(reader, start_frame, end_frame)

        keyframes: List[Frame] = []
        metric = CompositeImportance()

        for i in range(len(scene_boundaries) - 1):
            scene_start = scene_boundaries[i]
            scene_end = scene_boundaries[i + 1]
            scene_frames = list(reader.read_frames(scene_start, scene_end))

            if not scene_frames:
                continue

            # Score frames in this scene
            scored = []
            for frame in scene_frames:
                score = metric.score(frame.data)
                scored.append((score, frame))

            # Take top frames from this scene
            scored.sort(key=lambda x: x[0], reverse=True)
            for j in range(min(frames_per_scene, len(scored))):
                keyframes.append(scored[j][1])

        # Limit to num_keyframes
        if len(keyframes) > self.num_keyframes:
            # Use greedy temporal spacing
            keyframes = self._temporal_subset(keyframes, self.num_keyframes)

        return sorted(keyframes, key=lambda f: f.idx)

    def _detect_scene_boundaries(
        self, reader: VideoReader, start_frame: int, end_frame: int
    ) -> List[int]:
        """Simple scene boundary detection using histogram comparison."""
        boundaries = [start_frame]
        prev_hist: Optional[np.ndarray] = None
        threshold = 0.3  # Normalized correlation threshold

        for frame in reader.read_frames(start_frame, end_frame, step=5):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < threshold:
                    boundaries.append(frame.idx)

            prev_hist = hist

        boundaries.append(end_frame)
        return boundaries

    def _temporal_subset(self, frames: List[Frame], n: int) -> List[Frame]:
        """Select n frames with best temporal coverage using greedy algorithm."""
        if len(frames) <= n:
            return frames

        # Greedy: always pick the frame furthest from already selected
        selected = [frames[0]]
        remaining = frames[1:]

        while len(selected) < n and remaining:
            # Find frame with max min-distance to selected
            best_idx = 0
            best_dist = -1.0

            for i, frame in enumerate(remaining):
                min_dist = min(abs(frame.idx - s.idx) for s in selected)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return sorted(selected, key=lambda f: f.idx)


# ---------------------------------------------------------------------------
# Video Summarization
# ---------------------------------------------------------------------------


@dataclass
class VideoSummary:
    """Result of video summarization."""

    keyframes: List[Frame]
    segments: List[Tuple[float, float]]  # (start, end) time ranges
    total_duration: float
    summary_duration: float
    compression_ratio: float


class VideoSummarizer:
    """Create video summaries by selecting representative segments.

    Uses shot boundary detection and importance scoring to identify
    the most informative segments.
    """

    def __init__(
        self,
        target_duration: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        num_segments: int = 5,
        min_segment_duration: float = 1.0,
    ) -> None:
        self.target_duration = target_duration
        self.compression_ratio = compression_ratio
        self.num_segments = num_segments
        self.min_segment_duration = min_segment_duration

    def summarize(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> VideoSummary:
        """Generate video summary.

        Algorithm:
        1. Detect shot boundaries
        2. Score each shot by importance
        3. Select top segments
        4. Return keyframes and time ranges
        """
        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        total_duration = (end_time or meta.duration) - (start_time or 0)

        # Determine target summary duration
        if self.target_duration:
            target = self.target_duration
        elif self.compression_ratio:
            target = total_duration * self.compression_ratio
        else:
            target = total_duration * 0.3  # Default 30%

        # Detect shots
        shots = self._detect_shots(reader, start_frame, end_frame)
        logger.info("summarizer_shots_detected", num_shots=len(shots))

        if not shots:
            # No shots detected, use uniform segmentation
            shot_duration = total_duration / self.num_segments
            shots = [(i * shot_duration, (i + 1) * shot_duration) for i in range(self.num_segments)]

        # Score shots
        scored_shots = self._score_shots(reader, shots, fps)

        # Select top segments to meet target duration
        selected = self._select_segments(scored_shots, target)

        # Extract keyframes from selected segments
        keyframes = self._extract_keyframes_from_segments(reader, selected, fps)

        summary_duration = sum(end - start for start, end in selected)
        compression = summary_duration / total_duration if total_duration > 0 else 0

        return VideoSummary(
            keyframes=keyframes,
            segments=selected,
            total_duration=total_duration,
            summary_duration=summary_duration,
            compression_ratio=compression,
        )

    def _detect_shots(
        self, reader: VideoReader, start_frame: int, end_frame: int
    ) -> List[Tuple[float, float]]:
        """Detect shot boundaries using histogram comparison."""
        boundaries = []
        prev_hist: Optional[np.ndarray] = None
        shot_start = start_frame

        for frame in reader.read_frames(start_frame, end_frame, step=3):
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                if diff > 20.0:  # Threshold for scene change
                    boundaries.append(
                        (shot_start / reader.metadata.fps, frame.idx / reader.metadata.fps)
                    )
                    shot_start = frame.idx

            prev_hist = hist

        # Add final shot
        boundaries.append((shot_start / reader.metadata.fps, end_frame / reader.metadata.fps))

        # Filter short shots
        return [
            (start, end) for start, end in boundaries if end - start >= self.min_segment_duration
        ]

    def _score_shots(
        self, reader: VideoReader, shots: List[Tuple[float, float]], fps: float
    ) -> List[Tuple[float, Tuple[float, float]]]:
        """Score each shot by visual importance."""
        scored = []
        metric = CompositeImportance()

        for start, end in shots:
            start_f = int(start * fps)
            end_f = int(end * fps)

            # Sample a few frames from the shot
            scores = []
            for frame in reader.read_frames(start_f, end_f, max(1, (end_f - start_f) // 3)):
                scores.append(metric.score(frame.data))

            avg_score = sum(scores) / len(scores) if scores else 0.0
            scored.append((avg_score, (start, end)))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _select_segments(
        self, scored_shots: List[Tuple[float, Tuple[float, float]]], target_duration: float
    ) -> List[Tuple[float, float]]:
        """Select segments to meet target duration."""
        selected = []
        current_duration = 0.0

        for score, (start, end) in scored_shots:
            duration = end - start
            if current_duration + duration <= target_duration:
                selected.append((start, end))
                current_duration += duration
            elif current_duration < target_duration * 0.8:
                # Partial fill if we're under 80% of target
                selected.append((start, end))
                current_duration += duration
                break
            else:
                break

        # Sort by time
        selected.sort(key=lambda x: x[0])
        return selected

    def _extract_keyframes_from_segments(
        self,
        reader: VideoReader,
        segments: List[Tuple[float, float]],
        fps: float,
    ) -> List[Frame]:
        """Extract one keyframe per segment."""
        keyframes = []
        metric = HistogramEntropyImportance()

        for start, end in segments:
            start_f = int(start * fps)
            end_f = int(end * fps)

            best_frame: Optional[Frame] = None
            best_score = -1.0

            for frame in reader.read_frames(start_f, end_f, max(1, (end_f - start_f) // 5)):
                score = metric.score(frame.data)
                if score > best_score:
                    best_score = score
                    best_frame = frame

            if best_frame:
                keyframes.append(best_frame)

        return keyframes


# ---------------------------------------------------------------------------
# Semantic-Aware Processing
# ---------------------------------------------------------------------------


class SemanticSegmenter:
    """Segment video based on semantic content changes.

    Uses low-level features as proxies for semantic changes:
    - Color palette shifts (new objects/scenes)
    - Texture complexity changes (different activities)
    - Motion pattern changes (action transitions)
    """

    def __init__(
        self,
        num_segments: int = 8,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.num_segments = num_segments
        self.feature_weights = feature_weights or {
            "color": 0.4,
            "texture": 0.3,
            "motion": 0.3,
        }

    def segment(
        self,
        reader: VideoReader,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """Segment video into semantically coherent chunks.

        Returns list of (start_time, end_time, confidence) tuples.
        """
        meta = reader.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)

        # Extract features at regular intervals
        features: List[Tuple[int, Dict[str, float]]] = []
        step = max(1, (end_frame - start_frame) // 100)  # Sample ~100 frames

        prev_features: Optional[Dict[str, float]] = None
        prev_gray: Optional[np.ndarray] = None

        for frame in reader.read_frames(start_frame, end_frame, step):
            feat = self._extract_features(frame.data, prev_gray)

            if prev_features is not None:
                # Calculate feature distance
                distance = self._feature_distance(feat, prev_features)
                features.append((frame.idx, distance))

            prev_features = feat
            prev_gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

        if not features:
            # Fallback: uniform segments
            duration = (end_time or meta.duration) - (start_time or 0)
            seg_duration = duration / self.num_segments
            return [
                (i * seg_duration, (i + 1) * seg_duration, 1.0) for i in range(self.num_segments)
            ]

        # Find peaks in feature distance (semantic boundaries)
        boundaries = self._find_boundaries(features, self.num_segments)

        # Convert frame indices to time
        segments = []
        boundaries = [start_frame] + boundaries + [end_frame]
        for i in range(len(boundaries) - 1):
            start_t = boundaries[i] / fps
            end_t = boundaries[i + 1] / fps
            confidence = 1.0  # Could be based on boundary strength
            segments.append((start_t, end_t, confidence))

        return segments

    def _extract_features(
        self, frame: np.ndarray, prev_gray: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Extract semantic features from frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Color features (dominant colors)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        color_entropy = -np.sum(color_hist * np.log2(color_hist + 1e-10))

        # Texture features (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = float(laplacian.var())

        # Motion features
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion = float(magnitude.mean())

        return {
            "color": color_entropy,
            "texture": texture / 1000.0,  # Normalize
            "motion": motion,
        }

    def _feature_distance(self, f1: Dict[str, float], f2: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted distance between feature vectors."""
        distance = {}
        for key in self.feature_weights:
            if key in f1 and key in f2:
                diff = abs(f1[key] - f2[key])
                # Normalize by average magnitude
                avg = (abs(f1[key]) + abs(f2[key])) / 2.0 + 1e-10
                distance[key] = (diff / avg) * self.feature_weights[key]
        return distance

    def _find_boundaries(
        self, features: List[Tuple[int, Dict[str, float]]], num_boundaries: int
    ) -> List[int]:
        """Find semantic boundaries using peak detection."""
        # Calculate total distance at each point
        distances = []
        for idx, dist_dict in features:
            total_dist = sum(dist_dict.values())
            distances.append((idx, total_dist))

        # Smooth distances with moving average
        window = 3
        smoothed = []
        for i in range(len(distances)):
            start = max(0, i - window // 2)
            end = min(len(distances), i + window // 2 + 1)
            avg = sum(distances[j][1] for j in range(start, end)) / (end - start)
            smoothed.append((distances[i][0], avg))

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i][1] > smoothed[i - 1][1] and smoothed[i][1] > smoothed[i + 1][1]:
                peaks.append(smoothed[i])

        # Sort by peak height and take top N
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[: num_boundaries - 1]

        # Sort by frame index
        top_peaks.sort(key=lambda x: x[0])
        return [idx for idx, _ in top_peaks]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AlgorithmRegistry:
    """Registry for algorithm implementations."""

    _samplers: Dict[str, type] = {
        "adaptive": AdaptiveSampler,
    }

    _extractors: Dict[str, type] = {
        "uniform": KeyframeExtractor,
    }

    _summarizers: Dict[str, type] = {
        "default": VideoSummarizer,
    }

    _segmenters: Dict[str, type] = {
        "semantic": SemanticSegmenter,
    }

    @classmethod
    def get_sampler(cls, name: str) -> type:
        if name not in cls._samplers:
            raise ValueError(f"Unknown sampler: {name}")
        return cls._samplers[name]

    @classmethod
    def get_extractor(cls, name: str) -> type:
        if name not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return cls._extractors[name]

    @classmethod
    def get_summarizer(cls, name: str) -> type:
        if name not in cls._summarizers:
            raise ValueError(f"Unknown summarizer: {name}")
        return cls._summarizers[name]

    @classmethod
    def get_segmenter(cls, name: str) -> type:
        if name not in cls._segmenters:
            raise ValueError(f"Unknown segmenter: {name}")
        return cls._segmenters[name]

    @classmethod
    def register_sampler(cls, name: str, sampler_cls: type) -> None:
        cls._samplers[name] = sampler_cls

    @classmethod
    def register_extractor(cls, name: str, extractor_cls: type) -> None:
        cls._extractors[name] = extractor_cls

    @classmethod
    def list_algorithms(cls) -> Dict[str, List[str]]:
        return {
            "samplers": list(cls._samplers.keys()),
            "extractors": list(cls._extractors.keys()),
            "summarizers": list(cls._summarizers.keys()),
            "segmenters": list(cls._segmenters.keys()),
        }
