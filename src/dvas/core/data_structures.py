"""Data structures for DVAS video processing.

Provides efficient data structures for video analysis including scored frames,
min-max heaps, and sliding window buffers.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


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
        # _min_heap stores original scores (min-heap), so its root is the lowest score
        if self._min_heap:
            heapq.heappop(self._min_heap)
        elif self._max_heap:
            # Fallback: remove from _max_heap if _min_heap is empty
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


__all__ = ["ScoredFrame", "MinMaxHeap", "SlidingWindowBuffer"]
