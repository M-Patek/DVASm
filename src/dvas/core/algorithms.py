"""Advanced algorithms and data structures for DVAS video processing.

.. deprecated::
   This module is kept for backward compatibility. Import from the
   specialized submodules directly:
   - `dvas.core.data_structures` — ScoredFrame, MinMaxHeap, SlidingWindowBuffer
   - `dvas.core.frame_metrics` — FrameImportanceMetric, MotionImportance, etc.
   - `dvas.core.video_summary` — AdaptiveSampler, KeyframeExtractor, VideoSummarizer, etc.
"""

from __future__ import annotations

# Data structures
from dvas.core.data_structures import MinMaxHeap, ScoredFrame, SlidingWindowBuffer

# Frame metrics
from dvas.core.frame_metrics import (
    ColorVarianceImportance,
    CompositeImportance,
    EdgeImportance,
    FrameImportanceMetric,
    HistogramEntropyImportance,
    MotionImportance,
)

# Video summary
from dvas.core.video_summary import (
    AdaptiveSampler,
    KeyframeExtractor,
    SemanticSegmenter,
    VideoSummary,
    VideoSummarizer,
)

__all__ = [
    # Data structures
    "ScoredFrame",
    "MinMaxHeap",
    "SlidingWindowBuffer",
    # Frame metrics
    "FrameImportanceMetric",
    "MotionImportance",
    "EdgeImportance",
    "ColorVarianceImportance",
    "HistogramEntropyImportance",
    "CompositeImportance",
    # Video summary
    "AdaptiveSampler",
    "KeyframeExtractor",
    "VideoSummary",
    "VideoSummarizer",
    "SemanticSegmenter",
]
