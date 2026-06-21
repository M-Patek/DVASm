"""Tests for CoarseFineSampler adaptive sampling."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.core.adaptive_sampler import CoarseFineSampler, Region, fast_uniform_sample
from dvas.data.video_reader import Frame


class TestFrameDiff:
    """Test frame difference computation."""

    def test_identical_frames(self):
        sampler = CoarseFineSampler()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        diff = sampler._frame_diff(frame, frame)
        assert diff == 0.0

    def test_different_frames(self):
        sampler = CoarseFineSampler()
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        diff = sampler._frame_diff(frame1, frame2)
        assert diff > 0.0

    def test_downsampling(self):
        sampler = CoarseFineSampler()
        # Large frame should trigger downsampling
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        frame2 = np.ones((400, 400, 3), dtype=np.uint8) * 255
        diff = sampler._frame_diff(frame1, frame2)
        assert diff > 0.0


class TestCoarseSampling:
    """Test coarse sampling phase."""

    def test_coarse_sampling_basic(self):
        sampler = CoarseFineSampler(coarse_step=5)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(20)
        ]

        coarse_frames, diffs = sampler._coarse_sampling(frames)

        # Should sample every 5th frame: 0, 5, 10, 15
        assert len(coarse_frames) == 4
        assert coarse_frames[0].idx == 0
        assert coarse_frames[1].idx == 5
        assert len(diffs) == 4
        assert diffs[0] == 0.0  # First frame has no previous

    def test_coarse_sampling_single_frame(self):
        sampler = CoarseFineSampler()
        frames = [Frame(idx=0, timestamp=0.0, data=np.zeros((100, 100, 3), dtype=np.uint8))]

        coarse_frames, diffs = sampler._coarse_sampling(frames)

        assert len(coarse_frames) == 1
        assert len(diffs) == 1


class TestRegionDetection:
    """Test region detection."""

    def test_detect_single_region(self):
        sampler = CoarseFineSampler(diff_threshold=50.0)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(5)
        ]
        diffs = [0.0, 10.0, 20.0, 30.0, 40.0]  # All below threshold

        regions = sampler._detect_regions(frames, diffs)

        assert len(regions) == 1
        assert regions[0].start_frame == 0
        assert regions[0].end_frame == 4

    def test_detect_multiple_regions(self):
        sampler = CoarseFineSampler(diff_threshold=20.0)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(6)
        ]
        # High diff at index 3 triggers new region
        diffs = [0.0, 10.0, 15.0, 50.0, 10.0, 5.0]

        regions = sampler._detect_regions(frames, diffs)

        assert len(regions) >= 1

    def test_detect_regions_empty(self):
        sampler = CoarseFineSampler()
        frames = [Frame(idx=0, timestamp=0.0, data=np.zeros((100, 100, 3), dtype=np.uint8))]
        diffs = [0.0]

        regions = sampler._detect_regions(frames, diffs)

        assert len(regions) == 1


class TestFineSampling:
    """Test fine sampling phase."""

    def test_fine_sampling_basic(self):
        sampler = CoarseFineSampler(target_frames=4)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(20)
        ]
        regions = [Region(start_frame=0, end_frame=20, avg_diff=10.0, importance=1.0)]

        keyframes = sampler._fine_sampling(frames, regions)

        assert len(keyframes) <= 4
        assert len(keyframes) > 0

    def test_fine_sampling_empty(self):
        sampler = CoarseFineSampler(target_frames=4)
        frames = []
        regions = [Region(start_frame=0, end_frame=10, avg_diff=10.0, importance=1.0)]

        keyframes = sampler._fine_sampling(frames, regions)

        assert keyframes == []


class TestCoarseFineSampler:
    """Test full CoarseFineSampler."""

    def test_sample_basic(self):
        sampler = CoarseFineSampler(target_frames=4, coarse_step=5)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(30)
        ]

        keyframes = sampler.sample(frames)

        assert len(keyframes) <= 4
        assert len(keyframes) > 0
        # Should be in temporal order
        for i in range(len(keyframes) - 1):
            assert keyframes[i].idx < keyframes[i + 1].idx

    def test_sample_fewer_than_target(self):
        sampler = CoarseFineSampler(target_frames=10)
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(5)
        ]

        keyframes = sampler.sample(frames)

        assert len(keyframes) == 5

    def test_sample_empty(self):
        sampler = CoarseFineSampler()
        keyframes = sampler.sample([])
        assert keyframes == []

    def test_sample_with_motion(self):
        sampler = CoarseFineSampler(target_frames=4, coarse_step=5)
        frames = []
        for i in range(30):
            # Create frames with varying content
            if 10 <= i < 20:
                data = np.ones((100, 100, 3), dtype=np.uint8) * 255
            else:
                data = np.zeros((100, 100, 3), dtype=np.uint8)
            frames.append(Frame(idx=i, timestamp=i / 30.0, data=data))

        keyframes = sampler.sample(frames)

        assert len(keyframes) <= 4
        assert len(keyframes) > 0


class TestFastUniformSample:
    """Test fast uniform sampling fallback."""

    def test_uniform_sample(self):
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(10)
        ]

        sampled = fast_uniform_sample(frames, 4)

        assert len(sampled) == 4
        assert sampled[0].idx == 0
        assert sampled[-1].idx == 9

    def test_uniform_sample_fewer_than_target(self):
        frames = [
            Frame(idx=i, timestamp=i / 30.0, data=np.zeros((100, 100, 3), dtype=np.uint8))
            for i in range(3)
        ]

        sampled = fast_uniform_sample(frames, 5)

        assert len(sampled) == 3
