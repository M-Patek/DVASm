"""Tests for GPUFrameEncoder batch frame encoding."""

import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.core.gpu_encoder import GPUFrameEncoder, _bgr_to_rgb_vectorized, encode_frames_fast


class TestBGRtoRGBVectorized:
    """Test vectorized BGR->RGB conversion."""

    def test_converts_bgr_to_rgb(self):
        # Create a BGR frame
        bgr = np.array([[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]], dtype=np.uint8)

        rgb_frames = _bgr_to_rgb_vectorized([bgr])

        assert len(rgb_frames) == 1
        # BGR [0,0,255] -> RGB [255,0,0]
        assert rgb_frames[0][0, 0, 0] == 255  # R
        assert rgb_frames[0][0, 0, 1] == 0  # G
        assert rgb_frames[0][0, 0, 2] == 0  # B

    def test_handles_non_rgb_frames(self):
        # Grayscale frame (no conversion needed)
        gray = np.array([[128, 200], [50, 255]], dtype=np.uint8)
        result = _bgr_to_rgb_vectorized([gray])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], gray)

    def test_multiple_frames(self):
        bgr1 = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        rgb_frames = _bgr_to_rgb_vectorized([bgr1, bgr2])
        assert len(rgb_frames) == 2


class TestGPUFrameEncoder:
    """Test GPUFrameEncoder."""

    def test_init(self):
        encoder = GPUFrameEncoder(max_batch_size=16)
        assert encoder.max_batch_size == 16
        assert encoder._encoded_count == 0

    def test_encode_single_frame_rgb(self):
        encoder = GPUFrameEncoder()
        # RGB frame
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        encoded = encoder.encode_frame(frame, convert_bgr_to_rgb=False)

        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    def test_encode_single_frame_bgr(self):
        encoder = GPUFrameEncoder()
        # BGR frame (OpenCV default)
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        encoded = encoder.encode_frame(frame, convert_bgr_to_rgb=True)

        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_encode_frames_batch(self):
        encoder = GPUFrameEncoder(max_batch_size=4)
        # Create 5 frames
        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        encoded = encoder.encode_frames(frames, convert_bgr_to_rgb=False)

        assert len(encoded) == 5
        for e in encoded:
            assert isinstance(e, str)
            assert len(e) > 0

    def test_encode_empty_list(self):
        encoder = GPUFrameEncoder()
        result = encoder.encode_frames([])
        assert result == []

    def test_stats(self):
        encoder = GPUFrameEncoder()
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        encoder.encode_frame(frame)

        stats = encoder.stats
        assert stats["encoded_count"] == 1
        assert stats["total_bytes"] > 0
        assert stats["avg_bytes"] > 0

    def test_different_qualities(self):
        encoder = GPUFrameEncoder()
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        low_q = encoder.encode_frame(frame.copy(), quality=10)
        high_q = encoder.encode_frame(frame.copy(), quality=95)

        # Higher quality should produce larger base64 (more data)
        assert len(high_q) >= len(low_q)

    def test_max_batch_size_splitting(self):
        encoder = GPUFrameEncoder(max_batch_size=2)
        # Create 5 frames - should be split into 3 batches (2+2+1)
        frames = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(5)]

        encoded = encoder.encode_frames(frames, convert_bgr_to_rgb=False)
        assert len(encoded) == 5


class TestEncodeFramesFast:
    """Test convenience function."""

    def test_encode_frames_fast(self):
        frames = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        encoded = encode_frames_fast(frames, convert_bgr_to_rgb=False)
        assert len(encoded) == 3
        for e in encoded:
            assert isinstance(e, str)
            assert len(e) > 0
