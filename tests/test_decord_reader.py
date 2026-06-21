"""Tests for DecordVideoReader hardware-accelerated video reader.

These tests mock decord to avoid requiring the actual library.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock decord before importing our module
decord_mock = MagicMock()
decord_mock.cpu = MagicMock(return_value="cpu_ctx")
decord_mock.gpu = MagicMock(return_value="gpu_ctx")

# Mock VideoReader class
decord_video_reader_mock = MagicMock()
decord_video_reader_mock.get_avg_fps.return_value = 30.0
decord_video_reader_mock.__len__ = MagicMock(return_value=300)

# Mock frame data
mock_frame = MagicMock()
mock_frame.asnumpy.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
decord_video_reader_mock.__getitem__ = MagicMock(return_value=mock_frame)

# Mock batch result
mock_batch = MagicMock()
mock_batch.asnumpy.return_value = np.zeros((4, 1080, 1920, 3), dtype=np.uint8)
decord_video_reader_mock.get_batch.return_value = mock_batch

decord_mock.VideoReader = MagicMock(return_value=decord_video_reader_mock)

sys.modules["decord"] = decord_mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.data.decord_reader import DecordVideoReader, _get_decord_ctx, create_video_reader
from dvas.data.schemas import VideoMetadata


class TestDecordContext:
    """Test decord context helper."""

    def test_cpu_context(self):
        ctx = _get_decord_ctx("cpu")
        assert ctx == "cpu_ctx"
        decord_mock.cpu.assert_called_with(0)

    def test_cuda_context(self):
        ctx = _get_decord_ctx("cuda:0")
        assert ctx == "gpu_ctx"
        decord_mock.gpu.assert_called_with(0)

    def test_cuda_without_index(self):
        ctx = _get_decord_ctx("cuda")
        assert ctx == "gpu_ctx"
        decord_mock.gpu.assert_called_with(0)

    def test_int_context(self):
        ctx = _get_decord_ctx(0)
        assert ctx == "cpu_ctx"
        decord_mock.cpu.assert_called_with(0)


class TestDecordVideoReaderInit:
    """Test DecordVideoReader initialization."""

    def test_init_with_cpu(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            reader = DecordVideoReader(str(video_file), ctx="cpu")
            assert reader.ctx_str == "cpu"
            assert reader.video_path == video_file

    def test_init_with_cuda(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            reader = DecordVideoReader(str(video_file), ctx="cuda:0")
            assert reader.ctx_str == "cuda:0"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DecordVideoReader("/nonexistent/video.mp4")

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.wmv"
            video_file.write_text("fake video")

            with pytest.raises(ValueError, match="Unsupported video format"):
                DecordVideoReader(str(video_file))


class TestDecordVideoReaderMetadata:
    """Test metadata reading."""

    def test_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            reader = DecordVideoReader(str(video_file))
            meta = reader.metadata

            assert isinstance(meta, VideoMetadata)
            assert meta.fps == 30.0
            assert meta.total_frames == 300
            assert meta.resolution == [1920, 1080]
            assert meta.duration == 10.0  # 300 / 30


class TestDecordVideoReaderFrames:
    """Test frame reading."""

    def test_get_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                frames = reader.get_batch([0, 10, 20, 30])

            assert len(frames) == 4
            assert frames[0].idx == 0
            assert frames[1].idx == 10
            assert frames[2].idx == 20
            assert frames[3].idx == 30
            assert frames[0].timestamp == 0.0
            assert frames[1].timestamp == pytest.approx(0.333, rel=0.01)

    def test_get_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                frame = reader.get_frame(50)

            assert frame is not None
            assert frame.idx == 50
            assert frame.timestamp == pytest.approx(1.667, rel=0.01)

    def test_get_frame_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                frame = reader.get_frame(500)  # > 300

            assert frame is None

    def test_read_frames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                frames = list(reader.read_frames(start_frame=0, end_frame=10, step=2))

            assert len(frames) == 4  # 0, 2, 4, 6 (10 is exclusive)
            assert frames[0].idx == 0
            assert frames[1].idx == 2
            assert frames[2].idx == 4
            assert frames[3].idx == 6

    def test_read_frames_empty_range(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                frames = list(reader.read_frames(start_frame=100, end_frame=100))

            assert len(frames) == 0


class TestDecordVideoReaderKeyframes:
    """Test keyframe extraction."""

    def test_get_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            with DecordVideoReader(str(video_file)) as reader:
                keyframes = reader.get_keyframes(max_frames=10)

            assert len(keyframes) <= 10
            assert keyframes[0] == 0  # First frame always included


class TestCreateVideoReader:
    """Test factory function."""

    def test_returns_decord_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            reader = create_video_reader(str(video_file), use_decord=True)
            assert isinstance(reader, DecordVideoReader)

    def test_returns_standard_when_decord_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            reader = create_video_reader(str(video_file), use_decord=False)
            from dvas.data.video_reader import VideoReader

            assert isinstance(reader, VideoReader)

    def test_fallback_on_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test.mp4"
            video_file.write_text("fake video")

            # Force decord to fail by patching DecordVideoReader init
            with patch.object(
                sys.modules["dvas.data.decord_reader"],
                "DecordVideoReader",
                side_effect=RuntimeError("GPU not available"),
            ):
                reader = create_video_reader(str(video_file), use_decord=True)
                from dvas.data.video_reader import VideoReader

                assert isinstance(reader, VideoReader)
