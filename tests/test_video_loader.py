"""Tests for video loading and processing utilities."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dvas.data.video_loader import EPICKitchensLoader, VideoLoader
from dvas.data.video_reader import Frame, VideoReader
from dvas.data.frame_sampler import UniformSampler, SamplerConfig
from dvas.data.scene_detector import HistogramSceneDetector
from dvas.data.motion_estimator import OpticalFlowEstimator


class TestVideoReader:
    """Test VideoReader (new focused component)."""

    def test_video_reader_context_manager(self, tmp_path):
        """Test VideoReader context manager opens and closes properly."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                assert reader._cap is not None
                assert reader.video_path == video_path

    def test_video_reader_file_not_found(self, tmp_path):
        """Test VideoReader raises FileNotFoundError for missing file."""
        missing_path = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            VideoReader(missing_path)

    def test_video_reader_metadata(self, tmp_path):
        """Test metadata extraction."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,   # CAP_PROP_FPS
                    3: 1920.0, # CAP_PROP_FRAME_WIDTH
                    4: 1080.0, # CAP_PROP_FRAME_HEIGHT
                    7: 300.0,  # CAP_PROP_FRAME_COUNT
                    6: 1684566380,  # CAP_PROP_FOURCC
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                meta = reader.metadata
                assert meta.fps == 30.0
                assert meta.resolution == [1920, 1080]
                assert meta.total_frames == 300
                assert meta.duration == 10.0

    def test_video_reader_read_frames(self, tmp_path):
        """Test frame reading."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 5:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                frames = list(reader.read_frames())
                assert len(frames) == 5
                assert all(isinstance(f, Frame) for f in frames)


class TestFrameSampler:
    """Test FrameSampler strategies."""

    def test_uniform_sampler(self, tmp_path):
        """Test UniformSampler."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 10:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                config = SamplerConfig(num_frames=3)
                sampler = UniformSampler(config)
                frames = list(sampler.sample(reader))
                assert len(frames) <= 3
                assert all(isinstance(f, Frame) for f in frames)


class TestSceneDetector:
    """Test SceneDetector strategies."""

    def test_histogram_detector(self, tmp_path):
        """Test HistogramSceneDetector."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            # Generate frames with varying colors to trigger scene changes
            frame_idx = [0]
            def mock_read():
                if frame_idx[0] < 10:
                    idx = frame_idx[0]
                    frame_idx[0] += 1
                    # Alternate between two different colors
                    color = np.full((100, 100, 3), idx * 25, dtype=np.uint8)
                    return (True, color)
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                detector = HistogramSceneDetector(threshold=5.0)
                boundaries = detector.detect(reader)
                assert len(boundaries) > 0
                assert all(hasattr(b, "start_time") for b in boundaries)
                assert all(hasattr(b, "end_time") for b in boundaries)


class TestMotionEstimator:
    """Test MotionEstimator strategies."""

    def test_optical_flow_estimator(self, tmp_path):
        """Test OpticalFlowEstimator."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 5:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoReader(video_path) as reader:
                estimator = OpticalFlowEstimator()
                result = estimator.estimate(reader)
                assert 0.0 <= result.score <= 1.0
                assert result.method == "optical_flow"


class TestVideoLoader:
    """Test VideoLoader (coordinator)."""

    def test_video_loader_context_manager(self, tmp_path):
        """Test VideoLoader context manager."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance

            with VideoLoader(video_path) as loader:
                assert loader.video_path == video_path
                assert loader._reader is not None

    def test_video_loader_file_not_found(self, tmp_path):
        """Test VideoLoader raises FileNotFoundError for missing file."""
        missing_path = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            VideoLoader(missing_path)

    def test_video_loader_metadata(self, tmp_path):
        """Test metadata extraction through VideoLoader."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get
            mock_cap.return_value = mock_instance

            with VideoLoader(video_path) as loader:
                meta = loader.metadata
                assert meta.fps == 30.0
                assert meta.resolution == [1920, 1080]

    def test_iter_frames(self, tmp_path):
        """Test frame iteration through VideoLoader."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 5:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoLoader(video_path) as loader:
                frames = list(loader.iter_frames())
                assert len(frames) == 5
                assert all(isinstance(f, Frame) for f in frames)

    def test_read_frames_with_sampling(self, tmp_path):
        """Test read_frames with num_frames sampling."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 10:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoLoader(video_path) as loader:
                frames = list(loader.read_frames(num_frames=3))
                assert len(frames) <= 3

    @pytest.mark.asyncio
    async def test_aiter_frames_async_streaming(self, tmp_path):
        """Test aiter_frames produces frames asynchronously."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True

            def mock_get(prop):
                props = {
                    5: 30.0,
                    3: 1920.0,
                    4: 1080.0,
                    7: 300.0,
                    6: 1684566380,
                }
                return props.get(prop, 0.0)

            mock_instance.get.side_effect = mock_get

            frames_read = [0]
            def mock_read():
                if frames_read[0] < 3:
                    frames_read[0] += 1
                    return (True, np.zeros((100, 100, 3), dtype=np.uint8))
                return (False, None)

            mock_instance.read.side_effect = mock_read
            mock_cap.return_value = mock_instance

            with VideoLoader(video_path) as loader:
                frames = []
                async for frame in loader.aiter_frames():
                    frames.append(frame)

                assert len(frames) == 3
                assert all(isinstance(f, Frame) for f in frames)


class TestEPICKitchensLoader:
    """Test EPIC-KITCHENS loader."""

    def test_init_with_path(self, tmp_path):
        """Test initialization with explicit path."""
        root = tmp_path / "epic"
        root.mkdir()

        loader = EPICKitchensLoader(root_path=root)
        assert loader.root_path == root

    def test_get_video_path(self, tmp_path):
        """Test video path resolution with extension auto-detection."""
        root = tmp_path / "epic"
        (root / "P01" / "videos").mkdir(parents=True)

        # Create a mock video file
        video_file = root / "P01" / "videos" / "P01_01.mp4"
        video_file.write_bytes(b"fake")

        loader = EPICKitchensLoader(root_path=root)

        # Should find the video
        path = loader.get_video_path("P01_01")
        assert path is not None
        assert path.exists()

    def test_get_video_path_not_found(self, tmp_path):
        """Test video path resolution when video doesn't exist."""
        root = tmp_path / "epic"
        root.mkdir()

        loader = EPICKitchensLoader(root_path=root)

        path = loader.get_video_path("P99_99")
        assert path is None

    def test_get_actions_for_video_no_pandas(self, tmp_path):
        """Test get_actions_for_video when pandas is not available."""
        root = tmp_path / "epic"
        root.mkdir()

        loader = EPICKitchensLoader(root_path=root)

        # Should return empty list when pandas is not available
        with patch("dvas.data.video_loader._lazy_import_pandas", return_value=None):
            actions = loader.get_actions_for_video("P01_01")
            assert actions == []

    def test_load_video(self, tmp_path):
        """Test loading a specific video."""
        root = tmp_path / "epic"
        (root / "P01" / "videos").mkdir(parents=True)
        video_file = root / "P01" / "videos" / "P01_01.mp4"
        video_file.write_bytes(b"fake")

        loader = EPICKitchensLoader(root_path=root)

        with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance

            video_loader = loader.load_video("P01_01")
            assert video_loader is not None
