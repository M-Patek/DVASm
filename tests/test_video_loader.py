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


class TestVideoFormatSupport:
    """Test multi-format video support (resolves known_gap: MP4-only)."""

    def test_supported_formats_constant_exposes_common_types(self):
        """The SUPPORTED_VIDEO_FORMATS constant must cover at least the formats
        the EPIC-KITCHENS loader resolves to."""
        from dvas.data.video_reader import SUPPORTED_VIDEO_FORMATS

        for ext in ("mp4", "mov", "avi", "mkv", "webm", "m4v"):
            assert ext in SUPPORTED_VIDEO_FORMATS, f"{ext} missing from supported formats"

    @pytest.mark.parametrize("ext", ["mp4", "MP4", "mov", "MOV", "avi", "mkv", "webm", "m4v"])
    def test_video_reader_accepts_supported_extension(self, tmp_path, ext):
        """VideoReader must accept any supported extension without erroring
        on format (it can still fail on open, but format check is early)."""
        video_path = tmp_path / f"sample.{ext}"
        video_path.write_bytes(b"fake_video_data")

        with patch("dvas.data.video_reader.cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance

            reader = VideoReader(video_path)
            assert reader.video_path == video_path

    @pytest.mark.parametrize("ext", ["wmv", "rm", "asf", "vob"])
    def test_video_reader_rejects_unsupported_extension(self, tmp_path, ext):
        """VideoReader must fail fast on truly unsupported formats with a
        clear error message listing valid alternatives."""
        video_path = tmp_path / f"sample.{ext}"
        video_path.write_bytes(b"fake_video_data")

        with pytest.raises(ValueError) as exc_info:
            VideoReader(video_path)

        assert "Unsupported video format" in str(exc_info.value)
        assert ext in str(exc_info.value)

    @pytest.mark.parametrize("ext", [".mkv", ".webm", ".m4v"])
    def test_epic_loader_resolves_additional_formats(self, tmp_path, ext):
        """EPICKitchensLoader.get_video_path must find videos in MKV/WebM/M4V,
        not just MP4/AVI/MOV."""
        root = tmp_path / "epic"
        (root / "P01" / "videos").mkdir(parents=True)
        video_file = root / "P01" / "videos" / f"P01_01{ext}"
        video_file.write_bytes(b"fake")

        loader = EPICKitchensLoader(root_path=root)
        path = loader.get_video_path("P01_01")

        assert path is not None
        assert path.suffix == ext
        assert path.exists()

    def test_epic_loader_prefers_mp4_over_alternates(self, tmp_path):
        """When multiple formats exist, MP4 should win (most common in EPIC)."""
        root = tmp_path / "epic"
        (root / "P01" / "videos").mkdir(parents=True)

        # Create an MKV first
        (root / "P01" / "videos" / "P01_01.mkv").write_bytes(b"fake_mkv")
        # Then the MP4 - should be preferred
        (root / "P01" / "videos" / "P01_01.MP4").write_bytes(b"fake_mp4")

        loader = EPICKitchensLoader(root_path=root)
        path = loader.get_video_path("P01_01")

        assert path.suffix.lower() == ".mp4"
