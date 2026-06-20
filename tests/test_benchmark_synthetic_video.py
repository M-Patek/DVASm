"""Tests for synthetic video benchmark."""

import tempfile

import pytest

from dvas.benchmarks.synthetic_video import SyntheticVideo, SyntheticVideoBenchmark


class TestSyntheticVideo:
    """Test SyntheticVideo dataclass."""

    def test_creation(self):
        """Test basic creation."""
        video = SyntheticVideo(
            video_id="sv_001",
            scene_type="kitchen",
            num_objects=5,
            actions=["cut", "wash", "stir"],
            duration=30.0,
            complexity=7,
        )
        assert video.video_id == "sv_001"
        assert video.scene_type == "kitchen"
        assert video.num_objects == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        video = SyntheticVideo(
            video_id="sv_002",
            scene_type="office",
            num_objects=3,
            actions=["type", "click"],
            duration=15.0,
            complexity=4,
        )
        data = video.to_dict()
        assert data["video_id"] == "sv_002"
        assert data["scene_type"] == "office"
        assert data["num_objects"] == 3

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "video_id": "sv_003",
            "scene_type": "garden",
            "num_objects": 8,
            "actions": ["plant", "water", "harvest"],
            "duration": 45.0,
            "complexity": 9,
        }
        video = SyntheticVideo.from_dict(data)
        assert video.video_id == "sv_003"
        assert video.scene_type == "garden"
        assert video.num_objects == 8


class TestSyntheticVideoBenchmark:
    """Test SyntheticVideoBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SyntheticVideoBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "synthetic_video"
        assert temp_benchmark.results_dir.exists()

    def test_evaluate_by_complexity(self, temp_benchmark):
        """Test complexity-based evaluation."""
        videos = [
            SyntheticVideo("v1", "kitchen", 3, ["cut"], 10.0, 3),
            SyntheticVideo("v2", "kitchen", 5, ["wash"], 20.0, 5),
            SyntheticVideo("v3", "office", 8, ["type"], 30.0, 8),
        ]
        predictions = ["cutting", "washing", "typing"]
        ground_truth = ["cutting", "washing", "typing"]
        result = temp_benchmark.evaluate_by_complexity(predictions, ground_truth, videos)
        assert isinstance(result, dict)

    def test_evaluate_by_scene_type(self, temp_benchmark):
        """Test scene type evaluation."""
        videos = [
            SyntheticVideo("v1", "kitchen", 3, ["cut"], 10.0, 3),
            SyntheticVideo("v2", "kitchen", 5, ["wash"], 20.0, 5),
            SyntheticVideo("v3", "office", 8, ["type"], 30.0, 8),
        ]
        predictions = ["cutting", "washing", "typing"]
        ground_truth = ["cutting", "washing", "typing"]
        result = temp_benchmark.evaluate_by_scene_type(predictions, ground_truth, videos)
        assert isinstance(result, dict)

    def test_evaluate_object_count_accuracy(self, temp_benchmark):
        """Test object count accuracy."""
        predicted = [3, 5, 8]
        ground_truth = [3, 5, 8]
        result = temp_benchmark.evaluate_object_count_accuracy(predicted, ground_truth)
        assert "exact_match" in result
        assert "mean_absolute_error" in result
        assert result["exact_match"] == 1.0

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        videos = [
            SyntheticVideo("v1", "kitchen", 3, ["cut"], 10.0, 3),
            SyntheticVideo("v2", "office", 5, ["type"], 20.0, 5),
        ]
        predictions = ["cutting", "typing"]
        ground_truth = ["cutting", "typing"]
        result = temp_benchmark.run_benchmark("test_model", predictions, ground_truth, videos)
        assert result.benchmark_name == "synthetic_video"
        assert result.model_id == "test_model"

    def test_empty_videos(self, temp_benchmark):
        """Test with empty video list."""
        result = temp_benchmark.run_benchmark("test_model", [], [], [])
        assert result.benchmark_name == "synthetic_video"
