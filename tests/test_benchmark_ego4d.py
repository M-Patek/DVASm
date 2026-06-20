"""Tests for Ego4D benchmark."""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.benchmarks.ego4d import Ego4DBenchmark, Ego4DMoment


class TestEgo4DMoment:
    """Test Ego4DMoment dataclass."""

    def test_creation(self):
        """Test basic creation."""
        moment = Ego4DMoment(
            video_id="video_001",
            start_time=10.0,
            end_time=20.0,
            query="When does the person open the fridge?",
            label=True,
        )
        assert moment.video_id == "video_001"
        assert moment.start_time == 10.0
        assert moment.end_time == 20.0
        assert moment.label is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        moment = Ego4DMoment(
            video_id="video_002",
            start_time=5.0,
            end_time=15.0,
            query="Find the cooking scene",
        )
        data = moment.to_dict()
        assert data["video_id"] == "video_002"
        assert data["start_time"] == 5.0
        assert data["query"] == "Find the cooking scene"
        assert data["label"] is True

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "video_id": "video_003",
            "start_time": 30.0,
            "end_time": 45.0,
            "query": "When does the person wash their hands?",
            "label": False,
        }
        moment = Ego4DMoment.from_dict(data)
        assert moment.video_id == "video_003"
        assert moment.start_time == 30.0
        assert moment.end_time == 45.0
        assert moment.label is False


class TestEgo4DBenchmark:
    """Test Ego4DBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Ego4DBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "ego4d"
        assert temp_benchmark.results_dir.exists()

    def test_evaluate_moment_retrieval_perfect(self, temp_benchmark):
        """Test moment retrieval with perfect predictions."""
        predictions = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]
        ground_truth = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]

        result = temp_benchmark.evaluate_moment_retrieval(predictions, ground_truth)
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["mean_iou"] == 1.0

    def test_evaluate_moment_retrieval_partial(self, temp_benchmark):
        """Test moment retrieval with partial overlap."""
        predictions = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]
        ground_truth = [
            Ego4DMoment("v1", 12.0, 18.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]

        result = temp_benchmark.evaluate_moment_retrieval(predictions, ground_truth)
        assert 0.0 < result["mean_iou"] < 1.0
        assert result["recall"] > 0.0

    def test_evaluate_moment_retrieval_different_videos(self, temp_benchmark):
        """Test moment retrieval with different video IDs."""
        predictions = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
        ]
        ground_truth = [
            Ego4DMoment("v2", 10.0, 20.0, "query1"),
        ]

        result = temp_benchmark.evaluate_moment_retrieval(predictions, ground_truth)
        assert result["mean_iou"] == 0.0

    def test_evaluate_moment_retrieval_empty(self, temp_benchmark):
        """Test moment retrieval with empty inputs."""
        result = temp_benchmark.evaluate_moment_retrieval([], [])
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0

    def test_evaluate_action_anticipation(self, temp_benchmark):
        """Test action anticipation evaluation."""
        predictions = ["open", "wash", "cut"]
        ground_truth = ["open", "wash", "cut"]

        result = temp_benchmark.evaluate_action_anticipation(predictions, ground_truth)
        assert result["top1_accuracy"] == 1.0

    def test_evaluate_action_anticipation_partial(self, temp_benchmark):
        """Test action anticipation with partial matches."""
        predictions = ["open", "wash", "cut"]
        ground_truth = ["open", "pour", "cut"]

        result = temp_benchmark.evaluate_action_anticipation(predictions, ground_truth)
        assert result["top1_accuracy"] == pytest.approx(2/3, abs=0.01)

    def test_evaluate_action_anticipation_empty(self, temp_benchmark):
        """Test action anticipation with empty inputs."""
        result = temp_benchmark.evaluate_action_anticipation([], [])
        assert result["top1_accuracy"] == 0.0

    def test_run_benchmark_moment_retrieval(self, temp_benchmark):
        """Test running moment retrieval benchmark."""
        predictions = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]
        ground_truth = [
            Ego4DMoment("v1", 10.0, 20.0, "query1"),
            Ego4DMoment("v2", 5.0, 15.0, "query2"),
        ]

        result = temp_benchmark.run_benchmark(
            "test_model",
            "moment_retrieval",
            predictions,
            ground_truth,
        )

        assert result.benchmark_name == "ego4d_moment_retrieval"
        assert result.model_id == "test_model"
        assert "recall" in result.metrics

    def test_run_benchmark_action_anticipation(self, temp_benchmark):
        """Test running action anticipation benchmark."""
        predictions = ["open", "wash", "cut"]
        ground_truth = ["open", "wash", "cut"]

        result = temp_benchmark.run_benchmark(
            "test_model",
            "action_anticipation",
            predictions,
            ground_truth,
        )

        assert result.benchmark_name == "ego4d_action_anticipation"
        assert result.model_id == "test_model"
        assert result.metrics["top1_accuracy"] == 1.0

    def test_run_benchmark_unsupported_task(self, temp_benchmark):
        """Test error on unsupported task."""
        with pytest.raises(ValueError):
            temp_benchmark.run_benchmark(
                "test_model",
                "unsupported_task",
                [],
                [],
            )

    def test_run_benchmark_moment_type_error(self, temp_benchmark):
        """Test type error for moment retrieval."""
        with pytest.raises(TypeError):
            temp_benchmark.run_benchmark(
                "test_model",
                "moment_retrieval",
                ["not_a_moment"],
                ["not_a_moment"],
            )

    def test_load_moment_annotations_no_root(self, temp_benchmark):
        """Test loading annotations without dataset root."""
        annotations = temp_benchmark.load_moment_annotations()
        assert annotations == []
