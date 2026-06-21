"""Tests for Open X-Embodiment benchmark."""

import tempfile

import pytest

from dvas.benchmarks.open_x_embodiment import OpenXEmbodimentBenchmark, RobotAction


class TestRobotAction:
    """Test RobotAction dataclass."""

    def test_creation(self):
        """Test basic creation."""
        action = RobotAction(
            gripper_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper_state="open",
            action_type="reach",
        )
        assert action.gripper_pose == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        assert action.gripper_state == "open"
        assert action.action_type == "reach"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        action = RobotAction(
            gripper_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper_state="close",
            action_type="grasp",
        )
        data = action.to_dict()
        assert data["gripper_state"] == "close"
        assert data["action_type"] == "grasp"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "gripper_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper_state": "open",
            "action_type": "place",
        }
        action = RobotAction.from_dict(data)
        assert action.gripper_state == "open"
        assert action.action_type == "place"


class TestOpenXEmbodimentBenchmark:
    """Test OpenXEmbodimentBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield OpenXEmbodimentBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "open_x_embodiment"
        assert temp_benchmark.results_dir.exists()

    def test_evaluate_gripper_pose_perfect(self, temp_benchmark):
        """Test gripper pose evaluation with perfect predictions."""
        predictions = [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6, 0.1, 0.1, 0.1],
        ]
        ground_truth = [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6, 0.1, 0.1, 0.1],
        ]

        result = temp_benchmark.evaluate_gripper_pose(predictions, ground_truth)
        assert result["position_error"] == 0.0
        assert result["rotation_error"] == 0.0

    def test_evaluate_gripper_pose_imperfect(self, temp_benchmark):
        """Test gripper pose evaluation with imperfect predictions."""
        predictions = [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
        ]
        ground_truth = [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6, 0.1, 0.1, 0.1],
        ]

        result = temp_benchmark.evaluate_gripper_pose(predictions, ground_truth)
        assert result["position_error"] > 0.0
        assert result["rotation_error"] > 0.0

    def test_evaluate_gripper_pose_empty(self, temp_benchmark):
        """Test gripper pose evaluation with empty inputs."""
        result = temp_benchmark.evaluate_gripper_pose([], [])
        assert result["position_error"] == 0.0
        assert result["rotation_error"] == 0.0

    def test_evaluate_gripper_pose_mismatch_length(self, temp_benchmark):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            temp_benchmark.evaluate_gripper_pose(
                [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]],
                [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.4, 0.5, 0.6, 0.1, 0.1, 0.1]],
            )

    def test_evaluate_action_sequence_perfect(self, temp_benchmark):
        """Test action sequence evaluation with perfect predictions."""
        predictions = [
            ["reach", "grasp", "lift"],
            ["reach", "place", "release"],
        ]
        ground_truth = [
            ["reach", "grasp", "lift"],
            ["reach", "place", "release"],
        ]

        result = temp_benchmark.evaluate_action_sequence(predictions, ground_truth)
        assert result["sequence_accuracy"] == 1.0
        assert result["action_accuracy"] == 1.0

    def test_evaluate_action_sequence_partial(self, temp_benchmark):
        """Test action sequence evaluation with partial matches."""
        predictions = [
            ["reach", "grasp", "lift"],
            ["reach", "place", "release"],
        ]
        ground_truth = [
            ["reach", "grasp", "lift"],
            ["reach", "pour", "release"],
        ]

        result = temp_benchmark.evaluate_action_sequence(predictions, ground_truth)
        assert result["sequence_accuracy"] == 0.5
        assert 0.0 < result["action_accuracy"] < 1.0

    def test_evaluate_action_sequence_empty(self, temp_benchmark):
        """Test action sequence evaluation with empty inputs."""
        result = temp_benchmark.evaluate_action_sequence([], [])
        assert result["sequence_accuracy"] == 0.0
        assert result["action_accuracy"] == 0.0

    def test_evaluate_embodiment_understanding(self, temp_benchmark):
        """Test embodiment understanding evaluation."""
        predictions = ["reach and grasp", "place object"]
        ground_truth = ["reach and grasp", "place object"]

        result = temp_benchmark.evaluate_embodiment_understanding(predictions, ground_truth)
        assert result["accuracy"] == 1.0

    def test_evaluate_embodiment_understanding_empty(self, temp_benchmark):
        """Test embodiment understanding with empty inputs."""
        result = temp_benchmark.evaluate_embodiment_understanding([], [])
        assert result["accuracy"] == 0.0

    def test_run_benchmark_gripper_pose(self, temp_benchmark):
        """Test running gripper pose benchmark."""
        predictions = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]]
        ground_truth = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]]

        result = temp_benchmark.run_benchmark(
            "test_model",
            "gripper_pose_prediction",
            predictions,
            ground_truth,
        )

        assert result.benchmark_name == "open_x_embodiment_gripper_pose_prediction"
        assert result.model_id == "test_model"
        assert "position_error" in result.metrics

    def test_run_benchmark_action_sequence(self, temp_benchmark):
        """Test running action sequence benchmark."""
        predictions = [["reach", "grasp", "lift"]]
        ground_truth = [["reach", "grasp", "lift"]]

        result = temp_benchmark.run_benchmark(
            "test_model",
            "action_sequence_generation",
            predictions,
            ground_truth,
        )

        assert result.benchmark_name == "open_x_embodiment_action_sequence_generation"
        assert result.model_id == "test_model"
        assert result.metrics["sequence_accuracy"] == 1.0

    def test_run_benchmark_unsupported_task(self, temp_benchmark):
        """Test error on unsupported task."""
        with pytest.raises(ValueError):
            temp_benchmark.run_benchmark(
                "test_model",
                "unsupported_task",
                [],
                [],
            )

    def test_load_robot_actions_no_root(self, temp_benchmark):
        """Test loading actions without dataset root."""
        actions = temp_benchmark.load_robot_actions()
        assert actions == []
