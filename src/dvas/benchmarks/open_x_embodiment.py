"""Open X-Embodiment benchmark for robotics embodiment tasks.

Evaluates models on robotic manipulation tasks including
gripper pose prediction, action sequence generation, and
embodiment understanding.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.benchmarks.base import BaseBenchmark, BenchmarkResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RobotAction:
    """Robot action for Open X-Embodiment evaluation.

    Attributes:
        gripper_pose: 6-DOF pose [x, y, z, rx, ry, rz]
        joint_positions: Joint positions
        gripper_state: "open" or "close"
        action_type: Type of action ("reach", "grasp", "place", etc.)
    """

    gripper_pose: List[float]
    joint_positions: List[float]
    gripper_state: str
    action_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gripper_pose": self.gripper_pose,
            "joint_positions": self.joint_positions,
            "gripper_state": self.gripper_state,
            "action_type": self.action_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotAction":
        return cls(
            gripper_pose=data["gripper_pose"],
            joint_positions=data["joint_positions"],
            gripper_state=data["gripper_state"],
            action_type=data["action_type"],
        )


class OpenXEmbodimentBenchmark(BaseBenchmark):
    """Open X-Embodiment robotics benchmark.

    Evaluates models on robotic manipulation tasks including
    gripper pose prediction, action sequence generation,
    and embodiment understanding.

    Args:
        benchmark_dir: Directory for storing benchmark data
        dataset_root: Path to Open X-Embodiment dataset root
    """

    # Supported tasks
    TASKS = {
        "gripper_pose_prediction",
        "action_sequence_generation",
        "embodiment_understanding",
    }

    def __init__(
        self,
        benchmark_dir: Union[str, Path],
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__(benchmark_dir, "open_x_embodiment")
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def load_robot_actions(self, split: str = "test") -> List[RobotAction]:
        """Load robot action annotations.

        Args:
            split: Dataset split

        Returns:
            List of RobotAction objects
        """
        if not self.dataset_root or not self.dataset_root.exists():
            logger.warning("Dataset root not found, returning empty actions")
            return []

        action_file = self.dataset_root / f"actions_{split}.json"
        if not action_file.exists():
            logger.warning(f"Action file not found: {action_file}")
            return []

        with open(action_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        actions = []
        for item in data.get("actions", []):
            try:
                actions.append(RobotAction.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid action: {e}")
                continue

        return actions

    def evaluate_gripper_pose(
        self,
        predictions: List[List[float]],
        ground_truth: List[List[float]],
    ) -> Dict[str, float]:
        """Evaluate gripper pose prediction accuracy.

        Args:
            predictions: Predicted 6-DOF poses
            ground_truth: Ground truth 6-DOF poses

        Returns:
            Dictionary with position and rotation errors
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return {"position_error": 0.0, "rotation_error": 0.0}

        position_errors = []
        rotation_errors = []

        for pred, gt in zip(predictions, ground_truth):
            pred = np.array(pred)
            gt = np.array(gt)

            if len(pred) < 6 or len(gt) < 6:
                continue

            # Position error (Euclidean distance)
            pos_error = np.linalg.norm(pred[:3] - gt[:3])
            position_errors.append(pos_error)

            # Rotation error (angular difference)
            rot_error = np.linalg.norm(pred[3:6] - gt[3:6])
            rotation_errors.append(rot_error)

        return {
            "position_error": float(np.mean(position_errors)) if position_errors else 0.0,
            "position_error_std": float(np.std(position_errors)) if position_errors else 0.0,
            "rotation_error": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
            "rotation_error_std": float(np.std(rotation_errors)) if rotation_errors else 0.0,
        }

    def evaluate_action_sequence(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate action sequence generation.

        Args:
            predictions: Predicted action sequences
            ground_truth: Ground truth action sequences

        Returns:
            Dictionary with sequence accuracy metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return {"sequence_accuracy": 0.0, "action_accuracy": 0.0}

        exact_matches = 0
        total_actions = 0
        correct_actions = 0

        for pred_seq, gt_seq in zip(predictions, ground_truth):
            if pred_seq == gt_seq:
                exact_matches += 1

            for pred_action, gt_action in zip(pred_seq, gt_seq):
                total_actions += 1
                if pred_action.strip().lower() == gt_action.strip().lower():
                    correct_actions += 1

        return {
            "sequence_accuracy": exact_matches / len(predictions) if predictions else 0.0,
            "action_accuracy": correct_actions / total_actions if total_actions > 0 else 0.0,
        }

    def evaluate_embodiment_understanding(
        self,
        predictions: List[str],
        ground_truth: List[str],
    ) -> Dict[str, float]:
        """Evaluate embodiment understanding (text-based).

        Args:
            predictions: Predicted descriptions
            ground_truth: Ground truth descriptions

        Returns:
            Dictionary with accuracy and BLEU score
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if not predictions:
            return {"accuracy": 0.0, "bleu": 0.0}

        # Exact match accuracy
        exact_matches = sum(
            1
            for pred, gt in zip(predictions, ground_truth)
            if pred.strip().lower() == gt.strip().lower()
        )

        return {
            "accuracy": exact_matches / len(predictions),
            "bleu": self.compute_bleu(predictions, ground_truth),
        }

    def run_benchmark(
        self,
        model_id: str,
        task: str,
        predictions: Any,
        ground_truth: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run the Open X-Embodiment benchmark for a specific task.

        Args:
            model_id: Identifier for the model
            task: Task type (gripper_pose_prediction, action_sequence_generation, embodiment_understanding)
            predictions: Model predictions
            ground_truth: Ground truth annotations
            metadata: Optional additional metadata

        Returns:
            BenchmarkResult with task-specific metrics

        Raises:
            ValueError: If task is not supported
        """
        if task not in self.TASKS:
            raise ValueError(f"Unsupported task: {task}. Must be one of {self.TASKS}")

        logger.info("Running Open X-Embodiment benchmark", model=model_id, task=task)

        if task == "gripper_pose_prediction":
            metrics = self.evaluate_gripper_pose(predictions, ground_truth)
            pred_texts = [str(p) for p in predictions]
            ref_texts = [str(g) for g in ground_truth]
        elif task == "action_sequence_generation":
            metrics = self.evaluate_action_sequence(predictions, ground_truth)
            pred_texts = [" ".join(p) for p in predictions]
            ref_texts = [" ".join(g) for g in ground_truth]
        elif task == "embodiment_understanding":
            metrics = self.evaluate_embodiment_understanding(predictions, ground_truth)
            pred_texts = predictions
            ref_texts = ground_truth
        else:
            raise ValueError(f"Unsupported task: {task}")

        result = BenchmarkResult(
            benchmark_name=f"open_x_embodiment_{task}",
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            predictions=pred_texts,
            references=ref_texts,
            metadata=metadata or {},
        )

        self._save_result(result)
        logger.info("Open X-Embodiment benchmark complete", task=task, metrics=metrics)
        return result
