"""RLDS (Reinforcement Learning Datasets) format exporter.

Exports DVAS annotations to RLDS format used by TensorFlow Datasets
for storing and loading robot learning data efficiently.

Reference: https://github.com/google-research/rlds
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dvas.data.schemas import Annotation


class RLDSFormatter:
    """Formatter for RLDS (Reinforcement Learning Datasets) format.

    RLDS format stores episodes as sequences of steps with observations,
    actions, rewards, and metadata in a structured format optimized for
    RL training.
    """

    def __init__(
        self,
        dataset_name: str = "dvas_robot_dataset",
        description: str = "DVAS robot manipulation dataset",
        include_images: bool = True,
        image_encoding: str = "png",
    ):
        """Initialize formatter.

        Args:
            dataset_name: Name of the dataset
            description: Dataset description
            include_images: Whether to include image references
            image_encoding: Image format (png, jpg)
        """
        self.dataset_name = dataset_name
        self.description = description
        self.include_images = include_images
        self.image_encoding = image_encoding

    def format_annotation(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert a single annotation to RLDS episode format.

        Args:
            annotation: DVAS Annotation object

        Returns:
            RLDS episode dictionary
        """
        steps = []

        for i, segment in enumerate(annotation.segments):
            is_first = i == 0
            is_last = i == len(annotation.segments) - 1

            step = self._convert_segment_to_step(
                segment, annotation, is_first=is_first, is_terminal=is_last
            )
            steps.append(step)

        # Calculate episode metadata
        total_duration = sum(seg.duration for seg in annotation.segments)

        return {
            "steps": steps,
            "episode_metadata": {
                "video_id": annotation.video_id,
                "annotation_id": annotation.id,
                "source": annotation.source,
                "schema_version": annotation.schema_version,
                "total_duration": total_duration,
                "num_steps": len(steps),
                "fps": annotation.metadata.fps,
            },
        }

    def format_annotations(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Convert multiple annotations to RLDS episodes.

        Args:
            annotations: List of DVAS Annotation objects

        Returns:
            List of RLDS episode dictionaries
        """
        return [self.format_annotation(ann) for ann in annotations]

    def _convert_segment_to_step(
        self,
        segment,
        annotation: Annotation,
        is_first: bool = False,
        is_terminal: bool = False,
    ) -> Dict[str, Any]:
        """Convert a segment to RLDS step format.

        Args:
            segment: Segment object
            annotation: Parent annotation
            is_first: Whether this is the first step
            is_terminal: Whether this is the terminal step

        Returns:
            RLDS step dictionary
        """
        # Build observation
        observation = self._build_observation(segment, annotation)

        # Build action
        action = self._build_action(segment)

        # Build step following RLDS structure
        step = {
            # Standard RLDS fields
            "observation": observation,
            "action": action,
            "reward": 0.0,  # Placeholder - would be set from actual reward signal
            "discount": 0.0 if is_terminal else 1.0,
            "is_terminal": is_terminal,
            "is_first": is_first,
            "is_last": is_terminal,

            # Additional metadata
            "language_instruction": segment.caption,
        }

        # Add timestamp info
        step["timestamp"] = {
            "start": segment.start_time,
            "end": segment.end_time,
            "duration": segment.duration,
        }

        return step

    def _build_observation(self, segment, annotation: Annotation) -> Dict[str, Any]:
        """Build RLDS observation dictionary.

        Args:
            segment: Segment object
            annotation: Parent annotation

        Returns:
            Observation dictionary
        """
        observation: Dict[str, Any] = {}

        # Image observation (video reference + frame indices)
        if self.include_images:
            observation["image"] = {
                "video_path": annotation.video_path,
                "frame_indices": segment.key_frames if segment.key_frames else [],
                "encoding": self.image_encoding,
            }

        # Scene context
        observation["scene_context"] = {
            "scene_type": segment.scene_type or "unknown",
            "lighting": segment.lighting or "unknown",
        }

        # Objects in scene
        if segment.objects:
            observation["objects"] = [
                {
                    "name": obj.name,
                    "bbox": obj.bbox.to_list() if obj.bbox else None,
                    "confidence": obj.confidence,
                    "attributes": obj.attributes,
                    "state": obj.state,
                }
                for obj in segment.objects
            ]
        else:
            observation["objects"] = []

        # Robot state placeholder
        observation["robot_state"] = {
            "tcp_position": None,
            "tcp_orientation": None,
            "joint_positions": None,
            "joint_velocities": None,
            "gripper_position": None,
        }

        return observation

    def _build_action(self, segment) -> Dict[str, Any]:
        """Build RLDS action dictionary.

        Args:
            segment: Segment object

        Returns:
            Action dictionary
        """
        if not segment.actions:
            # Return empty/identity action
            return {
                "world_vector": [0.0, 0.0, 0.0],  # XYZ displacement
                "rotation_delta": [0.0, 0.0, 0.0],  # Rotation (Euler or axis-angle)
                "gripper_closedness": 0.0,  # 0=open, 1=closed
                "terminate_episode": 0,  # 0=continue, 1=terminate
                "actions": [],  # List of individual actions
            }

        # Aggregate actions from segment
        actions_list = []
        for action in segment.actions:
            action_dict = {
                "verb": action.verb,
                "noun": action.noun,
                "hand": action.hand.value if hasattr(action.hand, "value") else action.hand,
            }

            # Add embodiment data if available
            if action.embodiment:
                action_dict["gripper_pose"] = action.embodiment.gripper_pose
                action_dict["joint_target"] = action.embodiment.joint_target
                action_dict["gripper_state"] = action.embodiment.gripper_state

            actions_list.append(action_dict)

        # Build composite action representation
        return {
            # Standard RL action space
            "world_vector": self._extract_world_vector(segment.actions),
            "rotation_delta": self._extract_rotation(segment.actions),
            "gripper_closedness": self._extract_gripper_state(segment.actions),
            "terminate_episode": 0,

            # Semantic actions
            "actions": actions_list,
            "natural_language": segment.caption,
        }

    def _extract_world_vector(self, actions: List) -> List[float]:
        """Extract world vector from actions (placeholder).

        In real implementation, this would extract actual displacement
        from robot embodiment data.
        """
        # Placeholder - return zero vector
        return [0.0, 0.0, 0.0]

    def _extract_rotation(self, actions: List) -> List[float]:
        """Extract rotation from actions (placeholder)."""
        return [0.0, 0.0, 0.0]

    def _extract_gripper_state(self, actions: List) -> float:
        """Extract gripper closedness from actions.

        Returns:
            0.0 for open, 1.0 for closed
        """
        if not actions:
            return 0.0

        # Check last action for gripper state
        last_action = actions[-1]
        if last_action.embodiment and last_action.embodiment.gripper_state:
            return 1.0 if last_action.embodiment.gripper_state == "close" else 0.0

        return 0.0

    def export_dataset_info(self, output_path: Path) -> None:
        """Export RLDS dataset info file.

        Args:
            output_path: Path to write dataset_info.json
        """
        dataset_info = {
            "name": self.dataset_name,
            "description": self.description,
            "features": {
                "steps": {
                    "observation": {
                        "image": {"dtype": "string", "shape": []},
                        "scene_context": {"dtype": "string", "shape": []},
                        "objects": {"dtype": "object", "shape": []},
                        "robot_state": {"dtype": "object", "shape": []},
                    },
                    "action": {
                        "world_vector": {"dtype": "float32", "shape": [3]},
                        "rotation_delta": {"dtype": "float32", "shape": [3]},
                        "gripper_closedness": {"dtype": "float32", "shape": []},
                        "terminate_episode": {"dtype": "int32", "shape": []},
                    },
                    "reward": {"dtype": "float32", "shape": []},
                    "discount": {"dtype": "float32", "shape": []},
                    "is_terminal": {"dtype": "bool", "shape": []},
                    "is_first": {"dtype": "bool", "shape": []},
                    "is_last": {"dtype": "bool", "shape": []},
                }
            },
            "splits": {
                "train": {"num_examples": 0, "num_shards": 1},
                "val": {"num_examples": 0, "num_shards": 1},
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

    def export_to_file(
        self,
        annotations: List[Annotation],
        output_path: Path,
        format: str = "jsonl",
    ) -> int:
        """Export annotations to RLDS format files.

        Args:
            annotations: List of annotations
            output_path: Output file path
            format: "jsonl" or "json"

        Returns:
            Number of episodes exported
        """
        episodes = self.format_annotations(annotations)
        output_path = Path(output_path)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for episode in episodes:
                    f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        elif format == "json":
            output = {
                "episodes": episodes,
                "dataset_info": {
                    "name": self.dataset_name,
                    "description": self.description,
                    "num_episodes": len(episodes),
                },
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        return len(episodes)

    def validate_episode(self, episode: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate RLDS episode structure.

        Args:
            episode: Episode dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if "steps" not in episode:
            errors.append("Missing 'steps' field")
            return False, errors

        steps = episode["steps"]
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
            return False, errors

        if len(steps) == 0:
            errors.append("Episode has no steps")
            return False, errors

        for i, step in enumerate(steps):
            if "observation" not in step:
                errors.append(f"Step {i}: Missing 'observation'")
            if "action" not in step:
                errors.append(f"Step {i}: Missing 'action'")
            if "reward" not in step:
                errors.append(f"Step {i}: Missing 'reward'")
            if "discount" not in step:
                errors.append(f"Step {i}: Missing 'discount'")
            if "is_terminal" not in step:
                errors.append(f"Step {i}: Missing 'is_terminal'")
            if "is_first" not in step:
                errors.append(f"Step {i}: Missing 'is_first'")
            if "is_last" not in step:
                errors.append(f"Step {i}: Missing 'is_last'")

        # Check first/last step consistency
        if not steps[0].get("is_first"):
            errors.append("First step must have is_first=True")
        if not steps[-1].get("is_last"):
            errors.append("Last step must have is_last=True")

        return len(errors) == 0, errors
