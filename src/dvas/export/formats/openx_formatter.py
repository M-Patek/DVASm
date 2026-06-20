"""Open X-Embodiment format exporter.

Exports DVAS annotations to the Open X-Embodiment dataset format used by
Google's robotics research for training generalist robot policies.

Reference: https://github.com/google-deepmind/open_x_embodiment
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from dvas.data.schemas import Annotation, Action


class OpenXFormatter:
    """Formatter for Open X-Embodiment dataset format.

    The Open X-Embodiment format represents robot episodes as sequences of
    steps, each containing observations (images, state) and actions
    (joint positions, gripper commands).
    """

    # Standard field names in Open X format
    REQUIRED_FIELDS = [
        "steps",
    ]

    # Action space definitions
    ACTION_SPACE_ABSOLUTE = "absolute"
    ACTION_SPACE_DELTA = "delta"

    def __init__(
        self,
        action_space: str = "absolute",
        include_images: bool = True,
        include_depth: bool = False,
        include_state: bool = True,
        image_key: str = "image",
    ):
        """Initialize formatter.

        Args:
            action_space: "absolute" or "delta" action representation
            include_images: Whether to include image paths in output
            include_depth: Whether to include depth data
            include_state: Whether to include robot state
            image_key: Key for image field in observations
        """
        self.action_space = action_space
        self.include_images = include_images
        self.include_depth = include_depth
        self.include_state = include_state
        self.image_key = image_key

    def format_annotation(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert a single annotation to Open X-Embodiment format.

        Args:
            annotation: DVAS Annotation object

        Returns:
            Dictionary in Open X-Embodiment format
        """
        steps = []

        for segment in annotation.segments:
            step = self._convert_segment_to_step(segment, annotation)
            steps.append(step)

        # Build episode metadata
        metadata = {
            "video_id": annotation.video_id,
            "annotation_id": annotation.id,
            "schema_version": annotation.schema_version,
            "source": annotation.source,
            "fps": annotation.metadata.fps,
            "camera_type": annotation.metadata.camera_type or "unknown",
            "environment": annotation.metadata.environment or "unknown",
            "num_steps": len(steps),
        }

        if annotation.model_version:
            metadata["model_version"] = annotation.model_version

        if annotation.quality_score is not None:
            metadata["quality_score"] = annotation.quality_score

        return {
            "steps": steps,
            "metadata": metadata,
        }

    def format_annotations(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Convert multiple annotations to Open X-Embodiment format.

        Args:
            annotations: List of DVAS Annotation objects

        Returns:
            List of dictionaries in Open X-Embodiment format
        """
        return [self.format_annotation(ann) for ann in annotations]

    def _convert_segment_to_step(self, segment, annotation: Annotation) -> Dict[str, Any]:
        """Convert a segment to an Open X step.

        Args:
            segment: Segment object
            annotation: Parent annotation

        Returns:
            Step dictionary
        """
        # Build observation
        observation = self._build_observation(segment, annotation)

        # Build actions
        actions = self._build_actions(segment)

        # Build step
        step = {
            "observation": observation,
            "action": actions,
            "language_instruction": segment.caption,
            "is_terminal": False,  # Will be set True for last step
        }

        # Add timing info
        step["timestamp"] = {
            "start": segment.start_time,
            "end": segment.end_time,
        }

        # Add discount (standard RL discount)
        step["discount"] = 1.0

        # Add reward signal if available (placeholder)
        step["reward"] = 0.0

        return step

    def _build_observation(self, segment, annotation: Annotation) -> Dict[str, Any]:
        """Build observation dictionary for a segment.

        Args:
            segment: Segment object
            annotation: Parent annotation

        Returns:
            Observation dictionary
        """
        observation: Dict[str, Any] = {}

        # Image observations
        if self.include_images:
            observation[self.image_key] = annotation.video_path
            # Add frame indices if key frames are specified
            if segment.key_frames:
                observation["frame_indices"] = segment.key_frames

        # Robot state (placeholder - would be populated from actual robot data)
        if self.include_state:
            observation["state"] = {
                "tcp_pose": None,  # Would be extracted from segment
                "joint_positions": None,
                "gripper_position": None,
            }

        # Scene description
        observation["scene_type"] = segment.scene_type or "unknown"
        observation["lighting"] = segment.lighting or "unknown"

        # Objects in scene
        if segment.objects:
            observation["objects"] = [
                {
                    "name": obj.name,
                    "attributes": obj.attributes,
                    "state": obj.state,
                    "material": getattr(obj, "material", None),
                    "color": getattr(obj, "color", None),
                }
                for obj in segment.objects
            ]

        return observation

    def _build_actions(self, segment) -> List[Dict[str, Any]]:
        """Build action list for a segment.

        Args:
            segment: Segment object

        Returns:
            List of action dictionaries
        """
        actions = []

        for action in segment.actions:
            action_dict = self._convert_action(action)
            actions.append(action_dict)

        return actions

    def _convert_action(self, action: Action) -> Dict[str, Any]:
        """Convert a DVAS Action to Open X action format.

        Args:
            action: DVAS Action object

        Returns:
            Action dictionary
        """
        action_dict: Dict[str, Any] = {
            "verb": action.verb,
            "noun": action.noun,
            "hand": action.hand.value if hasattr(action.hand, "value") else action.hand,
        }

        # Add temporal info
        if action.start_time is not None:
            action_dict["start_time"] = action.start_time
        if action.end_time is not None:
            action_dict["end_time"] = action.end_time

        # Add embodiment action if available
        if action.embodiment:
            action_dict["embodiment_action"] = {
                "gripper_pose": action.embodiment.gripper_pose,
                "joint_target": action.embodiment.joint_target,
                "action_space": self.action_space,  # Use formatter's action_space
                "gripper_state": action.embodiment.gripper_state,
            }
        else:
            # Provide placeholder embodiment structure
            action_dict["embodiment_action"] = {
                "gripper_pose": None,
                "joint_target": None,
                "action_space": self.action_space,
                "gripper_state": None,
            }

        # Add instrument and state info
        if action.instrument:
            action_dict["instrument"] = action.instrument
        if action.source_state:
            action_dict["source_state"] = action.source_state
        if action.target_state:
            action_dict["target_state"] = action.target_state

        # Add physical properties
        if action.physical:
            action_dict["physical"] = {
                "force": action.physical.force,
                "trajectory": action.physical.trajectory,
                "contact_type": action.physical.contact_type,
                "tool": action.physical.tool,
            }

        # Add confidence
        if action.confidence is not None:
            action_dict["confidence"] = action.confidence

        return action_dict

    def export_to_file(
        self,
        annotations: List[Annotation],
        output_path: Path,
        format: str = "jsonl",
    ) -> int:
        """Export annotations to file in Open X-Embodiment format.

        Args:
            annotations: List of annotations to export
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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"episodes": episodes}, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'json'.")

        return len(episodes)

    def validate_output(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate output conforms to Open X-Embodiment format.

        Args:
            data: Output dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if "steps" not in data:
            errors.append("Missing required field: 'steps'")
            return False, errors

        if not isinstance(data["steps"], list):
            errors.append("'steps' must be a list")
            return False, errors

        for i, step in enumerate(data["steps"]):
            if "observation" not in step:
                errors.append(f"Step {i}: Missing 'observation'")
            if "action" not in step:
                errors.append(f"Step {i}: Missing 'action'")
            if "language_instruction" not in step:
                errors.append(f"Step {i}: Missing 'language_instruction'")

        return len(errors) == 0, errors
