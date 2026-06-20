"""Ego4D format exporter.

Exports DVAS annotations to Ego4D-compatible format for egocentric
video understanding and robotics research.

Reference: https://ego4d-data.org/
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from dvas.data.schemas import Annotation, AnnotationStandard


class Ego4DFormatter:
    """Formatter for Ego4D dataset format.

    Ego4D format focuses on egocentric video understanding with rich
    narrations, object interactions, and temporal annotations.
    """

    def __init__(
        self,
        include_narrations: bool = True,
        include_3d_annotations: bool = False,
        include_hand_tracking: bool = True,
        include_object_interactions: bool = True,
    ):
        """Initialize formatter.

        Args:
            include_narrations: Include text narrations
            include_3d_annotations: Include 3D spatial annotations
            include_hand_tracking: Include hand pose tracking
            include_object_interactions: Include hand-object interaction data
        """
        self.include_narrations = include_narrations
        self.include_3d_annotations = include_3d_annotations
        self.include_hand_tracking = include_hand_tracking
        self.include_object_interactions = include_object_interactions

    def format_annotation(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert annotation to Ego4D format.

        Args:
            annotation: DVAS Annotation object

        Returns:
            Ego4D-compatible dictionary
        """
        # Build narrations from segments
        narrations = []
        for segment in annotation.segments:
            narration = self._convert_segment_to_narration(segment)
            narrations.append(narration)

        # Build video metadata
        video_metadata = {
            "video_id": annotation.video_id,
            "video_path": annotation.video_path,
            "fps": annotation.metadata.fps,
            "resolution": annotation.metadata.resolution,
            "duration": annotation.metadata.duration,
            "total_frames": annotation.metadata.total_frames,
            "has_audio": annotation.metadata.has_audio,
        }

        # Add camera info if available
        if annotation.metadata.camera_type:
            video_metadata["camera_type"] = annotation.metadata.camera_type
        if annotation.metadata.environment:
            video_metadata["environment"] = annotation.metadata.environment

        # Build Ego4D structure
        ego4d_data = {
            "video_id": annotation.video_id,
            "annotation_id": annotation.id,
            "schema_version": annotation.schema_version,
            "annotation_standard": AnnotationStandard.EGO4D.value,
            "video_metadata": video_metadata,
            "narrations": narrations,
        }

        # Add quality info
        if annotation.quality_score is not None:
            ego4d_data["quality_score"] = annotation.quality_score

        # Add source info
        ego4d_data["source"] = {
            "type": annotation.source,
            "model_version": annotation.model_version,
            "created_at": annotation.created_at.isoformat() if annotation.created_at else None,
        }

        return ego4d_data

    def format_annotations(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Convert multiple annotations to Ego4D format.

        Args:
            annotations: List of DVAS Annotation objects

        Returns:
            List of Ego4D-compatible dictionaries
        """
        return [self.format_annotation(ann) for ann in annotations]

    def _convert_segment_to_narration(self, segment) -> Dict[str, Any]:
        """Convert a segment to Ego4D narration format.

        Args:
            segment: Segment object

        Returns:
            Narration dictionary
        """
        narration = {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "timestamp": (segment.start_time + segment.end_time) / 2,
        }

        # Add narration text
        if self.include_narrations:
            narration["narration_text"] = segment.caption
            if segment.caption_dense:
                narration["narration_dense"] = segment.caption_dense

        # Add key frames
        if segment.key_frames:
            narration["key_frames"] = segment.key_frames

        # Convert actions to Ego4D format
        if segment.actions:
            narration["actions"] = self._convert_actions(segment.actions)

        # Convert objects to Ego4D format
        if segment.objects:
            narration["objects"] = self._convert_objects(segment.objects)

        # Add scene context
        narration["scene_context"] = {
            "scene_type": segment.scene_type or "unknown",
            "lighting": segment.lighting or "unknown",
        }

        # Add temporal relations
        if segment.temporal_relations:
            narration["temporal_relations"] = [
                {
                    "relation": rel.relation,
                    "target_segment_id": rel.target_segment_id,
                    "description": rel.description,
                }
                for rel in segment.temporal_relations
            ]

        # Add hand tracking if enabled
        if self.include_hand_tracking:
            narration["hand_tracking"] = self._extract_hand_tracking(segment)

        # Add object interactions
        if self.include_object_interactions:
            narration["object_interactions"] = self._extract_object_interactions(segment)

        return narration

    def _convert_actions(self, actions: List) -> List[Dict[str, Any]]:
        """Convert actions to Ego4D format.

        Args:
            actions: List of Action objects

        Returns:
            List of action dictionaries
        """
        result = []
        for action in actions:
            action_dict = {
                "verb": action.verb,
                "noun": action.noun,
                "hand": action.hand.value if hasattr(action.hand, "value") else action.hand,
            }

            # Add temporal info
            if action.start_time is not None:
                action_dict["start_time"] = action.start_time
            if action.end_time is not None:
                action_dict["end_time"] = action.end_time

            # Add instrument (Ego4D specific)
            if action.instrument:
                action_dict["instrument"] = action.instrument

            # Add state changes (Ego4D specific)
            if action.source_state or action.target_state:
                action_dict["state_change"] = {
                    "from": action.source_state,
                    "to": action.target_state,
                }

            # Add physical properties
            if action.physical:
                action_dict["physical"] = {
                    "force": action.physical.force,
                    "trajectory": action.physical.trajectory,
                    "contact_type": action.physical.contact_type,
                    "tool": action.physical.tool,
                }

            # Add embodiment data
            if action.embodiment:
                action_dict["embodiment"] = {
                    "gripper_pose": action.embodiment.gripper_pose,
                    "joint_target": action.embodiment.joint_target,
                    "action_space": action.embodiment.action_space,
                    "gripper_state": action.embodiment.gripper_state,
                }

            # Add confidence
            if action.confidence is not None:
                action_dict["confidence"] = action.confidence

            result.append(action_dict)

        return result

    def _convert_objects(self, objects: List) -> List[Dict[str, Any]]:
        """Convert objects to Ego4D format.

        Args:
            objects: List of Object objects

        Returns:
            List of object dictionaries
        """
        result = []
        for obj in objects:
            obj_dict = {
                "name": obj.name,
                "attributes": obj.attributes,
            }

            # Add bounding box
            if obj.bbox:
                obj_dict["bbox"] = {
                    "x1": obj.bbox.x1,
                    "y1": obj.bbox.y1,
                    "x2": obj.bbox.x2,
                    "y2": obj.bbox.y2,
                }

            # Add detection confidence
            if obj.confidence is not None:
                obj_dict["confidence"] = obj.confidence

            # Add state
            if obj.state:
                obj_dict["state"] = obj.state

            # Add material and color (VLA v2.0)
            if obj.material:
                obj_dict["material"] = obj.material
            if obj.color:
                obj_dict["color"] = obj.color

            result.append(obj_dict)

        return result

    def _extract_hand_tracking(self, segment) -> Dict[str, Any]:
        """Extract hand tracking data from segment.

        Args:
            segment: Segment object

        Returns:
            Hand tracking dictionary
        """
        # Placeholder - would extract from segment or action data
        return {
            "left_hand": {
                "visible": False,
                "pose": None,
            },
            "right_hand": {
                "visible": False,
                "pose": None,
            },
        }

    def _extract_object_interactions(self, segment) -> List[Dict[str, Any]]:
        """Extract hand-object interactions from segment.

        Args:
            segment: Segment object

        Returns:
            List of interaction dictionaries
        """
        interactions = []

        for action in segment.actions:
            if action.noun:  # Object being acted upon
                interaction = {
                    "action": action.verb,
                    "object": action.noun,
                    "hand": action.hand.value if hasattr(action.hand, "value") else action.hand,
                    "start_time": action.start_time,
                    "end_time": action.end_time,
                }

                # Add contact info
                if action.physical and action.physical.contact_type:
                    interaction["contact_type"] = action.physical.contact_type

                interactions.append(interaction)

        return interactions

    def export_to_file(
        self,
        annotations: List[Annotation],
        output_path: Path,
        format: str = "jsonl",
    ) -> int:
        """Export annotations to Ego4D format file.

        Args:
            annotations: List of annotations
            output_path: Output file path
            format: "jsonl" or "json"

        Returns:
            Number of videos exported
        """
        ego4d_data = self.format_annotations(annotations)
        output_path = Path(output_path)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ego4d_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif format == "json":
            output = {
                "videos": ego4d_data,
                "metadata": {
                    "format": "ego4d",
                    "version": "1.0",
                    "num_videos": len(ego4d_data),
                },
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        return len(ego4d_data)

    def validate_output(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate Ego4D output structure.

        Args:
            data: Output dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        required_fields = ["video_id", "video_metadata", "narrations"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        if "narrations" in data:
            if not isinstance(data["narrations"], list):
                errors.append("'narrations' must be a list")
            else:
                for i, narr in enumerate(data["narrations"]):
                    if "start_time" not in narr:
                        errors.append(f"Narration {i}: Missing 'start_time'")
                    if "end_time" not in narr:
                        errors.append(f"Narration {i}: Missing 'end_time'")

        return len(errors) == 0, errors
