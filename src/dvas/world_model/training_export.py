"""Training data export for World Model training formats.

Supports export to multiple world model training formats:
- Sarlo format (Stanford robotics world model)
- SAPIEN format (part-based manipulation)
- Generic trajectory format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvas.data.schemas import Action, Annotation, Segment
from dvas.utils.logging import get_logger
from dvas.world_model.dynamics import ContactDynamics, MotionPrediction, PhysicalDynamics
from dvas.world_model.state_repr import ObjectState, ObjectRole, WorldState
from dvas.world_model.temporal_graph import MultiObjectTransitionGraph, TemporalEventGraph

logger = get_logger(__name__)


@dataclass
class TrajectorySample:
    """Single trajectory sample for world model training.

    Represents a complete trajectory with state transitions,
    actions, and outcomes.

    Attributes:
        trajectory_id: Unique identifier
        video_id: Source video ID
        start_time: Start timestamp
        end_time: End timestamp
        states: List of world states over time
        actions: Actions performed
        dynamics: Physical dynamics annotations
        transitions: State transition graphs
        metadata: Additional metadata
    """

    trajectory_id: str
    video_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    states: List[WorldState] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    dynamics: List[PhysicalDynamics] = field(default_factory=list)
    transitions: Optional[MultiObjectTransitionGraph] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "states": [s.to_dict() for s in self.states],
            "actions": [
                {
                    "verb": a.verb,
                    "noun": a.noun,
                    "hand": a.hand.value,
                    "start_time": a.start_time,
                    "end_time": a.end_time,
                }
                for a in self.actions
            ],
            "dynamics": [d.to_dict() for d in self.dynamics],
            "transitions": self.transitions.to_dict() if self.transitions else None,
            "metadata": self.metadata,
        }


class SarloExporter:
    """Export to Sarlo (Stanford Robotics) format.

    Sarlo format focuses on:
    - Language-conditioned trajectory generation
    - Object-centric state representations
    - Action-conditioned dynamics
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_trajectory(
        self,
        sample: TrajectorySample,
        language_instruction: Optional[str] = None,
    ) -> Path:
        """Export a single trajectory in Sarlo format.

        Args:
            sample: Trajectory sample to export
            language_instruction: Natural language task description

        Returns:
            Path to exported file
        """
        # Build Sarlo format
        sarlo_data: Dict[str, Any] = {
            "trajectory_id": sample.trajectory_id,
            "video_id": sample.video_id,
            "language_instruction": language_instruction or self._generate_instruction(sample),
            "duration": sample.end_time - sample.start_time,
            "fps": 30,  # Assumed, should come from metadata
            "objects": self._extract_objects_sarlo(sample),
            "trajectory": self._extract_trajectory_sarlo(sample),
            "actions": self._extract_actions_sarlo(sample),
        }

        output_path = self.output_dir / f"{sample.trajectory_id}_sarlo.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sarlo_data, f, indent=2)

        logger.info("sarlo_export_complete", path=str(output_path))
        return output_path

    def export_batch(
        self,
        samples: List[TrajectorySample],
        split: str = "train",
    ) -> Path:
        """Export a batch of trajectories.

        Args:
            samples: List of trajectory samples
            split: Dataset split (train/val/test)

        Returns:
            Path to batch directory
        """
        batch_dir = self.output_dir / split
        batch_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict[str, Any]] = []

        for sample in samples:
            path = self.export_trajectory(sample)
            manifest.append({
                "trajectory_id": sample.trajectory_id,
                "file": str(path.relative_to(batch_dir)),
                "duration": sample.end_time - sample.start_time,
            })

        # Write manifest
        manifest_path = batch_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "sarlo_batch_export_complete",
            split=split,
            count=len(samples),
            path=str(batch_dir),
        )
        return batch_dir

    def _extract_objects_sarlo(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract object descriptions in Sarlo format."""
        objects: Dict[str, Dict[str, Any]] = {}

        for state in sample.states:
            for obj_id, obj in state.scene_graph.objects.items():
                if obj_id not in objects:
                    objects[obj_id] = {
                        "object_id": obj_id,
                        "object_name": obj.name,
                        "category": self._categorize_object(obj),
                        "properties": {
                            "mass": obj.mass,
                            "material": obj.material,
                            "affordances": [a.value for a in obj.affordances],
                        },
                    }

        return list(objects.values())

    def _extract_trajectory_sarlo(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract trajectory in Sarlo format."""
        frames = []

        for i, state in enumerate(sample.states):
            frame = {
                "frame_idx": i,
                "timestamp": state.timestamp,
                "object_states": [
                    {
                        "object_id": obj_id,
                        "position": obj.position.tolist(),
                        "orientation": obj.orientation.tolist(),
                        "state": obj.state,
                        "is_visible": obj.is_visible,
                    }
                    for obj_id, obj in state.scene_graph.objects.items()
                ],
                "agent_state": self._get_agent_state(state),
            }
            frames.append(frame)

        return frames

    def _extract_actions_sarlo(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract actions in Sarlo format."""
        return [
            {
                "verb": a.verb,
                "noun": a.noun,
                "hand": a.hand.value,
                "start_time": a.start_time,
                "end_time": a.end_time,
                "descriptor": f"{a.verb} the {a.noun}",
            }
            for a in sample.actions
        ]

    def _get_agent_state(self, state: WorldState) -> Optional[Dict[str, Any]]:
        """Extract agent state."""
        agent = state.get_agent()
        if agent is None:
            return None
        return {
            "position": agent.position.tolist(),
            "hand_position": state.get_agent_hand_position("right"),
        }

    def _categorize_object(self, obj: ObjectState) -> str:
        """Categorize object for Sarlo."""
        if obj.role == ObjectRole.INSTRUMENT:
            return "tool"
        elif obj.role == ObjectRole.CONTAINER:
            return "container"
        elif obj.role == ObjectRole.SUPPORT:
            return "surface"
        elif obj.affordances:
            if any(a.value in ["graspable", "liftable"] for a in obj.affordances):
                return "manipulable"
        return "static"

    def _generate_instruction(self, sample: TrajectorySample) -> str:
        """Generate language instruction from actions."""
        if sample.actions:
            actions = [f"{a.verb} the {a.noun}" for a in sample.actions[:3]]
            return ", then ".join(actions)
        return "Perform the manipulation task"


class SapienExporter:
    """Export to SAPIEN format for part-based manipulation.

    SAPIEN format focuses on:
    - Articulated part semantics
    - Contact-rich manipulation
    - Physics simulation parameters
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_trajectory(
        self,
        sample: TrajectorySample,
        urdf_path: Optional[str] = None,
    ) -> Path:
        """Export in SAPIEN format.

        Args:
            sample: Trajectory sample
            urdf_path: Optional path to URDF model

        Returns:
            Path to exported file
        """
        sapien_data: Dict[str, Any] = {
            "metadata": {
                "task_id": sample.trajectory_id,
                "video_id": sample.video_id,
                "duration": sample.end_time - sample.start_time,
                "urdf_path": urdf_path,
            },
            "objects": self._extract_objects_sapien(sample),
            "articulations": self._extract_articulations(sample),
            "contacts": self._extract_contacts_sapien(sample),
            "motion": self._extract_motion_sapien(sample),
        }

        output_path = self.output_dir / f"{sample.trajectory_id}_sapien.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sapien_data, f, indent=2)

        logger.info("sapien_export_complete", path=str(output_path))
        return output_path

    def export_batch(
        self,
        samples: List[TrajectorySample],
        split: str = "train",
    ) -> Path:
        """Export batch to SAPIEN format."""
        batch_dir = self.output_dir / split
        batch_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            self.export_trajectory(sample)

        logger.info(
            "sapien_batch_export_complete",
            split=split,
            count=len(samples),
        )
        return batch_dir

    def _extract_objects_sapien(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract objects in SAPIEN format."""
        objects: Dict[str, Dict[str, Any]] = {}

        for state in sample.states:
            for obj_id, obj in state.scene_graph.objects.items():
                if obj_id not in objects:
                    objects[obj_id] = {
                        "name": obj.name,
                        "id": obj_id,
                        "part_semantics": self._infer_part_semantics(obj),
                        "initial_state": {
                            "position": obj.position.tolist(),
                            "orientation": obj.orientation.tolist(),
                        },
                        "properties": {
                            "static": not obj.is_moving(0.01),
                            "articulated": "joint" in obj.attributes,
                        },
                    }

        return list(objects.values())

    def _extract_articulations(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract articulation information."""
        articulations = []

        # Look for objects with joint information
        for state in sample.states:
            for obj_id, obj in state.scene_graph.objects.items():
                if "joint" in obj.attributes:
                    articulations.append({
                        "object_id": obj_id,
                        "joint_type": obj.attributes.get("joint_type", "revolute"),
                        "joint_limits": obj.attributes.get("joint_limits", [-3.14, 3.14]),
                        "initial_qpos": obj.attributes.get("initial_qpos", 0.0),
                    })

        return articulations

    def _extract_contacts_sapien(self, sample: TrajectorySample) -> List[Dict[str, Any]]:
        """Extract contact information."""
        contacts = []

        for dynamics in sample.dynamics:
            for event in dynamics.contact_events:
                contacts.append({
                    "object_a": event.subject_id,
                    "object_b": event.object_id,
                    "contact_type": event.contact_type.value,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "force_magnitude": event.force.magnitude if event.force else None,
                })

        return contacts

    def _extract_motion_sapien(self, sample: TrajectorySample) -> Dict[str, Any]:
        """Extract motion trajectories."""
        motion = {
            "timestamps": [s.timestamp for s in sample.states],
            "object_motions": {},
        }

        # Aggregate trajectories
        for state in sample.states:
            for obj_id, obj in state.scene_graph.objects.items():
                if obj_id not in motion["object_motions"]:
                    motion["object_motions"][obj_id] = {
                        "positions": [],
                        "orientations": [],
                        "velocities": [],
                    }
                motion["object_motions"][obj_id]["positions"].append(
                    obj.position.tolist()
                )
                motion["object_motions"][obj_id]["orientations"].append(
                    obj.orientation.tolist()
                )
                motion["object_motions"][obj_id]["velocities"].append(
                    obj.velocity.tolist()
                )

        return motion

    def _infer_part_semantics(self, obj: ObjectState) -> str:
        """Infer part semantics from object properties."""
        name_lower = obj.name.lower()

        if "handle" in name_lower or "knob" in name_lower:
            return "handle"
        elif "lid" in name_lower or "cap" in name_lower:
            return "lid"
        elif "drawer" in name_lower:
            return "drawer"
        elif "door" in name_lower:
            return "door"
        elif "button" in name_lower:
            return "button"
        elif "lever" in name_lower:
            return "lever"

        return "base"


class GenericExporter:
    """Generic trajectory exporter supporting multiple formats."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sarlo = SarloExporter(output_dir / "sarlo")
        self.sapien = SapienExporter(output_dir / "sapien")

    def export_annotation(
        self,
        annotation: Annotation,
        format: str = "sarlo",
    ) -> Path:
        """Export annotation in specified format.

        Args:
            annotation: Annotation to export
            format: Export format ("sarlo", "sapien", or "json")

        Returns:
            Path to exported file
        """
        # Convert annotation to trajectory sample(s)
        samples = self._annotation_to_samples(annotation)

        if format == "sarlo":
            return self.sarlo.export_batch(samples, split="train")
        elif format == "sapien":
            return self.sapien.export_batch(samples, split="train")
        elif format == "json":
            return self._export_json(samples, annotation.id)
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_samples(
        self,
        samples: List[TrajectorySample],
        format: str = "sarlo",
        split: str = "train",
    ) -> Path:
        """Export trajectory samples.

        Args:
            samples: List of samples
            format: Export format
            split: Dataset split

        Returns:
            Path to exported data
        """
        if format == "sarlo":
            return self.sarlo.export_batch(samples, split=split)
        elif format == "sapien":
            return self.sapien.export_batch(samples, split=split)
        elif format == "json":
            return self._export_json(samples, f"batch_{split}")
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_json(
        self,
        samples: List[TrajectorySample],
        name: str,
    ) -> Path:
        """Export as generic JSON."""
        output_path = self.output_dir / f"{name}.json"

        data = {
            "version": "1.0",
            "count": len(samples),
            "trajectories": [s.to_dict() for s in samples],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("json_export_complete", path=str(output_path))
        return output_path

    def _annotation_to_samples(
        self,
        annotation: Annotation,
    ) -> List[TrajectorySample]:
        """Convert annotation to trajectory samples."""
        samples = []

        for i, segment in enumerate(annotation.segments):
            # Build world states from segment
            states = self._segment_to_states(segment)

            sample = TrajectorySample(
                trajectory_id=f"{annotation.id}_seg{i}",
                video_id=annotation.video_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                states=states,
                actions=segment.actions,
                metadata={
                    "caption": segment.caption,
                    "objects": [obj.name for obj in segment.objects],
                },
            )
            samples.append(sample)

        return samples

    def _segment_to_states(self, segment: Segment) -> List[WorldState]:
        """Convert segment to world states."""
        states = []

        # Create world state from segment
        # This is a simplified version - real implementation would
        # extract actual object states from video
        state = WorldState(
            timestamp=segment.start_time,
            environment={
                "scene_type": segment.scene_type,
                "lighting": segment.lighting,
            },
        )

        # Add objects from segment
        for obj in segment.objects:
            obj_state = ObjectState(
                object_id=f"obj_{obj.name}",
                name=obj.name,
                state=obj.state or "unknown",
                material=obj.material or "unknown",
            )
            state.scene_graph.add_object(obj_state)

        states.append(state)
        return states


def export_trajectories(
    annotations: List[Annotation],
    output_dir: Union[str, Path],
    format: str = "sarlo",
    split: str = "train",
) -> Path:
    """Convenience function to export multiple annotations.

    Args:
        annotations: List of annotations to export
        output_dir: Output directory
        format: Export format ("sarlo", "sapien", "json")
        split: Dataset split

    Returns:
        Path to output directory
    """
    exporter = GenericExporter(output_dir)

    for annotation in annotations:
        exporter.export_annotation(annotation, format=format)

    logger.info(
        "trajectory_export_complete",
        format=format,
        count=len(annotations),
        output_dir=str(output_dir),
    )

    return Path(output_dir)
