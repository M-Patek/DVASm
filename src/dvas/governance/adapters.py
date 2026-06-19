"""Governance — Annotation standard management and multi-standard adapters.

Provides adapters to convert internal Annotation to/from different
annotation standards (EPIC-KITCHENS, Ego4D, Open X-Embodiment, custom).

Usage:
    from dvas.governance import get_adapter, AnnotationStandard
    from dvas.data.schemas import Annotation

    adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
    epic_data = adapter.to_standard(annotation)
    annotation = adapter.from_standard(epic_data)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dvas.data.schemas import (
    Action,
    Annotation,
    AnnotationStandard,
    Hand,
    Object,
    Segment,
    VideoMetadata,
)


class StandardAdapter(ABC):
    """Abstract base for annotation standard adapters."""

    @property
    @abstractmethod
    def standard(self) -> AnnotationStandard:
        """Return the standard this adapter handles."""
        pass

    @abstractmethod
    def to_standard(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert internal Annotation to this standard's format."""
        pass

    @abstractmethod
    def from_standard(self, data: Dict[str, Any]) -> Annotation:
        """Convert this standard's format to internal Annotation."""
        pass


class EPICAdapter(StandardAdapter):
    """EPIC-KITCHENS format adapter.

    EPIC only supports verb+noun+hand. All v2.0 extensions are discarded.
    """

    @property
    def standard(self) -> AnnotationStandard:
        return AnnotationStandard.EPIC_KITCHENS

    def to_standard(self, annotation: Annotation) -> Dict[str, Any]:
        """Export to EPIC-KITCHENS format."""
        segments = []
        for seg in annotation.segments:
            actions = []
            for a in seg.actions:
                actions.append({
                    "verb": a.verb,
                    "noun": a.noun,
                    "hand": a.hand.value,
                    "start_time": a.start_time,
                    "end_time": a.end_time,
                })
            segments.append({
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "actions": actions,
            })

        return {
            "id": annotation.id,
            "video_id": annotation.video_id,
            "segments": segments,
            "metadata": {
                "fps": annotation.metadata.fps,
                "resolution": annotation.metadata.resolution,
                "duration": annotation.metadata.duration,
            },
        }

    def from_standard(self, data: Dict[str, Any]) -> Annotation:
        """Import from EPIC-KITCHENS format."""
        segments = []
        for seg_data in data.get("segments", []):
            actions = []
            for a_data in seg_data.get("actions", []):
                actions.append(Action(
                    verb=a_data["verb"],
                    noun=a_data["noun"],
                    hand=Hand(a_data.get("hand", "unknown")),
                    start_time=a_data.get("start_time"),
                    end_time=a_data.get("end_time"),
                ))
            segments.append(Segment(
                start_time=seg_data["start_time"],
                end_time=seg_data["end_time"],
                caption="",  # EPIC has no captions
                actions=actions,
            ))

        meta = data.get("metadata", {})
        return Annotation(
            id=data["id"],
            video_id=data["video_id"],
            video_path="",  # EPIC doesn't store path in annotation
            segments=segments,
            metadata=VideoMetadata(
                fps=meta.get("fps", 30.0),
                resolution=meta.get("resolution", [1920, 1080]),
                duration=meta.get("duration", 0.0),
                total_frames=0,
            ),
            annotation_standard=AnnotationStandard.EPIC_KITCHENS,
        )


class Ego4DAdapter(StandardAdapter):
    """Ego4D format adapter.

    Ego4D supports: narration, 3D spatial info, hand-object interaction,
    instrument, state changes.
    """

    @property
    def standard(self) -> AnnotationStandard:
        return AnnotationStandard.EGO4D

    def to_standard(self, annotation: Annotation) -> Dict[str, Any]:
        """Export to Ego4D format."""
        narrations = []
        for seg in annotation.segments:
            narrations.append({
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "narration": seg.caption,
                "actions": [
                    {
                        "verb": a.verb,
                        "noun": a.noun,
                        "hand": a.hand.value,
                        "instrument": a.instrument,
                        "source_state": a.source_state,
                        "target_state": a.target_state,
                    }
                    for a in seg.actions
                ],
                "objects": [
                    {
                        "name": obj.name,
                        "attributes": obj.attributes,
                        "state": obj.state,
                    }
                    for obj in seg.objects
                ],
            })

        return {
            "id": annotation.id,
            "video_id": annotation.video_id,
            "narrations": narrations,
            "metadata": {
                "camera_type": annotation.metadata.camera_type or "egocentric",
                "environment": annotation.metadata.environment or "indoor",
            },
        }

    def from_standard(self, data: Dict[str, Any]) -> Annotation:
        """Import from Ego4D format."""
        segments = []
        for narr in data.get("narrations", []):
            actions = []
            for a_data in narr.get("actions", []):
                actions.append(Action(
                    verb=a_data["verb"],
                    noun=a_data["noun"],
                    hand=Hand(a_data.get("hand", "unknown")),
                    instrument=a_data.get("instrument"),
                    source_state=a_data.get("source_state"),
                    target_state=a_data.get("target_state"),
                ))
            objects = [
                Object(
                    name=obj["name"],
                    attributes=obj.get("attributes", {}),
                    state=obj.get("state"),
                )
                for obj in narr.get("objects", [])
            ]
            segments.append(Segment(
                start_time=narr["start_time"],
                end_time=narr["end_time"],
                caption=narr.get("narration", ""),
                actions=actions,
                objects=objects,
            ))

        return Annotation(
            id=data["id"],
            video_id=data["video_id"],
            video_path="",
            segments=segments,
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=0.0,
                total_frames=0,
                camera_type="egocentric",
            ),
            annotation_standard=AnnotationStandard.EGO4D,
        )


class OpenXAdapter(StandardAdapter):
    """Open X-Embodiment format adapter.

    Open X focuses on: embodiment actions, gripper poses, joint targets,
    action space definitions for robot training.
    """

    @property
    def standard(self) -> AnnotationStandard:
        return AnnotationStandard.OPEN_X_EMBODIMENT

    def to_standard(self, annotation: Annotation) -> Dict[str, Any]:
        """Export to Open X-Embodiment format."""
        steps = []
        for seg in annotation.segments:
            for action in seg.actions:
                step = {
                    "language_instruction": f"{action.verb} {action.noun}",
                    "verb": action.verb,
                    "noun": action.noun,
                }
                if action.embodiment:
                    step["action"] = {
                        "gripper_pose": action.embodiment.gripper_pose,
                        "joint_target": action.embodiment.joint_target,
                        "action_space": action.embodiment.action_space,
                        "gripper_state": action.embodiment.gripper_state,
                    }
                steps.append(step)

        return {
            "id": annotation.id,
            "video_id": annotation.video_id,
            "steps": steps,
            "metadata": {
                "camera_type": annotation.metadata.camera_type,
                "environment": annotation.metadata.environment,
            },
        }

    def from_standard(self, data: Dict[str, Any]) -> Annotation:
        """Import from Open X-Embodiment format."""
        from dvas.data.schemas import EmbodimentAction

        actions = []
        for step in data.get("steps", []):
            embodiment = None
            if "action" in step:
                act = step["action"]
                embodiment = EmbodimentAction(
                    gripper_pose=act.get("gripper_pose"),
                    joint_target=act.get("joint_target"),
                    action_space=act.get("action_space"),
                    gripper_state=act.get("gripper_state"),
                )
            actions.append(Action(
                verb=step["verb"],
                noun=step["noun"],
                embodiment=embodiment,
            ))

        return Annotation(
            id=data["id"],
            video_id=data["video_id"],
            video_path="",
            segments=[Segment(
                start_time=0.0,
                end_time=0.0,
                caption="",
                actions=actions,
            )],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=0.0,
                total_frames=0,
            ),
            annotation_standard=AnnotationStandard.OPEN_X_EMBODIMENT,
        )


# ── Registry ─────────────────────────────────────────────────────────

_ADAPTERS: Dict[AnnotationStandard, StandardAdapter] = {
    AnnotationStandard.EPIC_KITCHENS: EPICAdapter(),
    AnnotationStandard.EGO4D: Ego4DAdapter(),
    AnnotationStandard.OPEN_X_EMBODIMENT: OpenXAdapter(),
}


def get_adapter(standard: AnnotationStandard) -> StandardAdapter:
    """Get adapter for a specific annotation standard.

    Args:
        standard: The annotation standard to adapt to/from.

    Returns:
        StandardAdapter instance.

    Raises:
        ValueError: If the standard is not supported.
    """
    if standard not in _ADAPTERS:
        raise ValueError(f"Unsupported annotation standard: {standard}")
    return _ADAPTERS[standard]


def list_standards() -> List[AnnotationStandard]:
    """List all supported annotation standards."""
    return list(_ADAPTERS.keys())
