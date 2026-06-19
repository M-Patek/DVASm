"""Export adapters for different training formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from dvas.data.schemas import Annotation


class ExportAdapter(ABC):
    """Abstract base class for export format adapters."""

    @abstractmethod
    def export(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Convert annotations to target format."""
        pass


class LLaVAAdapter(ExportAdapter):
    """Export to LLaVA training format."""

    def export(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Export to LLaVA format."""
        return [ann.to_llava_format() for ann in annotations]


class OpenAIAdapter(ExportAdapter):
    """Export to OpenAI fine-tuning format."""

    def export(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Export to OpenAI format."""
        return [ann.to_openai_format() for ann in annotations]


class ShareGPTAdapter(ExportAdapter):
    """Export to ShareGPT (vicuna) format."""

    def export(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Export to ShareGPT format."""
        results = []
        for ann in annotations:
            conversations = []
            for seg in ann.segments:
                conversations.append(
                    {"from": "human", "value": f"<video>\n{ann.video_path}\nDescribe this video."}
                )
                conversations.append(
                    {
                        "from": "gpt",
                        "value": seg.caption,
                    }
                )
            results.append(
                {
                    "id": ann.id,
                    "video": ann.video_path,
                    "conversations": conversations,
                }
            )
        return results


class WorldModelAdapter(ExportAdapter):
    """Export to World Model training format.

    Format: sequence of (observation, action, next_observation) tuples
    for world model training (e.g., Dreamer, IRIS, or similar).
    """

    def export(self, annotations: List[Annotation]) -> List[Dict[str, Any]]:
        """Export to World Model training format."""
        results = []
        for ann in annotations:
            episodes = []
            for i, seg in enumerate(ann.segments):
                # Current observation
                observation = {
                    "caption": seg.caption,
                    "caption_dense": seg.caption_dense,
                    "scene_type": seg.scene_type,
                    "lighting": seg.lighting,
                    "objects": [
                        {"name": obj.name, "attributes": obj.attributes, "state": obj.state}
                        for obj in seg.objects
                    ],
                }

                # Action representation
                actions = []
                for action in seg.actions:
                    action_repr = {
                        "verb": action.verb,
                        "noun": action.noun,
                        "hand": action.hand.value if action.hand else None,
                    }
                    if action.instrument:
                        action_repr["instrument"] = action.instrument
                    if action.physical:
                        action_repr["physical"] = action.physical.model_dump(exclude_none=True)
                    if action.embodiment:
                        action_repr["embodiment"] = action.embodiment.model_dump(exclude_none=True)
                    actions.append(action_repr)

                # Next observation (from next segment or state prediction)
                next_observation = None
                if i + 1 < len(ann.segments):
                    next_seg = ann.segments[i + 1]
                    next_observation = {
                        "caption": next_seg.caption,
                        "scene_type": next_seg.scene_type,
                    }
                elif ann.state_predictions:
                    next_observation = {
                        "predicted_desc": ann.state_predictions.predicted_next_frame_desc,
                        "expected_change": ann.state_predictions.expected_state_change,
                    }

                episodes.append({
                    "observation": observation,
                    "actions": actions,
                    "next_observation": next_observation,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                })

            # World model metadata
            dynamics = None
            if ann.dynamics:
                dynamics = {
                    "physical_constraints": ann.dynamics.physical_constraints,
                    "causal_links": ann.dynamics.causal_links,
                }

            results.append({
                "id": ann.id,
                "video_id": ann.video_id,
                "video_path": ann.video_path,
                "episodes": episodes,
                "dynamics": dynamics,
                "metadata": {
                    "schema_version": ann.schema_version,
                    "annotation_standard": ann.annotation_standard.value,
                    "camera_type": ann.metadata.camera_type,
                    "environment": ann.metadata.environment,
                },
            })

        return results


# Registry of available adapters
ADAPTERS = {
    "llava": LLaVAAdapter,
    "openai": OpenAIAdapter,
    "sharegpt": ShareGPTAdapter,
    "world_model": WorldModelAdapter,
}


def export_annotations(
    annotations: List[Annotation],
    output_path: Path,
    format: str = "llava",
) -> int:
    """
    Export annotations to specified format.

    Args:
        annotations: List of annotations to export
        output_path: Path to output JSONL file
        format: Export format (llava, openai, sharegpt, world_model)

    Returns:
        Number of annotations exported
    """
    if format not in ADAPTERS:
        raise ValueError(f"Unknown format: {format}. Available: {list(ADAPTERS.keys())}")

    adapter = ADAPTERS[format]()
    data = adapter.export(annotations)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(data)
