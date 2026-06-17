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
                conversations.append({
                    "from": "human",
                    "value": f"<video>\n{ann.video_path}\nDescribe this video."
                })
                conversations.append({
                    "from": "gpt",
                    "value": seg.caption,
                })
            results.append({
                "id": ann.id,
                "video": ann.video_path,
                "conversations": conversations,
            })
        return results


# Registry of available adapters
ADAPTERS = {
    "llava": LLaVAAdapter,
    "openai": OpenAIAdapter,
    "sharegpt": ShareGPTAdapter,
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
        format: Export format (llava, openai, sharegpt)

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
