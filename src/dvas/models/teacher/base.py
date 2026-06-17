"""Base class for teacher models."""

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class TeacherModel(ABC):
    """Abstract base class for teacher (gold-standard) models."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate annotation for a video or frames.

        Args:
            video_path: Path to video file
            frames: List of numpy arrays (BGR images)
            prompt: Custom prompt for annotation
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing annotation results
        """
        pass

    @abstractmethod
    async def annotate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch annotation for efficiency."""
        pass

    def _encode_image(self, image: np.ndarray, format: str = "JPEG") -> str:
        """Encode numpy image to base64 string."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image

        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _encode_frames(self, frames: List[np.ndarray]) -> List[str]:
        """Encode multiple frames to base64 strings."""
        return [self._encode_image(frame) for frame in frames]

    def _get_default_prompt(self, task: str = "caption") -> str:
        """Get default prompt for a task."""
        prompts = {
            "caption": self._caption_prompt(),
            "dense_caption": self._dense_caption_prompt(),
            "qa": self._qa_prompt(),
            "temporal": self._temporal_prompt(),
            "fine_grained": self._fine_grained_prompt(),
        }
        return prompts.get(task, prompts["caption"])

    def _caption_prompt(self) -> str:
        """Basic captioning prompt."""
        return """Describe what is happening in this video. Be concise but accurate."""

    def _dense_caption_prompt(self) -> str:
        """Dense captioning with temporal information."""
        return """Provide a detailed description of the video, including:
1. Overall scene and context
2. Sequential actions performed
3. Tools or objects being used
4. Hand movements and interactions
5. Temporal progression of activities

Format your response as a coherent paragraph."""

    def _qa_prompt(self) -> str:
        """Question-answer generation prompt."""
        return """Based on this video, generate 3-5 question-answer pairs:

Format:
Q: [Question about actions, objects, or sequence]
A: [Concise answer]

Cover different aspects: what is happening, how it's done, what tools are used."""

    def _temporal_prompt(self) -> str:
        """Temporal localization prompt."""
        return """Analyze this video and identify distinct action segments.

For each segment, provide:
- Start and end time (in seconds)
- Action label (verb + noun)
- Brief description

Format as JSON-like structure."""

    def _fine_grained_prompt(self) -> str:
        """Fine-grained robotic-focused prompt (our main use case)."""
        return """You are an expert in robotic manipulation and egocentric video understanding.

Provide a detailed analysis of this first-person video, focusing on:

1. **Scene Understanding**: Describe the environment, workspace layout, and context
2. **Hand Actions**: Identify left and right hand movements separately
3. **Object Interactions**: List all objects touched/manipulated and how
4. **Temporal Sequence**: Break down the action into chronological steps
5. **Motion Details**: Describe grip types, arm positions, movement trajectories
6. **Tool Usage**: If tools are used, describe how they are held and operated

Output format (JSON-like):
{
  "scene_description": "...",
  "hand_actions": [
    {"hand": "left|right", "action": "...", "target": "...", "time": "start-end"}
  ],
  "objects": [
    {"name": "...", "state": "...", "interacted": true/false}
  ],
  "steps": [
    {"order": 1, "action": "...", "details": "..."}
  ]
}

Be precise and detailed - this annotation will be used for training robotic manipulation models."""
