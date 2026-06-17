"""Mock teacher model for end-to-end testing without API keys."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dvas.models.teacher.base import TeacherModel


class MockTeacher(TeacherModel):
    """Mock teacher that generates synthetic annotations for testing.

    This allows end-to-end pipeline testing without requiring:
    - OpenAI API key
    - Network access
    - Real model inference
    """

    VERBS = ["grasp", "pick", "place", "rotate", "cut", "pour", "stir", "hold", "release"]
    NOUNS = ["cup", "bowl", "plate", "knife", "spoon", "bottle", "box", "tool", "object"]
    STATES = ["empty", "full", "open", "closed", "hot", "cold", "clean", "dirty"]

    def __init__(self, model_name: str = "mock-teacher", **kwargs):
        super().__init__(model_name, **kwargs)
        self.call_count = 0
        self.total_frames_processed = 0

    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate synthetic annotation."""
        self.call_count += 1
        if frames:
            self.total_frames_processed += len(frames)

        # Generate deterministic but varied output based on call count
        random.seed(self.call_count)

        num_actions = random.randint(2, 5)
        num_objects = random.randint(2, 4)
        num_steps = random.randint(3, 6)

        hand_actions = []
        for i in range(num_actions):
            hand_actions.append({
                "hand": random.choice(["left", "right", "both"]),
                "action": random.choice(self.VERBS),
                "target": random.choice(self.NOUNS),
                "time": f"{i*2}-{(i+1)*2}s"
            })

        objects = []
        for i in range(num_objects):
            objects.append({
                "name": random.choice(self.NOUNS),
                "state": random.choice(self.STATES),
                "interacted": random.random() > 0.3
            })

        steps = []
        for i in range(num_steps):
            steps.append({
                "order": i + 1,
                "action": random.choice(self.VERBS) + " " + random.choice(self.NOUNS),
                "details": f"Step {i+1} of the manipulation sequence"
            })

        result = {
            "scene_description": f"A robotic manipulation scene with {num_objects} objects and {num_actions} hand actions.",
            "hand_actions": hand_actions,
            "objects": objects,
            "steps": steps,
        }

        return {
            "text": json.dumps(result, indent=2),
            "model": self.model_name,
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            "finish_reason": "stop",
        }

    async def annotate_batch(
        self,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch annotation (just calls annotate sequentially)."""
        results = []
        for item in items:
            result = await self.annotate(
                frames=item.get("frames"),
                prompt=item.get("prompt"),
                **kwargs
            )
            results.append(result)
        return results
