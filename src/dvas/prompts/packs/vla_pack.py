"""VLA/Robot manipulation prompt packs.

Provides specialized prompts for Vision-Language-Action models and
robotic manipulation task annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.prompts.registry import PromptDomain, PromptTemplate
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VLAPromptPack:
    """Prompt pack for VLA/Robot manipulation tasks."""

    name: str = "vla_robot"
    domain: PromptDomain = PromptDomain.VLA

    # Standard VLA prompt templates
    TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "vla_grasp_analysis": """Analyze this robotic manipulation video for VLA training.

Focus on grasp analysis:
1. GRASP TYPE: Pinch, power, tripod, hook, lateral, precision, etc.
2. CONTACT POINTS: Where fingers/gripper contact the object
3. FORCE: Estimated force level (gentle/moderate/firm)
4. APPROACH: Direction and angle of approach
5. LIFT: How the object is lifted and stabilized

For each grasp event, provide:
- Timestamp
- Grasp type
- Object name
- Success/failure
- Adjustment (if any)

Caption: """,

        "vla_trajectory": """Describe the trajectory of the end effector in this video.

Include:
1. START POSITION: Initial position in workspace
2. END POSITION: Final position after action
3. PATH: Straight line, arc, or complex curve
4. VELOCITY: Constant, accelerating, or decelerating
5. ORIENTATION: How gripper orientation changes
6. OBSTACLES: Any obstacles avoided

Use this format:
{{
  "trajectory": {{
    "type": "arc|straight|complex",
    "start": [x, y, z],
    "end": [x, y, z],
    "waypoints": [[x, y, z], ...],
    "orientation_changes": ["rotate_x", "rotate_y", ...]
  }}
}}

Caption: """,

        "vla_action_sequence": """Provide a step-by-step action sequence for this manipulation task.

For each step:
1. ACTION: What the robot does (pick, place, push, rotate, etc.)
2. OBJECT: The primary object being manipulated
3. PRECONDITION: What must be true before this step
4. POSTCONDITION: What becomes true after this step
5. DURATION: Estimated time in seconds

Format as a numbered list with JSON for each step.

Caption: """,

        "vla_affordance": """Identify object affordances in this video for VLA training.

For each object:
1. OBJECT NAME: What is the object
2. AFFORDANCES: What actions are possible (grasp, push, stack, etc.)
3. GRIP POINTS: Where to grasp
4. STABLE CONFIGURATIONS: How the object can rest stably
5. CONSTRAINTS: Physical constraints (weight, fragility, etc.)

Caption: """,

        "vla_fine_motor": """Provide fine-grained motor control description.

Detail level:
1. FINGER CONFIGURATION: Which fingers used, joint angles
2. CONTACT FORCE: Pressure at contact points
3. SLIP DETECTION: Any slipping or readjustment
4. COMPLIANCE: How the hand/object deforms under pressure
5. PRECISION: Accuracy of placement/alignment

Caption: """,
    })

    def get_template(self, name: str) -> Optional[str]:
        """Get a template by name."""
        return self.TEMPLATES.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.TEMPLATES.keys())

    def create_prompt_template(
        self,
        template_name: str,
        custom_variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptTemplate]:
        """Create a PromptTemplate from a named template.

        Args:
            template_name: Name of the template to use.
            custom_variables: Optional custom variables to substitute.

        Returns:
            PromptTemplate or None if name not found.
        """
        template_text = self.get_template(template_name)
        if template_text is None:
            return None

        if custom_variables:
            for key, value in custom_variables.items():
                template_text = template_text.replace(f"{{{key}}}", value)

        return PromptTemplate(
            id=f"vla_{template_name}",
            metadata=PromptTemplate.__dataclass_fields__["metadata"].type(
                name=template_name,
                version="1.0.0",
                domain=self.domain,
                description=f"VLA prompt: {template_name}",
                tags=["vla", "robot", template_name],
            ),
            template=template_text,
            variables=list(custom_variables.keys()) if custom_variables else [],
        )

    def get_grasp_prompt(self, object_name: Optional[str] = None) -> str:
        """Get a grasp analysis prompt, optionally focused on a specific object."""
        base = self.TEMPLATES["vla_grasp_analysis"]
        if object_name:
            base += f"\n\nFocus specifically on: {object_name}"
        return base

    def get_trajectory_prompt(self, workspace_bounds: Optional[List[float]] = None) -> str:
        """Get a trajectory description prompt."""
        base = self.TEMPLATES["vla_trajectory"]
        if workspace_bounds:
            base += f"\n\nWorkspace bounds: {workspace_bounds}"
        return base

    def get_action_sequence_prompt(self, num_steps_hint: Optional[int] = None) -> str:
        """Get an action sequence prompt."""
        base = self.TEMPLATES["vla_action_sequence"]
        if num_steps_hint:
            base += f"\n\nExpected approximately {num_steps_hint} steps."
        return base
