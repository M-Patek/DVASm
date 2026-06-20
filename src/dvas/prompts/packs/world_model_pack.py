"""World model state prediction prompt packs.

Provides specialized prompts for world model training data generation,
including state prediction, dynamics annotation, and counterfactual reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.prompts.registry import PromptDomain, PromptTemplate
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorldModelPromptPack:
    """Prompt pack for world model state prediction tasks."""

    name: str = "world_model"
    domain: PromptDomain = PromptDomain.WORLD_MODEL

    TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "wm_state_prediction": """Predict the next state of the scene after the action in this video.

Given the current frame, predict:
1. OBJECT POSITIONS: Where will each object be in the next frame?
2. OBJECT STATES: How will object states change? (position, orientation, velocity)
3. CONTACT: Will any new contacts occur? Which objects will touch?
4. DEFORMATION: Will any objects deform, break, or change shape?
5. OCCLUSION: Will any objects become occluded or visible?

For each object, provide:
- Name
- Current state
- Predicted next state
- Confidence (0.0-1.0)

Caption: """,

        "wm_dynamics": """Describe the physical dynamics in this video.

Identify and describe:
1. FORCES: What forces are acting? (gravity, friction, applied, contact)
2. MOMENTUM: How does momentum transfer between objects?
3. COLLISIONS: Describe any collisions (elastic, inelastic, etc.)
4. STABILITY: Is the system stable or unstable?
5. CONSTRAINTS: What physical constraints are present?

Use physics terminology and estimate magnitudes where possible.

Caption: """,

        "wm_counterfactual": """Generate counterfactual scenarios for this video.

For the action shown, answer:
1. What if the action had NOT been performed? (no-op counterfactual)
2. What if a DIFFERENT action had been performed? (alternative action)
3. What if the action had been performed DIFFERENTLY? (parameter variation)
4. What would happen if an OBSTACLE were present?
5. What would happen if the OBJECT PROPERTIES were different?

For each counterfactual, describe the expected outcome.

Caption: """,

        "wm_causal_chain": """Identify the causal chain of events in this video.

Map out:
1. INITIAL STATE: What is true before any action
2. ACTION: What action is performed
3. DIRECT EFFECTS: Immediate consequences (within 1 second)
4. INDIRECT EFFECTS: Consequences that follow from direct effects
5. FINAL STATE: What is true after all effects propagate

For each causal link, identify:
- Cause
- Effect
- Mechanism (how cause leads to effect)
- Time delay

Caption: """,

        "wm_scene_graph": """Generate a scene graph for this video frame.

Nodes (objects):
- List all objects with properties (position, size, color, material)

Edges (relations):
- Spatial: left_of, right_of, above, below, inside, on_top_of
- Functional: supports, contains, attached_to
- Temporal: moving_toward, moving_away, stationary

Format as JSON:
{{
  "nodes": [{{"id": "obj1", "name": "cup", "properties": {{...}}}}],
  "edges": [{{"from": "obj1", "to": "obj2", "relation": "on_top_of"}}]
}}

Caption: """,

        "wm_temporal_reasoning": """Analyze temporal relationships in this video.

Identify:
1. EVENT ORDERING: What happens first, second, third?
2. TEMPORAL OVERLAPS: Which events happen simultaneously?
3. DURATIONS: How long does each event last?
4. TEMPORAL CONSTRAINTS: Must A happen before B?
5. TEMPORAL GAPS: Are there pauses or waiting periods?

Use Allen's interval algebra where applicable (before, meets, overlaps, etc.)

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

        from dvas.prompts.registry import PromptMetadata
        return PromptTemplate(
            id=f"wm_{template_name}",
            metadata=PromptMetadata(
                name=template_name,
                version="1.0.0",
                domain=self.domain,
                description=f"World model prompt: {template_name}",
                tags=["world_model", "state_prediction", template_name],
            ),
            template=template_text,
            variables=list(custom_variables.keys()) if custom_variables else [],
        )

    def get_state_prediction_prompt(self, objects_hint: Optional[List[str]] = None) -> str:
        """Get a state prediction prompt, optionally with object hints."""
        base = self.TEMPLATES["wm_state_prediction"]
        if objects_hint:
            base += f"\n\nKey objects to track: {', '.join(objects_hint)}"
        return base

    def get_dynamics_prompt(self, physics_type: Optional[str] = None) -> str:
        """Get a dynamics description prompt."""
        base = self.TEMPLATES["wm_dynamics"]
        if physics_type:
            base += f"\n\nFocus on {physics_type} physics."
        return base
