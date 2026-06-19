"""World Model — Placeholder module for future state prediction and dynamics annotation.

This module is currently a thin shell. It defines interfaces that will be
implemented when World Model training data generation is needed.

Current status: interfaces defined, implementations return empty/placeholder data.

Usage:
    from dvas.world_model import WorldModelAnnotator

    annotator = WorldModelAnnotator()
    prediction = await annotator.generate_state_prediction(segment, action)
    dynamics = await annotator.generate_dynamics(segment)
"""

from typing import Optional

from dvas.data.schemas import (
    Action,
    DynamicsAnnotation,
    Segment,
    StatePrediction,
)


class WorldModelAnnotator:
    """Generate World Model training annotations.

    Currently returns placeholder data. Future implementation will:
    - Use a world model (or teacher model with specialized prompts) to predict
      next-frame descriptions given current state + action
    - Generate counterfactual scenarios ("what if action B instead of A")
    - Extract physical constraints and causal links
    """

    async def generate_state_prediction(
        self,
        segment: Segment,
        action: Optional[Action] = None,
    ) -> StatePrediction:
        """Generate state prediction annotation for a segment.

        Args:
            segment: The video segment to analyze
            action: Optional action to simulate

        Returns:
            StatePrediction with predicted next state.
            Currently returns empty placeholder.
        """
        # Placeholder: future implementation will call a world model
        # or use a specialized teacher prompt to predict next state
        return StatePrediction(
            predicted_next_frame_desc=None,
            expected_state_change=None,
            preconditions=[],
            effects=[],
        )

    async def generate_dynamics(
        self,
        segment: Segment,
    ) -> DynamicsAnnotation:
        """Generate physical dynamics annotation for a segment.

        Args:
            segment: The video segment to analyze

        Returns:
            DynamicsAnnotation with physical constraints and causal links.
            Currently returns empty placeholder.
        """
        # Placeholder: future implementation will analyze segment
        # to extract physical constraints (gravity, friction) and causal chains
        return DynamicsAnnotation(
            physical_constraints=[],
            causal_links=[],
            counterfactuals=[],
        )

    async def generate_counterfactual(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_action: Action,
    ) -> DynamicsAnnotation:
        """Generate counterfactual annotation ("what if" scenario).

        Args:
            segment: The video segment
            actual_action: The action that was actually performed
            alternative_action: The alternative action to simulate

        Returns:
            DynamicsAnnotation with counterfactual results.
            Currently returns empty placeholder.
        """
        # Placeholder: future implementation will simulate alternative action
        # and predict the resulting state difference
        return DynamicsAnnotation(
            physical_constraints=[],
            causal_links=[],
            counterfactuals=[
                {
                    "if": f"{alternative_action.verb} {alternative_action.noun}",
                    "then": "placeholder: predicted outcome",
                    "instead_of": f"{actual_action.verb} {actual_action.noun}",
                }
            ],
        )
