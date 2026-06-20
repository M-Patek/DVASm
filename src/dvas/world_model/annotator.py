"""World Model annotation module for state prediction and dynamics.

Generates World Model training annotations including:
- State predictions (before/after action)
- Physical dynamics annotations
- Counterfactual scenarios
- Causal relation extraction

Supports both Teacher model-based and rule-based annotation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from dvas.data.schemas import Action, Annotation, DynamicsAnnotation, Segment, StatePrediction
from dvas.models.teacher.base import TeacherModel
from dvas.utils.logging import get_logger
from dvas.world_model.dynamics import ContactDynamics, ContactType, PhysicalDynamics
from dvas.world_model.state_repr import (
    AffordanceState,
    ObjectRole,
    ObjectState,
    Relationship,
    WorldState,
)
from dvas.world_model.temporal_graph import (
    EventType,
    TemporalEvent,
    TemporalEventGraph,
)

logger = get_logger(__name__)


class WorldModelAnnotator:
    """Generate World Model training annotations.

    Provides comprehensive world model annotations including state predictions,
    physical dynamics, counterfactuals, and causal relations.

    Can use:
    - Teacher model: For high-quality, LLM-based annotations
    - Rule-based: For fast, heuristic-based annotations

    Example:
        # Teacher-based annotation
        teacher = TeacherModel(model_name="gpt-5.5")
        annotator = WorldModelAnnotator(teacher_model=teacher)

        # Rule-based annotation
        annotator = WorldModelAnnotator()  # No teacher model

        # Generate annotations
        state_before = await annotator.generate_state_before(segment)
        state_after = await annotator.generate_state_after(segment)
        prediction = await annotator.predict_next_state(state_before, action)
    """

    def __init__(
        self,
        teacher_model: Optional[TeacherModel] = None,
        use_teacher: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """Initialize the annotator.

        Args:
            teacher_model: Optional teacher model for high-quality annotations
            use_teacher: Whether to use teacher model when available
            confidence_threshold: Minimum confidence for annotations
        """
        self.teacher_model = teacher_model
        self.use_teacher = use_teacher and teacher_model is not None
        self.confidence_threshold = confidence_threshold
        self._annotation_count = 0
        self._cache: Dict[str, Any] = {}

    async def generate_state_before(
        self,
        segment: Segment,
        video_frames: Optional[List[Any]] = None,
    ) -> WorldState:
        """Generate scene description before action.

        Analyzes the initial state of the scene before any action occurs.

        Args:
            segment: Video segment to analyze
            video_frames: Optional video frames for visual analysis

        Returns:
            WorldState representing scene before action
        """
        if self.use_teacher and video_frames:
            return await self._generate_state_before_teacher(segment, video_frames)

        return self._generate_state_before_rule_based(segment)

    async def generate_state_after(
        self,
        segment: Segment,
        video_frames: Optional[List[Any]] = None,
    ) -> WorldState:
        """Generate scene description after action.

        Analyzes the final state of the scene after action completion.

        Args:
            segment: Video segment to analyze
            video_frames: Optional video frames for visual analysis

        Returns:
            WorldState representing scene after action
        """
        if self.use_teacher and video_frames:
            return await self._generate_state_after_teacher(segment, video_frames)

        return self._generate_state_after_rule_based(segment)

    async def predict_next_state(
        self,
        current_state: WorldState,
        action: Action,
        prediction_horizon: float = 1.0,
    ) -> WorldState:
        """Predict future state given current state and action.

        Uses physics-based heuristics or teacher model to predict
        how the scene will evolve after the action.

        Args:
            current_state: Current world state
            action: Action to simulate
            prediction_horizon: How far ahead to predict (seconds)

        Returns:
            Predicted WorldState
        """
        if self.use_teacher:
            return await self._predict_next_state_teacher(current_state, action)

        return self._predict_next_state_rule_based(current_state, action, prediction_horizon)

    async def annotate_physical_dynamics(
        self,
        segment: Segment,
        video_frames: Optional[List[Any]] = None,
    ) -> List[PhysicalDynamics]:
        """Extract physical properties and dynamics annotations.

        Identifies physical properties like mass, friction, and
        tracks contact events and forces.

        Args:
            segment: Video segment to analyze
            video_frames: Optional video frames

        Returns:
            List of PhysicalDynamics annotations
        """
        if self.use_teacher and video_frames:
            return await self._annotate_dynamics_teacher(segment, video_frames)

        return self._annotate_dynamics_rule_based(segment)

    async def extract_causal_relations(
        self,
        segment: Segment,
        temporal_graph: Optional[TemporalEventGraph] = None,
    ) -> List[Dict[str, str]]:
        """Identify causal links between actions and outcomes.

        Extracts causal relations like "push causes movement" or
        "grasp enables lifting".

        Args:
            segment: Video segment to analyze
            temporal_graph: Optional pre-computed temporal graph

        Returns:
            List of causal relations as dictionaries
        """
        if self.use_teacher:
            return await self._extract_causal_teacher(segment)

        return self._extract_causal_rule_based(segment, temporal_graph)

    async def generate_counterfactuals(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_actions: Optional[List[Action]] = None,
    ) -> List[Dict[str, str]]:
        """Generate what-if scenarios.

        Creates counterfactual scenarios by simulating alternative
        actions and predicting their outcomes.

        Args:
            segment: Video segment
            actual_action: The action that was actually performed
            alternative_actions: List of alternative actions to simulate

        Returns:
            List of counterfactual scenarios
        """
        if alternative_actions is None:
            alternative_actions = self._generate_alternative_actions(actual_action)

        if self.use_teacher:
            return await self._generate_counterfactuals_teacher(
                segment, actual_action, alternative_actions
            )

        return self._generate_counterfactuals_rule_based(
            segment, actual_action, alternative_actions
        )

    async def generate_state_prediction(
        self,
        segment: Segment,
        action: Optional[Action] = None,
    ) -> StatePrediction:
        """Generate state prediction annotation (legacy compatibility).

        Args:
            segment: Video segment
            action: Optional action

        Returns:
            StatePrediction schema object
        """
        # Generate before and after states
        state_before = await self.generate_state_before(segment)
        state_after = await self.generate_state_after(segment)

        # Generate prediction
        if action is None and segment.actions:
            action = segment.actions[0]

        if action:
            predicted = await self.predict_next_state(state_before, action)
            predicted_desc = predicted.describe()
        else:
            predicted_desc = "No action specified"

        # Extract preconditions and effects
        preconditions = self._extract_preconditions(state_before)
        effects = self._extract_effects(state_before, state_after)

        return StatePrediction(
            predicted_next_frame_desc=predicted_desc,
            expected_state_change=state_after.describe(),
            preconditions=preconditions,
            effects=effects,
        )

    async def generate_dynamics(
        self,
        segment: Segment,
    ) -> DynamicsAnnotation:
        """Generate dynamics annotation (legacy compatibility).

        Args:
            segment: Video segment

        Returns:
            DynamicsAnnotation schema object
        """
        dynamics_list = await self.annotate_physical_dynamics(segment)

        # Convert to schema format
        physical_constraints = []
        causal_links = []
        counterfactuals = []

        for dyn in dynamics_list:
            # Extract physical constraints
            if dyn.properties.mass is not None:
                physical_constraints.append(
                    f"{dyn.object_id} mass: {dyn.properties.mass:.2f}kg"
                )

            # Extract causal links from contact events
            for event in dyn.contact_events:
                causal_links.append({
                    "cause": f"{event.subject_id} {event.contact_type.value}",
                    "effect": f"{event.object_id} response",
                    "confidence": str(event.is_stable),
                })

        return DynamicsAnnotation(
            physical_constraints=physical_constraints,
            causal_links=causal_links,
            counterfactuals=counterfactuals,
        )

    async def generate_counterfactual(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_action: Action,
    ) -> DynamicsAnnotation:
        """Generate counterfactual annotation (legacy compatibility).

        Args:
            segment: Video segment
            actual_action: Action that occurred
            alternative_action: Alternative action to simulate

        Returns:
            DynamicsAnnotation with counterfactual results
        """
        counterfactuals = await self.generate_counterfactuals(
            segment, actual_action, [alternative_action]
        )

        return DynamicsAnnotation(
            physical_constraints=[],
            causal_links=[],
            counterfactuals=counterfactuals,
        )

    async def annotate_full(
        self,
        segment: Segment,
        video_frames: Optional[List[Any]] = None,
    ) -> Annotation:
        """Generate complete world model annotation for a segment.

        Args:
            segment: Video segment
            video_frames: Optional video frames

        Returns:
            Complete Annotation with world model data
        """
        start_time = time.time()

        # Generate all annotations
        state_pred = await self.generate_state_prediction(segment)
        await self.generate_state_before(segment, video_frames)
        await self.generate_state_after(segment, video_frames)

        dynamics = await self.generate_dynamics(segment)

        # Extract causal relations
        await self.extract_causal_relations(segment)

        # Generate counterfactuals if actions exist
        counterfactuals = []
        if segment.actions:
            counterfactuals = await self.generate_counterfactuals(
                segment, segment.actions[0]
            )

        # Build annotation
        annotation = Annotation(
            id=f"wm_{int(time.time())}_{self._annotation_count}",
            video_id="unknown",
            video_path="unknown",
            segments=[segment],
            metadata={
                "fps": 30.0,
                "resolution": [1920, 1080],
                "duration": segment.duration,
                "total_frames": int(segment.duration * 30),
            },
            state_predictions=state_pred,
            dynamics=dynamics,
        )

        # Add counterfactuals to dynamics
        annotation.dynamics.counterfactuals = counterfactuals

        self._annotation_count += 1

        duration = time.time() - start_time
        logger.info(
            "full_annotation_complete",
            annotation_id=annotation.id,
            duration=duration,
            has_teacher=self.use_teacher,
        )

        return annotation

    # Teacher-based implementations

    async def _generate_state_before_teacher(
        self,
        segment: Segment,
        video_frames: List[Any],
    ) -> WorldState:
        """Use teacher model to generate state description."""
        prompt = self._build_state_prompt(segment, "before")

        result = await self.teacher_model.annotate(
            frames=video_frames,
            prompt=prompt,
            task="world_model_state",
        )

        # Parse teacher response into WorldState
        return self._parse_state_description(result.text)

    async def _generate_state_after_teacher(
        self,
        segment: Segment,
        video_frames: List[Any],
    ) -> WorldState:
        """Use teacher model to generate state description."""
        prompt = self._build_state_prompt(segment, "after")

        result = await self.teacher_model.annotate(
            frames=video_frames,
            prompt=prompt,
            task="world_model_state",
        )

        return self._parse_state_description(result.text)

    async def _predict_next_state_teacher(
        self,
        current_state: WorldState,
        action: Action,
    ) -> WorldState:
        """Use teacher model for state prediction."""
        _ = f"""
        Given the current scene state:
        {current_state.describe()}

        And the action: {action.verb} {action.noun}

        Predict the next state. Describe:
        1. Object positions after the action
        2. Any state changes (open/closed, full/empty)
        3. Contact relationships
        """

        # Note: This would need frames for actual teacher inference
        # For now, fall back to rule-based
        return self._predict_next_state_rule_based(current_state, action)

    async def _annotate_dynamics_teacher(
        self,
        segment: Segment,
        video_frames: List[Any],
    ) -> List[PhysicalDynamics]:
        """Use teacher model for dynamics annotation."""
        prompt = """
        Analyze the physical dynamics in this video segment:
        1. Identify contact events and forces
        2. Estimate object masses and materials
        3. Describe motion trajectories
        4. Identify friction and elasticity properties
        """

        result = await self.teacher_model.annotate(
            frames=video_frames,
            prompt=prompt,
            task="world_model_dynamics",
        )

        return self._parse_dynamics_description(result.text, segment)

    async def _extract_causal_teacher(
        self,
        segment: Segment,
    ) -> List[Dict[str, str]]:
        """Use teacher model for causal extraction."""
        # Would need frames for actual implementation
        return self._extract_causal_rule_based(segment)

    async def _generate_counterfactuals_teacher(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_actions: List[Action],
    ) -> List[Dict[str, str]]:
        """Use teacher model for counterfactual generation."""
        counterfactuals = []

        for alt_action in alternative_actions:
            _ = f"""
            In the video, the action was: {actual_action.verb} {actual_action.noun}

            What would happen if instead the action was: {alt_action.verb} {alt_action.noun}?

            Describe the likely outcome in one sentence.
            """

            # Would need frames for actual implementation
            # For now, use rule-based
            cf = self._generate_single_counterfactual(
                segment, actual_action, alt_action
            )
            counterfactuals.append(cf)

        return counterfactuals

    # Rule-based implementations

    def _generate_state_before_rule_based(self, segment: Segment) -> WorldState:
        """Generate state using heuristics from segment data."""
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
                object_id=f"obj_{obj.name}_{id(obj)}",
                name=obj.name,
                state=obj.state or "unknown",
                material=obj.material or "unknown",
            )
            # Store color in attributes if available
            if obj.color:
                obj_state.attributes["color"] = obj.color

            # Infer affordances from object name
            obj_state.affordances = self._infer_affordances(obj.name)

            # Set role based on actions
            for action in segment.actions:
                if obj.name in action.noun:
                    obj_state.role = ObjectRole.TARGET
                    break
                if obj.name == action.instrument:
                    obj_state.role = ObjectRole.INSTRUMENT
                    break

            state.scene_graph.add_object(obj_state)

        # Add relationships based on scene context
        objects = list(state.scene_graph.objects.values())
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                # Simple proximity-based relationships
                rel = self._infer_relationship(obj1, obj2)
                if rel:
                    state.scene_graph.add_relationship(rel)

        return state

    def _generate_state_after_rule_based(self, segment: Segment) -> WorldState:
        """Generate state after action using heuristics."""
        # Start with before state
        state = self._generate_state_before_rule_based(segment)

        # Apply action effects
        for action in segment.actions:
            self._apply_action_effects(state, action)

        state.timestamp = segment.end_time
        return state

    def _predict_next_state_rule_based(
        self,
        current_state: WorldState,
        action: Action,
        horizon: float,
    ) -> WorldState:
        """Predict next state using physics heuristics."""
        predicted = current_state.copy()

        # Apply action effects
        self._apply_action_effects(predicted, action)

        # Add time evolution
        predicted.timestamp += horizon

        # Apply simple physics
        for obj in predicted.scene_graph.objects.values():
            if obj.is_moving():
                # Update position based on velocity
                obj.position += obj.velocity * horizon

        return predicted

    def _annotate_dynamics_rule_based(
        self,
        segment: Segment,
    ) -> List[PhysicalDynamics]:
        """Generate dynamics using heuristics."""
        dynamics_list = []

        for action in segment.actions:
            # Find target object
            target_obj = None
            for obj in segment.objects:
                if obj.name in action.noun:
                    target_obj = obj
                    break

            if target_obj:
                dyn = PhysicalDynamics(
                    object_id=f"obj_{target_obj.name}",
                    source="heuristic",
                )

                # Infer physical properties
                dyn.properties.mass = self._estimate_mass(target_obj.name)
                dyn.properties.material = target_obj.material or "unknown"
                dyn.properties.friction = self._estimate_friction(target_obj.name)

                # Create contact event for action
                if action.physical and action.physical.contact_type:
                    contact = ContactDynamics(
                        subject_id="agent",
                        object_id=f"obj_{target_obj.name}",
                        primary_contact_type=ContactType(action.physical.contact_type),
                    )
                    dyn.contact_events.append(contact)

                dynamics_list.append(dyn)

        return dynamics_list

    def _extract_causal_rule_based(
        self,
        segment: Segment,
        temporal_graph: Optional[TemporalEventGraph] = None,
    ) -> List[Dict[str, str]]:
        """Extract causal relations using heuristics."""
        causal_relations = []

        # Build temporal graph if not provided
        if temporal_graph is None:
            temporal_graph = self._build_temporal_graph(segment)

        # Extract causal chains
        for action in segment.actions:
            # Action -> Object state change
            for obj in segment.objects:
                if obj.name in action.noun:
                    causal_relations.append({
                        "cause": f"{action.verb} {action.noun}",
                        "effect": f"{obj.name} state change",
                        "type": "action_effect",
                    })

            # Tool -> Action enablement
            if action.instrument:
                causal_relations.append({
                    "cause": f"grasp {action.instrument}",
                    "effect": f"enable {action.verb}",
                    "type": "enablement",
                })

        return causal_relations

    def _generate_counterfactuals_rule_based(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_actions: List[Action],
    ) -> List[Dict[str, str]]:
        """Generate counterfactuals using heuristics."""
        counterfactuals = []

        for alt_action in alternative_actions:
            cf = self._generate_single_counterfactual(
                segment, actual_action, alt_action
            )
            counterfactuals.append(cf)

        return counterfactuals

    def _generate_single_counterfactual(
        self,
        segment: Segment,
        actual_action: Action,
        alternative_action: Action,
    ) -> Dict[str, str]:
        """Generate a single counterfactual scenario."""
        # Simple rule-based outcome prediction
        verb_outcomes = {
            "push": "moves away",
            "pull": "moves toward",
            "lift": "rises up",
            "place": "is set down",
            "open": "reveals contents",
            "close": "covers contents",
            "pour": "empties contents",
            "fill": "contains liquid",
        }

        outcome = verb_outcomes.get(
            alternative_action.verb,
            "changes state"
        )

        return {
            "if": f"{alternative_action.verb} {alternative_action.noun}",
            "then": f"the {alternative_action.noun} {outcome}",
            "instead_of": f"{actual_action.verb} {actual_action.noun}",
        }

    # Helper methods

    def _infer_affordances(self, object_name: str) -> set:
        """Infer affordances from object name."""
        affordances = set()
        name_lower = object_name.lower()

        # Common affordances
        if any(x in name_lower for x in ["cup", "mug", "glass", "bottle", "container"]):
            affordances.add(AffordanceState.GRASPABLE)
            affordances.add(AffordanceState.LIFTABLE)
            affordances.add(AffordanceState.CONTAINER)

        if any(x in name_lower for x in ["drawer", "door", "lid", "cap"]):
            affordances.add(AffordanceState.OPENABLE)
            affordances.add(AffordanceState.CLOSEABLE)

        if any(x in name_lower for x in ["spoon", "fork", "knife", "tool"]):
            affordances.add(AffordanceState.GRASPABLE)
            affordances.add(AffordanceState.PUSHABLE)

        if any(x in name_lower for x in ["pot", "pan", "plate", "bowl"]):
            affordances.add(AffordanceState.GRASPABLE)
            affordances.add(AffordanceState.LIFTABLE)

        if any(x in name_lower for x in ["button", "knob", "switch"]):
            affordances.add(AffordanceState.PUSHABLE)
            affordances.add(AffordanceState.ROTATABLE)

        return affordances

    def _infer_relationship(
        self,
        obj1: ObjectState,
        obj2: ObjectState,
    ) -> Optional[Relationship]:
        """Infer spatial relationship between objects."""
        # Simple name-based heuristics
        name1 = obj1.name.lower()
        name2 = obj2.name.lower()

        # Container relationships
        if any(x in name1 for x in ["cup", "bowl", "pot", "pan"]):
            if any(x in name2 for x in ["spoon", "fork", "liquid", "food"]):
                return Relationship(
                    subject_id=obj2.object_id,
                    object_id=obj1.object_id,
                    relation_type="contained_by",
                )

        # Support relationships
        if any(x in name1 for x in ["table", "counter", "surface"]):
            return Relationship(
                subject_id=obj2.object_id,
                object_id=obj1.object_id,
                relation_type="supported_by",
            )

        return None

    def _apply_action_effects(self, state: WorldState, action: Action) -> None:
        """Apply action effects to state."""
        # Find target object
        target_id = None
        for obj_id, obj in state.scene_graph.objects.items():
            if obj.name in action.noun:
                target_id = obj_id
                break

        if target_id is None:
            return

        target = state.scene_graph.objects[target_id]

        # Apply verb-specific effects
        verb = action.verb.lower()

        if verb in ["open", "unlock"]:
            target.state = "open"
        elif verb in ["close", "lock", "shut"]:
            target.state = "closed"
        elif verb in ["fill", "pour"]:
            target.state = "full"
        elif verb in ["empty", "drain", "clear"]:
            target.state = "empty"
        elif verb in ["turn_on", "activate", "start"]:
            target.state = "on"
        elif verb in ["turn_off", "deactivate", "stop"]:
            target.state = "off"

        # Position changes for movement verbs
        if verb in ["push", "move", "slide"]:
            target.velocity = np.array([0.1, 0.0, 0.0])  # Simple movement
        elif verb in ["lift", "pick_up", "raise"]:
            target.velocity = np.array([0.0, 0.1, 0.0])  # Upward movement

    def _estimate_mass(self, object_name: str) -> float:
        """Estimate object mass from name."""
        name_lower = object_name.lower()

        mass_estimates = {
            "spoon": 0.05,
            "fork": 0.05,
            "knife": 0.08,
            "cup": 0.3,
            "mug": 0.4,
            "glass": 0.3,
            "plate": 0.5,
            "bowl": 0.4,
            "bottle": 0.5,
            "pot": 1.0,
            "pan": 1.2,
            "book": 0.5,
            "phone": 0.2,
        }

        for key, mass in mass_estimates.items():
            if key in name_lower:
                return mass

        return 0.5  # Default

    def _estimate_friction(self, object_name: str) -> float:
        """Estimate friction coefficient from object name."""
        name_lower = object_name.lower()

        if any(x in name_lower for x in ["glass", "ceramic", "smooth"]):
            return 0.1
        elif any(x in name_lower for x in ["wood", "plastic"]):
            return 0.3
        elif any(x in name_lower for x in ["rubber", "grip"]):
            return 0.7
        elif any(x in name_lower for x in ["fabric", "cloth"]):
            return 0.5

        return 0.3  # Default

    def _build_temporal_graph(self, segment: Segment) -> TemporalEventGraph:
        """Build temporal event graph from segment."""
        graph = TemporalEventGraph()

        # Add action events
        for i, action in enumerate(segment.actions):
            event = TemporalEvent(
                event_id=f"action_{i}",
                event_type=EventType.ACTION_START,
                timestamp=action.start_time or segment.start_time,
                description=f"{action.verb} {action.noun}",
                objects=[action.noun],
            )
            graph.add_event(event)

            if action.end_time:
                end_event = TemporalEvent(
                    event_id=f"action_{i}_end",
                    event_type=EventType.ACTION_END,
                    timestamp=action.end_time,
                    description=f"end {action.verb} {action.noun}",
                    objects=[action.noun],
                )
                graph.add_event(end_event)

        # Infer temporal relations
        inferred = graph.infer_relations()
        for rel in inferred:
            graph.add_relation(rel)

        return graph

    def _build_state_prompt(self, segment: Segment, timing: str) -> str:
        """Build prompt for state generation."""
        return f"""
        Describe the scene state {timing} the action.

        Segment: {segment.caption}
        Objects: {', '.join(obj.name for obj in segment.objects)}
        Actions: {', '.join(f"{a.verb} {a.noun}" for a in segment.actions)}

        Describe:
        1. Object positions and states
        2. Spatial relationships
        3. Contact information
        """

    def _parse_state_description(self, text: str) -> WorldState:
        """Parse teacher model response into WorldState."""
        # Simplified parsing - would need more sophisticated NLP
        state = WorldState()

        # Extract object mentions
        lines = text.split("\n")
        for line in lines:
            # Very basic parsing
            if "object" in line.lower() or "position" in line.lower():
                # Would extract actual object info
                pass

        return state

    def _parse_dynamics_description(
        self,
        text: str,
        segment: Segment,
    ) -> List[PhysicalDynamics]:
        """Parse teacher model response into PhysicalDynamics."""
        # Simplified parsing
        dynamics_list = []

        for obj in segment.objects:
            dyn = PhysicalDynamics(
                object_id=f"obj_{obj.name}",
                source="teacher",
            )
            dynamics_list.append(dyn)

        return dynamics_list

    def _extract_preconditions(self, state: WorldState) -> List[str]:
        """Extract preconditions from state."""
        preconditions = []

        for obj in state.scene_graph.objects.values():
            if obj.state != "unknown":
                preconditions.append(f"{obj.name} is {obj.state}")

        return preconditions

    def _extract_effects(self, before: WorldState, after: WorldState) -> List[str]:
        """Extract effects by comparing states."""
        effects = []

        for obj_id, after_obj in after.scene_graph.objects.items():
            if obj_id in before.scene_graph.objects:
                before_obj = before.scene_graph.objects[obj_id]
                if before_obj.state != after_obj.state:
                    effects.append(
                        f"{after_obj.name} changed from {before_obj.state} to {after_obj.state}"
                    )

        return effects

    def _generate_alternative_actions(self, actual: Action) -> List[Action]:
        """Generate alternative actions for counterfactuals."""
        alternatives = []

        # Verb substitutions
        verb_pairs = [
            ("push", "pull"),
            ("lift", "lower"),
            ("open", "close"),
            ("fill", "empty"),
            ("turn_on", "turn_off"),
        ]

        for v1, v2 in verb_pairs:
            if actual.verb == v1:
                alternatives.append(Action(verb=v2, noun=actual.noun))
            elif actual.verb == v2:
                alternatives.append(Action(verb=v1, noun=actual.noun))

        # Add generic alternatives
        if actual.verb not in ["push", "pull"]:
            alternatives.append(Action(verb="push", noun=actual.noun))
            alternatives.append(Action(verb="pull", noun=actual.noun))

        return alternatives[:3]  # Limit to 3 alternatives
