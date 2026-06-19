"""Tests for world_model annotator module."""

import pytest

from dvas.data.schemas import Action, Annotation, Segment, VideoMetadata
from dvas.world_model.annotator import WorldModelAnnotator
from dvas.world_model.state_repr import (
    AffordanceState,
    ObjectRole,
    ObjectState,
    SceneGraph,
    WorldState,
)
from dvas.world_model.temporal_graph import TemporalEventGraph


class TestWorldModelAnnotator:
    """Tests for WorldModelAnnotator class."""

    @pytest.fixture
    def annotator(self):
        """Create a basic annotator."""
        return WorldModelAnnotator()

    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment for testing."""
        return Segment(
            start_time=0.0,
            end_time=2.0,
            caption="Pick up the cup",
            actions=[
                Action(
                    verb="pick_up",
                    noun="cup",
                    hand="right",
                    start_time=0.5,
                    end_time=1.5,
                ),
            ],
            objects=[
                {"name": "cup", "state": "on_table"},
                {"name": "table"},
            ],
        )

    @pytest.mark.asyncio
    async def test_generate_state_before(self, annotator, sample_segment):
        """Test generating state before action."""
        state = await annotator.generate_state_before(sample_segment)

        assert isinstance(state, WorldState)
        assert state.timestamp == sample_segment.start_time
        assert len(state.scene_graph.objects) > 0

    @pytest.mark.asyncio
    async def test_generate_state_after(self, annotator, sample_segment):
        """Test generating state after action."""
        state = await annotator.generate_state_after(sample_segment)

        assert isinstance(state, WorldState)
        assert state.timestamp == sample_segment.end_time

    @pytest.mark.asyncio
    async def test_predict_next_state(self, annotator):
        """Test predicting next state."""
        current_state = WorldState(timestamp=0.0)
        current_state.scene_graph.add_object(
            ObjectState(object_id="cup_1", name="cup", position=[0, 0, 0])
        )

        action = Action(verb="push", noun="cup")

        predicted = await annotator.predict_next_state(current_state, action)

        assert isinstance(predicted, WorldState)
        assert predicted.timestamp > current_state.timestamp

    @pytest.mark.asyncio
    async def test_annotate_physical_dynamics(self, annotator, sample_segment):
        """Test annotating physical dynamics."""
        from dvas.world_model.dynamics import PhysicalDynamics

        dynamics = await annotator.annotate_physical_dynamics(sample_segment)

        assert isinstance(dynamics, list)
        for d in dynamics:
            assert isinstance(d, PhysicalDynamics)

    @pytest.mark.asyncio
    async def test_extract_causal_relations(self, annotator, sample_segment):
        """Test extracting causal relations."""
        relations = await annotator.extract_causal_relations(sample_segment)

        assert isinstance(relations, list)
        for r in relations:
            assert "cause" in r
            assert "effect" in r

    @pytest.mark.asyncio
    async def test_generate_counterfactuals(self, annotator, sample_segment):
        """Test generating counterfactuals."""
        actual_action = Action(verb="push", noun="cup")
        alternative = Action(verb="pull", noun="cup")

        counterfactuals = await annotator.generate_counterfactuals(
            sample_segment, actual_action, [alternative]
        )

        assert isinstance(counterfactuals, list)
        assert len(counterfactuals) > 0

        for cf in counterfactuals:
            assert "if" in cf
            assert "then" in cf

    @pytest.mark.asyncio
    async def test_generate_state_prediction(self, annotator, sample_segment):
        """Test generating state prediction schema."""
        from dvas.data.schemas import StatePrediction

        prediction = await annotator.generate_state_prediction(sample_segment)

        assert isinstance(prediction, StatePrediction)

    @pytest.mark.asyncio
    async def test_generate_dynamics(self, annotator, sample_segment):
        """Test generating dynamics annotation."""
        from dvas.data.schemas import DynamicsAnnotation

        dynamics = await annotator.generate_dynamics(sample_segment)

        assert isinstance(dynamics, DynamicsAnnotation)

    @pytest.mark.asyncio
    async def test_generate_counterfactual(self, annotator, sample_segment):
        """Test generating counterfactual annotation."""
        from dvas.data.schemas import DynamicsAnnotation

        actual_action = Action(verb="push", noun="cup")
        alternative_action = Action(verb="pull", noun="cup")

        cf = await annotator.generate_counterfactual(
            sample_segment, actual_action, alternative_action
        )

        assert isinstance(cf, DynamicsAnnotation)

    @pytest.mark.asyncio
    async def test_annotate_full(self, annotator, sample_segment):
        """Test generating complete annotation."""
        from dvas.data.schemas import Annotation

        annotation = await annotator.annotate_full(sample_segment)

        assert isinstance(annotation, Annotation)
        assert annotation.state_predictions is not None
        assert annotation.dynamics is not None

    def test_infer_affordances(self, annotator):
        """Test affordance inference."""
        affordances = annotator._infer_affordances("cup")

        assert AffordanceState.GRASPABLE in affordances
        assert AffordanceState.LIFTABLE in affordances

    def test_infer_affordances_drawer(self, annotator):
        """Test affordance inference for drawer."""
        affordances = annotator._infer_affordances("drawer")

        assert AffordanceState.OPENABLE in affordances
        assert AffordanceState.CLOSEABLE in affordances

    def test_estimate_mass(self, annotator):
        """Test mass estimation."""
        cup_mass = annotator._estimate_mass("cup")
        spoon_mass = annotator._estimate_mass("spoon")

        assert cup_mass > spoon_mass
        assert cup_mass == 0.3  # From mass_estimates

    def test_estimate_friction(self, annotator):
        """Test friction estimation."""
        glass_friction = annotator._estimate_friction("glass")
        rubber_friction = annotator._estimate_friction("rubber_grip")

        assert rubber_friction > glass_friction

    def test_generate_alternative_actions(self, annotator):
        """Test generating alternative actions."""
        actual = Action(verb="push", noun="cup")
        alternatives = annotator._generate_alternative_actions(actual)

        assert len(alternatives) > 0
        assert all(isinstance(a, Action) for a in alternatives)


class TestWorldModelAnnotatorHeuristics:
    """Tests for rule-based/heuristic implementations."""

    @pytest.fixture
    def annotator(self):
        """Create a rule-based annotator."""
        return WorldModelAnnotator(use_teacher=False)

    def test_apply_action_effects_open(self, annotator):
        """Test applying open action effects."""
        state = WorldState()
        drawer = ObjectState(
            object_id="drawer_1",
            name="drawer",
            state="closed",
        )
        state.scene_graph.add_object(drawer)

        action = Action(verb="open", noun="drawer")
        annotator._apply_action_effects(state, action)

        assert drawer.state == "open"

    def test_apply_action_effects_fill(self, annotator):
        """Test applying fill action effects."""
        state = WorldState()
        cup = ObjectState(
            object_id="cup_1",
            name="cup",
            state="empty",
        )
        state.scene_graph.add_object(cup)

        action = Action(verb="fill", noun="cup")
        annotator._apply_action_effects(state, action)

        assert cup.state == "full"

    def test_apply_action_effects_movement(self, annotator):
        """Test applying movement action effects."""
        state = WorldState()
        cup = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[0, 0, 0],
            velocity=[0, 0, 0],
        )
        state.scene_graph.add_object(cup)

        action = Action(verb="push", noun="cup")
        annotator._apply_action_effects(state, action)

        assert cup.speed() > 0

    def test_build_temporal_graph(self, annotator):
        """Test building temporal graph."""
        segment = Segment(
            start_time=0.0,
            end_time=2.0,
            caption="Test",
            actions=[
                Action(
                    verb="pick_up",
                    noun="cup",
                    start_time=0.5,
                    end_time=1.5,
                ),
            ],
        )

        graph = annotator._build_temporal_graph(segment)

        assert isinstance(graph, TemporalEventGraph)
        assert len(graph.events) >= 2  # Start and end events

    def test_extract_preconditions(self, annotator):
        """Test precondition extraction."""
        state = WorldState()
        state.scene_graph.add_object(
            ObjectState(object_id="cup_1", name="cup", state="on_table")
        )
        state.scene_graph.add_object(
            ObjectState(object_id="lid_1", name="lid", state="closed")
        )

        preconditions = annotator._extract_preconditions(state)

        assert len(preconditions) > 0
        assert any("on_table" in p for p in preconditions)

    def test_extract_effects(self, annotator):
        """Test effect extraction."""
        before = WorldState()
        before.scene_graph.add_object(
            ObjectState(object_id="cup_1", name="cup", state="empty")
        )

        after = WorldState()
        after.scene_graph.add_object(
            ObjectState(object_id="cup_1", name="cup", state="full")
        )

        effects = annotator._extract_effects(before, after)

        assert len(effects) > 0
        assert any("full" in e for e in effects)


class TestWorldModelAnnotatorIntegration:
    """Integration tests for WorldModelAnnotator."""

    @pytest.mark.asyncio
    async def test_full_annotation_pipeline(self):
        """Test complete annotation pipeline."""
        annotator = WorldModelAnnotator(use_teacher=False)

        # Create test segment
        segment = Segment(
            start_time=0.0,
            end_time=3.0,
            caption="Open the drawer and take out a spoon",
            actions=[
                Action(verb="open", noun="drawer", start_time=0.5, end_time=1.5),
                Action(verb="take", noun="spoon", start_time=1.5, end_time=2.5),
            ],
            objects=[
                {"name": "drawer", "state": "closed"},
                {"name": "spoon"},
                {"name": "table"},
            ],
        )

        # Generate full annotation
        annotation = await annotator.annotate_full(segment)

        # Verify structure
        assert annotation.id.startswith("wm_")
        assert len(annotation.segments) == 1

        # Verify state predictions
        assert annotation.state_predictions is not None
        assert annotation.state_predictions.predicted_next_frame_desc is not None

        # Verify dynamics
        assert annotation.dynamics is not None

    @pytest.mark.asyncio
    async def test_state_consistency(self):
        """Test that before/after states are consistent."""
        annotator = WorldModelAnnotator(use_teacher=False)

        segment = Segment(
            start_time=0.0,
            end_time=2.0,
            caption="Open the drawer",
            actions=[Action(verb="open", noun="drawer")],
            objects=[{"name": "drawer", "state": "closed"}],
        )

        state_before = await annotator.generate_state_before(segment)
        state_after = await annotator.generate_state_after(segment)

        # Check that timestamp ordering is correct
        assert state_before.timestamp <= state_after.timestamp
