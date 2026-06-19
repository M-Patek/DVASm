"""Tests for world_model temporal_graph module."""

import pytest

from dvas.world_model.temporal_graph import (
    EventType,
    MultiObjectTransitionGraph,
    ObjectStateTransitionGraph,
    StateTransition,
    TemporalEvent,
    TemporalEventGraph,
    TemporalRelation,
    TemporalRelationType,
)


class TestTemporalEvent:
    """Tests for TemporalEvent class."""

    def test_create(self):
        """Test creating temporal event."""
        event = TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=1.0,
        )
        assert event.event_id == "event_1"
        assert event.event_type == EventType.ACTION_START

    def test_end_time(self):
        """Test end time calculation."""
        event = TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=1.0,
            duration=2.0,
        )
        assert event.end_time == 3.0

    def test_is_instantaneous(self):
        """Test instantaneous check."""
        event1 = TemporalEvent(
            event_id="event_1",
            event_type=EventType.COLLISION,
            timestamp=1.0,
        )
        event2 = TemporalEvent(
            event_id="event_2",
            event_type=EventType.ACTION_START,
            timestamp=1.0,
            duration=2.0,
        )
        assert event1.is_instantaneous
        assert not event2.is_instantaneous

    def test_overlaps(self):
        """Test overlap detection."""
        event1 = TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=0.0,
            duration=3.0,
        )
        event2 = TemporalEvent(
            event_id="event_2",
            event_type=EventType.CONTACT_START,
            timestamp=2.0,
            duration=2.0,
        )
        assert event1.overlaps(event2)

    def test_to_dict(self):
        """Test serialization."""
        event = TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=1.0,
            description="Test event",
        )
        data = event.to_dict()
        assert data["event_id"] == "event_1"
        assert data["event_type"] == "action_start"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "event_id": "event_1",
            "event_type": "collision",
            "timestamp": 2.0,
            "duration": 0.0,
            "objects": ["obj1", "obj2"],
        }
        event = TemporalEvent.from_dict(data)
        assert event.event_type == EventType.COLLISION
        assert event.objects == ["obj1", "obj2"]


class TestTemporalRelation:
    """Tests for TemporalRelation class."""

    def test_create(self):
        """Test creating temporal relation."""
        rel = TemporalRelation(
            event_a_id="event_1",
            event_b_id="event_2",
            relation_type=TemporalRelationType.BEFORE,
        )
        assert rel.event_a_id == "event_1"
        assert rel.relation_type == TemporalRelationType.BEFORE

    def test_to_dict(self):
        """Test serialization."""
        rel = TemporalRelation(
            event_a_id="event_1",
            event_b_id="event_2",
            relation_type=TemporalRelationType.CAUSES,
            confidence=0.9,
        )
        data = rel.to_dict()
        assert data["relation_type"] == "causes"
        assert data["confidence"] == 0.9

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "event_a_id": "event_1",
            "event_b_id": "event_2",
            "relation_type": "during",
            "confidence": 0.8,
        }
        rel = TemporalRelation.from_dict(data)
        assert rel.relation_type == TemporalRelationType.DURING


class TestTemporalEventGraph:
    """Tests for TemporalEventGraph class."""

    def test_create_empty(self):
        """Test creating empty graph."""
        graph = TemporalEventGraph()
        assert len(graph.events) == 0

    def test_add_event(self):
        """Test adding events."""
        graph = TemporalEventGraph()
        event = TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=1.0,
        )
        graph.add_event(event)
        assert "event_1" in graph.events

    def test_add_relation(self):
        """Test adding relations."""
        graph = TemporalEventGraph()
        event1 = TemporalEvent(event_id="event_1", event_type=EventType.ACTION_START, timestamp=0.0)
        event2 = TemporalEvent(event_id="event_2", event_type=EventType.ACTION_END, timestamp=2.0)
        graph.add_event(event1)
        graph.add_event(event2)

        rel = TemporalRelation(
            event_a_id="event_1",
            event_b_id="event_2",
            relation_type=TemporalRelationType.BEFORE,
        )
        graph.add_relation(rel)
        assert len(graph.relations) == 1

    def test_get_events_at_time(self):
        """Test getting events at specific time."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(
            event_id="event_1",
            event_type=EventType.ACTION_START,
            timestamp=0.0,
            duration=3.0,
        ))
        graph.add_event(TemporalEvent(
            event_id="event_2",
            event_type=EventType.CONTACT_START,
            timestamp=2.0,
            duration=1.0,
        ))

        events = graph.get_events_at_time(1.5)
        assert len(events) == 1

        events = graph.get_events_at_time(2.5)
        assert len(events) == 2

    def test_get_events_in_range(self):
        """Test getting events in time range."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0))
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=5.0))
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.ACTION_START, timestamp=10.0))

        events = graph.get_events_in_range(2.0, 8.0)
        assert len(events) == 1
        assert events[0].event_id == "e2"

    def test_get_event_order(self):
        """Test getting events in chronological order."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=2.0))
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=1.0))
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.ACTION_START, timestamp=3.0))

        order = graph.get_event_order()
        assert order == ["e1", "e2", "e3"]

    def test_get_parallel_events(self):
        """Test grouping parallel events."""
        graph = TemporalEventGraph()
        # Sequential events
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0, duration=1.0))
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=2.0, duration=1.0))
        # Parallel events
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.ACTION_START, timestamp=3.0, duration=2.0))
        graph.add_event(TemporalEvent(event_id="e4", event_type=EventType.ACTION_START, timestamp=3.5, duration=1.0))

        groups = graph.get_parallel_events()
        assert len(groups) == 3  # [e1], [e2], [e3, e4]
        assert len(groups[2]) == 2  # e3 and e4 are parallel

    def test_infer_relations(self):
        """Test relation inference."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0, duration=1.0))
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=2.0, duration=1.0))
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.ACTION_START, timestamp=0.5, duration=2.0))

        inferred = graph.infer_relations()
        assert len(inferred) > 0

        # Check that e1 is before e2
        before_rel = [r for r in inferred if r.event_a_id == "e1" and r.event_b_id == "e2"]
        assert len(before_rel) > 0
        assert before_rel[0].relation_type == TemporalRelationType.BEFORE

    def test_get_causal_chain(self):
        """Test causal chain extraction."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0))
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.STATE_CHANGE, timestamp=1.0))
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.GOAL_REACHED, timestamp=2.0))

        graph.add_relation(TemporalRelation("e1", "e2", TemporalRelationType.CAUSES))
        graph.add_relation(TemporalRelation("e2", "e3", TemporalRelationType.CAUSES))

        chain = graph.get_causal_chain("e1")
        assert chain == ["e1", "e2", "e3"]

    def test_traverse(self):
        """Test graph traversal."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0))
        graph.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=1.0))
        graph.add_event(TemporalEvent(event_id="e3", event_type=EventType.ACTION_START, timestamp=2.0))

        graph.add_relation(TemporalRelation("e1", "e2", TemporalRelationType.BEFORE))
        graph.add_relation(TemporalRelation("e2", "e3", TemporalRelationType.BEFORE))

        traversed = list(graph.traverse("e1"))
        assert len(traversed) == 3

    def test_to_dict(self):
        """Test serialization."""
        graph = TemporalEventGraph()
        graph.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0))
        graph.add_relation(TemporalRelation("e1", "e2", TemporalRelationType.BEFORE))

        data = graph.to_dict()
        assert "events" in data
        assert "relations" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "events": {
                "e1": {
                    "event_id": "e1",
                    "event_type": "action_start",
                    "timestamp": 0.0,
                    "duration": 0.0,
                    "objects": [],
                    "source": "unknown",
                }
            },
            "relations": [],
        }
        graph = TemporalEventGraph.from_dict(data)
        assert "e1" in graph.events

    def test_merge(self):
        """Test graph merging."""
        graph1 = TemporalEventGraph()
        graph1.add_event(TemporalEvent(event_id="e1", event_type=EventType.ACTION_START, timestamp=0.0))

        graph2 = TemporalEventGraph()
        graph2.add_event(TemporalEvent(event_id="e2", event_type=EventType.ACTION_START, timestamp=1.0))

        merged = graph1.merge(graph2)
        assert "e1" in merged.events
        assert "e2" in merged.events


class TestStateTransition:
    """Tests for StateTransition class."""

    def test_create(self):
        """Test creating state transition."""
        transition = StateTransition(
            object_id="cup_1",
            from_state="empty",
            to_state="full",
            timestamp=1.0,
        )
        assert transition.object_id == "cup_1"
        assert transition.from_state == "empty"
        assert transition.to_state == "full"

    def test_to_dict(self):
        """Test serialization."""
        transition = StateTransition(
            object_id="cup_1",
            from_state="closed",
            to_state="open",
            timestamp=2.0,
            confidence=0.95,
        )
        data = transition.to_dict()
        assert data["from_state"] == "closed"
        assert data["confidence"] == 0.95

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "object_id": "cup_1",
            "from_state": "empty",
            "to_state": "full",
            "timestamp": 1.5,
            "trigger_event_id": "pour_action",
        }
        transition = StateTransition.from_dict(data)
        assert transition.trigger_event_id == "pour_action"


class TestObjectStateTransitionGraph:
    """Tests for ObjectStateTransitionGraph class."""

    def test_create(self):
        """Test creating transition graph."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        assert graph.object_id == "cup_1"
        assert len(graph.transitions) == 0

    def test_add_transition(self):
        """Test adding transitions."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        transition = StateTransition(
            object_id="cup_1",
            from_state="empty",
            to_state="full",
            timestamp=1.0,
        )
        graph.add_transition(transition)
        assert len(graph.transitions) == 1

    def test_add_transition_wrong_object(self):
        """Test adding transition for wrong object is rejected."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        transition = StateTransition(
            object_id="cup_2",  # Wrong object
            from_state="empty",
            to_state="full",
            timestamp=1.0,
        )
        graph.add_transition(transition)
        assert len(graph.transitions) == 0

    def test_add_transition_chronological_order(self):
        """Test transitions are kept in chronological order."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=2.0))
        graph.add_transition(StateTransition("cup_1", "b", "c", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "c", "d", timestamp=3.0))

        timestamps = [t.timestamp for t in graph.transitions]
        assert timestamps == [1.0, 2.0, 3.0]

    def test_get_state_at_time(self):
        """Test getting state at specific time."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "unknown", "empty", timestamp=0.0))
        graph.add_transition(StateTransition("cup_1", "empty", "full", timestamp=2.0))
        graph.add_transition(StateTransition("cup_1", "full", "empty", timestamp=5.0))

        assert graph.get_state_at_time(1.0) == "empty"
        assert graph.get_state_at_time(3.0) == "full"
        assert graph.get_state_at_time(6.0) == "empty"

    def test_get_state_sequence(self):
        """Test getting state sequence."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "closed", "open", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "open", "closed", timestamp=3.0))

        sequence = graph.get_state_sequence()
        assert len(sequence) == 3  # initial + 2 transitions
        assert sequence[1] == (1.0, "open")

    def test_get_transition_counts(self):
        """Test transition counting."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "b", "a", timestamp=2.0))
        graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=3.0))

        counts = graph.get_transition_counts()
        assert counts["b"] == 2
        assert counts["a"] == 1

    def test_get_transition_matrix(self):
        """Test transition matrix."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "closed", "open", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "open", "closed", timestamp=2.0))
        graph.add_transition(StateTransition("cup_1", "closed", "open", timestamp=3.0))

        matrix = graph.get_transition_matrix()
        assert matrix[("closed", "open")] == 2
        assert matrix[("open", "closed")] == 1

    def test_find_cycles(self):
        """Test cycle detection."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "b", "c", timestamp=2.0))
        graph.add_transition(StateTransition("cup_1", "c", "a", timestamp=3.0))  # Cycle

        cycles = graph.find_cycles()
        assert len(cycles) > 0

    def test_predict_next_state(self):
        """Test next state prediction."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "closed", "open", timestamp=1.0))
        graph.add_transition(StateTransition("cup_1", "closed", "open", timestamp=3.0))
        graph.add_transition(StateTransition("cup_1", "open", "closed", timestamp=2.0))

        next_state = graph.predict_next_state("closed")
        assert next_state == "open"

        next_state = graph.predict_next_state("unknown")
        assert next_state is None

    def test_to_dict(self):
        """Test serialization."""
        graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))

        data = graph.to_dict()
        assert data["object_id"] == "cup_1"
        assert len(data["transitions"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "object_id": "cup_1",
            "transitions": [
                {
                    "object_id": "cup_1",
                    "from_state": "empty",
                    "to_state": "full",
                    "timestamp": 1.0,
                    "duration": 0.0,
                    "confidence": 1.0,
                }
            ],
        }
        graph = ObjectStateTransitionGraph.from_dict(data)
        assert len(graph.transitions) == 1
        assert graph.transitions[0].from_state == "empty"

    def test_merge(self):
        """Test graph merging."""
        graph1 = ObjectStateTransitionGraph(object_id="cup_1")
        graph1.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))

        graph2 = ObjectStateTransitionGraph(object_id="cup_1")
        graph2.add_transition(StateTransition("cup_1", "c", "d", timestamp=2.0))

        merged = graph1.merge(graph2)
        assert len(merged.transitions) == 2

    def test_merge_different_objects_raises(self):
        """Test merging graphs for different objects raises error."""
        graph1 = ObjectStateTransitionGraph(object_id="cup_1")
        graph2 = ObjectStateTransitionGraph(object_id="cup_2")

        with pytest.raises(ValueError):
            graph1.merge(graph2)


class TestMultiObjectTransitionGraph:
    """Tests for MultiObjectTransitionGraph class."""

    def test_create(self):
        """Test creating multi-object graph."""
        graph = MultiObjectTransitionGraph()
        assert len(graph.object_graphs) == 0

    def test_add_object_graph(self):
        """Test adding object graphs."""
        graph = MultiObjectTransitionGraph()
        obj_graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_object_graph(obj_graph)
        assert "cup_1" in graph.object_graphs

    def test_get_object_graph(self):
        """Test getting object graph."""
        graph = MultiObjectTransitionGraph()
        obj_graph = ObjectStateTransitionGraph(object_id="cup_1")
        graph.add_object_graph(obj_graph)

        retrieved = graph.get_object_graph("cup_1")
        assert retrieved is not None
        assert retrieved.object_id == "cup_1"

    def test_get_states_at_time(self):
        """Test getting all states at time."""
        graph = MultiObjectTransitionGraph()

        cup_graph = ObjectStateTransitionGraph(object_id="cup_1")
        cup_graph.add_transition(StateTransition("cup_1", "unknown", "empty", timestamp=0.0))
        cup_graph.add_transition(StateTransition("cup_1", "empty", "full", timestamp=2.0))
        graph.add_object_graph(cup_graph)

        lid_graph = ObjectStateTransitionGraph(object_id="lid_1")
        lid_graph.add_transition(StateTransition("lid_1", "unknown", "off", timestamp=0.0))
        lid_graph.add_transition(StateTransition("lid_1", "off", "on", timestamp=1.0))
        graph.add_object_graph(lid_graph)

        states = graph.get_states_at_time(1.5)
        assert states["cup_1"] == "empty"
        assert states["lid_1"] == "on"

    def test_find_correlated_transitions(self):
        """Test finding correlated transitions."""
        graph = MultiObjectTransitionGraph()

        cup_graph = ObjectStateTransitionGraph(object_id="cup_1")
        cup_graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))
        graph.add_object_graph(cup_graph)

        lid_graph = ObjectStateTransitionGraph(object_id="lid_1")
        lid_graph.add_transition(StateTransition("lid_1", "x", "y", timestamp=1.02))
        graph.add_object_graph(lid_graph)

        # Also add a third object with a transition far away
        spoon_graph = ObjectStateTransitionGraph(object_id="spoon_1")
        spoon_graph.add_transition(StateTransition("spoon_1", "m", "n", timestamp=5.0))
        graph.add_object_graph(spoon_graph)

        correlated = graph.find_correlated_transitions(time_window=0.1)
        # cup_1 and lid_1 transitions should be correlated
        assert len(correlated) >= 1
        # First group should contain cup_1 and lid_1
        assert "cup_1" in correlated[0]
        assert "lid_1" in correlated[0]

    def test_to_dict(self):
        """Test serialization."""
        graph = MultiObjectTransitionGraph()
        obj_graph = ObjectStateTransitionGraph(object_id="cup_1")
        obj_graph.add_transition(StateTransition("cup_1", "a", "b", timestamp=1.0))
        graph.add_object_graph(obj_graph)

        data = graph.to_dict()
        assert "object_graphs" in data
        assert "cup_1" in data["object_graphs"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "object_graphs": {
                "cup_1": {
                    "object_id": "cup_1",
                    "transitions": [
                        {
                            "object_id": "cup_1",
                            "from_state": "empty",
                            "to_state": "full",
                            "timestamp": 1.0,
                            "duration": 0.0,
                            "confidence": 1.0,
                        }
                    ],
                }
            },
        }
        graph = MultiObjectTransitionGraph.from_dict(data)
        assert "cup_1" in graph.object_graphs
        assert len(graph.object_graphs["cup_1"].transitions) == 1
