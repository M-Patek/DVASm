"""Tests for world_model state representation module."""

import numpy as np
import pytest

from dvas.world_model.state_repr import (
    AffordanceState,
    ContactState,
    ObjectRole,
    ObjectState,
    Relationship,
    SceneGraph,
    WorldState,
)


class TestObjectState:
    """Tests for ObjectState class."""

    def test_create_basic(self):
        """Test creating a basic ObjectState."""
        obj = ObjectState(
            object_id="cup_1",
            name="cup",
        )
        assert obj.object_id == "cup_1"
        assert obj.name == "cup"
        assert np.allclose(obj.position, [0, 0, 0])
        assert obj.is_visible

    def test_create_with_position(self):
        """Test creating ObjectState with position."""
        obj = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[1.0, 2.0, 3.0],
        )
        assert np.allclose(obj.position, [1.0, 2.0, 3.0])

    def test_distance_to(self):
        """Test distance calculation between objects."""
        obj1 = ObjectState(
            object_id="obj1",
            name="obj1",
            position=[0, 0, 0],
        )
        obj2 = ObjectState(
            object_id="obj2",
            name="obj2",
            position=[3, 4, 0],
        )
        assert obj1.distance_to(obj2) == 5.0

    def test_speed(self):
        """Test speed calculation."""
        obj = ObjectState(
            object_id="obj1",
            name="obj1",
            velocity=[3, 4, 0],
        )
        assert obj.speed() == 5.0

    def test_is_moving(self):
        """Test is_moving check."""
        obj1 = ObjectState(
            object_id="obj1",
            name="obj1",
            velocity=[0.1, 0, 0],
        )
        obj2 = ObjectState(
            object_id="obj2",
            name="obj2",
            velocity=[0.001, 0, 0],
        )
        assert obj1.is_moving(threshold=0.01)
        assert not obj2.is_moving(threshold=0.01)

    def test_to_dict(self):
        """Test serialization to dict."""
        obj = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[1.0, 2.0, 3.0],
            affordances={AffordanceState.GRASPABLE, AffordanceState.LIFTABLE},
        )
        data = obj.to_dict()
        assert data["object_id"] == "cup_1"
        assert data["name"] == "cup"
        assert data["position"] == [1.0, 2.0, 3.0]
        assert "graspable" in data["affordances"]
        assert "liftable" in data["affordances"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "object_id": "cup_1",
            "name": "cup",
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.2, 0.3],
            "affordances": ["graspable", "liftable"],
            "role": "target",
        }
        obj = ObjectState.from_dict(data)
        assert obj.object_id == "cup_1"
        assert obj.name == "cup"
        assert np.allclose(obj.position, [1.0, 2.0, 3.0])
        assert AffordanceState.GRASPABLE in obj.affordances
        assert obj.role == ObjectRole.TARGET

    def test_copy(self):
        """Test deep copy."""
        obj = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[1.0, 2.0, 3.0],
        )
        copy = obj.copy()
        assert copy.object_id == obj.object_id
        assert np.allclose(copy.position, obj.position)
        # Ensure independent
        copy.position[0] = 100
        assert not np.allclose(copy.position, obj.position)


class TestRelationship:
    """Tests for Relationship class."""

    def test_create(self):
        """Test creating a Relationship."""
        rel = Relationship(
            subject_id="cup_1",
            object_id="table_1",
            relation_type="on",
        )
        assert rel.subject_id == "cup_1"
        assert rel.object_id == "table_1"
        assert rel.relation_type == "on"

    def test_to_dict(self):
        """Test serialization."""
        rel = Relationship(
            subject_id="cup_1",
            object_id="table_1",
            relation_type="on",
            contact_state=ContactState.SUPPORTED_BY,
        )
        data = rel.to_dict()
        assert data["subject_id"] == "cup_1"
        assert data["contact_state"] == "supported_by"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "subject_id": "cup_1",
            "object_id": "table_1",
            "relation_type": "on",
            "contact_state": "supported_by",
            "confidence": 0.9,
        }
        rel = Relationship.from_dict(data)
        assert rel.subject_id == "cup_1"
        assert rel.contact_state == ContactState.SUPPORTED_BY
        assert rel.confidence == 0.9


class TestSceneGraph:
    """Tests for SceneGraph class."""

    def test_create_empty(self):
        """Test creating empty SceneGraph."""
        graph = SceneGraph()
        assert len(graph.objects) == 0
        assert len(graph.relationships) == 0

    def test_add_object(self):
        """Test adding objects."""
        graph = SceneGraph()
        obj = ObjectState(object_id="cup_1", name="cup")
        graph.add_object(obj)
        assert "cup_1" in graph.objects
        assert graph.objects["cup_1"].name == "cup"

    def test_remove_object(self):
        """Test removing objects."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="cup_1", name="cup")
        obj2 = ObjectState(object_id="table_1", name="table")
        graph.add_object(obj1)
        graph.add_object(obj2)

        rel = Relationship("cup_1", "table_1", "on")
        graph.add_relationship(rel)

        graph.remove_object("cup_1")
        assert "cup_1" not in graph.objects
        assert len(graph.relationships) == 0

    def test_add_relationship(self):
        """Test adding relationships."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="cup_1", name="cup")
        obj2 = ObjectState(object_id="table_1", name="table")
        graph.add_object(obj1)
        graph.add_object(obj2)

        rel = Relationship("cup_1", "table_1", "on")
        graph.add_relationship(rel)
        assert len(graph.relationships) == 1

    def test_add_relationship_missing_object(self):
        """Test adding relationship with missing object."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="cup_1", name="cup")
        graph.add_object(obj1)

        rel = Relationship("cup_1", "missing_1", "on")
        graph.add_relationship(rel)  # Should log warning but not crash
        assert len(graph.relationships) == 0

    def test_get_relationships(self):
        """Test getting relationships for an object."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="cup_1", name="cup")
        obj2 = ObjectState(object_id="table_1", name="table")
        obj3 = ObjectState(object_id="spoon_1", name="spoon")
        graph.add_object(obj1)
        graph.add_object(obj2)
        graph.add_object(obj3)

        graph.add_relationship(Relationship("cup_1", "table_1", "on"))
        graph.add_relationship(Relationship("spoon_1", "cup_1", "in"))

        rels = graph.get_relationships("cup_1")
        assert len(rels) == 2

    def test_get_related_objects(self):
        """Test getting related objects."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="cup_1", name="cup")
        obj2 = ObjectState(object_id="table_1", name="table")
        graph.add_object(obj1)
        graph.add_object(obj2)
        graph.add_relationship(Relationship("cup_1", "table_1", "on"))

        related = graph.get_related_objects("cup_1")
        assert len(related) == 1
        assert related[0][0] == "table_1"

    def test_find_object_by_name(self):
        """Test finding objects by name."""
        graph = SceneGraph()
        graph.add_object(ObjectState(object_id="cup_1", name="cup"))
        graph.add_object(ObjectState(object_id="cup_2", name="cup"))
        graph.add_object(ObjectState(object_id="table_1", name="table"))

        cups = graph.find_object_by_name("cup")
        assert len(cups) == 2

    def test_get_objects_by_role(self):
        """Test getting objects by role."""
        graph = SceneGraph()
        obj1 = ObjectState(object_id="agent_1", name="hand", role=ObjectRole.AGENT)
        obj2 = ObjectState(object_id="cup_1", name="cup", role=ObjectRole.TARGET)
        graph.add_object(obj1)
        graph.add_object(obj2)

        agents = graph.get_objects_by_role(ObjectRole.AGENT)
        assert len(agents) == 1
        assert agents[0].name == "hand"

    def test_to_dict(self):
        """Test serialization."""
        graph = SceneGraph()
        graph.add_object(ObjectState(object_id="cup_1", name="cup"))
        graph.add_relationship(Relationship("cup_1", "table_1", "on"))

        data = graph.to_dict()
        assert "objects" in data
        assert "relationships" in data
        assert "cup_1" in data["objects"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "objects": {
                "cup_1": {
                    "object_id": "cup_1",
                    "name": "cup",
                    "position": [0, 0, 0],
                    "affordances": [],
                    "role": "target",
                }
            },
            "relationships": [
                {
                    "subject_id": "cup_1",
                    "object_id": "table_1",
                    "relation_type": "on",
                    "contact_state": "no_contact",
                }
            ],
            "timestamp": 1.0,
        }
        graph = SceneGraph.from_dict(data)
        assert "cup_1" in graph.objects
        assert len(graph.relationships) == 1

    def test_copy(self):
        """Test deep copy."""
        graph = SceneGraph()
        graph.add_object(ObjectState(object_id="cup_1", name="cup"))
        graph.add_object(ObjectState(object_id="table_1", name="table"))
        graph.add_relationship(Relationship("cup_1", "table_1", "on"))

        copy = graph.copy()
        assert len(copy.objects) == 2
        assert len(copy.relationships) == 1
        # Ensure independent
        copy.objects["cup_1"].name = "modified"
        assert graph.objects["cup_1"].name == "cup"


class TestWorldState:
    """Tests for WorldState class."""

    def test_create_empty(self):
        """Test creating empty WorldState."""
        state = WorldState()
        assert state.timestamp == 0.0
        assert state.agent_id is None

    def test_create_with_data(self):
        """Test creating WorldState with data."""
        graph = SceneGraph()
        graph.add_object(ObjectState(object_id="cup_1", name="cup"))

        state = WorldState(
            scene_graph=graph,
            timestamp=1.5,
            environment={"scene_type": "kitchen"},
        )
        assert state.timestamp == 1.5
        assert state.environment["scene_type"] == "kitchen"

    def test_get_agent(self):
        """Test getting agent from state."""
        state = WorldState()
        agent = ObjectState(
            object_id="hand_1",
            name="hand",
            role=ObjectRole.AGENT,
        )
        state.scene_graph.add_object(agent)

        found_agent = state.get_agent()
        assert found_agent is not None
        assert found_agent.object_id == "hand_1"

    def test_get_agent_by_id(self):
        """Test getting agent by explicit ID."""
        state = WorldState(agent_id="hand_1")
        agent = ObjectState(
            object_id="hand_1",
            name="hand",
        )
        state.scene_graph.add_object(agent)

        found_agent = state.get_agent()
        assert found_agent.object_id == "hand_1"

    def test_get_target_objects(self):
        """Test getting target objects."""
        state = WorldState()
        state.scene_graph.add_object(ObjectState(
            object_id="cup_1",
            name="cup",
            role=ObjectRole.TARGET,
        ))
        state.scene_graph.add_object(ObjectState(
            object_id="table_1",
            name="table",
            role=ObjectRole.CONTEXT,
        ))

        targets = state.get_target_objects()
        assert len(targets) == 1
        assert targets[0].name == "cup"

    def test_describe(self):
        """Test scene description generation."""
        state = WorldState()
        state.scene_graph.add_object(ObjectState(
            object_id="cup_1",
            name="cup",
            state="empty",
            role=ObjectRole.TARGET,
        ))

        desc = state.describe()
        assert "cup" in desc
        assert "empty" in desc

    def test_to_dict(self):
        """Test serialization."""
        state = WorldState(
            timestamp=1.0,
            agent_id="hand_1",
        )
        state.scene_graph.add_object(ObjectState(object_id="cup_1", name="cup"))

        data = state.to_dict()
        assert data["timestamp"] == 1.0
        assert data["agent_id"] == "hand_1"
        assert "scene_graph" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "scene_graph": {
                "objects": {
                    "cup_1": {
                        "object_id": "cup_1",
                        "name": "cup",
                        "position": [0, 0, 0],
                        "affordances": [],
                        "role": "target",
                    }
                },
                "relationships": [],
            },
            "agent_id": "hand_1",
            "timestamp": 1.0,
            "environment": {"scene_type": "kitchen"},
            "metadata": {},
        }
        state = WorldState.from_dict(data)
        assert state.agent_id == "hand_1"
        assert state.timestamp == 1.0
        assert "cup_1" in state.scene_graph.objects

    def test_copy(self):
        """Test deep copy."""
        state = WorldState(timestamp=1.0)
        state.scene_graph.add_object(ObjectState(object_id="cup_1", name="cup"))

        copy = state.copy()
        assert copy.timestamp == 1.0
        assert "cup_1" in copy.scene_graph.objects
        # Ensure independent
        copy.timestamp = 2.0
        assert state.timestamp == 1.0

    def test_interpolate(self):
        """Test state interpolation."""
        state1 = WorldState(timestamp=0.0)
        obj1 = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[0.0, 0.0, 0.0],
        )
        state1.scene_graph.add_object(obj1)

        state2 = WorldState(timestamp=1.0)
        obj2 = ObjectState(
            object_id="cup_1",
            name="cup",
            position=[2.0, 0.0, 0.0],
        )
        state2.scene_graph.add_object(obj2)

        interpolated = state1.interpolate(state2, 0.5)
        cup = interpolated.scene_graph.objects["cup_1"]
        assert np.allclose(cup.position, [1.0, 0.0, 0.0])
        assert interpolated.timestamp == 0.5
