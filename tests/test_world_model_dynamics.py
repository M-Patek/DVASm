"""Tests for world_model dynamics module."""

import numpy as np
import pytest

from dvas.world_model.dynamics import (
    ContactDynamics,
    ContactEvent,
    ContactType,
    DynamicsType,
    ForceVector,
    MotionPrediction,
    PhysicalDynamics,
    PhysicalProperties,
    Trajectory,
)


class TestPhysicalProperties:
    """Tests for PhysicalProperties class."""

    def test_create_default(self):
        """Test creating with defaults."""
        props = PhysicalProperties()
        assert props.mass is None
        assert props.is_rigid is True
        assert props.material_type == "unknown"

    def test_create_with_values(self):
        """Test creating with specific values."""
        props = PhysicalProperties(
            mass=1.5,
            density=800.0,
            friction=0.3,
            restitution=0.5,
        )
        assert props.mass == 1.5
        assert props.density == 800.0
        assert props.friction == 0.3
        assert props.restitution == 0.5

    def test_to_dict(self):
        """Test serialization."""
        props = PhysicalProperties(mass=1.0, friction=0.3)
        data = props.to_dict()
        assert data["mass"] == 1.0
        assert data["friction"] == 0.3
        assert data["is_rigid"] is True

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "mass": 2.0,
            "density": 1000.0,
            "friction": 0.4,
            "is_rigid": False,
            "material_type": "rubber",
        }
        props = PhysicalProperties.from_dict(data)
        assert props.mass == 2.0
        assert props.is_rigid is False
        assert props.material_type == "rubber"


class TestForceVector:
    """Tests for ForceVector class."""

    def test_create_default(self):
        """Test creating with defaults."""
        force = ForceVector()
        assert np.allclose(force.force, [0, 0, 0])
        assert force.magnitude == 0.0

    def test_create_with_force(self):
        """Test creating with force vector."""
        force = ForceVector(force=[3.0, 4.0, 0.0])
        assert force.magnitude == 5.0

    def test_direction(self):
        """Test direction calculation."""
        force = ForceVector(force=[0, 0, 10])
        direction = force.direction
        assert np.allclose(direction, [0, 0, 1])

    def test_to_dict(self):
        """Test serialization."""
        force = ForceVector(
            force=[1.0, 2.0, 3.0],
            timestamp=1.5,
        )
        data = force.to_dict()
        assert data["force"] == [1.0, 2.0, 3.0]
        assert data["timestamp"] == 1.5

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "force": [10.0, 0.0, 0.0],
            "position": [1.0, 2.0, 3.0],
            "timestamp": 2.0,
        }
        force = ForceVector.from_dict(data)
        assert np.allclose(force.force, [10.0, 0.0, 0.0])
        assert force.position is not None


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_create(self):
        """Test creating trajectory."""
        traj = Trajectory(object_id="cup_1")
        assert traj.object_id == "cup_1"
        assert len(traj.positions) == 0

    def test_add_point(self):
        """Test adding trajectory points."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(
            position=[0.0, 0.0, 0.0],
            timestamp=0.0,
        )
        traj.add_point(
            position=[1.0, 0.0, 0.0],
            timestamp=1.0,
        )
        assert len(traj.positions) == 2
        assert len(traj.timestamps) == 2

    def test_get_point_at_time(self):
        """Test getting point at specific time."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)
        traj.add_point(position=[2.0, 0.0, 0.0], timestamp=2.0)

        point = traj.get_point_at_time(1.0)
        assert point is not None
        assert np.allclose(point[0], [1.0, 0.0, 0.0])

    def test_get_point_at_time_bounds(self):
        """Test getting point at boundary times."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)
        traj.add_point(position=[2.0, 0.0, 0.0], timestamp=2.0)

        # Before start
        point = traj.get_point_at_time(-1.0)
        assert np.allclose(point[0], [0.0, 0.0, 0.0])

        # After end
        point = traj.get_point_at_time(3.0)
        assert np.allclose(point[0], [2.0, 0.0, 0.0])

    def test_length(self):
        """Test trajectory length calculation."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)
        traj.add_point(position=[3.0, 4.0, 0.0], timestamp=1.0)

        assert traj.length() == 5.0

    def test_to_dict(self):
        """Test serialization."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)

        data = traj.to_dict()
        assert data["object_id"] == "cup_1"
        assert len(data["positions"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "object_id": "cup_1",
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "orientations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            "timestamps": [0.0, 1.0],
            "velocities": [],
            "is_predicted": True,
        }
        traj = Trajectory.from_dict(data)
        assert traj.object_id == "cup_1"
        assert len(traj.positions) == 2
        assert traj.is_predicted is True


class TestContactEvent:
    """Tests for ContactEvent class."""

    def test_create(self):
        """Test creating contact event."""
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            contact_type=ContactType.GRASPING,
            start_time=1.0,
        )
        assert event.subject_id == "hand_1"
        assert event.contact_type == ContactType.GRASPING

    def test_duration_property(self):
        """Test duration calculation."""
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            start_time=1.0,
            end_time=3.0,
        )
        assert event.duration == 2.0

    def test_duration_none(self):
        """Test duration is None for ongoing contact."""
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            start_time=1.0,
        )
        assert event.duration is None
        assert event.is_active is True

    def test_to_dict(self):
        """Test serialization."""
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            contact_type=ContactType.GRASPING,
            start_time=1.0,
            end_time=2.0,
            is_stable=True,
        )
        data = event.to_dict()
        assert data["subject_id"] == "hand_1"
        assert data["contact_type"] == "grasping"
        assert data["is_stable"] is True

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "subject_id": "hand_1",
            "object_id": "cup_1",
            "contact_type": "sliding",
            "start_time": 1.0,
            "end_time": 2.0,
            "is_stable": False,
        }
        event = ContactEvent.from_dict(data)
        assert event.contact_type == ContactType.SLIDING
        assert event.is_stable is False


class TestPhysicalDynamics:
    """Tests for PhysicalDynamics class."""

    def test_create(self):
        """Test creating physical dynamics."""
        dyn = PhysicalDynamics(object_id="cup_1")
        assert dyn.object_id == "cup_1"
        assert dyn.dynamics_type == DynamicsType.RIGID_BODY

    def test_add_contact_event(self):
        """Test adding contact events."""
        dyn = PhysicalDynamics(object_id="cup_1")
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            contact_type=ContactType.GRASPING,
        )
        dyn.add_contact_event(event)
        assert len(dyn.contact_events) == 1

    def test_add_force(self):
        """Test adding forces."""
        dyn = PhysicalDynamics(object_id="cup_1")
        force = ForceVector(force=[0, 0, 10])
        dyn.add_force(force)
        assert len(dyn.forces) == 1

    def test_get_peak_force(self):
        """Test peak force calculation."""
        dyn = PhysicalDynamics(object_id="cup_1")
        dyn.add_force(ForceVector(force=[0, 0, 5]))
        dyn.add_force(ForceVector(force=[0, 0, 15]))
        assert dyn.get_peak_force() == 15.0

    def test_to_dict(self):
        """Test serialization."""
        dyn = PhysicalDynamics(
            object_id="cup_1",
            dynamics_type=DynamicsType.RIGID_BODY,
            source="teacher",
            confidence=0.9,
        )
        data = dyn.to_dict()
        assert data["object_id"] == "cup_1"
        assert data["dynamics_type"] == "rigid_body"
        assert data["confidence"] == 0.9

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "object_id": "cup_1",
            "dynamics_type": "soft_body",
            "properties": {"mass": 1.0},
            "contact_events": [],
            "forces": [],
            "timestamp": 1.0,
            "source": "heuristic",
        }
        dyn = PhysicalDynamics.from_dict(data)
        assert dyn.dynamics_type == DynamicsType.SOFT_BODY
        assert dyn.source == "heuristic"


class TestContactDynamics:
    """Tests for ContactDynamics class."""

    def test_create(self):
        """Test creating contact dynamics."""
        contact = ContactDynamics(
            subject_id="hand_1",
            object_id="cup_1",
            primary_contact_type=ContactType.GRASPING,
        )
        assert contact.subject_id == "hand_1"
        assert contact.is_grasp is False

    def test_add_event(self):
        """Test adding contact events."""
        contact = ContactDynamics(subject_id="hand_1", object_id="cup_1")
        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            contact_type=ContactType.SLIDING,
        )
        contact.add_event(event)
        assert len(contact.events) == 1
        assert ContactType.SLIDING in contact.contact_types

    def test_has_sliding(self):
        """Test sliding detection."""
        contact = ContactDynamics(subject_id="hand_1", object_id="cup_1")
        assert not contact.has_sliding

        event = ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            contact_type=ContactType.SLIDING,
        )
        contact.add_event(event)
        assert contact.has_sliding

    def test_total_contact_duration(self):
        """Test total contact duration."""
        contact = ContactDynamics(subject_id="hand_1", object_id="cup_1")
        contact.add_event(ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            start_time=0.0,
            end_time=1.0,
        ))
        contact.add_event(ContactEvent(
            subject_id="hand_1",
            object_id="cup_1",
            start_time=2.0,
            end_time=3.5,
        ))
        assert contact.total_contact_duration == 2.5

    def test_to_dict(self):
        """Test serialization."""
        contact = ContactDynamics(
            subject_id="hand_1",
            object_id="cup_1",
            primary_contact_type=ContactType.GRASPING,
            is_grasp=True,
            grasp_quality=0.85,
        )
        data = contact.to_dict()
        assert data["is_grasp"] is True
        assert data["grasp_quality"] == 0.85

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "subject_id": "hand_1",
            "object_id": "cup_1",
            "primary_contact_type": "grasping",
            "contact_types": ["sliding", "rolling"],
            "events": [],
            "normal_force": 10.0,
            "is_grasp": True,
        }
        contact = ContactDynamics.from_dict(data)
        assert contact.is_grasp is True
        assert contact.normal_force == 10.0
        assert len(contact.contact_types) == 2


class TestMotionPrediction:
    """Tests for MotionPrediction class."""

    def test_create(self):
        """Test creating motion prediction."""
        pred = MotionPrediction(object_id="cup_1", prediction_horizon=1.0)
        assert pred.object_id == "cup_1"
        assert pred.prediction_horizon == 1.0

    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create predicted trajectory
        pred_traj = Trajectory(object_id="cup_1", is_predicted=True)
        pred_traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)
        pred_traj.add_point(position=[1.0, 0.0, 0.0], timestamp=1.0)

        # Create actual trajectory
        actual_traj = Trajectory(object_id="cup_1", is_predicted=False)
        actual_traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)
        actual_traj.add_point(position=[1.2, 0.0, 0.0], timestamp=1.0)

        pred = MotionPrediction(
            object_id="cup_1",
            predicted_trajectory=pred_traj,
            actual_trajectory=actual_traj,
        )

        metrics = pred.compute_metrics()
        assert "mean_position_error" in metrics
        # Float comparison with tolerance
        assert abs(metrics["mean_position_error"] - 0.1) < 0.01

    def test_is_accurate(self):
        """Test accuracy check."""
        pred = MotionPrediction(object_id="cup_1")
        pred.metrics = {"mean_position_error": 0.05}
        assert pred.is_accurate(threshold=0.1)

        pred.metrics = {"mean_position_error": 0.15}
        assert not pred.is_accurate(threshold=0.1)

    def test_to_dict(self):
        """Test serialization."""
        traj = Trajectory(object_id="cup_1")
        traj.add_point(position=[0.0, 0.0, 0.0], timestamp=0.0)

        pred = MotionPrediction(
            object_id="cup_1",
            predicted_trajectory=traj,
            model_name="test_model",
        )
        data = pred.to_dict()
        assert data["object_id"] == "cup_1"
        assert data["model_name"] == "test_model"
