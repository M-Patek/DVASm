"""Tests for Enhanced Robot Action schema.

Tests the robot_action.py module including pose representations,
contact events, policy hints, and causal chains.
"""

import pytest
from datetime import datetime

from dvas.data.robot_schemas.robot_action import (
    ActionCondition,
    ActionPrimitive,
    CausalLink,
    ContactEvent,
    EnhancedRobotAction,
    FailureAnnotation,
    FailureMode,
    GripperData,
    GripperState,
    HandPose,
    Pose6D,
    RobotPolicyHint,
)


class TestPose6D:
    """Test 6D pose representation."""

    def test_pose_initialization(self):
        """Test basic pose initialization."""
        pose = Pose6D(x=1.0, y=2.0, z=3.0, rx=0.1, ry=0.2, rz=0.3)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0

    def test_pose_to_list_euler(self):
        """Test conversion to list (Euler angles)."""
        pose = Pose6D(x=1.0, y=2.0, z=3.0, rx=0.1, ry=0.2, rz=0.3)
        result = pose.to_list()
        assert len(result) == 6
        assert result == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    def test_pose_to_list_quaternion(self):
        """Test conversion to list (quaternion)."""
        pose = Pose6D(x=1.0, y=2.0, z=3.0, rx=0.0, ry=0.0, rz=0.0, rw=1.0)
        result = pose.to_list()
        assert len(result) == 7
        assert result[-1] == 1.0

    def test_pose_to_dict(self):
        """Test conversion to dictionary."""
        pose = Pose6D(x=1.0, y=2.0, z=3.0)
        result = pose.to_dict()
        assert result["x"] == 1.0
        assert result["frame_id"] == "base_link"

    def test_pose_from_list_6dof(self):
        """Test creation from 6-DOF list."""
        pose = Pose6D.from_list([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        assert pose.x == 1.0
        assert pose.rw is None

    def test_pose_from_list_7dof(self):
        """Test creation from 7-DOF (quaternion) list."""
        pose = Pose6D.from_list([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        assert pose.rw == 1.0

    def test_pose_from_dict(self):
        """Test creation from dictionary."""
        data = {"x": 1.5, "y": 2.5, "z": 3.5, "frame_id": "gripper_frame"}
        pose = Pose6D.from_dict(data)
        assert pose.x == 1.5
        assert pose.frame_id == "gripper_frame"

    def test_invalid_list_length(self):
        """Test error on invalid list length."""
        with pytest.raises(ValueError):
            Pose6D.from_list([1.0, 2.0, 3.0])  # Too short


class TestHandPose:
    """Test hand pose tracking."""

    def test_hand_pose_initialization(self):
        """Test hand pose initialization."""
        left = Pose6D(x=0.1, y=0.2, z=0.3)
        right = Pose6D(x=0.4, y=0.5, z=0.6)
        hand_pose = HandPose(left_hand=left, right_hand=right)

        assert hand_pose.left_hand is not None
        assert hand_pose.right_hand is not None
        assert hand_pose.left_hand.x == 0.1
        assert hand_pose.right_hand.x == 0.4

    def test_hand_pose_to_dict(self):
        """Test hand pose serialization."""
        left = Pose6D(x=0.1, y=0.2, z=0.3)
        hand_pose = HandPose(left_hand=left, timestamp=5.0)

        result = hand_pose.to_dict()
        assert result["timestamp"] == 5.0
        assert result["left_hand"]["x"] == 0.1
        assert result["right_hand"] is None

    def test_hand_pose_from_dict(self):
        """Test hand pose deserialization."""
        data = {
            "left_hand": {"x": 0.1, "y": 0.2, "z": 0.3},
            "right_hand": None,
            "left_confidence": 0.95,
            "right_confidence": 0.0,
            "timestamp": 3.5,
        }
        hand_pose = HandPose.from_dict(data)
        assert hand_pose.timestamp == 3.5
        assert hand_pose.left_confidence == 0.95


class TestGripperData:
    """Test gripper state representation."""

    def test_gripper_initialization(self):
        """Test gripper data initialization."""
        gripper = GripperData(
            state=GripperState.CLOSED,
            aperture=0.0,
            force=15.0,
            position=0.02,
        )
        assert gripper.state == GripperState.CLOSED
        assert gripper.force == 15.0

    def test_gripper_states(self):
        """Test all gripper states."""
        for state in GripperState:
            gripper = GripperData(state=state)
            assert gripper.state == state

    def test_gripper_serialization(self):
        """Test gripper data round-trip."""
        original = GripperData(
            state=GripperState.PARTIAL,
            aperture=0.5,
            force=8.0,
            is_slipping=True,
        )
        data = original.to_dict()
        restored = GripperData.from_dict(data)

        assert restored.state == original.state
        assert restored.aperture == original.aperture
        assert restored.is_slipping == original.is_slipping


class TestContactEvent:
    """Test contact event representation."""

    def test_contact_event_initialization(self):
        """Test contact event initialization."""
        event = ContactEvent(
            timestamp=2.5,
            contact_point=(0.1, 0.2, 0.3),
            force_magnitude=20.0,
            contact_type="grasp",
            object_name="cup",
        )
        assert event.timestamp == 2.5
        assert event.force_magnitude == 20.0
        assert event.object_name == "cup"

    def test_contact_event_serialization(self):
        """Test contact event round-trip."""
        event = ContactEvent(
            timestamp=3.0,
            contact_point=(0.1, 0.2, 0.3),
            contact_normal=(0.0, 0.0, 1.0),
            force_magnitude=25.0,
            force_vector=(0.0, 0.0, -25.0),
            contact_type="push",
            is_sticky=True,
            duration=1.5,
        )
        data = event.to_dict()
        restored = ContactEvent.from_dict(data)

        assert restored.timestamp == event.timestamp
        assert restored.contact_point == event.contact_point
        assert restored.is_sticky == event.is_sticky


class TestRobotPolicyHint:
    """Test robot policy hint representation."""

    def test_policy_hint_initialization(self):
        """Test policy hint initialization."""
        hint = RobotPolicyHint(
            policy_type="diffusion",
            speed="slow",
            precision="precise",
            compliant_mode=True,
        )
        assert hint.policy_type == "diffusion"
        assert hint.compliant_mode is True

    def test_policy_hint_defaults(self):
        """Test policy hint default values."""
        hint = RobotPolicyHint()
        assert hint.speed == "normal"
        assert hint.precision == "standard"
        assert hint.trajectory_type == "straight"

    def test_policy_hint_serialization(self):
        """Test policy hint round-trip."""
        original = RobotPolicyHint(
            policy_type="bc",
            custom_params={"temperature": 0.5, "top_p": 0.9},
        )
        data = original.to_dict()
        restored = RobotPolicyHint.from_dict(data)

        assert restored.policy_type == original.policy_type
        assert restored.custom_params == original.custom_params


class TestActionCondition:
    """Test action condition (pre/post) representation."""

    def test_precondition_creation(self):
        """Test precondition creation."""
        pre = ActionCondition(
            condition_type="precondition",
            description="Object must be within reach",
            object_states={"cup": "visible"},
            is_mandatory=True,
        )
        assert pre.condition_type == "precondition"
        assert pre.is_mandatory is True

    def test_postcondition_creation(self):
        """Test postcondition creation."""
        post = ActionCondition(
            condition_type="postcondition",
            description="Object is grasped",
            object_states={"cup": "held"},
            confidence=0.9,
        )
        assert post.condition_type == "postcondition"
        assert post.confidence == 0.9

    def test_condition_with_pose(self):
        """Test condition with pose requirements."""
        pose = Pose6D(x=0.5, y=0.3, z=0.2)
        condition = ActionCondition(
            condition_type="precondition",
            description="Hand at hover position",
            robot_pose_requirements=pose,
        )
        data = condition.to_dict()
        assert data["robot_pose_requirements"] is not None


class TestFailureAnnotation:
    """Test failure mode annotation."""

    def test_failure_initialization(self):
        """Test failure annotation initialization."""
        failure = FailureAnnotation(
            failure_mode=FailureMode.SLIP,
            description="Object slipped from gripper",
            is_recoverable=True,
            likelihood=0.15,
        )
        assert failure.failure_mode == FailureMode.SLIP
        assert failure.occurred is False  # Default

    def test_failure_modes(self):
        """Test all failure modes."""
        for mode in FailureMode:
            failure = FailureAnnotation(
                failure_mode=mode,
                description=f"Test {mode.value}",
            )
            assert failure.failure_mode == mode

    def test_failure_serialization(self):
        """Test failure annotation round-trip."""
        original = FailureAnnotation(
            failure_mode=FailureMode.COLLISION,
            description="Collision with obstacle",
            recovery_action="retreat_and_replan",
            likelihood=0.05,
            occurred=True,
        )
        data = original.to_dict()
        restored = FailureAnnotation.from_dict(data)

        assert restored.failure_mode == original.failure_mode
        assert restored.recovery_action == original.recovery_action
        assert restored.occurred == original.occurred


class TestCausalLink:
    """Test causal link representation."""

    def test_causal_link_initialization(self):
        """Test causal link initialization."""
        link = CausalLink(
            source_action_id="action_001",
            target_action_id="action_002",
            relation_type="enables",
            strength=0.9,
        )
        assert link.source_action_id == "action_001"
        assert link.relation_type == "enables"

    def test_causal_relations(self):
        """Test different causal relation types."""
        relations = ["enables", "requires", "conflicts_with", "follows"]
        for rel in relations:
            link = CausalLink(
                source_action_id="a1",
                target_action_id="a2",
                relation_type=rel,
            )
            assert link.relation_type == rel

    def test_causal_link_serialization(self):
        """Test causal link round-trip."""
        original = CausalLink(
            source_action_id="pick_001",
            target_action_id="place_001",
            relation_type="enables",
            strength=0.95,
            description="Must pick before placing",
            temporal_gap=2.0,
        )
        data = original.to_dict()
        restored = CausalLink.from_dict(data)

        assert restored.source_action_id == original.source_action_id
        assert restored.temporal_gap == original.temporal_gap


class TestEnhancedRobotAction:
    """Test enhanced robot action representation."""

    def test_action_initialization(self):
        """Test basic action initialization."""
        action = EnhancedRobotAction(
            action_id="action_001",
            verb="pick",
            noun="cup",
            hand="right",
            start_time=1.0,
            end_time=3.0,
        )
        assert action.action_id == "action_001"
        assert action.verb == "pick"

    def test_action_with_primitive(self):
        """Test action with primitive type."""
        action = EnhancedRobotAction(
            action_id="action_001",
            verb="grasp",
            noun="bottle",
            action_primitive=ActionPrimitive.GRASP,
        )
        assert action.action_primitive == ActionPrimitive.GRASP

    def test_action_with_poses(self):
        """Test action with hand and target poses."""
        hand_pose = HandPose(left_hand=Pose6D(x=0.1, y=0.2, z=0.3))
        target_pose = Pose6D(x=0.5, y=0.5, z=0.1)

        action = EnhancedRobotAction(
            action_id="action_002",
            verb="reach",
            noun="target",
            hand_pose=hand_pose,
            target_pose=target_pose,
        )
        assert action.hand_pose is not None
        assert action.target_pose is not None

    def test_action_with_gripper(self):
        """Test action with gripper states."""
        initial = GripperData(state=GripperState.OPEN)
        final = GripperData(state=GripperState.CLOSED)

        action = EnhancedRobotAction(
            action_id="action_003",
            verb="grasp",
            noun="object",
            initial_gripper=initial,
            final_gripper=final,
            gripper_state=final,
        )
        assert action.initial_gripper.state == GripperState.OPEN
        assert action.final_gripper.state == GripperState.CLOSED

    def test_action_with_contacts(self):
        """Test action with contact events."""
        contact = ContactEvent(
            timestamp=2.5,
            contact_point=(0.0, 0.0, 0.0),
            force_magnitude=10.0,
        )
        action = EnhancedRobotAction(
            action_id="action_004",
            verb="push",
            noun="box",
        )
        action.add_contact_event(contact)

        assert len(action.contact_events) == 1
        assert action.has_contact() is True
        assert action.get_max_force() == 10.0

    def test_action_with_policy_hint(self):
        """Test action with policy hint."""
        hint = RobotPolicyHint(
            policy_type="diffusion",
            speed="slow",
            compliant_mode=True,
        )
        action = EnhancedRobotAction(
            action_id="action_005",
            verb="insert",
            noun="peg",
            robot_policy_hint=hint,
        )
        assert action.robot_policy_hint is not None
        assert action.robot_policy_hint.policy_type == "diffusion"

    def test_action_with_conditions(self):
        """Test action with pre/post conditions."""
        pre = ActionCondition(
            condition_type="precondition",
            description="Object visible",
        )
        post = ActionCondition(
            condition_type="postcondition",
            description="Object grasped",
        )
        action = EnhancedRobotAction(
            action_id="action_006",
            verb="grasp",
            noun="object",
            preconditions=[pre],
            postconditions=[post],
        )
        assert len(action.preconditions) == 1
        assert len(action.postconditions) == 1

    def test_action_with_failures(self):
        """Test action with failure modes."""
        failure = FailureAnnotation(
            failure_mode=FailureMode.MISGRASP,
            description="Grasped wrong part of object",
        )
        action = EnhancedRobotAction(
            action_id="action_007",
            verb="grasp",
            noun="tool",
            failure_modes=[failure],
        )
        assert len(action.failure_modes) == 1

    def test_action_with_causal_chain(self):
        """Test action with causal links."""
        link = CausalLink(
            source_action_id="action_007",
            target_action_id="action_008",
            relation_type="enables",
        )
        action = EnhancedRobotAction(
            action_id="action_007",
            verb="pick",
            noun="ingredient",
        )
        action.add_causal_link(link)

        assert len(action.causal_chain) == 1

    def test_action_duration(self):
        """Test action duration calculation."""
        action = EnhancedRobotAction(
            action_id="action_008",
            verb="wait",
            noun="none",
            start_time=5.0,
            end_time=8.5,
        )
        assert action.get_duration() == 3.5

    def test_action_validation_success(self):
        """Test validation of valid action."""
        action = EnhancedRobotAction(
            action_id="action_009",
            verb="move",
            noun="arm",
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
        )
        is_valid, errors = action.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_action_validation_missing_id(self):
        """Test validation catches missing action_id."""
        action = EnhancedRobotAction(
            action_id="",
            verb="move",
            noun="arm",
        )
        is_valid, errors = action.validate()
        assert is_valid is False
        assert any("action_id" in e for e in errors)

    def test_action_validation_invalid_time(self):
        """Test validation catches invalid timestamps."""
        action = EnhancedRobotAction(
            action_id="action_010",
            verb="move",
            noun="arm",
            start_time=5.0,
            end_time=3.0,  # end before start
        )
        is_valid, errors = action.validate()
        assert is_valid is False
        assert any("end_time" in e for e in errors)

    def test_action_validation_invalid_confidence(self):
        """Test validation catches invalid confidence."""
        action = EnhancedRobotAction(
            action_id="action_011",
            verb="move",
            noun="arm",
            confidence=1.5,  # > 1.0
        )
        is_valid, errors = action.validate()
        assert is_valid is False
        assert any("confidence" in e for e in errors)

    def test_action_validation_condition_type(self):
        """Test validation catches wrong condition types."""
        pre = ActionCondition(
            condition_type="postcondition",  # Wrong type for precondition list
            description="Test",
        )
        action = EnhancedRobotAction(
            action_id="action_012",
            verb="test",
            noun="object",
            preconditions=[pre],
        )
        is_valid, errors = action.validate()
        assert is_valid is False
        assert any("precondition" in e.lower() for e in errors)

    def test_action_serialization(self):
        """Test full action serialization round-trip."""
        original = EnhancedRobotAction(
            action_id="action_full",
            verb="pick",
            noun="cup",
            hand="right",
            start_time=1.0,
            end_time=3.0,
            hand_pose=HandPose(
                right_hand=Pose6D(x=0.1, y=0.2, z=0.3),
                timestamp=2.0,
            ),
            gripper_state=GripperData(state=GripperState.CLOSED),
            action_primitive=ActionPrimitive.PICK,
            robot_policy_hint=RobotPolicyHint(policy_type="diffusion"),
            confidence=0.92,
            source="demo",
        )
        data = original.to_dict()
        restored = EnhancedRobotAction.from_dict(data)

        assert restored.action_id == original.action_id
        assert restored.verb == original.verb
        assert restored.confidence == original.confidence
        assert restored.hand_pose.timestamp == original.hand_pose.timestamp

    def test_action_primitives(self):
        """Test all action primitives."""
        primitives = [
            ActionPrimitive.PICK,
            ActionPrimitive.PLACE,
            ActionPrimitive.PUSH,
            ActionPrimitive.PULL,
            ActionPrimitive.GRASP,
            ActionPrimitive.RELEASE,
        ]
        for i, prim in enumerate(primitives):
            action = EnhancedRobotAction(
                action_id=f"action_{i}",
                verb=prim.value,
                noun="object",
                action_primitive=prim,
            )
            assert action.action_primitive == prim
