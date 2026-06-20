"""Enhanced robot action schema for VLA/Robot learning.

Extends the base Action schema with robot-specific annotations including
hand poses, gripper states, contact events, action primitives, and causal
relationships for robotic manipulation learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActionPrimitive(str, Enum):
    """Low-level action primitives for robotic manipulation."""

    PICK = "pick"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    GRASP = "grasp"
    RELEASE = "release"
    LIFT = "lift"
    ROTATE = "rotate"
    SLIDE = "slide"
    INSERT = "insert"
    EXTRACT = "extract"
    WIPE = "wipe"
    POUR = "pour"
    CUT = "cut"
    STIR = "stir"
    TWIST = "twist"
    PRESS = "press"
    TAP = "tap"
    MOVE = "move"
    REACH = "reach"
    UNKNOWN = "unknown"


class GripperState(str, Enum):
    """Gripper state enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class FailureMode(str, Enum):
    """Known failure modes for robot actions."""

    SLIP = "slip"
    COLLISION = "collision"
    MISGRASP = "misgrasp"
    TIMEOUT = "timeout"
    OUT_OF_REACH = "out_of_reach"
    OCCLUSION = "occlusion"
    OBJECT_NOT_FOUND = "object_not_found"
    INSUFFICIENT_FORCE = "insufficient_force"
    EXCESSIVE_FORCE = "excessive_force"
    KINEMATIC_ERROR = "kinematic_error"
    PLANNING_FAILURE = "planning_failure"
    GRIPPER_MALFUNCTION = "gripper_malfunction"


@dataclass
class Pose6D:
    """6D pose representation (position + orientation).

    Position is in meters (x, y, z).
    Orientation is in quaternion format (qx, qy, qz, qw) or Euler angles (rx, ry, rz)
    depending on the convention flag.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0  # rotation around x (roll) or qx
    ry: float = 0.0  # rotation around y (pitch) or qy
    rz: float = 0.0  # rotation around z (yaw) or qz
    rw: Optional[float] = None  # qw for quaternion (None implies Euler angles)
    frame_id: str = "base_link"  # Reference frame

    def to_list(self) -> List[float]:
        """Convert to list representation."""
        if self.rw is not None:
            return [self.x, self.y, self.z, self.rx, self.ry, self.rz, self.rw]
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rx": self.rx,
            "ry": self.ry,
            "rz": self.rz,
            "frame_id": self.frame_id,
        }
        if self.rw is not None:
            result["rw"] = self.rw
        return result

    @classmethod
    def from_list(cls, values: List[float], frame_id: str = "base_link") -> "Pose6D":
        """Create from list representation."""
        if len(values) == 6:
            return cls(
                x=values[0],
                y=values[1],
                z=values[2],
                rx=values[3],
                ry=values[4],
                rz=values[5],
                frame_id=frame_id,
            )
        elif len(values) == 7:
            return cls(
                x=values[0],
                y=values[1],
                z=values[2],
                rx=values[3],
                ry=values[4],
                rz=values[5],
                rw=values[6],
                frame_id=frame_id,
            )
        raise ValueError(f"Expected 6 or 7 values, got {len(values)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pose6D":
        """Create from dictionary."""
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            rx=data.get("rx", 0.0),
            ry=data.get("ry", 0.0),
            rz=data.get("rz", 0.0),
            rw=data.get("rw"),
            frame_id=data.get("frame_id", "base_link"),
        )


@dataclass
class HandPose:
    """Hand position and orientation tracking.

    Tracks both left and right hands with full 6D pose information.
    """

    left_hand: Optional[Pose6D] = None
    right_hand: Optional[Pose6D] = None
    left_confidence: float = 1.0
    right_confidence: float = 1.0
    timestamp: Optional[float] = None  # seconds from segment start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "left_hand": self.left_hand.to_dict() if self.left_hand else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand else None,
            "left_confidence": self.left_confidence,
            "right_confidence": self.right_confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandPose":
        """Create from dictionary."""
        return cls(
            left_hand=Pose6D.from_dict(data["left_hand"]) if data.get("left_hand") else None,
            right_hand=Pose6D.from_dict(data["right_hand"]) if data.get("right_hand") else None,
            left_confidence=data.get("left_confidence", 1.0),
            right_confidence=data.get("right_confidence", 1.0),
            timestamp=data.get("timestamp"),
        )


@dataclass
class GripperData:
    """Gripper state with force and position information."""

    state: GripperState = GripperState.UNKNOWN
    aperture: Optional[float] = None  # 0.0 (closed) to 1.0 (fully open)
    force: Optional[float] = None  # Newtons
    force_limit: Optional[float] = None  # Maximum force limit
    position: Optional[float] = None  # Linear position in meters
    velocity: Optional[float] = None  # Velocity in m/s
    is_slipping: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "aperture": self.aperture,
            "force": self.force,
            "force_limit": self.force_limit,
            "position": self.position,
            "velocity": self.velocity,
            "is_slipping": self.is_slipping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GripperData":
        """Create from dictionary."""
        return cls(
            state=GripperState(data.get("state", "unknown")),
            aperture=data.get("aperture"),
            force=data.get("force"),
            force_limit=data.get("force_limit"),
            position=data.get("position"),
            velocity=data.get("velocity"),
            is_slipping=data.get("is_slipping", False),
        )


@dataclass
class ContactEvent:
    """Contact event tracking for forceful interactions.

    Records when and where contact occurs during manipulation.
    """

    timestamp: float  # seconds from segment start
    contact_point: Tuple[float, float, float]  # 3D contact location
    contact_normal: Optional[Tuple[float, float, float]] = None  # Surface normal
    force_magnitude: float = 0.0  # Newtons
    force_vector: Optional[Tuple[float, float, float]] = None
    contact_type: str = "unknown"  # "grasp", "push", "slide", "impact"
    object_name: Optional[str] = None  # Object being contacted
    is_sticky: bool = False  # Whether contact persists
    duration: float = 0.0  # Contact duration in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "contact_point": list(self.contact_point),
            "contact_normal": list(self.contact_normal) if self.contact_normal else None,
            "force_magnitude": self.force_magnitude,
            "force_vector": list(self.force_vector) if self.force_vector else None,
            "contact_type": self.contact_type,
            "object_name": self.object_name,
            "is_sticky": self.is_sticky,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContactEvent":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            contact_point=tuple(data["contact_point"]),
            contact_normal=tuple(data["contact_normal"]) if data.get("contact_normal") else None,
            force_magnitude=data.get("force_magnitude", 0.0),
            force_vector=tuple(data["force_vector"]) if data.get("force_vector") else None,
            contact_type=data.get("contact_type", "unknown"),
            object_name=data.get("object_name"),
            is_sticky=data.get("is_sticky", False),
            duration=data.get("duration", 0.0),
        )


@dataclass
class RobotPolicyHint:
    """Suggested policy behavior for robot learning.

    Provides guidance for policy networks on how to execute the action.
    """

    policy_type: str = "unknown"  # "diffusion", "bc", "rl", "hybrid"
    speed: str = "normal"  # "slow", "normal", "fast"
    precision: str = "standard"  # "loose", "standard", "precise"
    compliant_mode: bool = False  # Use force compliance
    use_visual_feedback: bool = True
    use_force_feedback: bool = False
    trajectory_type: str = "straight"  # "straight", "arc", "spline"
    target_tolerance: float = 0.01  # meters
    orientation_tolerance: float = 0.1  # radians
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_type": self.policy_type,
            "speed": self.speed,
            "precision": self.precision,
            "compliant_mode": self.compliant_mode,
            "use_visual_feedback": self.use_visual_feedback,
            "use_force_feedback": self.use_force_feedback,
            "trajectory_type": self.trajectory_type,
            "target_tolerance": self.target_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotPolicyHint":
        """Create from dictionary."""
        return cls(
            policy_type=data.get("policy_type", "unknown"),
            speed=data.get("speed", "normal"),
            precision=data.get("precision", "standard"),
            compliant_mode=data.get("compliant_mode", False),
            use_visual_feedback=data.get("use_visual_feedback", True),
            use_force_feedback=data.get("use_force_feedback", False),
            trajectory_type=data.get("trajectory_type", "straight"),
            target_tolerance=data.get("target_tolerance", 0.01),
            orientation_tolerance=data.get("orientation_tolerance", 0.1),
            custom_params=data.get("custom_params", {}),
        )


@dataclass
class ActionCondition:
    """Preconditions and postconditions for actions.

    Defines required state before action and expected state after.
    """

    condition_type: str  # "precondition" or "postcondition"
    description: str
    object_states: Dict[str, str] = field(default_factory=dict)  # object -> state
    robot_pose_requirements: Optional[Pose6D] = None
    gripper_requirements: Optional[GripperData] = None
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)
    is_mandatory: bool = True
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_type": self.condition_type,
            "description": self.description,
            "object_states": self.object_states,
            "robot_pose_requirements": (
                self.robot_pose_requirements.to_dict() if self.robot_pose_requirements else None
            ),
            "gripper_requirements": (
                self.gripper_requirements.to_dict() if self.gripper_requirements else None
            ),
            "spatial_relations": self.spatial_relations,
            "is_mandatory": self.is_mandatory,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionCondition":
        """Create from dictionary."""
        return cls(
            condition_type=data["condition_type"],
            description=data["description"],
            object_states=data.get("object_states", {}),
            robot_pose_requirements=(
                Pose6D.from_dict(data["robot_pose_requirements"])
                if data.get("robot_pose_requirements")
                else None
            ),
            gripper_requirements=(
                GripperData.from_dict(data["gripper_requirements"])
                if data.get("gripper_requirements")
                else None
            ),
            spatial_relations=data.get("spatial_relations", []),
            is_mandatory=data.get("is_mandatory", True),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class FailureAnnotation:
    """Failure mode annotation for robust learning."""

    failure_mode: FailureMode
    description: str
    is_recoverable: bool = True
    recovery_action: Optional[str] = None
    likelihood: float = 0.1  # 0.0 to 1.0
    mitigation: Optional[str] = None  # How to avoid this failure
    occurred: bool = False  # Whether this failure actually occurred

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_mode": self.failure_mode.value,
            "description": self.description,
            "is_recoverable": self.is_recoverable,
            "recovery_action": self.recovery_action,
            "likelihood": self.likelihood,
            "mitigation": self.mitigation,
            "occurred": self.occurred,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureAnnotation":
        """Create from dictionary."""
        return cls(
            failure_mode=FailureMode(data.get("failure_mode", "collision")),
            description=data["description"],
            is_recoverable=data.get("is_recoverable", True),
            recovery_action=data.get("recovery_action"),
            likelihood=data.get("likelihood", 0.1),
            mitigation=data.get("mitigation"),
            occurred=data.get("occurred", False),
        )


@dataclass
class CausalLink:
    """Causal dependency between actions.

    Represents how one action enables or affects another.
    """

    source_action_id: str
    target_action_id: str
    relation_type: str  # "enables", "requires", "conflicts_with", "follows"
    strength: float = 1.0  # 0.0 to 1.0
    description: Optional[str] = None
    temporal_gap: float = 0.0  # Expected time gap in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "description": self.description,
            "temporal_gap": self.temporal_gap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalLink":
        """Create from dictionary."""
        return cls(
            source_action_id=data["source_action_id"],
            target_action_id=data["target_action_id"],
            relation_type=data["relation_type"],
            strength=data.get("strength", 1.0),
            description=data.get("description"),
            temporal_gap=data.get("temporal_gap", 0.0),
        )


@dataclass
class EnhancedRobotAction:
    """Enhanced action with robot-specific annotations.

    This extends the base Action concept with rich information for
    robotic manipulation learning, including hand poses, contact events,
    causal relationships, and policy hints.
    """

    # Core identification
    action_id: str
    verb: str
    noun: str
    hand: str = "unknown"  # "left", "right", "both"

    # Temporal information
    start_time: float = 0.0
    end_time: float = 0.0

    # Pose and motion
    hand_pose: Optional[HandPose] = None
    target_pose: Optional[Pose6D] = None  # Target position for the action
    waypoints: List[Pose6D] = field(default_factory=list)  # Intermediate poses

    # Gripper information
    gripper_state: Optional[GripperData] = None
    initial_gripper: Optional[GripperData] = None
    final_gripper: Optional[GripperData] = None

    # Contact and force
    contact_events: List[ContactEvent] = field(default_factory=list)

    # Action semantics
    action_primitive: ActionPrimitive = ActionPrimitive.UNKNOWN
    description: Optional[str] = None

    # Policy guidance
    robot_policy_hint: Optional[RobotPolicyHint] = None

    # Preconditions and postconditions
    preconditions: List[ActionCondition] = field(default_factory=list)
    postconditions: List[ActionCondition] = field(default_factory=list)

    # Failure handling
    failure_modes: List[FailureAnnotation] = field(default_factory=list)

    # Causal relationships
    causal_chain: List[CausalLink] = field(default_factory=list)

    # Metadata
    confidence: float = 1.0
    source: str = "annotation"  # "annotation", "demo", "simulation"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "verb": self.verb,
            "noun": self.noun,
            "hand": self.hand,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "hand_pose": self.hand_pose.to_dict() if self.hand_pose else None,
            "target_pose": self.target_pose.to_dict() if self.target_pose else None,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "gripper_state": self.gripper_state.to_dict() if self.gripper_state else None,
            "initial_gripper": self.initial_gripper.to_dict() if self.initial_gripper else None,
            "final_gripper": self.final_gripper.to_dict() if self.final_gripper else None,
            "contact_events": [e.to_dict() for e in self.contact_events],
            "action_primitive": self.action_primitive.value,
            "description": self.description,
            "robot_policy_hint": (
                self.robot_policy_hint.to_dict() if self.robot_policy_hint else None
            ),
            "preconditions": [p.to_dict() for p in self.preconditions],
            "postconditions": [p.to_dict() for p in self.postconditions],
            "failure_modes": [f.to_dict() for f in self.failure_modes],
            "causal_chain": [c.to_dict() for c in self.causal_chain],
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedRobotAction":
        """Create from dictionary."""
        return cls(
            action_id=data["action_id"],
            verb=data["verb"],
            noun=data["noun"],
            hand=data.get("hand", "unknown"),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            hand_pose=HandPose.from_dict(data["hand_pose"]) if data.get("hand_pose") else None,
            target_pose=Pose6D.from_dict(data["target_pose"]) if data.get("target_pose") else None,
            waypoints=[Pose6D.from_dict(w) for w in data.get("waypoints", [])],
            gripper_state=(
                GripperData.from_dict(data["gripper_state"]) if data.get("gripper_state") else None
            ),
            initial_gripper=(
                GripperData.from_dict(data["initial_gripper"])
                if data.get("initial_gripper")
                else None
            ),
            final_gripper=(
                GripperData.from_dict(data["final_gripper"]) if data.get("final_gripper") else None
            ),
            contact_events=[ContactEvent.from_dict(e) for e in data.get("contact_events", [])],
            action_primitive=ActionPrimitive(data.get("action_primitive", "unknown")),
            description=data.get("description"),
            robot_policy_hint=(
                RobotPolicyHint.from_dict(data["robot_policy_hint"])
                if data.get("robot_policy_hint")
                else None
            ),
            preconditions=[ActionCondition.from_dict(p) for p in data.get("preconditions", [])],
            postconditions=[ActionCondition.from_dict(p) for p in data.get("postconditions", [])],
            failure_modes=[FailureAnnotation.from_dict(f) for f in data.get("failure_modes", [])],
            causal_chain=[CausalLink.from_dict(c) for c in data.get("causal_chain", [])],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "annotation"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this action annotation.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Basic requirements
        if not self.action_id:
            errors.append("action_id is required")
        if not self.verb:
            errors.append("verb is required")
        if not self.noun:
            errors.append("noun is required")

        # Time validation
        if self.end_time < self.start_time:
            errors.append("end_time must be >= start_time")

        # Confidence validation
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0.0 and 1.0")

        # Validate preconditions
        for i, pre in enumerate(self.preconditions):
            if pre.condition_type != "precondition":
                errors.append(f"precondition[{i}] has incorrect condition_type")

        # Validate postconditions
        for i, post in enumerate(self.postconditions):
            if post.condition_type != "postcondition":
                errors.append(f"postcondition[{i}] has incorrect condition_type")

        # Validate causal chain references
        for link in self.causal_chain:
            if link.source_action_id != self.action_id and link.target_action_id != self.action_id:
                errors.append(
                    f"causal link {link.source_action_id} -> {link.target_action_id} "
                    f"does not reference this action ({self.action_id})"
                )

        return len(errors) == 0, errors

    def get_duration(self) -> float:
        """Get action duration in seconds."""
        return self.end_time - self.start_time

    def add_contact_event(self, event: ContactEvent) -> None:
        """Add a contact event to this action."""
        self.contact_events.append(event)

    def add_causal_link(self, link: CausalLink) -> None:
        """Add a causal link to this action."""
        self.causal_chain.append(link)

    def has_contact(self) -> bool:
        """Check if this action involves contact."""
        return len(self.contact_events) > 0

    def get_max_force(self) -> float:
        """Get maximum contact force during action."""
        if not self.contact_events:
            return 0.0
        return max(e.force_magnitude for e in self.contact_events)
