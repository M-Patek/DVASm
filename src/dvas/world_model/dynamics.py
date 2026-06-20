"""Physical dynamics annotations for World Model training.

Provides classes for representing:
- Physical dynamics (mass, friction, elasticity)
- Contact dynamics (forces, trajectories)
- Motion predictions (predicted vs actual)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DynamicsType(str, Enum):
    """Type of physical dynamics."""

    RIGID_BODY = "rigid_body"
    SOFT_BODY = "soft_body"
    FLUID = "fluid"
    GRANULAR = "granular"
    ARTICULATED = "articulated"
    CLOTH = "cloth"
    UNKNOWN = "unknown"


class ContactType(str, Enum):
    """Type of contact interaction."""

    COLLISION = "collision"
    SLIDING = "sliding"
    ROLLING = "rolling"
    SANDWICHING = "sandwiching"  # Two surfaces pinching an object
    GRASPING = "grasping"
    PUSHING = "pushing"
    PULLING = "pulling"
    LIFTING = "lifting"
    DROPPING = "dropping"
    CONTAINMENT = "containment"  # Object inside container
    POURING = "pouring"
    STABLE = "stable"  # Static contact
    UNKNOWN = "unknown"


@dataclass
class PhysicalProperties:
    """Physical properties of an object or material.

    Attributes:
        mass: Mass in kg (None if unknown)
        density: Density in kg/m³
        friction: Coefficient of friction (static)
        friction_dynamic: Coefficient of dynamic friction
        restitution: Elasticity/restitution (0-1, 1 = perfectly elastic)
        stiffness: Spring constant for deformable objects
        damping: Damping coefficient
        surface_roughness: Surface roughness metric
        deformability: How easily the object deforms (0-1)
        is_rigid: Whether object is treated as rigid body
        material_type: Material classification
        attributes: Additional physical properties
    """

    mass: Optional[float] = None
    density: Optional[float] = None  # kg/m³
    friction: Optional[float] = None  # Static friction coefficient
    friction_dynamic: Optional[float] = None
    restitution: Optional[float] = None  # 0-1
    stiffness: Optional[float] = None  # N/m
    damping: Optional[float] = None
    surface_roughness: Optional[float] = None
    deformability: Optional[float] = None  # 0-1
    is_rigid: bool = True
    material_type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mass": self.mass,
            "density": self.density,
            "friction": self.friction,
            "friction_dynamic": self.friction_dynamic,
            "restitution": self.restitution,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "surface_roughness": self.surface_roughness,
            "deformability": self.deformability,
            "is_rigid": self.is_rigid,
            "material_type": self.material_type,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhysicalProperties:
        """Create from dictionary."""
        return cls(
            mass=data.get("mass"),
            density=data.get("density"),
            friction=data.get("friction"),
            friction_dynamic=data.get("friction_dynamic"),
            restitution=data.get("restitution"),
            stiffness=data.get("stiffness"),
            damping=data.get("damping"),
            surface_roughness=data.get("surface_roughness"),
            deformability=data.get("deformability"),
            is_rigid=data.get("is_rigid", True),
            material_type=data.get("material_type", "unknown"),
            attributes=data.get("attributes", {}),
        )


@dataclass
class ForceVector:
    """3D force vector with application point and direction.

    Attributes:
        force: Force vector [fx, fy, fz] in Newtons
        position: Application point (None for body-centered force)
        torque: Torque vector [tx, ty, tz] in N·m (optional)
        duration: Force duration in seconds (None for instantaneous)
        timestamp: When the force is applied
    """

    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position: Optional[np.ndarray] = None
    torque: Optional[np.ndarray] = None
    duration: Optional[float] = None
    timestamp: float = 0.0

    def __post_init__(self):
        self.force = np.asarray(self.force, dtype=np.float32)
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=np.float32)
        if self.torque is not None:
            self.torque = np.asarray(self.torque, dtype=np.float32)

    @property
    def magnitude(self) -> float:
        """Get force magnitude in Newtons."""
        return float(np.linalg.norm(self.force))

    @property
    def direction(self) -> np.ndarray:
        """Get normalized force direction."""
        mag = self.magnitude
        if mag > 1e-6:
            return self.force / mag
        return np.array([0.0, 0.0, 1.0])  # Default up

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "force": self.force.tolist(),
            "position": self.position.tolist() if self.position is not None else None,
            "torque": self.torque.tolist() if self.torque is not None else None,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ForceVector:
        """Create from dictionary."""
        return cls(
            force=np.array(data.get("force", [0, 0, 0])),
            position=np.array(data["position"]) if data.get("position") else None,
            torque=np.array(data["torque"]) if data.get("torque") else None,
            duration=data.get("duration"),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class Trajectory:
    """Object trajectory over time.

    Attributes:
        object_id: ID of the object
        positions: List of position vectors
        orientations: List of orientation quaternions/vectors
        timestamps: List of timestamps for each point
        velocities: Optional velocity at each point
        is_predicted: Whether this is a prediction or ground truth
    """

    object_id: str
    positions: List[np.ndarray] = field(default_factory=list)
    orientations: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    is_predicted: bool = False

    def add_point(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Add a trajectory point."""
        self.positions.append(np.asarray(position, dtype=np.float32))
        if orientation is not None:
            self.orientations.append(np.asarray(orientation, dtype=np.float32))
        else:
            self.orientations.append(np.array([0.0, 0.0, 0.0, 1.0]))
        self.timestamps.append(timestamp)
        if velocity is not None:
            self.velocities.append(np.asarray(velocity, dtype=np.float32))
        else:
            self.velocities.append(np.zeros(3))

    def get_point_at_time(self, timestamp: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get interpolated position and orientation at given time."""
        if not self.timestamps:
            return None

        # Find surrounding timestamps
        if timestamp <= self.timestamps[0]:
            return self.positions[0], self.orientations[0]
        if timestamp >= self.timestamps[-1]:
            return self.positions[-1], self.orientations[-1]

        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= timestamp <= self.timestamps[i + 1]:
                # Linear interpolation
                t = (timestamp - self.timestamps[i]) / (
                    self.timestamps[i + 1] - self.timestamps[i]
                )
                pos = (1 - t) * self.positions[i] + t * self.positions[i + 1]
                # Simple slerp approximation
                orient = (1 - t) * self.orientations[i] + t * self.orientations[i + 1]
                if np.linalg.norm(orient) > 0:
                    orient = orient / np.linalg.norm(orient)
                return pos, orient

        return None

    def length(self) -> float:
        """Calculate total trajectory length."""
        if len(self.positions) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.positions) - 1):
            total += np.linalg.norm(self.positions[i + 1] - self.positions[i])
        return float(total)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "positions": [p.tolist() for p in self.positions],
            "orientations": [o.tolist() for o in self.orientations],
            "timestamps": self.timestamps,
            "velocities": [v.tolist() for v in self.velocities],
            "is_predicted": self.is_predicted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trajectory:
        """Create from dictionary."""
        traj = cls(
            object_id=data["object_id"],
            is_predicted=data.get("is_predicted", False),
        )
        for i, pos in enumerate(data.get("positions", [])):
            traj.add_point(
                position=np.array(pos),
                orientation=np.array(data["orientations"][i]) if data.get("orientations") else None,
                timestamp=data["timestamps"][i] if data.get("timestamps") else 0.0,
                velocity=np.array(data["velocities"][i]) if data.get("velocities") else None,
            )
        return traj


@dataclass
class ContactEvent:
    """A contact event between two objects.

    Attributes:
        subject_id: First object ID
        object_id: Second object ID
        contact_type: Type of contact interaction
        start_time: When contact started
        end_time: When contact ended (None if ongoing)
        force: Estimated contact force
        contact_points: List of contact point positions
        impulse: Change in momentum from collision
        is_stable: Whether contact is stable (vs transient)
    """

    subject_id: str
    object_id: str
    contact_type: ContactType = ContactType.UNKNOWN
    start_time: float = 0.0
    end_time: Optional[float] = None
    force: Optional[ForceVector] = None
    contact_points: List[np.ndarray] = field(default_factory=list)
    impulse: Optional[np.ndarray] = None
    is_stable: bool = False

    @property
    def duration(self) -> Optional[float]:
        """Get contact duration if ended."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def is_active(self) -> bool:
        """Check if contact is currently active."""
        return self.end_time is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "contact_type": self.contact_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "force": self.force.to_dict() if self.force else None,
            "contact_points": [cp.tolist() for cp in self.contact_points],
            "impulse": self.impulse.tolist() if self.impulse is not None else None,
            "is_stable": self.is_stable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContactEvent:
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            contact_type=ContactType(data.get("contact_type", "unknown")),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time"),
            force=ForceVector.from_dict(data["force"]) if data.get("force") else None,
            contact_points=[np.array(cp) for cp in data.get("contact_points", [])],
            impulse=np.array(data["impulse"]) if data.get("impulse") else None,
            is_stable=data.get("is_stable", False),
        )


@dataclass
class PhysicalDynamics:
    """Complete physical dynamics annotation for an object or interaction.

    Combines physical properties, contact events, and trajectories into
    a comprehensive dynamics annotation.

    Attributes:
        object_id: Target object ID
        dynamics_type: Type of dynamics model
        properties: Physical properties
        trajectory: Object trajectory
        contact_events: List of contact events involving this object
        forces: Applied forces
        constraints: Physical constraints (e.g., joint limits)
        timestamp: Annotation timestamp
        source: Source of annotation ("teacher", "simulation", "heuristic")
        confidence: Overall confidence in the annotation
    """

    object_id: str
    dynamics_type: DynamicsType = DynamicsType.RIGID_BODY
    properties: PhysicalProperties = field(default_factory=PhysicalProperties)
    trajectory: Optional[Trajectory] = None
    contact_events: List[ContactEvent] = field(default_factory=list)
    forces: List[ForceVector] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = "unknown"
    confidence: float = 1.0

    def add_contact_event(self, event: ContactEvent) -> None:
        """Add a contact event."""
        self.contact_events.append(event)

    def add_force(self, force: ForceVector) -> None:
        """Add an applied force."""
        self.forces.append(force)

    def get_total_impulse(self) -> np.ndarray:
        """Calculate total impulse from all contact events."""
        total = np.zeros(3)
        for event in self.contact_events:
            if event.impulse is not None:
                total += event.impulse
        return total

    def get_peak_force(self) -> float:
        """Get the maximum force magnitude."""
        max_force = 0.0
        for force in self.forces:
            max_force = max(max_force, force.magnitude)
        for event in self.contact_events:
            if event.force is not None:
                max_force = max(max_force, event.force.magnitude)
        return max_force

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "dynamics_type": self.dynamics_type.value,
            "properties": self.properties.to_dict(),
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "contact_events": [e.to_dict() for e in self.contact_events],
            "forces": [f.to_dict() for f in self.forces],
            "constraints": self.constraints,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhysicalDynamics:
        """Create from dictionary."""
        return cls(
            object_id=data["object_id"],
            dynamics_type=DynamicsType(data.get("dynamics_type", "rigid_body")),
            properties=PhysicalProperties.from_dict(data.get("properties", {})),
            trajectory=Trajectory.from_dict(data["trajectory"]) if data.get("trajectory") else None,
            contact_events=[ContactEvent.from_dict(e) for e in data.get("contact_events", [])],
            forces=[ForceVector.from_dict(f) for f in data.get("forces", [])],
            constraints=data.get("constraints", {}),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ContactDynamics:
    """Contact dynamics between two objects.

    Specialized annotation for describing contact interactions,
    including forces, friction, and contact geometry.

    Attributes:
        subject_id: First object ID (typically agent/instrument)
        object_id: Second object ID (typically target)
        primary_contact_type: Main type of contact
        contact_types: All detected contact types over interaction
        events: Chronological list of contact events
        normal_force: Force perpendicular to contact surface
        tangential_force: Force parallel to contact surface
        friction_force: Friction force magnitude
        contact_area: Estimated contact area in m²
        penetration_depth: How much objects overlap/indent
        slip_velocity: Relative sliding velocity
        is_grasp: Whether this represents a grasp
        grasp_quality: Quality metric for grasps (0-1)
        stability: Contact stability metric (0-1)
    """

    subject_id: str
    object_id: str
    primary_contact_type: ContactType = ContactType.UNKNOWN
    contact_types: List[ContactType] = field(default_factory=list)
    events: List[ContactEvent] = field(default_factory=list)
    normal_force: Optional[float] = None  # Newtons
    tangential_force: Optional[float] = None
    friction_force: Optional[float] = None
    contact_area: Optional[float] = None  # m²
    penetration_depth: Optional[float] = None  # meters
    slip_velocity: Optional[float] = None  # m/s
    is_grasp: bool = False
    grasp_quality: Optional[float] = None  # 0-1
    stability: Optional[float] = None  # 0-1

    def add_event(self, event: ContactEvent) -> None:
        """Add a contact event."""
        self.events.append(event)
        if event.contact_type not in self.contact_types:
            self.contact_types.append(event.contact_type)

    @property
    def total_contact_duration(self) -> float:
        """Total duration of all contact events."""
        total = 0.0
        for event in self.events:
            if event.duration is not None:
                total += event.duration
        return total

    @property
    def has_sliding(self) -> bool:
        """Check if sliding occurred during contact."""
        return ContactType.SLIDING in self.contact_types or (
            self.slip_velocity is not None and self.slip_velocity > 0.01
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "primary_contact_type": self.primary_contact_type.value,
            "contact_types": [ct.value for ct in self.contact_types],
            "events": [e.to_dict() for e in self.events],
            "normal_force": self.normal_force,
            "tangential_force": self.tangential_force,
            "friction_force": self.friction_force,
            "contact_area": self.contact_area,
            "penetration_depth": self.penetration_depth,
            "slip_velocity": self.slip_velocity,
            "is_grasp": self.is_grasp,
            "grasp_quality": self.grasp_quality,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContactDynamics:
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            primary_contact_type=ContactType(data.get("primary_contact_type", "unknown")),
            contact_types=[ContactType(ct) for ct in data.get("contact_types", [])],
            events=[ContactEvent.from_dict(e) for e in data.get("events", [])],
            normal_force=data.get("normal_force"),
            tangential_force=data.get("tangential_force"),
            friction_force=data.get("friction_force"),
            contact_area=data.get("contact_area"),
            penetration_depth=data.get("penetration_depth"),
            slip_velocity=data.get("slip_velocity"),
            is_grasp=data.get("is_grasp", False),
            grasp_quality=data.get("grasp_quality"),
            stability=data.get("stability"),
        )


@dataclass
class MotionPrediction:
    """Predicted vs actual motion comparison.

    Used for evaluating and training world models by comparing
    predicted future states against ground truth.

    Attributes:
        object_id: Target object
        prediction_horizon: How far ahead prediction was made (seconds)
        predicted_trajectory: Model's predicted trajectory
        actual_trajectory: Ground truth trajectory
        start_state: World state at prediction time
        prediction_time: When prediction was made
        model_name: Name of model that made prediction
        metrics: Computed accuracy metrics
    """

    object_id: str
    prediction_horizon: float = 0.0
    predicted_trajectory: Optional[Trajectory] = None
    actual_trajectory: Optional[Trajectory] = None
    start_state: Optional[Dict[str, Any]] = None
    prediction_time: float = 0.0
    model_name: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute prediction accuracy metrics.

        Returns:
            Dictionary of metric_name -> value
        """
        if self.predicted_trajectory is None or self.actual_trajectory is None:
            return {}

        metrics = {}

        # Position error at end of prediction
        if self.predicted_trajectory.positions and self.actual_trajectory.positions:
            pred_pos = self.predicted_trajectory.positions[-1]
            actual_pos = self.actual_trajectory.positions[-1]
            end_error = np.linalg.norm(pred_pos - actual_pos)
            metrics["end_position_error"] = float(end_error)

        # Average position error
        errors = []
        for pred_pos, actual_pos in zip(
            self.predicted_trajectory.positions,
            self.actual_trajectory.positions,
        ):
            errors.append(np.linalg.norm(pred_pos - actual_pos))

        if errors:
            metrics["mean_position_error"] = float(np.mean(errors))
            metrics["max_position_error"] = float(np.max(errors))
            metrics["final_displacement_error"] = float(errors[-1] if errors else 0)

        # Trajectory shape similarity (simplified)
        if len(self.predicted_trajectory.positions) > 1:
            pred_length = self.predicted_trajectory.length()
            actual_length = self.actual_trajectory.length()
            if actual_length > 0.001:
                metrics["length_ratio"] = float(pred_length / actual_length)

        # Velocity accuracy
        if self.predicted_trajectory.velocities and self.actual_trajectory.velocities:
            vel_errors = []
            for pred_vel, actual_vel in zip(
                self.predicted_trajectory.velocities,
                self.actual_trajectory.velocities,
            ):
                vel_errors.append(np.linalg.norm(pred_vel - actual_vel))
            if vel_errors:
                metrics["mean_velocity_error"] = float(np.mean(vel_errors))

        self.metrics = metrics
        return metrics

    def is_accurate(self, threshold: float = 0.1) -> bool:
        """Check if prediction is within threshold."""
        if "mean_position_error" in self.metrics:
            return self.metrics["mean_position_error"] < threshold
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "prediction_horizon": self.prediction_horizon,
            "predicted_trajectory": self.predicted_trajectory.to_dict() if self.predicted_trajectory else None,
            "actual_trajectory": self.actual_trajectory.to_dict() if self.actual_trajectory else None,
            "start_state": self.start_state,
            "prediction_time": self.prediction_time,
            "model_name": self.model_name,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MotionPrediction:
        """Create from dictionary."""
        prediction = cls(
            object_id=data["object_id"],
            prediction_horizon=data.get("prediction_horizon", 0.0),
            predicted_trajectory=Trajectory.from_dict(data["predicted_trajectory"]) if data.get("predicted_trajectory") else None,
            actual_trajectory=Trajectory.from_dict(data["actual_trajectory"]) if data.get("actual_trajectory") else None,
            start_state=data.get("start_state"),
            prediction_time=data.get("prediction_time", 0.0),
            model_name=data.get("model_name", "unknown"),
            metrics=data.get("metrics", {}),
        )
        return prediction
