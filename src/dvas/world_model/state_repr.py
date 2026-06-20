"""State representation for World Model training data generation.

Provides data structures for representing complete scene states, individual
object states, and scene graphs for spatial and relational understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AffordanceState(str, Enum):
    """Affordance state of an object - what actions it supports."""

    GRASPABLE = "graspable"
    PUSHABLE = "pushable"
    LIFTABLE = "liftable"
    OPENABLE = "openable"
    CLOSEABLE = "closeable"
    ROTATABLE = "rotatable"
    STATIONARY = "stationary"
    CONTAINER = "container"
    POURABLE = "pourable"
    UNKNOWN = "unknown"


class ContactState(str, Enum):
    """Contact state between objects."""

    TOUCHING = "touching"
    SEPARATED = "separated"
    SUPPORTED_BY = "supported_by"
    GRASPING = "grasping"
    PENETRATING = "penetrating"
    SLIDING = "sliding"
    ROLLING = "rolling"
    COLLIDING = "colliding"
    NO_CONTACT = "no_contact"


class ObjectRole(str, Enum):
    """Role of an object in the scene."""

    AGENT = "agent"  # The actor performing actions
    INSTRUMENT = "instrument"  # Tool being used
    TARGET = "target"  # Object being acted upon
    CONTEXT = "context"  # Background object
    OBSTACLE = "obstacle"  # Blocking object
    SUPPORT = "support"  # Surface/supporting object
    UNKNOWN = "unknown"


@dataclass
class ObjectState:
    """State of a single object in the scene.

    Represents the complete state of an object including:
    - Spatial: position, orientation, bounding box
    - Physical: velocity, mass, size
    - Semantic: affordances, state changes, role

    Attributes:
        object_id: Unique identifier for the object
        name: Object class/name (e.g., "cup", "spoon")
        position: 3D position [x, y, z] in world coordinates
        orientation: Quaternion [qx, qy, qz, qw] or Euler angles [rx, ry, rz]
        velocity: Linear velocity [vx, vy, vz]
        angular_velocity: Angular velocity [wx, wy, wz]
        bbox: Bounding box [x1, y1, z1, x2, y2, z2] or None
        mass: Mass in kg (0 if unknown)
        size: Approximate dimensions [width, height, depth]
        affordances: Set of possible affordances
        material: Material type (e.g., "ceramic", "metal")
        state: Current state (e.g., "empty", "full", "open")
        role: Object's role in the scene
        is_visible: Whether object is currently visible
        attributes: Additional custom attributes
        confidence: Detection confidence [0.0, 1.0]
    """

    object_id: str
    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox: Optional[np.ndarray] = None  # [x1, y1, z1, x2, y2, z2]
    mass: float = 0.0
    size: np.ndarray = field(default_factory=lambda: np.ones(3))
    affordances: Set[AffordanceState] = field(default_factory=set)
    material: str = "unknown"
    state: str = "unknown"
    role: ObjectRole = ObjectRole.UNKNOWN
    is_visible: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self):
        """Ensure numpy arrays are properly typed."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.orientation = np.asarray(self.orientation, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32)
        self.size = np.asarray(self.size, dtype=np.float32)
        if self.bbox is not None:
            self.bbox = np.asarray(self.bbox, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "velocity": self.velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "mass": self.mass,
            "size": self.size.tolist(),
            "affordances": [a.value for a in self.affordances],
            "material": self.material,
            "state": self.state,
            "role": self.role.value,
            "is_visible": self.is_visible,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectState:
        """Create ObjectState from dictionary."""
        return cls(
            object_id=data["object_id"],
            name=data["name"],
            position=np.array(data.get("position", [0, 0, 0])),
            orientation=np.array(data.get("orientation", [0, 0, 0, 1])),
            velocity=np.array(data.get("velocity", [0, 0, 0])),
            angular_velocity=np.array(data.get("angular_velocity", [0, 0, 0])),
            bbox=np.array(data["bbox"]) if data.get("bbox") else None,
            mass=data.get("mass", 0.0),
            size=np.array(data.get("size", [1, 1, 1])),
            affordances={AffordanceState(a) for a in data.get("affordances", [])},
            material=data.get("material", "unknown"),
            state=data.get("state", "unknown"),
            role=ObjectRole(data.get("role", "unknown")),
            is_visible=data.get("is_visible", True),
            attributes=data.get("attributes", {}),
            confidence=data.get("confidence", 1.0),
        )

    def distance_to(self, other: ObjectState) -> float:
        """Calculate Euclidean distance to another object."""
        return float(np.linalg.norm(self.position - other.position))

    def speed(self) -> float:
        """Calculate scalar speed."""
        return float(np.linalg.norm(self.velocity))

    def is_moving(self, threshold: float = 0.01) -> bool:
        """Check if object is moving above threshold."""
        return self.speed() > threshold

    def copy(self) -> ObjectState:
        """Create a deep copy of this object state."""
        return ObjectState(
            object_id=self.object_id,
            name=self.name,
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            bbox=self.bbox.copy() if self.bbox is not None else None,
            mass=self.mass,
            size=self.size.copy(),
            affordances=self.affordances.copy(),
            material=self.material,
            state=self.state,
            role=self.role,
            is_visible=self.is_visible,
            attributes=self.attributes.copy(),
            confidence=self.confidence,
        )


@dataclass
class Relationship:
    """Relationship between two objects.

    Attributes:
        subject_id: ID of the subject object
        object_id: ID of the object
        relation_type: Type of relationship (e.g., "on", "next_to", "holding")
        contact_state: Physical contact state
        confidence: Confidence in the relationship [0.0, 1.0]
        attributes: Additional relationship attributes
    """

    subject_id: str
    object_id: str
    relation_type: str  # e.g., "on", "next_to", "holding", "supporting"
    contact_state: ContactState = ContactState.NO_CONTACT
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation_type": self.relation_type,
            "contact_state": self.contact_state.value,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relationship:
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            object_id=data["object_id"],
            relation_type=data["relation_type"],
            contact_state=ContactState(data.get("contact_state", "no_contact")),
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
        )


class SceneGraph:
    """Scene graph representing spatial and semantic relationships between objects.

    A scene graph encodes the structure of a scene as nodes (objects) and
    edges (relationships), enabling reasoning about object interactions.

    Attributes:
        objects: Dictionary of object_id -> ObjectState
        relationships: List of Relationship instances
        timestamp: Optional timestamp for this scene state
    """

    def __init__(
        self,
        objects: Optional[Dict[str, ObjectState]] = None,
        relationships: Optional[List[Relationship]] = None,
        timestamp: Optional[float] = None,
    ):
        self.objects: Dict[str, ObjectState] = objects or {}
        self.relationships: List[Relationship] = relationships or []
        self.timestamp: Optional[float] = timestamp
        self._edge_index: Optional[Dict[str, List[Relationship]]] = None

    def add_object(self, obj: ObjectState) -> None:
        """Add an object to the scene graph."""
        self.objects[obj.object_id] = obj
        self._edge_index = None  # Invalidate cache

    def remove_object(self, object_id: str) -> None:
        """Remove an object and all related relationships."""
        if object_id in self.objects:
            del self.objects[object_id]
            self.relationships = [
                r
                for r in self.relationships
                if r.subject_id != object_id and r.object_id != object_id
            ]
            self._edge_index = None

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the scene graph."""
        if rel.subject_id in self.objects and rel.object_id in self.objects:
            self.relationships.append(rel)
            self._edge_index = None
        else:
            logger.warning(
                "scene_graph_add_relationship_skip",
                subject=rel.subject_id,
                object=rel.object_id,
                reason="object_not_found",
            )

    def get_relationships(self, object_id: str) -> List[Relationship]:
        """Get all relationships involving the given object."""
        return [
            r for r in self.relationships if r.subject_id == object_id or r.object_id == object_id
        ]

    def get_related_objects(
        self,
        object_id: str,
        relation_type: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Get objects related to the given object.

        Returns:
            List of (object_id, relation_type, confidence) tuples
        """
        related = []
        for r in self.relationships:
            if r.subject_id == object_id:
                if relation_type is None or r.relation_type == relation_type:
                    related.append((r.object_id, r.relation_type, r.confidence))
            elif r.object_id == object_id:
                if relation_type is None or r.relation_type == relation_type:
                    related.append((r.subject_id, f"inverse_{r.relation_type}", r.confidence))
        return related

    def find_object_by_name(self, name: str) -> List[ObjectState]:
        """Find all objects with the given name."""
        return [obj for obj in self.objects.values() if obj.name == name]

    def get_objects_by_role(self, role: ObjectRole) -> List[ObjectState]:
        """Get all objects with the given role."""
        return [obj for obj in self.objects.values() if obj.role == role]

    def get_supporting_surfaces(self) -> List[ObjectState]:
        """Get objects that serve as supporting surfaces."""
        surfaces = []
        for obj in self.objects.values():
            if AffordanceState.CONTAINER in obj.affordances or obj.role == ObjectRole.SUPPORT:
                surfaces.append(obj)
            elif any(
                r.relation_type == "supported_by" and r.object_id == obj.object_id
                for r in self.relationships
            ):
                surfaces.append(obj)
        return surfaces

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneGraph:
        """Create from dictionary."""
        objects = {
            oid: ObjectState.from_dict(odata) for oid, odata in data.get("objects", {}).items()
        }
        relationships = [Relationship.from_dict(rdata) for rdata in data.get("relationships", [])]
        return cls(
            objects=objects,
            relationships=relationships,
            timestamp=data.get("timestamp"),
        )

    def copy(self) -> SceneGraph:
        """Create a deep copy of the scene graph."""
        return SceneGraph(
            objects={oid: obj.copy() for oid, obj in self.objects.items()},
            relationships=[
                Relationship(
                    subject_id=r.subject_id,
                    object_id=r.object_id,
                    relation_type=r.relation_type,
                    contact_state=r.contact_state,
                    confidence=r.confidence,
                    attributes=r.attributes.copy(),
                )
                for r in self.relationships
            ],
            timestamp=self.timestamp,
        )


@dataclass
class WorldState:
    """Complete state representation of the world/scene.

    Encapsulates all information needed to represent a single moment
    in time for World Model training:
    - Scene graph with all objects and relationships
    - Agent state (the entity performing actions)
    - Environment state (lighting, physics parameters)

    Attributes:
        scene_graph: SceneGraph with objects and relationships
        agent_id: ID of the agent/actor in the scene
        timestamp: Temporal position in the sequence
        frame_number: Frame index in video (if applicable)
        environment: Environment parameters
        metadata: Additional custom metadata
    """

    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    agent_id: Optional[str] = None
    timestamp: float = 0.0
    frame_number: Optional[int] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_agent(self) -> Optional[ObjectState]:
        """Get the agent object if present."""
        if self.agent_id and self.agent_id in self.scene_graph.objects:
            return self.scene_graph.objects[self.agent_id]
        # Try to find agent by role
        agents = self.scene_graph.get_objects_by_role(ObjectRole.AGENT)
        if agents:
            self.agent_id = agents[0].object_id
            return agents[0]
        return None

    def get_agent_hand_position(self, hand: str = "right") -> Optional[np.ndarray]:
        """Get the position of the agent's hand (if tracked)."""
        agent = self.get_agent()
        if agent is None:
            return None
        # Check if agent has hand positions in attributes
        hand_key = f"{hand}_hand_position"
        if hand_key in agent.attributes:
            return np.array(agent.attributes[hand_key])
        return agent.position  # Fall back to agent position

    def get_target_objects(self) -> List[ObjectState]:
        """Get objects marked as targets of interaction."""
        return self.scene_graph.get_objects_by_role(ObjectRole.TARGET)

    def get_instrument_objects(self) -> List[ObjectState]:
        """Get objects marked as instruments/tools."""
        return self.scene_graph.get_objects_by_role(ObjectRole.INSTRUMENT)

    def describe(self) -> str:
        """Generate a natural language description of the scene state."""
        parts = []

        # Agent description
        agent = self.get_agent()
        if agent:
            parts.append(f"Agent ({agent.name}) at position {agent.position.round(3)}")

        # Object descriptions
        visible_objects = [obj for obj in self.scene_graph.objects.values() if obj.is_visible]
        if visible_objects:
            obj_desc = ", ".join([f"{obj.name} ({obj.state})" for obj in visible_objects[:5]])
            parts.append(f"Objects: {obj_desc}")

        # Key relationships
        if self.scene_graph.relationships:
            key_rels = [
                f"{r.subject_id} {r.relation_type} {r.object_id}"
                for r in self.scene_graph.relationships[:3]
            ]
            parts.append(f"Relationships: {', '.join(key_rels)}")

        return ". ".join(parts) if parts else "Empty scene"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_graph": self.scene_graph.to_dict(),
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "environment": self.environment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldState:
        """Create from dictionary."""
        return cls(
            scene_graph=SceneGraph.from_dict(data.get("scene_graph", {})),
            agent_id=data.get("agent_id"),
            timestamp=data.get("timestamp", 0.0),
            frame_number=data.get("frame_number"),
            environment=data.get("environment", {}),
            metadata=data.get("metadata", {}),
        )

    def copy(self) -> WorldState:
        """Create a deep copy of this world state."""
        return WorldState(
            scene_graph=self.scene_graph.copy(),
            agent_id=self.agent_id,
            timestamp=self.timestamp,
            frame_number=self.frame_number,
            environment=self.environment.copy(),
            metadata=self.metadata.copy(),
        )

    def interpolate(self, other: WorldState, alpha: float) -> WorldState:
        """Interpolate between this state and another state.

        Args:
            other: Target WorldState
            alpha: Interpolation factor [0.0, 1.0]

        Returns:
            New interpolated WorldState
        """
        result = self.copy()
        result.timestamp = self.timestamp + alpha * (other.timestamp - self.timestamp)

        for obj_id, obj in other.scene_graph.objects.items():
            if obj_id in result.scene_graph.objects:
                # Interpolate position
                current = result.scene_graph.objects[obj_id]
                current.position = (1 - alpha) * current.position + alpha * obj.position
                current.orientation = self._slerp(current.orientation, obj.orientation, alpha)
                current.velocity = (1 - alpha) * current.velocity + alpha * obj.velocity

        return result

    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Simple linear interpolation for quaternions (approximate slerp).

        For proper slerp, use scipy.spatial.transform.Slerp.
        """
        if len(q1) == 3 and len(q2) == 3:
            # Euler angles - simple lerp
            return (1 - t) * q1 + t * q2
        # Quaternions
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q1) + (s1 * q2)
