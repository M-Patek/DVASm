"""Object affordance annotations for robot learning.

Defines affordance annotations that describe what actions are possible
with objects, including graspable regions, force requirements, and
spatial constraints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AffordanceType(str, Enum):
    """Types of object affordances."""

    GRASPABLE = "graspable"
    PUSHABLE = "pushable"
    PULLABLE = "pullable"
    LIFTABLE = "liftable"
    ROTATABLE = "rotatable"
    SLIDABLE = "slidable"
    INSERTABLE = "insertable"
    EXTRACTABLE = "extractable"
    POURABLE = "pourable"
    CONTAINABLE = "containable"
    STACKABLE = "stackable"
    HANGABLE = "hangable"
    CUTTABLE = "cuttable"
    WRITABLE = "writable"
    PRESSABLE = "pressable"
    TWISTABLE = "twistable"
    OPENABLE = "openable"
    CLOSEABLE = "closeable"
    SQUEEZABLE = "squeezable"
    SUPPORTABLE = "supportable"  # Can support other objects
    UNKNOWN = "unknown"


class Handedness(str, Enum):
    """Hand preference for affordance."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    EITHER = "either"
    NONE = "none"


@dataclass
class SpatialRegion:
    """Spatial region on an object (e.g., graspable area).

    Defines a region using normalized coordinates or 3D bounding box.
    """

    # 2D normalized coordinates [x1, y1, x2, y2] for image-based regions
    bbox_2d: Optional[Tuple[float, float, float, float]] = None

    # 3D bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
    bbox_3d: Optional[Tuple[float, float, float, float, float, float]] = None

    # Center point (normalized 2D or 3D coordinates)
    center: Optional[Tuple[float, ...]] = None

    # Region shape description
    shape: str = "unknown"  # "rectangular", "circular", "irregular"

    # Surface normal at center (for grasp planning)
    surface_normal: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d else None,
            "bbox_3d": list(self.bbox_3d) if self.bbox_3d else None,
            "center": list(self.center) if self.center else None,
            "shape": self.shape,
            "surface_normal": list(self.surface_normal) if self.surface_normal else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialRegion":
        """Create from dictionary."""
        return cls(
            bbox_2d=tuple(data["bbox_2d"]) if data.get("bbox_2d") else None,
            bbox_3d=tuple(data["bbox_3d"]) if data.get("bbox_3d") else None,
            center=tuple(data["center"]) if data.get("center") else None,
            shape=data.get("shape", "unknown"),
            surface_normal=tuple(data["surface_normal"]) if data.get("surface_normal") else None,
        )

    def get_area_2d(self) -> Optional[float]:
        """Calculate 2D area if bbox_2d is defined."""
        if self.bbox_2d:
            x1, y1, x2, y2 = self.bbox_2d
            return (x2 - x1) * (y2 - y1)
        return None

    def get_volume_3d(self) -> Optional[float]:
        """Calculate 3D volume if bbox_3d is defined."""
        if self.bbox_3d:
            x1, y1, z1, x2, y2, z2 = self.bbox_3d
            return (x2 - x1) * (y2 - y1) * (z2 - z1)
        return None


@dataclass
class ForceRequirements:
    """Force requirements for an affordance.

    Defines the forces needed to execute an action on an object.
    """

    # Minimum force required (Newtons)
    min_force: Optional[float] = None

    # Maximum force allowed (Newtons)
    max_force: Optional[float] = None

    # Optimal/target force (Newtons)
    target_force: Optional[float] = None

    # Force direction relative to object
    force_direction: Optional[Tuple[float, float, float]] = None

    # Whether force should be applied gradually
    gradual_application: bool = True

    # Whether sustained force is needed
    sustained: bool = False

    # Duration of force application (seconds)
    duration: Optional[float] = None

    # Torque requirements for rotational affordances (Nm)
    min_torque: Optional[float] = None
    max_torque: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_force": self.min_force,
            "max_force": self.max_force,
            "target_force": self.target_force,
            "force_direction": list(self.force_direction) if self.force_direction else None,
            "gradual_application": self.gradual_application,
            "sustained": self.sustained,
            "duration": self.duration,
            "min_torque": self.min_torque,
            "max_torque": self.max_torque,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForceRequirements":
        """Create from dictionary."""
        return cls(
            min_force=data.get("min_force"),
            max_force=data.get("max_force"),
            target_force=data.get("target_force"),
            force_direction=tuple(data["force_direction"]) if data.get("force_direction") else None,
            gradual_application=data.get("gradual_application", True),
            sustained=data.get("sustained", False),
            duration=data.get("duration"),
            min_torque=data.get("min_torque"),
            max_torque=data.get("max_torque"),
        )

    def validate_force(self, applied_force: float) -> Tuple[bool, str]:
        """Check if applied force is within valid range.

        Returns:
            Tuple of (is_valid, message)
        """
        if self.min_force is not None and applied_force < self.min_force:
            return False, f"Force {applied_force}N below minimum {self.min_force}N"
        if self.max_force is not None and applied_force > self.max_force:
            return False, f"Force {applied_force}N exceeds maximum {self.max_force}N"
        return True, "Force within valid range"


@dataclass
class GraspConstraints:
    """Constraints specific to grasping affordances."""

    # Number of fingers required
    min_fingers: int = 2
    max_fingers: int = 5

    # Preferred hand orientation
    preferred_hand_orientation: Optional[str] = None  # "power", "precision", "hook"

    # Whether pinch grasp is possible
    pinch_grasp: bool = False

    # Whether power grasp is possible
    power_grasp: bool = True

    # Whether two-handed grasp is needed
    two_handed: bool = False

    # Minimum gripper aperture (meters)
    min_aperture: float = 0.0

    # Maximum gripper aperture (meters)
    max_aperture: float = 0.1

    # Surface friction coefficient (approximate)
    friction_coefficient: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_fingers": self.min_fingers,
            "max_fingers": self.max_fingers,
            "preferred_hand_orientation": self.preferred_hand_orientation,
            "pinch_grasp": self.pinch_grasp,
            "power_grasp": self.power_grasp,
            "two_handed": self.two_handed,
            "min_aperture": self.min_aperture,
            "max_aperture": self.max_aperture,
            "friction_coefficient": self.friction_coefficient,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraspConstraints":
        """Create from dictionary."""
        return cls(
            min_fingers=data.get("min_fingers", 2),
            max_fingers=data.get("max_fingers", 5),
            preferred_hand_orientation=data.get("preferred_hand_orientation"),
            pinch_grasp=data.get("pinch_grasp", False),
            power_grasp=data.get("power_grasp", True),
            two_handed=data.get("two_handed", False),
            min_aperture=data.get("min_aperture", 0.0),
            max_aperture=data.get("max_aperture", 0.1),
            friction_coefficient=data.get("friction_coefficient"),
        )


@dataclass
class SingleAffordance:
    """Single affordance annotation for an object.

    Describes one possible action that can be performed on/with an object.
    """

    # Affordance type
    affordance_type: AffordanceType

    # Confidence in this affordance (0.0 to 1.0)
    confidence: float = 1.0

    # Hand preference
    handedness: Handedness = Handedness.EITHER

    # Spatial region(s) where affordance applies
    regions: List[SpatialRegion] = field(default_factory=list)

    # Force requirements
    force_requirements: Optional[ForceRequirements] = None

    # Grasp-specific constraints (only for GRASPABLE)
    grasp_constraints: Optional[GraspConstraints] = None

    # Preconditions for this affordance
    preconditions: List[str] = field(default_factory=list)

    # Expected outcome
    outcome_description: Optional[str] = None

    # Whether this affordance requires a tool
    requires_tool: bool = False

    # Compatible tools (if requires_tool is True)
    compatible_tools: List[str] = field(default_factory=list)

    # Difficulty level (1-5, 1=easiest)
    difficulty: int = 3

    # Temporal duration estimate (seconds)
    estimated_duration: Optional[float] = None

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "affordance_type": self.affordance_type.value,
            "confidence": self.confidence,
            "handedness": self.handedness.value,
            "regions": [r.to_dict() for r in self.regions],
            "force_requirements": (
                self.force_requirements.to_dict() if self.force_requirements else None
            ),
            "grasp_constraints": (
                self.grasp_constraints.to_dict() if self.grasp_constraints else None
            ),
            "preconditions": self.preconditions,
            "outcome_description": self.outcome_description,
            "requires_tool": self.requires_tool,
            "compatible_tools": self.compatible_tools,
            "difficulty": self.difficulty,
            "estimated_duration": self.estimated_duration,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleAffordance":
        """Create from dictionary."""
        return cls(
            affordance_type=AffordanceType(data.get("affordance_type", "unknown")),
            confidence=data.get("confidence", 1.0),
            handedness=Handedness(data.get("handedness", "either")),
            regions=[SpatialRegion.from_dict(r) for r in data.get("regions", [])],
            force_requirements=(
                ForceRequirements.from_dict(data["force_requirements"])
                if data.get("force_requirements")
                else None
            ),
            grasp_constraints=(
                GraspConstraints.from_dict(data["grasp_constraints"])
                if data.get("grasp_constraints")
                else None
            ),
            preconditions=data.get("preconditions", []),
            outcome_description=data.get("outcome_description"),
            requires_tool=data.get("requires_tool", False),
            compatible_tools=data.get("compatible_tools", []),
            difficulty=data.get("difficulty", 3),
            estimated_duration=data.get("estimated_duration"),
            attributes=data.get("attributes", {}),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this affordance.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0.0 and 1.0")

        if not 1 <= self.difficulty <= 5:
            errors.append("difficulty must be between 1 and 5")

        if self.affordance_type == AffordanceType.GRASPABLE and not self.grasp_constraints:
            errors.append("GRASPABLE affordance should have grasp_constraints")

        return len(errors) == 0, errors

    def get_primary_region(self) -> Optional[SpatialRegion]:
        """Get the primary (largest) region for this affordance."""
        if not self.regions:
            return None
        # Return region with largest 2D area
        return max(self.regions, key=lambda r: r.get_area_2d() or 0.0)


@dataclass
class ObjectAffordance:
    """Complete affordance annotation for an object.

    Aggregates all affordances for a single object.
    """

    # Object identifier
    object_name: str

    # Object category
    category: Optional[str] = None

    # Object instance ID (for multiple objects of same type)
    instance_id: Optional[str] = None

    # List of affordances
    affordances: List[SingleAffordance] = field(default_factory=list)

    # Default/reference pose of the object
    reference_pose: Optional[Dict[str, Any]] = None

    # Object physical properties
    mass: Optional[float] = None  # kg
    dimensions: Optional[Tuple[float, float, float]] = None  # meters (l, w, h)
    material: Optional[str] = None

    # Whether object is movable
    is_movable: bool = True

    # Whether object is deformable
    is_deformable: bool = False

    # Whether object is fragile
    is_fragile: bool = False

    # Annotation metadata
    annotated_by: str = "auto"  # "auto", "human", "demo"
    annotation_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_name": self.object_name,
            "category": self.category,
            "instance_id": self.instance_id,
            "affordances": [a.to_dict() for a in self.affordances],
            "reference_pose": self.reference_pose,
            "mass": self.mass,
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "material": self.material,
            "is_movable": self.is_movable,
            "is_deformable": self.is_deformable,
            "is_fragile": self.is_fragile,
            "annotated_by": self.annotated_by,
            "annotation_confidence": self.annotation_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectAffordance":
        """Create from dictionary."""
        return cls(
            object_name=data["object_name"],
            category=data.get("category"),
            instance_id=data.get("instance_id"),
            affordances=[SingleAffordance.from_dict(a) for a in data.get("affordances", [])],
            reference_pose=data.get("reference_pose"),
            mass=data.get("mass"),
            dimensions=tuple(data["dimensions"]) if data.get("dimensions") else None,
            material=data.get("material"),
            is_movable=data.get("is_movable", True),
            is_deformable=data.get("is_deformable", False),
            is_fragile=data.get("is_fragile", False),
            annotated_by=data.get("annotated_by", "auto"),
            annotation_confidence=data.get("annotation_confidence", 1.0),
        )

    def get_affordance(self, aff_type: AffordanceType) -> List[SingleAffordance]:
        """Get all affordances of a specific type."""
        return [a for a in self.affordances if a.affordance_type == aff_type]

    def has_affordance(self, aff_type: AffordanceType) -> bool:
        """Check if object has a specific affordance type."""
        return any(a.affordance_type == aff_type for a in self.affordances)

    def get_graspable_regions(self) -> List[SpatialRegion]:
        """Get all graspable regions on this object."""
        regions = []
        for aff in self.affordances:
            if aff.affordance_type == AffordanceType.GRASPABLE:
                regions.extend(aff.regions)
        return regions

    def add_affordance(self, affordance: SingleAffordance) -> None:
        """Add an affordance to this object."""
        self.affordances.append(affordance)


@dataclass
class AffordanceAnnotation:
    """Container for affordance annotations in a scene/video.

    Maps object names to their affordance annotations.
    """

    # Scene/video identifier
    scene_id: str

    # Object affordances
    object_affordances: Dict[str, ObjectAffordance] = field(default_factory=dict)

    # Scene-level affordances (e.g., "walkable" for floor)
    scene_affordances: List[SingleAffordance] = field(default_factory=list)

    # Relationships between object affordances
    affordance_relations: List[Dict[str, Any]] = field(default_factory=list)

    # Annotation timestamp
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "object_affordances": {k: v.to_dict() for k, v in self.object_affordances.items()},
            "scene_affordances": [a.to_dict() for a in self.scene_affordances],
            "affordance_relations": self.affordance_relations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffordanceAnnotation":
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            object_affordances={
                k: ObjectAffordance.from_dict(v)
                for k, v in data.get("object_affordances", {}).items()
            },
            scene_affordances=[
                SingleAffordance.from_dict(a) for a in data.get("scene_affordances", [])
            ],
            affordance_relations=data.get("affordance_relations", []),
            timestamp=data.get("timestamp"),
        )

    def get_object_affordance(self, object_name: str) -> Optional[ObjectAffordance]:
        """Get affordance annotation for a specific object."""
        return self.object_affordances.get(object_name)

    def add_object_affordance(self, obj_aff: ObjectAffordance) -> None:
        """Add an object affordance annotation."""
        self.object_affordances[obj_aff.object_name] = obj_aff

    def get_all_affordances_of_type(
        self, aff_type: AffordanceType
    ) -> List[Tuple[str, SingleAffordance]]:
        """Get all affordances of a specific type across all objects.

        Returns:
            List of (object_name, affordance) tuples
        """
        results = []
        for obj_name, obj_aff in self.object_affordances.items():
            for aff in obj_aff.affordances:
                if aff.affordance_type == aff_type:
                    results.append((obj_name, aff))
        return results

    def find_graspable_objects(self) -> List[str]:
        """Find all objects that are graspable."""
        return [
            name
            for name, obj_aff in self.object_affordances.items()
            if obj_aff.has_affordance(AffordanceType.GRASPABLE)
        ]

    def find_pushable_objects(self) -> List[str]:
        """Find all objects that are pushable."""
        return [
            name
            for name, obj_aff in self.object_affordances.items()
            if obj_aff.has_affordance(AffordanceType.PUSHABLE)
        ]
