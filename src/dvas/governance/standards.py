"""Annotation standard management for DVAS governance.

Provides AnnotationStandard dataclass, StandardRegistry, and support for
EPIC-KITCHENS, Ego4D, Open X-Embodiment, Something-Something, and AVA standards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set



class StandardFieldType(str, Enum):
    """Types of fields in annotation standards."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"


@dataclass
class StandardField:
    """A field definition within an annotation standard."""

    name: str
    field_type: StandardFieldType
    description: str = ""
    validators: List[str] = field(default_factory=list)
    default: Any = None


@dataclass
class StandardVersion:
    """Version information for an annotation standard."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "StandardVersion":
        """Parse version from string."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))

    def __ge__(self, other: "StandardVersion") -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __gt__(self, other: "StandardVersion") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __le__(self, other: "StandardVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __lt__(self, other: "StandardVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


@dataclass
class AnnotationStandardDef:
    """Definition of an annotation standard."""

    name: str
    version: StandardVersion
    description: str
    fields: List[StandardField] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: Optional[float] = None
    deprecated: bool = False
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "fields": [
                {
                    "name": f.name,
                    "type": f.field_type.value,
                    "description": f.description,
                    "validators": f.validators,
                    "default": f.default,
                }
                for f in self.fields
            ],
            "supported_formats": self.supported_formats,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deprecated": self.deprecated,
            "replacement": self.replacement,
        }


class StandardRegistry:
    """Registry for managing multiple annotation standards.

    Usage::

        registry = StandardRegistry()
        registry.register(AnnotationStandardDef(
            name="EPIC-KITCHENS",
            version=StandardVersion(2, 0, 0),
            description="EPIC-KITCHENS annotation standard",
        ))

        epic = registry.get("EPIC-KITCHENS")
        standards = registry.list_standards()
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in standards."""
        self._standards: Dict[str, AnnotationStandardDef] = {}
        self._versions: Dict[str, List[StandardVersion]] = {}
        self._register_builtin_standards()

    def _register_builtin_standards(self) -> None:
        """Register built-in annotation standards."""
        # EPIC-KITCHENS
        self.register(AnnotationStandardDef(
            name="EPIC-KITCHENS",
            version=StandardVersion(2, 0, 0),
            description="EPIC-KITCHENS action recognition standard",
            fields=[
                StandardField("verb", StandardFieldType.REQUIRED, "Action verb"),
                StandardField("noun", StandardFieldType.REQUIRED, "Action noun"),
                StandardField("hand", StandardFieldType.REQUIRED, "Hand used"),
                StandardField("start_time", StandardFieldType.REQUIRED, "Start timestamp"),
                StandardField("end_time", StandardFieldType.REQUIRED, "End timestamp"),
                StandardField("participant_id", StandardFieldType.CONDITIONAL, "Participant ID"),
                StandardField("narration", StandardFieldType.OPTIONAL, "Text narration"),
            ],
            supported_formats=["json", "csv", "pkl"],
        ))

        # Ego4D
        self.register(AnnotationStandardDef(
            name="Ego4D",
            version=StandardVersion(1, 0, 0),
            description="Ego4D egocentric video understanding standard",
            fields=[
                StandardField("narration", StandardFieldType.REQUIRED, "Text narration"),
                StandardField("start_time", StandardFieldType.REQUIRED, "Start timestamp"),
                StandardField("end_time", StandardFieldType.REQUIRED, "End timestamp"),
                StandardField("spatial_info", StandardFieldType.OPTIONAL, "3D spatial information"),
                StandardField("hand_interaction", StandardFieldType.OPTIONAL, "Hand-object interaction"),
                StandardField("instrument", StandardFieldType.OPTIONAL, "Instrument used"),
                StandardField("state_changes", StandardFieldType.OPTIONAL, "State change annotations"),
            ],
            supported_formats=["json", "jsonl"],
        ))

        # Open X-Embodiment
        self.register(AnnotationStandardDef(
            name="Open X-Embodiment",
            version=StandardVersion(1, 0, 0),
            description="Open X-Embodiment robot learning standard",
            fields=[
                StandardField("language_instruction", StandardFieldType.REQUIRED, "Natural language instruction"),
                StandardField("action_space", StandardFieldType.REQUIRED, "Robot action space definition"),
                StandardField("gripper_pose", StandardFieldType.OPTIONAL, "Gripper pose"),
                StandardField("joint_target", StandardFieldType.OPTIONAL, "Joint target positions"),
                StandardField("gripper_state", StandardFieldType.OPTIONAL, "Gripper open/close state"),
                StandardField("embodiment_type", StandardFieldType.CONDITIONAL, "Robot embodiment type"),
            ],
            supported_formats=["json", "tfrecord"],
        ))

        # Something-Something
        self.register(AnnotationStandardDef(
            name="Something-Something",
            version=StandardVersion(2, 0, 0),
            description="Something-Something video understanding standard",
            fields=[
                StandardField("template", StandardFieldType.REQUIRED, "Action template"),
                StandardField("placeholders", StandardFieldType.REQUIRED, "Template placeholders"),
                StandardField("video_id", StandardFieldType.REQUIRED, "Video identifier"),
                StandardField("objects", StandardFieldType.OPTIONAL, "Object annotations"),
            ],
            supported_formats=["json", "csv"],
        ))

        # AVA
        self.register(AnnotationStandardDef(
            name="AVA",
            version=StandardVersion(2, 2, 0),
            description="AVA atomic visual actions standard",
            fields=[
                StandardField("action_id", StandardFieldType.REQUIRED, "Action class ID"),
                StandardField("person_id", StandardFieldType.REQUIRED, "Person identifier"),
                StandardField("bbox", StandardFieldType.REQUIRED, "Bounding box [x1,y1,x2,y2]"),
                StandardField("timestamp", StandardFieldType.REQUIRED, "Central timestamp"),
                StandardField("confidence", StandardFieldType.OPTIONAL, "Detection confidence"),
            ],
            supported_formats=["csv", "json"],
        ))

    def register(self, standard: AnnotationStandardDef) -> None:
        """Register a new annotation standard.

        Args:
            standard: The standard definition to register.

        Raises:
            ValueError: If a standard with the same name and version exists.
        """
        key = f"{standard.name}@{standard.version}"
        if key in self._standards:
            raise ValueError(f"Standard {key} already registered")

        self._standards[key] = standard

        if standard.name not in self._versions:
            self._versions[standard.name] = []
        self._versions[standard.name].append(standard.version)
        self._versions[standard.name].sort(reverse=True)

    def get(self, name: str, version: Optional[str] = None) -> AnnotationStandardDef:
        """Get a standard definition.

        Args:
            name: Standard name.
            version: Specific version. If None, returns latest.

        Returns:
            The standard definition.

        Raises:
            ValueError: If the standard is not found.
        """
        if version:
            key = f"{name}@{StandardVersion.from_string(version)}"
            if key in self._standards:
                return self._standards[key]
            raise ValueError(f"Standard {key} not found")

        # Return latest version
        versions = self._versions.get(name, [])
        if not versions:
            raise ValueError(f"Standard {name} not found")
        latest = versions[0]
        return self._standards[f"{name}@{latest}"]

    def list_standards(self) -> List[str]:
        """List all registered standard names."""
        return sorted(self._versions.keys())

    def get_versions(self, name: str) -> List[str]:
        """Get all versions for a standard."""
        versions = self._versions.get(name, [])
        return [str(v) for v in versions]

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """Unregister a standard.

        Args:
            name: Standard name.
            version: Specific version. If None, unregisters all versions.

        Returns:
            True if any standard was removed.
        """
        if version:
            key = f"{name}@{StandardVersion.from_string(version)}"
            removed = key in self._standards
            if removed:
                del self._standards[key]
                sv = StandardVersion.from_string(version)
                if name in self._versions and sv in self._versions[name]:
                    self._versions[name].remove(sv)
                    self._versions[name].sort(reverse=True)
                    # If no versions left, clean up
                    if not self._versions[name]:
                        del self._versions[name]
            return removed

        removed = False
        keys_to_remove = [k for k in self._standards if k.startswith(f"{name}@")]
        for key in keys_to_remove:
            del self._standards[key]
            removed = True
        if name in self._versions:
            del self._versions[name]
        return removed

    def validate_data(self, name: str, data: Dict[str, Any], version: Optional[str] = None) -> List[str]:
        """Validate data against a standard.

        Args:
            name: Standard name.
            data: Data to validate.
            version: Specific version. If None, uses latest.

        Returns:
            List of validation error messages. Empty if valid.
        """
        try:
            standard = self.get(name, version)
        except ValueError as e:
            return [str(e)]

        errors: List[str] = []
        required_fields = {f.name for f in standard.fields if f.field_type == StandardFieldType.REQUIRED}

        for field_name in required_fields:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        return errors

    def check_compliance(self, name: str, data: Dict[str, Any], version: Optional[str] = None) -> Dict[str, Any]:
        """Check compliance of data against a standard.

        Args:
            name: Standard name.
            data: Data to validate.
            version: Specific version. If None, uses latest.

        Returns:
            Compliance report dictionary.
        """
        errors = self.validate_data(name, data, version)
        try:
            standard = self.get(name, version)
        except ValueError:
            return {
                "compliant": False,
                "standard": name,
                "version": version,
                "errors": errors,
                "field_coverage": 0.0,
            }

        all_fields = {f.name for f in standard.fields}
        present_fields = {f for f in all_fields if f in data and data[f] is not None}
        coverage = len(present_fields) / len(all_fields) if all_fields else 1.0

        return {
            "compliant": len(errors) == 0,
            "standard": name,
            "version": str(standard.version),
            "errors": errors,
            "field_coverage": coverage,
            "present_fields": sorted(present_fields),
            "missing_fields": sorted(all_fields - present_fields),
        }

    def convert_between_standards(
        self,
        source_name: str,
        target_name: str,
        data: Dict[str, Any],
        source_version: Optional[str] = None,
        target_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert data between annotation standards.

        Args:
            source_name: Source standard name.
            target_name: Target standard name.
            data: Data to convert.
            source_version: Source version.
            target_version: Target version.

        Returns:
            Converted data.
        """
        source = self.get(source_name, source_version)
        target = self.get(target_name, target_version)

        result: Dict[str, Any] = {
            "_converted_from": source_name,
            "_converted_to": target_name,
            "_source_version": str(source.version),
            "_target_version": str(target.version),
        }

        # Map common fields
        source_fields = {f.name for f in source.fields}
        target_fields = {f.name for f in target.fields}
        common_fields = source_fields & target_fields

        for common_field in common_fields:
            if common_field in data:
                result[common_field] = data[common_field]

        for target_field in target.fields:
            if target_field.name not in result and target_field.default is not None:
                result[target_field.name] = target_field.default

        # Also copy fields from source data that exist in source but not in target
        for field_name in data:
            if field_name not in result and field_name not in ("_converted_from", "_converted_to", "_source_version", "_target_version"):
                result[field_name] = data[field_name]

        return result

    def get_standard_names(self) -> Set[str]:
        """Get all unique standard names."""
        return set(self._versions.keys())


__all__ = [
    "AnnotationStandardDef",
    "StandardField",
    "StandardFieldType",
    "StandardRegistry",
    "StandardVersion",
]
