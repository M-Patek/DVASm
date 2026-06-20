"""Tests for annotation standard management module.

Tests for AnnotationStandardDef, StandardRegistry, StandardVersion,
StandardField, and StandardFieldType.
"""

import time

import pytest

from dvas.governance.standards import (
    AnnotationStandardDef,
    StandardField,
    StandardFieldType,
    StandardRegistry,
    StandardVersion,
)


class TestStandardVersion:
    """Test StandardVersion dataclass."""

    def test_version_creation(self):
        """Test creating a version."""
        v = StandardVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_version_string(self):
        """Test version string representation."""
        v = StandardVersion(2, 0, 0)
        assert str(v) == "2.0.0"

    def test_version_from_string(self):
        """Test parsing version from string."""
        v = StandardVersion.from_string("1.5.2")
        assert v.major == 1
        assert v.minor == 5
        assert v.patch == 2

    def test_version_from_string_invalid(self):
        """Test parsing invalid version string."""
        with pytest.raises(ValueError):
            StandardVersion.from_string("invalid")

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = StandardVersion(1, 0, 0)
        v2 = StandardVersion(1, 1, 0)
        v3 = StandardVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 <= v1
        assert v2 >= v1
        assert v3 > v1


class TestStandardField:
    """Test StandardField dataclass."""

    def test_field_creation(self):
        """Test creating a standard field."""
        field = StandardField(
            name="verb",
            field_type=StandardFieldType.REQUIRED,
            description="Action verb",
        )
        assert field.name == "verb"
        assert field.field_type == StandardFieldType.REQUIRED
        assert field.description == "Action verb"

    def test_field_types(self):
        """Test field type enum values."""
        assert StandardFieldType.REQUIRED.value == "required"
        assert StandardFieldType.OPTIONAL.value == "optional"
        assert StandardFieldType.CONDITIONAL.value == "conditional"


class TestAnnotationStandardDef:
    """Test AnnotationStandardDef dataclass."""

    def test_standard_creation(self):
        """Test creating a standard definition."""
        standard = AnnotationStandardDef(
            name="Test Standard",
            version=StandardVersion(1, 0, 0),
            description="A test standard",
            fields=[
                StandardField("field1", StandardFieldType.REQUIRED),
            ],
            supported_formats=["json"],
        )
        assert standard.name == "Test Standard"
        assert str(standard.version) == "1.0.0"
        assert len(standard.fields) == 1

    def test_standard_to_dict(self):
        """Test converting standard to dict."""
        standard = AnnotationStandardDef(
            name="Test",
            version=StandardVersion(1, 0, 0),
            description="Test standard",
        )
        d = standard.to_dict()
        assert d["name"] == "Test"
        assert d["version"] == "1.0.0"
        assert d["deprecated"] is False

    def test_standard_deprecated(self):
        """Test deprecated standard."""
        standard = AnnotationStandardDef(
            name="Old Standard",
            version=StandardVersion(1, 0, 0),
            description="Old version",
            deprecated=True,
            replacement="New Standard",
        )
        assert standard.deprecated is True
        assert standard.replacement == "New Standard"


class TestStandardRegistry:
    """Test StandardRegistry class."""

    def test_init_with_builtins(self):
        """Test initialization with built-in standards."""
        registry = StandardRegistry()
        standards = registry.list_standards()
        assert "EPIC-KITCHENS" in standards
        assert "Ego4D" in standards
        assert "Open X-Embodiment" in standards
        assert "Something-Something" in standards
        assert "AVA" in standards

    def test_register_standard(self):
        """Test registering a custom standard."""
        registry = StandardRegistry()
        standard = AnnotationStandardDef(
            name="Custom",
            version=StandardVersion(1, 0, 0),
            description="Custom standard",
        )
        registry.register(standard)
        assert "Custom" in registry.list_standards()

    def test_register_duplicate(self):
        """Test registering duplicate standard."""
        registry = StandardRegistry()
        standard = AnnotationStandardDef(
            name="Duplicate",
            version=StandardVersion(1, 0, 0),
            description="Duplicate",
        )
        registry.register(standard)
        with pytest.raises(ValueError):
            registry.register(standard)

    def test_get_standard(self):
        """Test getting a standard."""
        registry = StandardRegistry()
        standard = registry.get("EPIC-KITCHENS")
        assert standard.name == "EPIC-KITCHENS"
        assert len(standard.fields) > 0

    def test_get_standard_with_version(self):
        """Test getting a specific version."""
        registry = StandardRegistry()
        standard = registry.get("EPIC-KITCHENS", "2.0.0")
        assert standard.name == "EPIC-KITCHENS"
        assert str(standard.version) == "2.0.0"

    def test_get_standard_not_found(self):
        """Test getting non-existent standard."""
        registry = StandardRegistry()
        with pytest.raises(ValueError):
            registry.get("NonExistent")

    def test_get_versions(self):
        """Test getting versions for a standard."""
        registry = StandardRegistry()
        versions = registry.get_versions("EPIC-KITCHENS")
        assert "2.0.0" in versions

    def test_unregister_standard(self):
        """Test unregistering a standard."""
        registry = StandardRegistry()
        registry.register(AnnotationStandardDef(
            name="Temp",
            version=StandardVersion(1, 0, 0),
            description="Temp",
        ))
        assert registry.unregister("Temp") is True
        assert "Temp" not in registry.list_standards()

    def test_unregister_not_found(self):
        """Test unregistering non-existent standard."""
        registry = StandardRegistry()
        assert registry.unregister("NonExistent") is False

    def test_validate_data_valid(self):
        """Test validating valid data."""
        registry = StandardRegistry()
        data = {
            "verb": "cut",
            "noun": "vegetable",
            "hand": "right",
            "start_time": 0.0,
            "end_time": 1.0,
        }
        errors = registry.validate_data("EPIC-KITCHENS", data)
        assert len(errors) == 0

    def test_validate_data_missing_fields(self):
        """Test validating data with missing fields."""
        registry = StandardRegistry()
        data = {
            "verb": "cut",
            # Missing noun, hand, start_time, end_time
        }
        errors = registry.validate_data("EPIC-KITCHENS", data)
        assert len(errors) > 0
        assert any("noun" in e for e in errors)
        assert any("hand" in e for e in errors)

    def test_validate_data_not_found(self):
        """Test validating against non-existent standard."""
        registry = StandardRegistry()
        errors = registry.validate_data("NonExistent", {})
        assert len(errors) == 1

    def test_check_compliance(self):
        """Test compliance checking."""
        registry = StandardRegistry()
        data = {
            "verb": "cut",
            "noun": "vegetable",
            "hand": "right",
            "start_time": 0.0,
            "end_time": 1.0,
            "narration": "optional field",
        }
        report = registry.check_compliance("EPIC-KITCHENS", data)
        assert report["compliant"] is True
        assert report["standard"] == "EPIC-KITCHENS"
        assert report["field_coverage"] > 0.0

    def test_check_compliance_nonexistent(self):
        """Test compliance for non-existent standard."""
        registry = StandardRegistry()
        report = registry.check_compliance("NonExistent", {})
        assert report["compliant"] is False

    def test_convert_between_standards(self):
        """Test converting between standards."""
        registry = StandardRegistry()
        data = {
            "verb": "cut",
            "noun": "vegetable",
            "hand": "right",
            "start_time": 0.0,
            "end_time": 1.0,
        }
        result = registry.convert_between_standards(
            "EPIC-KITCHENS", "Ego4D", data,
        )
        assert result["_converted_from"] == "EPIC-KITCHENS"
        assert result["_converted_to"] == "Ego4D"
        assert "verb" in result

    def test_get_standard_names(self):
        """Test getting all standard names."""
        registry = StandardRegistry()
        names = registry.get_standard_names()
        assert "EPIC-KITCHENS" in names
        assert "Ego4D" in names

    def test_built_in_standards_fields(self):
        """Test that built-in standards have expected fields."""
        registry = StandardRegistry()

        epic = registry.get("EPIC-KITCHENS")
        field_names = {f.name for f in epic.fields}
        assert "verb" in field_names
        assert "noun" in field_names
        assert "hand" in field_names

        ego4d = registry.get("Ego4D")
        field_names = {f.name for f in ego4d.fields}
        assert "narration" in field_names

        openx = registry.get("Open X-Embodiment")
        field_names = {f.name for f in openx.fields}
        assert "language_instruction" in field_names

        ssv = registry.get("Something-Something")
        field_names = {f.name for f in ssv.fields}
        assert "template" in field_names

        ava = registry.get("AVA")
        field_names = {f.name for f in ava.fields}
        assert "action_id" in field_names
        assert "bbox" in field_names


class TestStandardRegistryEdgeCases:
    """Test edge cases for StandardRegistry."""

    def test_empty_registry(self):
        """Test with empty registry."""
        registry = StandardRegistry()
        # Built-ins are registered by default
        assert len(registry.list_standards()) == 5

    def test_multiple_versions(self):
        """Test registering multiple versions."""
        registry = StandardRegistry()
        registry.register(AnnotationStandardDef(
            name="MultiVersion",
            version=StandardVersion(1, 0, 0),
            description="v1",
        ))
        registry.register(AnnotationStandardDef(
            name="MultiVersion",
            version=StandardVersion(2, 0, 0),
            description="v2",
        ))
        versions = registry.get_versions("MultiVersion")
        assert len(versions) == 2
        assert "2.0.0" in versions
        assert "1.0.0" in versions

    def test_unregister_version(self):
        """Test unregistering specific version."""
        registry = StandardRegistry()
        registry.register(AnnotationStandardDef(
            name="Versioned",
            version=StandardVersion(1, 0, 0),
            description="v1",
        ))
        assert registry.unregister("Versioned", "1.0.0") is True
        assert "Versioned" not in registry.list_standards()

    def test_get_latest_version(self):
        """Test getting latest version."""
        registry = StandardRegistry()
        registry.register(AnnotationStandardDef(
            name="Latest",
            version=StandardVersion(1, 0, 0),
            description="v1",
        ))
        registry.register(AnnotationStandardDef(
            name="Latest",
            version=StandardVersion(2, 0, 0),
            description="v2",
        ))
        latest = registry.get("Latest")
        assert str(latest.version) == "2.0.0"
