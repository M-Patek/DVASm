"""Tests for prompt versioning and lineage."""

import pytest

from dvas.prompts.versioning import (
    PromptVersion,
    PromptVersionManager,
    VersionError,
    compute_diff,
    is_compatible,
    suggest_version_bump,
)


class TestPromptVersion:
    """Test suite for PromptVersion."""

    def test_parse_simple_version(self):
        """Test parsing a simple version string."""
        v = PromptVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_with_v_prefix(self):
        """Test parsing version with 'v' prefix."""
        v = PromptVersion.parse("v2.0.0")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_prerelease(self):
        """Test parsing version with prerelease."""
        v = PromptVersion.parse("1.0.0-alpha")
        assert v.major == 1
        assert v.prerelease == "alpha"

    def test_parse_invalid_version(self):
        """Test parsing invalid version string."""
        with pytest.raises(VersionError):
            PromptVersion.parse("invalid")

    def test_parse_negative_component(self):
        """Test parsing version with negative component."""
        with pytest.raises(VersionError):
            PromptVersion.parse("-1.0.0")

    def test_version_comparison(self):
        """Test version comparison operators."""
        v1 = PromptVersion(1, 0, 0)
        v2 = PromptVersion(1, 1, 0)
        v3 = PromptVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 < v3
        assert v2 > v1
        assert v3 >= v1
        assert v1 <= v2

    def test_version_equality(self):
        """Test version equality."""
        v1 = PromptVersion(1, 2, 3)
        v2 = PromptVersion(1, 2, 3)
        assert v1 == v2

    def test_version_string(self):
        """Test version string representation."""
        v = PromptVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_version_string_with_prerelease(self):
        """Test version string with prerelease."""
        v = PromptVersion(1, 0, 0, prerelease="beta")
        assert str(v) == "1.0.0-beta"

    def test_bump_major(self):
        """Test bumping major version."""
        v = PromptVersion(1, 2, 3)
        bumped = v.bump_major()
        assert bumped.major == 2
        assert bumped.minor == 0
        assert bumped.patch == 0

    def test_bump_minor(self):
        """Test bumping minor version."""
        v = PromptVersion(1, 2, 3)
        bumped = v.bump_minor()
        assert bumped.major == 1
        assert bumped.minor == 3
        assert bumped.patch == 0

    def test_bump_patch(self):
        """Test bumping patch version."""
        v = PromptVersion(1, 2, 3)
        bumped = v.bump_patch()
        assert bumped.major == 1
        assert bumped.minor == 2
        assert bumped.patch == 4


class TestVersionCompatibility:
    """Test suite for version compatibility."""

    def test_exact_match(self):
        """Test exact version compatibility."""
        v1 = PromptVersion(1, 0, 0)
        v2 = PromptVersion(1, 0, 0)
        assert is_compatible(v1, v2, "exact") is True

    def test_major_compatibility(self):
        """Test major version compatibility."""
        v1 = PromptVersion(1, 2, 0)
        v2 = PromptVersion(1, 5, 0)
        assert is_compatible(v1, v2, "major") is True

    def test_major_incompatibility(self):
        """Test major version incompatibility."""
        v1 = PromptVersion(2, 0, 0)
        v2 = PromptVersion(1, 0, 0)
        assert is_compatible(v1, v2, "major") is False

    def test_minor_compatibility(self):
        """Test minor version compatibility."""
        v1 = PromptVersion(1, 2, 5)
        v2 = PromptVersion(1, 2, 0)
        assert is_compatible(v1, v2, "minor") is True

    def test_minor_incompatibility(self):
        """Test minor version incompatibility."""
        v1 = PromptVersion(1, 3, 0)
        v2 = PromptVersion(1, 2, 0)
        assert is_compatible(v1, v2, "minor") is False


class TestVersionDiff:
    """Test suite for version diff functionality."""

    def test_compute_diff(self):
        """Test computing diff between two templates."""
        old = "Line 1\nLine 2\nLine 3"
        new_template = "Line 1\nLine 2 modified\nLine 3\nLine 4"

        diff = compute_diff(old, new_template, "v1", "v2")
        assert diff.old_version == "v1"
        assert diff.new_version == "v2"
        assert len(diff.added_lines) > 0 or len(diff.removed_lines) > 0
        assert 0 < diff.similarity_ratio < 1

    def test_identical_templates(self):
        """Test diff of identical templates."""
        template = "Same content\nSecond line"
        diff = compute_diff(template, template)
        assert diff.similarity_ratio == 1.0
        assert len(diff.added_lines) == 0
        assert len(diff.removed_lines) == 0

    def test_completely_different(self):
        """Test diff of completely different templates."""
        old = "Completely different"
        new_template = "Nothing in common here"
        diff = compute_diff(old, new_template)
        assert diff.similarity_ratio < 0.5


class TestSuggestVersionBump:
    """Test suite for version bump suggestions."""

    def test_patch_bump(self):
        """Test suggesting patch bump for minor change."""
        old = "Template with some content"
        new = "Template with some modified content"
        current = PromptVersion(1, 0, 0)
        suggested = suggest_version_bump(old, new, current)
        assert suggested == PromptVersion(1, 0, 1)

    def test_minor_bump(self):
        """Test suggesting minor bump for moderate change."""
        old = "A" * 50 + "common" + "B" * 50
        new_template = "X" * 50 + "common" + "Y" * 50
        current = PromptVersion(1, 0, 0)
        suggested = suggest_version_bump(old, new_template, current)
        # Should be minor bump (some similarity but significant change)
        assert suggested.major == 1
        assert suggested.patch == 0


class TestPromptVersionManager:
    """Test suite for PromptVersionManager."""

    def test_register_and_get_latest(self):
        """Test registering versions and getting latest."""
        manager = PromptVersionManager()
        manager.register_version("prompt_1", PromptVersion(1, 0, 0))
        manager.register_version("prompt_1", PromptVersion(1, 1, 0))
        manager.register_version("prompt_1", PromptVersion(2, 0, 0))

        latest = manager.get_latest("prompt_1")
        assert latest is not None
        assert latest.major == 2

    def test_list_versions(self):
        """Test listing all versions for a prompt."""
        manager = PromptVersionManager()
        manager.register_version("p1", PromptVersion(1, 0, 0))
        manager.register_version("p1", PromptVersion(1, 1, 0))

        versions = manager.list_versions("p1")
        assert len(versions) == 2
        assert versions[0] < versions[1]

    def test_is_latest(self):
        """Test checking if a version is the latest."""
        manager = PromptVersionManager()
        manager.register_version("p1", PromptVersion(1, 0, 0))

        assert manager.is_latest("p1", PromptVersion(1, 0, 0)) is True
        assert manager.is_latest("p1", PromptVersion(2, 0, 0)) is True

    def test_get_latest_nonexistent(self):
        """Test getting latest for non-existent prompt."""
        manager = PromptVersionManager()
        assert manager.get_latest("nonexistent") is None
