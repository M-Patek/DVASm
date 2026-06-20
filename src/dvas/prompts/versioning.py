"""Prompt versioning with semantic versioning support.

Provides semantic versioning, version comparison, compatibility checks,
and diff functionality for prompt templates.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class VersionError(Exception):
    """Raised when version operations fail."""

    pass


@dataclass(frozen=True)
class PromptVersion:
    """Semantic version for prompt templates.

    Follows SemVer: major.minor.patch
    - major: breaking changes to prompt behavior
    - minor: new features/capabilities
    - patch: fixes, clarifications, non-behavioral changes
    """

    major: int
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise VersionError("Version components must be non-negative")

    @classmethod
    def parse(cls, version_str: str) -> "PromptVersion":
        """Parse a version string."""
        version_str = version_str.lstrip("v")

        prerelease: Optional[str] = None
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)

        parts = version_str.split(".")
        if len(parts) < 1 or len(parts) > 3:
            raise VersionError(f"Invalid version string: {version_str}")

        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            raise VersionError(f"Invalid version string: {version_str}")

        return cls(major=major, minor=minor, patch=patch, prerelease=prerelease)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        return base

    def __lt__(self, other: "PromptVersion") -> bool:
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)
        if self_tuple != other_tuple:
            return self_tuple < other_tuple
        # Prerelease versions are lower than release versions
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False

    def __le__(self, other: "PromptVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "PromptVersion") -> bool:
        return other < self

    def __ge__(self, other: "PromptVersion") -> bool:
        return self == other or self > other

    def bump_major(self) -> "PromptVersion":
        """Bump major version."""
        return PromptVersion(major=self.major + 1, minor=0, patch=0)

    def bump_minor(self) -> "PromptVersion":
        """Bump minor version."""
        return PromptVersion(major=self.major, minor=self.minor + 1, patch=0)

    def bump_patch(self) -> "PromptVersion":
        """Bump patch version."""
        return PromptVersion(major=self.major, minor=self.minor, patch=self.patch + 1)


def is_compatible(
    version: PromptVersion,
    target: PromptVersion,
    compatibility: str = "major",
) -> bool:
    """Check if two versions are compatible.

    Args:
        version: The version to check.
        target: The target version to compare against.
        compatibility: "major", "minor", or "exact".

    Returns:
        True if versions are compatible.
    """
    if compatibility == "exact":
        return version == target
    if compatibility == "major":
        return version.major == target.major and version >= target
    if compatibility == "minor":
        return version.major == target.major and version.minor == target.minor and version >= target
    return False


@dataclass
class VersionDiff:
    """Diff result between two prompt versions."""

    old_version: str
    new_version: str
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    unchanged_lines: List[str] = field(default_factory=list)
    similarity_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "unchanged_lines": self.unchanged_lines,
            "similarity_ratio": self.similarity_ratio,
        }


def compute_diff(
    old_template: str,
    new_template: str,
    old_version: str = "",
    new_version: str = "",
) -> VersionDiff:
    """Compute diff between two prompt templates versions.

    Args:
        old_template: The old template text.
        new_template: The new template text.
        old_version: Optional old version label.
        new_version: Optional new version label.

    Returns:
        VersionDiff with detailed comparison.
    """
    old_lines = old_template.splitlines(keepends=True)
    new_lines = new_template.splitlines(keepends=True)

    differ = difflib.Differ()
    diff = list(differ.compare(old_lines, new_lines))

    added: List[str] = []
    removed: List[str] = []
    unchanged: List[str] = []

    for line in diff:
        if line.startswith("+ "):
            added.append(line[2:].rstrip("\n"))
        elif line.startswith("- "):
            removed.append(line[2:].rstrip("\n"))
        elif line.startswith("  "):
            unchanged.append(line[2:].rstrip("\n"))

    sm = difflib.SequenceMatcher(None, old_template, new_template)
    similarity = sm.ratio()

    return VersionDiff(
        old_version=old_version,
        new_version=new_version,
        added_lines=added,
        removed_lines=removed,
        unchanged_lines=unchanged,
        similarity_ratio=similarity,
    )


def suggest_version_bump(
    old_template: str,
    new_template: str,
    current_version: PromptVersion,
) -> PromptVersion:
    """Suggest a version bump based on template changes.

    Analyzes the diff to determine if the change is major, minor, or patch.

    Returns:
        Suggested new version.
    """
    diff = compute_diff(old_template, new_template)

    if diff.similarity_ratio > 0.95:
        # Very minor change
        return current_version.bump_patch()
    elif diff.similarity_ratio > 0.8:
        # Moderate change
        return current_version.bump_minor()
    else:
        # Significant change
        return current_version.bump_major()


class PromptVersionManager:
    """Manages versioning for a collection of prompts."""

    def __init__(self) -> None:
        self._versions: dict = {}

    def register_version(self, prompt_id: str, version: PromptVersion) -> None:
        """Register a version for a prompt."""
        if prompt_id not in self._versions:
            self._versions[prompt_id] = []
        self._versions[prompt_id].append(version)
        self._versions[prompt_id].sort()

    def get_latest(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get the latest version for a prompt."""
        versions = self._versions.get(prompt_id, [])
        if not versions:
            return None
        return versions[-1]

    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions for a prompt."""
        return self._versions.get(prompt_id, []).copy()

    def is_latest(self, prompt_id: str, version: PromptVersion) -> bool:
        """Check if a version is the latest for a prompt."""
        latest = self.get_latest(prompt_id)
        if latest is None:
            return True
        return version >= latest
