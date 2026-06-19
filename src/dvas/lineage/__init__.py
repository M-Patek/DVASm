"""Lineage module — schema version management and data provenance tracking.

Public API:
    LineageTracker: Track annotation lifecycle and schema compatibility
    SchemaVersion: Enum of supported schema versions
    LineageStep: Single step in annotation provenance
    SchemaCompatibility: Result of compatibility check
"""

from dvas.lineage.lineage_tracker import (
    LineageStep,
    LineageTracker,
    SchemaCompatibility,
    SchemaVersion,
)

__all__ = [
    "LineageTracker",
    "LineageStep",
    "SchemaVersion",
    "SchemaCompatibility",
]
