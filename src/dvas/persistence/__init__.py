"""Persistence layer for DVAS.

Provides SQLite-backed metadata indexing, full-text search,
and data versioning on top of the file-based annotation storage.
"""

from dvas.persistence.indexed_store import (
    AnnotationIndex,
    AnnotationQuery,
    IndexStore,
    IndexStoreConfig,
    SearchResult,
    VersionInfo,
)

__all__ = [
    "AnnotationIndex",
    "AnnotationQuery",
    "IndexStore",
    "IndexStoreConfig",
    "SearchResult",
    "VersionInfo",
]
