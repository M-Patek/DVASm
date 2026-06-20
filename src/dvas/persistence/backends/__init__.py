"""Persistence backends for DVAS annotation storage.

Provides pluggable storage backends for different deployment scenarios:
- LocalFS: File-based storage for single-node deployments
- SQLite: Metadata indexing for fast queries
- PostgreSQL: Production metadata backend
- S3: Object storage for cloud deployments
"""

from dvas.persistence.backends.base import (
    Backend,
    BackendConfig,
    BackendStats,
    BackendType,
    DiffResult,
    IndexEntry,
    MetadataBackend,
    QueryFilter,
    SearchResult,
    StorageBackend,
    VersionInfo,
)
from dvas.persistence.backends.localfs import LocalFSBackend, LocalFSConfig
from dvas.persistence.backends.sqlite import SQLiteBackend, SQLiteConfig

__all__ = [
    # Base
    "Backend",
    "BackendConfig",
    "BackendStats",
    "BackendType",
    "MetadataBackend",
    "StorageBackend",
    "IndexEntry",
    "QueryFilter",
    "SearchResult",
    "VersionInfo",
    "DiffResult",
    # Storage backends
    "LocalFSBackend",
    "LocalFSConfig",
    # Metadata backends
    "SQLiteBackend",
    "SQLiteConfig",
]

# Optional backends (require additional dependencies)
try:
    from dvas.persistence.backends.postgresql import (  # noqa: F401
        PostgreSQLBackend,
        PostgreSQLConfig,
    )

    __all__.extend(["PostgreSQLBackend", "PostgreSQLConfig"])
except ImportError:
    pass

try:
    from dvas.persistence.backends.s3 import S3Backend, S3Config  # noqa: F401

    __all__.extend(["S3Backend", "S3Config"])
except ImportError:
    pass
