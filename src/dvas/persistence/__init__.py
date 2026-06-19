"""Persistence layer for DVAS.

Provides pluggable storage backends, metadata indexing, and data management tools.

Architecture:
    Backends (storage):
        - LocalFSBackend: File-based storage
        - S3Backend: S3/MinIO object storage

    Backends (metadata):
        - SQLiteBackend: SQLite-based indexing
        - PostgreSQLBackend: PostgreSQL indexing (optional)

    Tools:
        - IndexManager: Unified index management
        - MigrationManager: Schema migrations
        - DiffTool: Annotation comparison
        - RollbackManager: Version rollback
        - BackupManager: Backup/restore
        - CompactionManager: Storage compaction

Usage::

    from dvas.persistence import AnnotationStore
    from dvas.persistence.backends import LocalFSBackend, SQLiteBackend

    # Create backends
    storage = LocalFSBackend()
    storage.open()

    metadata = SQLiteBackend()
    metadata.open()

    # Use unified store
    store = AnnotationStore(storage_backend=storage, metadata_backend=metadata)
"""

# Backends
from dvas.persistence.backends import (
    Backend,
    BackendConfig,
    BackendStats,
    IndexEntry,
    LocalFSBackend,
    LocalFSConfig,
    MetadataBackend,
    PostgreSQLBackend,
    PostgreSQLConfig,
    QueryFilter,
    S3Backend,
    S3Config,
    SQLiteBackend,
    SQLiteConfig,
    StorageBackend,
    VersionInfo,
)

# Core store (backward compatible)
from dvas.persistence.indexed_store import (
    AnnotationIndex,
    AnnotationQuery,
    IndexStore,
    IndexStoreConfig,
    SearchResult,
    VersionInfo as IndexedVersionInfo,
)

# Tools
from dvas.persistence.backup_restore import BackupManager, BackupError
from dvas.persistence.compaction import CompactionManager, CompactionError
from dvas.persistence.diff_tool import (
    AnnotationDiff,
    compute_annotation_diff,
    diff_annotations,
    format_diff,
)
from dvas.persistence.index_manager import IndexManager, IndexStats
from dvas.persistence.migrations import (
    DVAS_MIGRATIONS,
    Migration,
    MigrationManager,
    SQLiteMigrationBackend,
    create_migration_manager,
)
from dvas.persistence.rollback_tool import RollbackManager, RollbackError

# Legacy compatibility
from dvas.persistence.legacy_store import AnnotationStore

__all__ = [
    # Backends - Base
    "Backend",
    "BackendConfig",
    "BackendStats",
    "StorageBackend",
    "MetadataBackend",
    "IndexEntry",
    "QueryFilter",
    "VersionInfo",
    # Backends - Storage
    "LocalFSBackend",
    "LocalFSConfig",
    "S3Backend",
    "S3Config",
    # Backends - Metadata
    "SQLiteBackend",
    "SQLiteConfig",
    "PostgreSQLBackend",
    "PostgreSQLConfig",
    # Legacy indexed store
    "AnnotationIndex",
    "AnnotationQuery",
    "IndexStore",
    "IndexStoreConfig",
    "SearchResult",
    "IndexedVersionInfo",
    # Tools
    "IndexManager",
    "IndexStats",
    "Migration",
    "MigrationManager",
    "SQLiteMigrationBackend",
    "DVAS_MIGRATIONS",
    "create_migration_manager",
    "compute_annotation_diff",
    "diff_annotations",
    "format_diff",
    "AnnotationDiff",
    "RollbackManager",
    "RollbackError",
    "BackupManager",
    "BackupError",
    "CompactionManager",
    "CompactionError",
    # Legacy compatibility
    "AnnotationStore",
]
