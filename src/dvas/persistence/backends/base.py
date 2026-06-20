"""Abstract base class for annotation storage backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from dvas.data.schemas import Annotation


class BackendType(str, Enum):
    """Types of storage backends."""

    LOCAL_FS = "local_fs"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    S3 = "s3"
    MINIO = "minio"


@dataclass
class BackendConfig:
    """Base configuration for storage backends."""

    backend_type: BackendType = BackendType.LOCAL_FS
    name: str = "default"
    read_only: bool = False
    compression: Optional[str] = None  # gzip, zstd, etc.
    encryption_key: Optional[str] = None  # Path to key file or key itself

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "backend_type": self.backend_type.value,
            "name": self.name,
            "read_only": self.read_only,
            "compression": self.compression,
        }


@dataclass
class IndexEntry:
    """Index entry for an annotation."""

    id: str
    video_id: str
    video_path: str
    source: str
    model_version: Optional[str]
    quality_score: Optional[float]
    created_at: datetime
    updated_at: Optional[datetime]
    num_segments: int
    total_duration: float
    tags: List[str]
    parent_id: Optional[str] = None
    content_hash: Optional[str] = None  # For integrity checking
    storage_path: Optional[str] = None  # Path in storage backend


@dataclass
class QueryFilter:
    """Query filter for searching annotations."""

    video_id: Optional[str] = None
    source: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    dataset_version: Optional[str] = None
    min_quality: Optional[float] = None
    max_quality: Optional[float] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    has_parent: Optional[bool] = None
    text_search: Optional[str] = None
    order_by: str = "created_at"
    order_desc: bool = True
    limit: int = 100
    offset: int = 0


@dataclass
class SearchResult:
    """Result from a search query."""

    annotation: Annotation
    score: float = 0.0
    highlights: List[str] = None

    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


@dataclass
class VersionInfo:
    """Information about a stored version."""

    id: str
    name: str
    description: str
    created_at: datetime
    annotation_count: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BackendStats:
    """Statistics for a backend."""

    total_annotations: int = 0
    by_source: Dict[str, int] = None
    by_model: Dict[str, int] = None
    storage_size_bytes: int = 0
    index_size_bytes: int = 0
    last_modified: Optional[datetime] = None

    def __post_init__(self):
        if self.by_source is None:
            self.by_source = {}
        if self.by_model is None:
            self.by_model = {}


@dataclass
class DiffResult:
    """Result of comparing two annotations or versions."""

    annotation_id: str
    field_changes: Dict[str, Tuple[Any, Any]]  # field -> (old, new)
    segments_added: List[int]  # indices of added segments
    segments_removed: List[int]  # indices of removed segments
    segments_modified: List[int]  # indices of modified segments
    unchanged: bool


class Backend(ABC):
    """Abstract base class for storage backends.

    All storage backends must implement this interface to ensure
    interoperability and consistent behavior across different
    storage systems.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._closed = False

    @property
    def is_read_only(self) -> bool:
        """Check if backend is read-only."""
        return self.config.read_only

    @abstractmethod
    def open(self) -> None:
        """Open the backend and initialize resources."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""
        pass

    def ensure_open(self) -> None:
        """Ensure the backend is open."""
        if self._closed:
            raise RuntimeError(f"Backend {self.config.name} is closed")

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """Check backend health.

        Returns:
            Tuple of (is_healthy, message)
        """
        pass


class StorageBackend(Backend):
    """Abstract base class for storage backends that store annotation data.

    This includes file-based backends (LocalFS, S3) that store the actual
    annotation JSON files.
    """

    @abstractmethod
    def save(
        self,
        annotation: Annotation,
        source: str = "model",
        overwrite: bool = False,
    ) -> str:
        """Save an annotation to storage.

        Args:
            annotation: The annotation to save
            source: Source category (gold, model, reviewed)
            overwrite: Whether to overwrite existing annotation

        Returns:
            Storage path/identifier
        """
        pass

    @abstractmethod
    def load(self, annotation_id: str, source: str = "model") -> Optional[Annotation]:
        """Load an annotation from storage.

        Args:
            annotation_id: The annotation ID
            source: Source category to search

        Returns:
            The annotation or None if not found
        """
        pass

    @abstractmethod
    def load_all(
        self,
        source: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Iterator[Annotation]:
        """Load all annotations as a generator.

        Args:
            source: Optional source filter
            video_id: Optional video ID filter

        Yields:
            Annotation objects
        """
        pass

    @abstractmethod
    def delete(self, annotation_id: str, source: str = "model") -> bool:
        """Delete an annotation from storage.

        Args:
            annotation_id: The annotation ID to delete
            source: Source category

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, annotation_id: str, source: str = "model") -> bool:
        """Check if an annotation exists."""
        pass

    @abstractmethod
    def get_storage_path(self, annotation_id: str, source: str = "model") -> str:
        """Get the storage path for an annotation."""
        pass

    @abstractmethod
    def get_statistics(self) -> BackendStats:
        """Get storage statistics."""
        pass


class MetadataBackend(Backend):
    """Abstract base class for metadata/index backends.

    This includes backends like SQLite and PostgreSQL that provide
    fast querying and indexing capabilities.
    """

    @abstractmethod
    def index_annotation(self, annotation: Annotation, storage_path: Optional[str] = None) -> None:
        """Index an annotation for fast querying."""
        pass

    @abstractmethod
    def get(self, annotation_id: str) -> Optional[IndexEntry]:
        """Get index entry for an annotation."""
        pass

    @abstractmethod
    def query(self, query_filter: QueryFilter) -> Tuple[List[IndexEntry], int]:
        """Query annotations using filters.

        Returns:
            Tuple of (entries, total_count)
        """
        pass

    @abstractmethod
    def search(self, query_text: str, limit: int = 100) -> List[SearchResult]:
        """Full-text search annotations."""
        pass

    @abstractmethod
    def delete_index(self, annotation_id: str) -> bool:
        """Delete an annotation from the index."""
        pass

    @abstractmethod
    def create_version(
        self,
        name: str,
        description: str = "",
        annotation_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a versioned snapshot."""
        pass

    @abstractmethod
    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Get version information."""
        pass

    @abstractmethod
    def list_versions(self) -> List[VersionInfo]:
        """List all versions."""
        pass

    @abstractmethod
    def restore_version(self, version_id: str) -> List[Dict[str, Any]]:
        """Restore annotations from a version.

        Returns:
            List of annotation data dictionaries
        """
        pass

    @abstractmethod
    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        pass

    @abstractmethod
    def diff_versions(
        self,
        version_id1: str,
        version_id2: str,
        annotation_id: Optional[str] = None,
    ) -> List[DiffResult]:
        """Compare two versions.

        Args:
            version_id1: First version to compare
            version_id2: Second version to compare
            annotation_id: Optional specific annotation to compare

        Returns:
            List of diff results
        """
        pass

    @abstractmethod
    def get_statistics(self) -> BackendStats:
        """Get index statistics."""
        pass

    @abstractmethod
    def vacuum(self) -> None:
        """Reclaim unused space."""
        pass

    @abstractmethod
    def optimize(self) -> None:
        """Optimize the index."""
        pass

    @abstractmethod
    def reindex(self) -> None:
        """Rebuild all indexes."""
        pass

    @abstractmethod
    def backup(self, destination: Path) -> Path:
        """Create a backup of the index."""
        pass

    @abstractmethod
    def restore(self, source: Path) -> None:
        """Restore from a backup."""
        pass


class MigrationBackend(ABC):
    """Interface for schema migration support."""

    @abstractmethod
    def get_schema_version(self) -> str:
        """Get current schema version."""
        pass

    @abstractmethod
    def migrate(self, target_version: str) -> bool:
        """Migrate to target schema version."""
        pass

    @abstractmethod
    def list_migrations(self) -> List[Dict[str, Any]]:
        """List available migrations."""
        pass
