"""Legacy AnnotationStore for backward compatibility.

This module provides a backward-compatible wrapper around the new
backend architecture. Existing code using AnnotationStore will continue
to work without modification.

New code should use the backend classes directly:
    from dvas.persistence.backends import LocalFSBackend, SQLiteBackend
    from dvas.persistence import IndexManager
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

from dvas.data.schemas import Annotation
from dvas.persistence.backends import LocalFSBackend, LocalFSConfig
from dvas.persistence.backends.base import QueryFilter
from dvas.persistence.index_manager import IndexManager
from dvas.persistence.indexed_store import IndexStore, IndexStoreConfig
from dvas.utils.logging import get_logger

if TYPE_CHECKING:
    from dvas.persistence.backends import SQLiteBackend

logger = get_logger(__name__)


class AnnotationStore:
    """File-based annotation storage with optional SQLite indexing.

    This is the legacy interface for backward compatibility.
    It wraps the new backend architecture internally.

    Provides file-based storage with automatic indexing for fast queries
    and full-text search.
    """

    def __init__(
        self,
        root_path: Optional[Path] = None,
        enable_index: bool = True,
    ):
        """Initialize the annotation store.

        Args:
            root_path: Root directory for storage
            enable_index: Whether to enable SQLite indexing
        """
        from dvas.config import settings

        self.root_path = Path(root_path or settings.DATA_ROOT / "annotations")
        self._enable_index = enable_index

        # Create storage backend
        storage_config = LocalFSConfig(root_path=self.root_path)
        self._storage = LocalFSBackend(storage_config)
        self._storage.open()

        # Create index backend (lazy initialization)
        self._index_manager: Optional[IndexManager] = None
        self._legacy_index: Optional[IndexStore] = None

    @property
    def index_store(self) -> Optional[IndexStore]:
        """Lazy-initialize the index store (legacy interface)."""
        if self._legacy_index is None and self._enable_index:
            config = IndexStoreConfig(
                db_path=self.root_path / "index.db",
                wal_mode=True,
                enable_fts=True,
                enable_versioning=True,
            )
            self._legacy_index = IndexStore(config)
            self._legacy_index.create_index()
        return self._legacy_index

    @property
    def index_manager(self) -> Optional[IndexManager]:
        """Lazy-initialize the index manager (new interface)."""
        if self._index_manager is None and self._enable_index:
            # Import here to avoid circular imports
            from dvas.persistence.backends import SQLiteBackend, SQLiteConfig

            config = SQLiteConfig(db_path=self.root_path / "index.db")
            metadata_backend = SQLiteBackend(config)
            metadata_backend.open()
            self._index_manager = IndexManager(metadata_backend)
        return self._index_manager

    def _get_storage_path(self, annotation_id: str, source: str = "model") -> Path:
        """Get storage path for an annotation."""
        return Path(self._storage.get_storage_path(annotation_id, source))

    def save(
        self,
        annotation: Annotation,
        source: str = "model",
        overwrite: bool = False,
    ) -> Path:
        """Save an annotation to storage and index."""
        storage_path = self._storage.save(annotation, source, overwrite)

        # Index the annotation
        if self._enable_index:
            try:
                if self.index_store:
                    self.index_store.index_annotation(annotation)
                if self.index_manager:
                    self.index_manager.index_annotation(annotation, storage_path)
            except Exception as e:
                logger.warning("index_annotation_failed", annotation_id=annotation.id, error=str(e))

        return Path(storage_path)

    def load(self, annotation_id: str, source: str = "model") -> Optional[Annotation]:
        """Load an annotation from storage."""
        return self._storage.load(annotation_id, source)

    def load_all(
        self,
        source: str = "model",
        video_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[Annotation]:
        """Load all annotations from a source as a generator."""
        # batch_size is kept for API compatibility but not used
        yield from self._storage.load_all(source, video_id)

    def _get_source_path(self, source: str) -> Path:
        """Get base path for a source."""
        return Path(self._storage._get_base_path(source))

    def export_to_jsonl(
        self,
        output_path: Path,
        source: str = "reviewed",
        format: str = "llava",
    ) -> int:
        """Export annotations to JSONL file."""
        return self._storage.export_to_jsonl(Path(output_path), source, format)

    def get_statistics(self) -> Dict:
        """Get storage statistics."""
        stats = self._storage.get_statistics()
        return {
            "gold": {"count": stats.by_source.get("gold", 0), "size_mb": 0},
            "model": {"count": stats.by_source.get("model", 0), "size_mb": 0},
            "reviewed": {"count": stats.by_source.get("reviewed", 0), "size_mb": 0},
        }

    def create_version(self, name: str) -> Path:
        """Create a versioned snapshot of reviewed annotations."""
        if self.index_store:
            return Path(self._storage.create_version(name))
        return self._storage.create_version(name)

    # -------------------------------------------------------------------
    # Indexed query methods (require enable_index=True)
    # -------------------------------------------------------------------

    def search(self, query_text: str, limit: int = 100) -> List[Dict]:
        """Full-text search annotations."""
        if not self._enable_index or not self.index_store:
            logger.warning("index_not_enabled")
            return []

        results = self.index_store.search(query_text, limit=limit)
        return [
            {
                "annotation": result.annotation,
                "score": result.score,
                "highlights": result.highlights,
            }
            for result in results
        ]

    def query_by_video(self, video_id: str, source: Optional[str] = None) -> List[Annotation]:
        """Query annotations by video ID."""
        if self._enable_index and self.index_store:
            from dvas.persistence.indexed_store import AnnotationQuery

            query = AnnotationQuery(video_id=video_id, source=source)
            results, _ = self.index_store.query(query)
            return results

        # Fallback to file-based search
        return list(self._storage.load_all(source=source, video_id=video_id))

    def query_by_quality(
        self, min_quality: float = 0.0, max_quality: float = 1.0, limit: int = 100
    ) -> List[Annotation]:
        """Query annotations by quality score."""
        if self._enable_index and self.index_store:
            from dvas.persistence.indexed_store import AnnotationQuery

            query = AnnotationQuery(
                min_quality=min_quality,
                max_quality=max_quality,
                limit=limit,
            )
            results, _ = self.index_store.query(query)
            return results

        # Fallback to file-based search
        annotations = []
        for ann in self._storage.load_all(source="model"):
            if ann.quality_score is not None and min_quality <= ann.quality_score <= max_quality:
                annotations.append(ann)
                if len(annotations) >= limit:
                    break
        return annotations

    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        if self._enable_index and self.index_store:
            return self.index_store.get_statistics()
        return {"index_enabled": False}

    def sync_index(self) -> Dict[str, int]:
        """Sync the index with the file store."""
        if not self._enable_index or not self.index_store:
            logger.warning("index_not_enabled")
            return {"synced": 0}

        return self.index_store.sync_from_store(self)

    def close(self) -> None:
        """Close the store and release resources."""
        if self._legacy_index:
            self._legacy_index.close()
            self._legacy_index = None

        if self._index_manager:
            self._index_manager.close()
            self._index_manager = None

        if self._storage:
            self._storage.close()
            self._storage = None
