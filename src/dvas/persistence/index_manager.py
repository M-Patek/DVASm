"""Index manager for annotation metadata.

Provides unified access to multiple index types:
- Annotation index (by metadata fields)
- Video hash index (content-based dedup)
- Frame hash index (frame-level dedup)
- Prompt version index
- Model version index
- Dataset version index
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import QueryFilter
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexStats:
    """Statistics for all indexes."""

    annotations_count: int = 0
    video_hashes_count: int = 0
    frame_hashes_count: int = 0
    model_versions: Dict[str, int] = None
    prompt_versions: Dict[str, int] = None
    dataset_versions: Dict[str, int] = None

    def __post_init__(self):
        if self.model_versions is None:
            self.model_versions = {}
        if self.prompt_versions is None:
            self.prompt_versions = {}
        if self.dataset_versions is None:
            self.dataset_versions = {}


class IndexManager:
    """Manages all annotation-related indexes.

    Provides a unified interface for:
    - Annotation metadata indexing
    - Content hash indexing (video/frame deduplication)
    - Version tracking

    Usage::

        manager = IndexManager(backend)
        manager.index_annotation(annotation)
        manager.index_video_hash(video_id, video_path, hash)

        # Query by version
        results = manager.query_by_model_version("gpt-4v-2024")
    """

    def __init__(self, backend):
        """Initialize with a metadata backend.

        Args:
            backend: MetadataBackend instance (SQLite, PostgreSQL, etc.)
        """
        self.backend = backend

    # -------------------------------------------------------------------------
    # Annotation Index
    # -------------------------------------------------------------------------

    def index_annotation(self, annotation: Annotation, storage_path: Optional[str] = None) -> None:
        """Index an annotation."""
        self.backend.index_annotation(annotation, storage_path)
        logger.debug("index_manager_annotation_indexed", id=annotation.id)

    def get_annotation(self, annotation_id: str) -> Optional[Dict[str, Any]]:
        """Get annotation index entry."""
        entry = self.backend.get(annotation_id)
        return entry.__dict__ if entry else None

    def query_annotations(self, query_filter: QueryFilter) -> Tuple[List[Dict], int]:
        """Query annotations."""
        entries, total = self.backend.query(query_filter)
        return [e.__dict__ for e in entries], total

    def search_annotations(self, query_text: str, limit: int = 100) -> List[Dict]:
        """Full-text search annotations."""
        results = self.backend.search(query_text, limit)
        return [
            {
                "annotation": r.annotation,
                "score": r.score,
                "highlights": r.highlights,
            }
            for r in results
        ]

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation from index."""
        return self.backend.delete_index(annotation_id)

    # -------------------------------------------------------------------------
    # Video Hash Index
    # -------------------------------------------------------------------------

    def index_video_hash(
        self,
        video_id: str,
        video_path: str,
        content_hash: str,
        frame_count: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Index video content hash for deduplication.

        Args:
            video_id: Video identifier
            video_path: Path to video file
            content_hash: Computed content hash
            frame_count: Optional frame count
            duration: Optional duration in seconds
        """
        if hasattr(self.backend, "index_video_hash"):
            self.backend.index_video_hash(video_id, video_path, content_hash, frame_count, duration)
            logger.debug("video_hash_indexed", video_id=video_id, hash=content_hash[:16])
        else:
            logger.warning("backend_does_not_support_video_hash_index")

    def find_video_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Find video by content hash.

        Returns:
            Video hash entry if found, None otherwise
        """
        if hasattr(self.backend, "get_video_by_hash"):
            return self.backend.get_video_by_hash(content_hash)
        return None

    def is_video_duplicate(self, content_hash: str) -> bool:
        """Check if a video with this hash already exists."""
        return self.find_video_by_hash(content_hash) is not None

    # -------------------------------------------------------------------------
    # Frame Hash Index
    # -------------------------------------------------------------------------

    def index_frame_hash(self, video_id: str, frame_index: int, content_hash: str) -> None:
        """Index frame content hash.

        Args:
            video_id: Video identifier
            frame_index: Frame index in video
            content_hash: Computed frame hash
        """
        if hasattr(self.backend, "index_frame_hash"):
            self.backend.index_frame_hash(video_id, frame_index, content_hash)
            logger.debug("frame_hash_indexed", video_id=video_id, frame=frame_index)

    def find_frames_by_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """Find frames by content hash.

        Returns:
            List of frame entries with matching hash
        """
        if hasattr(self.backend, "get_frames_by_hash"):
            return self.backend.get_frames_by_hash(content_hash)
        return []

    def find_duplicate_frames(self, content_hash: str, exclude_video_id: Optional[str] = None) -> List[Dict]:
        """Find duplicate frames across videos.

        Args:
            content_hash: Frame hash to search for
            exclude_video_id: Optional video to exclude from results

        Returns:
            List of matching frames from other videos
        """
        frames = self.find_frames_by_hash(content_hash)
        if exclude_video_id:
            frames = [f for f in frames if f.get("video_id") != exclude_video_id]
        return frames

    # -------------------------------------------------------------------------
    # Version Indexes
    # -------------------------------------------------------------------------

    def query_by_model_version(self, model_version: str, limit: int = 100) -> Tuple[List[Dict], int]:
        """Query annotations by model version."""
        query = QueryFilter(model_version=model_version, limit=limit)
        return self.query_annotations(query)

    def query_by_prompt_version(self, prompt_version: str, limit: int = 100) -> Tuple[List[Dict], int]:
        """Query annotations by prompt version."""
        query = QueryFilter(prompt_version=prompt_version, limit=limit)
        return self.query_annotations(query)

    def query_by_dataset_version(self, dataset_version: str, limit: int = 100) -> Tuple[List[Dict], int]:
        """Query annotations by dataset version."""
        query = QueryFilter(dataset_version=dataset_version, limit=limit)
        return self.query_annotations(query)

    def get_model_versions(self) -> Dict[str, int]:
        """Get all model versions and their counts."""
        stats = self.backend.get_statistics()
        return stats.by_model

    def get_prompt_versions(self) -> Dict[str, int]:
        """Get all prompt versions and their counts."""
        # This would require backend support
        logger.warning("get_prompt_versions_not_implemented")
        return {}

    def get_dataset_versions(self) -> Dict[str, int]:
        """Get all dataset versions and their counts."""
        # This would require backend support
        logger.warning("get_dataset_versions_not_implemented")
        return {}

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> IndexStats:
        """Get comprehensive index statistics."""
        backend_stats = self.backend.get_statistics()

        stats = IndexStats()
        stats.annotations_count = backend_stats.total_annotations
        stats.model_versions = backend_stats.by_model

        # Get video hash count if available
        if hasattr(self.backend, "_get_connection"):
            try:
                conn = self.backend._get_connection()
                row = conn.execute("SELECT COUNT(*) FROM video_hashes").fetchone()
                stats.video_hashes_count = row[0] if row else 0

                row = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()
                stats.frame_hashes_count = row[0] if row else 0
            except Exception:
                pass

        return stats

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def vacuum(self) -> None:
        """Reclaim unused space."""
        self.backend.vacuum()

    def optimize(self) -> None:
        """Optimize indexes."""
        self.backend.optimize()

    def reindex(self) -> None:
        """Rebuild all indexes."""
        self.backend.reindex()

    def close(self) -> None:
        """Close the backend."""
        self.backend.close()
