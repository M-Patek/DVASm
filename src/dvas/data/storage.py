"""Storage backend for annotations."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import orjson

from dvas.config import settings
from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

if TYPE_CHECKING:
    from dvas.persistence.indexed_store import AnnotationQuery, IndexStore, IndexStoreConfig

logger = get_logger(__name__)


class AnnotationStore:
    """File-based annotation storage with optional SQLite indexing.

    Provides file-based storage with automatic indexing for fast queries
    and full-text search.
    """

    def __init__(
        self,
        root_path: Optional[Path] = None,
        enable_index: bool = True,
    ):
        self.root_path = Path(root_path or settings.DATA_ROOT / "annotations")
        self.gold_path = self.root_path / "gold"
        self.model_path = self.root_path / "model"
        self.reviewed_path = self.root_path / "reviewed"
        self._enable_index = enable_index
        self._index_store: Optional[IndexStore] = None

        # Create directories
        for path in [self.gold_path, self.model_path, self.reviewed_path]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def index_store(self) -> Optional["IndexStore"]:
        """Lazy-initialize the index store."""
        if self._index_store is None and self._enable_index:
            from dvas.persistence.indexed_store import IndexStore, IndexStoreConfig

            config = IndexStoreConfig(
                db_path=self.root_path / "index.db",
                wal_mode=True,
                enable_fts=True,
                enable_versioning=True,
            )
            self._index_store = IndexStore(config)
            self._index_store.create_index()
        return self._index_store

    def _get_storage_path(
        self, annotation_id: str, source: str = "model"
    ) -> Path:
        """Get storage path for an annotation."""
        if source in ("gold", "teacher"):
            base_path = self.gold_path
        elif source in ("reviewed", "human"):
            base_path = self.reviewed_path
        else:
            base_path = self.model_path

        # Use first 2 chars of ID as subdirectory for better organization
        subdir = annotation_id[:2]
        return base_path / subdir / f"{annotation_id}.json"

    def save(
        self,
        annotation: Annotation,
        source: str = "model",
        overwrite: bool = False,
    ) -> Path:
        """Save an annotation to storage and index."""
        storage_path = self._get_storage_path(annotation.id, source)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        if storage_path.exists() and not overwrite:
            raise FileExistsError(f"Annotation already exists: {storage_path}")

        # Update timestamp
        annotation.updated_at = datetime.now(timezone.utc)

        # Save using orjson for speed
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write(orjson.dumps(annotation.model_dump(), option=orjson.OPT_INDENT_2).decode("utf-8"))

        # Index the annotation
        if self._enable_index and self.index_store:
            try:
                self.index_store.index_annotation(annotation)
            except Exception as e:
                logger.warning("index_annotation_failed", annotation_id=annotation.id, error=str(e))

        return storage_path

    def load(self, annotation_id: str, source: str = "model") -> Optional[Annotation]:
        """Load an annotation from storage."""
        storage_path = self._get_storage_path(annotation_id, source)

        if not storage_path.exists():
            # Try other sources
            for src in ["model", "gold", "reviewed"]:
                storage_path = self._get_storage_path(annotation_id, src)
                if storage_path.exists():
                    break
            else:
                return None

        with open(storage_path, "r", encoding="utf-8") as f:
            data = orjson.loads(f.read())

        return Annotation.model_validate(data)

    def load_all(
        self,
        source: str = "model",
        video_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[Annotation]:
        """Load all annotations from a source as a generator.

        Args:
            source: Source to load from (gold/model/reviewed)
            video_id: Optional filter by video_id
            batch_size: Number of annotations to yield before yielding control

        Yields:
            Annotation objects one at a time
        """
        base_path = self._get_source_path(source)
        count = 0

        for json_file in base_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = orjson.loads(f.read())
                annotation = Annotation.model_validate(data)

                if video_id is None or annotation.video_id == video_id:
                    yield annotation
                    count += 1

                    if count >= batch_size:
                        count = 0

            except (orjson.JSONDecodeError, OSError, PermissionError) as e:
                # Log error but continue processing other files
                import logging
                logging.getLogger(__name__).warning("Failed to load annotation", file=str(json_file), error=str(e))

    def _get_source_path(self, source: str) -> Path:
        """Get base path for a source."""
        if source in ("gold", "teacher"):
            return self.gold_path
        elif source in ("reviewed", "human"):
            return self.reviewed_path
        return self.model_path

    def export_to_jsonl(
        self,
        output_path: Path,
        source: str = "reviewed",
        format: str = "llava",
    ) -> int:
        """Export annotations to JSONL file."""
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for ann in self.load_all(source):
                if format == "llava":
                    data = ann.to_llava_format()
                elif format == "openai":
                    data = ann.to_openai_format()
                else:
                    data = ann.model_dump()

                f.write(orjson.dumps(data).decode("utf-8") + "\n")
                count += 1

        return count

    def get_statistics(self) -> Dict:
        """Get storage statistics."""
        stats = {
            "gold": {"count": 0, "size_mb": 0},
            "model": {"count": 0, "size_mb": 0},
            "reviewed": {"count": 0, "size_mb": 0},
        }

        for source in ["gold", "model", "reviewed"]:
            path = self._get_source_path(source)
            if path.exists():
                json_files = list(path.rglob("*.json"))
                stats[source]["count"] = len(json_files)
                stats[source]["size_mb"] = sum(
                    f.stat().st_size for f in json_files
                ) / (1024 * 1024)

        return stats

    def create_version(self, name: str) -> Path:
        """Create a versioned snapshot of reviewed annotations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.root_path / "versions" / f"{name}_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy reviewed annotations
        for json_file in self.reviewed_path.rglob("*.json"):
            rel_path = json_file.relative_to(self.reviewed_path)
            dest_path = version_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_file, dest_path)

        # Write manifest
        manifest = {
            "name": name,
            "timestamp": timestamp,
            "count": len(list(self.reviewed_path.rglob("*.json"))),
        }
        with open(version_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return version_dir

    # -------------------------------------------------------------------
    # Indexed query methods (require enable_index=True)
    # -------------------------------------------------------------------

    def search(self, query_text: str, limit: int = 100) -> List[Dict]:
        """Full-text search annotations.

        Args:
            query_text: Search query
            limit: Max results

        Returns:
            List of search results with annotation and score
        """
        if not self._enable_index or not self.index_store:
            logger.warning("index_not_enabled")
            return []

        from dvas.persistence.indexed_store import SearchResult

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
        """Query annotations by video ID.

        Args:
            video_id: Video ID to search for
            source: Optional source filter

        Returns:
            List of matching annotations
        """
        if self._enable_index and self.index_store:
            from dvas.persistence.indexed_store import AnnotationQuery

            query = AnnotationQuery(video_id=video_id, source=source)
            results, _ = self.index_store.query(query)
            return results

        # Fallback to file-based search
        annotations = []
        for ann in self.load_all(source=source or "model"):
            if ann.video_id == video_id:
                annotations.append(ann)
        return annotations

    def query_by_quality(
        self, min_quality: float = 0.0, max_quality: float = 1.0, limit: int = 100
    ) -> List[Annotation]:
        """Query annotations by quality score.

        Args:
            min_quality: Minimum quality score
            max_quality: Maximum quality score
            limit: Max results

        Returns:
            List of matching annotations
        """
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
        for ann in self.load_all(source="model"):
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
        """Sync the index with the file store.

        Returns:
            Dict with sync statistics
        """
        if not self._enable_index or not self.index_store:
            logger.warning("index_not_enabled")
            return {"synced": 0}

        return self.index_store.sync_from_store(self)

    def close(self) -> None:
        """Close the store and release resources."""
        if self._index_store:
            self._index_store.close()
            self._index_store = None
