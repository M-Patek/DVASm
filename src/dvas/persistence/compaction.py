"""Compaction and rebuild utilities for annotation storage.

Provides tools for:
- Compacting storage (removing duplicates, reclaiming space)
- Rebuilding indexes from storage
- Verifying data integrity
- Repairing corrupted data
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import StorageBackend, MetadataBackend
from dvas.utils.hash import compute_annotation_hash
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class CompactionError(Exception):
    """Error during compaction operation."""

    pass


class CompactionManager:
    """Manages compaction and rebuild operations.

    Usage::

        manager = CompactionManager(storage_backend, metadata_backend)
        # Compact storage
        stats = manager.compact()
        # Rebuild index
        stats = manager.rebuild_index()
        # Verify integrity
        report = manager.verify_integrity()
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        metadata_backend: Optional[MetadataBackend] = None,
    ):
        self.storage = storage_backend
        self.metadata = metadata_backend

    def compact(
        self,
        remove_duplicates: bool = True,
        remove_orphans: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Compact storage by removing duplicates and orphans.

        Args:
            remove_duplicates: Remove duplicate annotations
            remove_orphans: Remove orphaned files not in index
            dry_run: If True, only report what would be done

        Returns:
            Compaction statistics
        """
        stats = {
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "orphans_found": 0,
            "orphans_removed": 0,
            "space_reclaimed_bytes": 0,
        }

        if remove_duplicates and self.storage:
            dup_stats = self._remove_duplicates(dry_run)
            stats["duplicates_found"] = dup_stats["found"]
            stats["duplicates_removed"] = dup_stats["removed"]
            stats["space_reclaimed_bytes"] += dup_stats["space_reclaimed"]

        if remove_orphans and self.storage and self.metadata:
            orphan_stats = self._remove_orphans(dry_run)
            stats["orphans_found"] = orphan_stats["found"]
            stats["orphans_removed"] = orphan_stats["removed"]
            stats["space_reclaimed_bytes"] += orphan_stats["space_reclaimed"]

        logger.info(
            "compaction_completed",
            duplicates=stats["duplicates_removed"],
            orphans=stats["orphans_removed"],
            space_reclaimed=stats["space_reclaimed_bytes"],
        )

        return stats

    def _remove_duplicates(self, dry_run: bool) -> Dict[str, Any]:
        """Remove duplicate annotations based on content hash."""
        if not self.storage:
            return {"found": 0, "removed": 0, "space_reclaimed": 0}

        # Build hash -> annotation mapping
        hash_map: Dict[str, List[Annotation]] = {}

        for annotation in self.storage.load_all():
            content_hash = compute_annotation_hash(annotation)
            if content_hash not in hash_map:
                hash_map[content_hash] = []
            hash_map[content_hash].append(annotation)

        # Find and remove duplicates
        found = 0
        removed = 0
        space_reclaimed = 0

        for content_hash, annotations in hash_map.items():
            if len(annotations) > 1:
                found += len(annotations) - 1

                if not dry_run:
                    # Keep the first, remove the rest
                    for dup in annotations[1:]:
                        # Get file size before deletion
                        storage_path = self.storage.get_storage_path(dup.id, dup.source)
                        try:
                            if Path(storage_path).exists():
                                space_reclaimed += Path(storage_path).stat().st_size
                        except Exception:
                            pass

                        self.storage.delete(dup.id, dup.source)
                        removed += 1

        return {"found": found, "removed": removed, "space_reclaimed": space_reclaimed}

    def _remove_orphans(self, dry_run: bool) -> Dict[str, Any]:
        """Remove storage files not in the index."""
        if not self.storage or not self.metadata:
            return {"found": 0, "removed": 0, "space_reclaimed": 0}

        # Get all indexed IDs (inefficient for large datasets - would need backend support)
        # For now, iterate through storage and check index
        found = 0
        removed = 0
        space_reclaimed = 0

        # Implementation depends on storage backend type
        # For LocalFS, we can scan directories
        if hasattr(self.storage, "root_path"):
            root = Path(self.storage.root_path)
            for source in ["gold", "model", "reviewed"]:
                source_path = root / source
                if not source_path.exists():
                    continue

                for json_file in source_path.rglob("*.json"):
                    try:
                        annotation_id = json_file.stem
                        entry = self.metadata.get(annotation_id)

                        if entry is None:
                            found += 1
                            space_reclaimed += json_file.stat().st_size

                            if not dry_run:
                                json_file.unlink()
                                removed += 1
                    except Exception as e:
                        logger.warning("orphan_check_failed", file=str(json_file), error=str(e))

        return {"found": found, "removed": removed, "space_reclaimed": space_reclaimed}

    def rebuild_index(self, batch_size: int = 100) -> Dict[str, Any]:
        """Rebuild the index from storage.

        Args:
            batch_size: Number of annotations to process per batch

        Returns:
            Rebuild statistics
        """
        if not self.storage or not self.metadata:
            raise CompactionError("Both storage and metadata backends required")

        stats = {
            "annotations_processed": 0,
            "annotations_indexed": 0,
            "errors": [],
        }

        # Clear existing index (if supported)
        # This would require backend-specific implementation

        # Re-index all annotations
        batch = []

        for annotation in self.storage.load_all():
            stats["annotations_processed"] += 1
            batch.append(annotation)

            if len(batch) >= batch_size:
                self._index_batch(batch, stats)
                batch = []

        # Process remaining
        if batch:
            self._index_batch(batch, stats)

        # Optimize index
        if hasattr(self.metadata, "optimize"):
            self.metadata.optimize()

        logger.info(
            "index_rebuild_completed",
            processed=stats["annotations_processed"],
            indexed=stats["annotations_indexed"],
            errors=len(stats["errors"]),
        )

        return stats

    def _index_batch(self, annotations: List[Annotation], stats: Dict) -> None:
        """Index a batch of annotations."""
        for annotation in annotations:
            try:
                storage_path = self.storage.get_storage_path(annotation.id, annotation.source)
                self.metadata.index_annotation(annotation, storage_path)
                stats["annotations_indexed"] += 1
            except Exception as e:
                logger.error("index_failed", id=annotation.id, error=str(e))
                stats["errors"].append({"id": annotation.id, "error": str(e)})

    def verify_integrity(self, fix: bool = False) -> Dict[str, Any]:
        """Verify data integrity.

        Args:
            fix: If True, attempt to fix issues

        Returns:
            Integrity report
        """
        report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "annotations_checked": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "issues": [],
        }

        if not self.storage:
            return report

        for annotation in self.storage.load_all():
            report["annotations_checked"] += 1

            # Check for required fields
            if not annotation.id:
                report["issues_found"] += 1
                report["issues"].append({
                    "type": "missing_id",
                    "annotation": annotation,
                })

            # Check for hash consistency
            if self.metadata:
                entry = self.metadata.get(annotation.id)
                if entry and entry.content_hash:
                    current_hash = compute_annotation_hash(annotation)
                    if current_hash != entry.content_hash:
                        report["issues_found"] += 1
                        report["issues"].append({
                            "type": "hash_mismatch",
                            "id": annotation.id,
                            "expected": entry.content_hash,
                            "actual": current_hash,
                        })

                        if fix:
                            self.metadata.index_annotation(annotation)
                            report["issues_fixed"] += 1

        return report

    def rebuild_fts_index(self) -> Dict[str, Any]:
        """Rebuild the full-text search index.

        Returns:
            Rebuild statistics
        """
        if not self.metadata:
            return {"error": "No metadata backend"}

        # This requires backend-specific implementation
        if hasattr(self.metadata, "reindex"):
            self.metadata.reindex()
            return {"status": "reindexed"}

        return {"status": "not_supported"}
