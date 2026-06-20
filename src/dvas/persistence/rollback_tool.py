"""Annotation rollback tool for reverting changes.

Provides utilities for:
- Rolling back to previous versions
- Rolling back by time range
- Rolling back specific changes
"""

from datetime import datetime
from typing import Any, Dict, Optional

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import MetadataBackend
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class RollbackError(Exception):
    """Error during rollback operation."""

    pass


class RollbackManager:
    """Manages annotation rollbacks.

    Usage::

        manager = RollbackManager(backend, storage_backend)
        # Rollback to specific version
        manager.rollback_to_version("version-id")
        # Rollback by time
        manager.rollback_to_time(datetime(2024, 1, 1))
    """

    def __init__(
        self,
        metadata_backend: MetadataBackend,
        storage_backend=None,
    ):
        self.metadata = metadata_backend
        self.storage = storage_backend

    def rollback_to_version(self, version_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Rollback annotations to a specific version.

        Args:
            version_id: Version to rollback to
            dry_run: If True, only report what would be done

        Returns:
            Rollback statistics
        """
        version = self.metadata.get_version(version_id)
        if version is None:
            raise RollbackError(f"Version not found: {version_id}")

        # Get annotations from version
        annotations_data = self.metadata.restore_version(version_id)

        stats = {
            "version_id": version_id,
            "version_name": version.name,
            "annotations_restored": 0,
            "annotations_skipped": 0,
            "errors": [],
        }

        for data in annotations_data:
            try:
                annotation = Annotation.model_validate(data)

                if dry_run:
                    stats["annotations_restored"] += 1
                    continue

                # Restore to storage if available
                if self.storage:
                    self.storage.save(annotation, source=annotation.source, overwrite=True)

                # Update index
                self.metadata.index_annotation(annotation)
                stats["annotations_restored"] += 1

            except Exception as e:
                logger.error("rollback_failed", annotation_id=data.get("id"), error=str(e))
                stats["errors"].append({"id": data.get("id"), "error": str(e)})

        logger.info(
            "rollback_completed",
            version_id=version_id,
            restored=stats["annotations_restored"],
            errors=len(stats["errors"]),
        )

        return stats

    def rollback_annotation(
        self,
        annotation_id: str,
        version_id: str,
        dry_run: bool = False,
    ) -> Optional[Annotation]:
        """Rollback a single annotation to a specific version.

        Args:
            annotation_id: Annotation to rollback
            version_id: Version to rollback to
            dry_run: If True, only report what would be done

        Returns:
            The restored annotation or None
        """
        # Get annotations from version
        annotations_data = self.metadata.restore_version(version_id)

        for data in annotations_data:
            if data.get("id") == annotation_id:
                if dry_run:
                    return Annotation.model_validate(data)

                annotation = Annotation.model_validate(data)

                # Restore to storage if available
                if self.storage:
                    self.storage.save(annotation, source=annotation.source, overwrite=True)

                # Update index
                self.metadata.index_annotation(annotation)

                logger.info("annotation_rolled_back", id=annotation_id, version_id=version_id)
                return annotation

        logger.warning("annotation_not_found_in_version", id=annotation_id, version_id=version_id)
        return None

    def rollback_by_time(
        self,
        before: datetime,
        video_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Rollback annotations created before a specific time.

        This removes annotations that were created before the given time.
        Use with caution!

        Args:
            before: Rollback annotations created before this time
            video_id: Optional video ID filter
            dry_run: If True, only report what would be done

        Returns:
            Rollback statistics
        """
        from dvas.persistence.backends.base import QueryFilter

        query = QueryFilter(created_before=before, video_id=video_id, limit=10000)
        entries, total = self.metadata.query(query)

        stats = {
            "before": before.isoformat(),
            "total_found": total,
            "annotations_deleted": 0,
            "errors": [],
        }

        if dry_run:
            stats["annotations_deleted"] = total
            return stats

        for entry in entries:
            try:
                # Delete from index
                self.metadata.delete_index(entry.id)

                # Delete from storage if available
                if self.storage:
                    self.storage.delete(entry.id, source=entry.source)

                stats["annotations_deleted"] += 1

            except Exception as e:
                logger.error("rollback_delete_failed", id=entry.id, error=str(e))
                stats["errors"].append({"id": entry.id, "error": str(e)})

        logger.info(
            "time_rollback_completed",
            before=before.isoformat(),
            deleted=stats["annotations_deleted"],
            errors=len(stats["errors"]),
        )

        return stats

    def preview_rollback(self, version_id: str) -> Dict[str, Any]:
        """Preview what a rollback would do.

        Args:
            version_id: Version to preview

        Returns:
            Preview information
        """
        return self.rollback_to_version(version_id, dry_run=True)
