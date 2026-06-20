"""Local filesystem backend for annotation storage."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import orjson

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import BackendStats, BackendConfig, StorageBackend
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFSConfig(BackendConfig):
    """Configuration for LocalFS backend."""

    def __init__(
        self,
        root_path: Optional[Path] = None,
        name: str = "local_fs",
        read_only: bool = False,
        compression: Optional[str] = None,
    ):
        from dvas.config import settings

        super().__init__(
            backend_type=None,  # Will be set below
            name=name,
            read_only=read_only,
            compression=compression,
        )
        self.backend_type = None  # type: ignore
        self.root_path = Path(root_path or settings.DATA_ROOT / "annotations")


class LocalFSBackend(StorageBackend):
    """File-based annotation storage backend.

    Stores annotations as JSON files organized by source:
        {root}/
            gold/      - Teacher/gold annotations
            model/     - Model-generated annotations
            reviewed/  - Human-reviewed annotations
            versions/  - Versioned snapshots
    """

    def __init__(self, config: Optional[LocalFSConfig] = None):
        from dvas.persistence.backends.base import BackendType

        config = config or LocalFSConfig()
        config.backend_type = BackendType.LOCAL_FS
        super().__init__(config)
        self.config: LocalFSConfig = config
        self.root_path = config.root_path
        self.gold_path = self.root_path / "gold"
        self.model_path = self.root_path / "model"
        self.reviewed_path = self.root_path / "reviewed"
        self.versions_path = self.root_path / "versions"

    def open(self) -> None:
        """Create directories if they don't exist."""
        if not self.config.read_only:
            for path in [self.gold_path, self.model_path, self.reviewed_path, self.versions_path]:
                path.mkdir(parents=True, exist_ok=True)
        self._closed = False
        logger.info("localfs_backend_opened", root=str(self.root_path))

    def close(self) -> None:
        """Close the backend (no-op for file-based storage)."""
        self._closed = True
        logger.info("localfs_backend_closed")

    def health_check(self) -> Tuple[bool, str]:
        """Check if the storage is accessible."""
        try:
            if not self.root_path.exists():
                if self.config.read_only:
                    return False, f"Root path does not exist: {self.root_path}"
                self.root_path.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = self.root_path / ".health_check"
            if not self.config.read_only:
                test_file.write_text("ok")
                test_file.unlink()

            return True, "healthy"
        except Exception as e:
            return False, str(e)

    def _get_storage_path(self, annotation_id: str, source: str = "model") -> Path:
        """Get storage path for an annotation."""
        base_path = self._get_base_path(source)
        # Use first 2 chars of ID as subdirectory for better organization
        subdir = annotation_id[:2]
        return base_path / subdir / f"{annotation_id}.json"

    def _get_base_path(self, source: str) -> Path:
        """Get base path for a source category."""
        if source in ("gold", "teacher"):
            return self.gold_path
        elif source in ("reviewed", "human"):
            return self.reviewed_path
        return self.model_path

    def save(
        self,
        annotation: Annotation,
        source: str = "model",
        overwrite: bool = False,
    ) -> str:
        """Save an annotation to storage."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot save to read-only backend")

        storage_path = self._get_storage_path(annotation.id, source)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        if storage_path.exists() and not overwrite:
            raise FileExistsError(f"Annotation already exists: {storage_path}")

        # Update timestamp
        annotation.updated_at = datetime.now(timezone.utc)

        # Save using orjson for speed
        data = annotation.model_dump()
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))

        logger.debug("annotation_saved", id=annotation.id, path=str(storage_path))
        return str(storage_path)

    def load(self, annotation_id: str, source: str = "model") -> Optional[Annotation]:
        """Load an annotation from storage."""
        self.ensure_open()

        storage_path = self._get_storage_path(annotation_id, source)

        if not storage_path.exists():
            # Try other sources in order
            for src in ["model", "gold", "reviewed"]:
                if src != source:
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
        source: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Iterator[Annotation]:
        """Load all annotations as a generator."""
        self.ensure_open()

        sources = [source] if source else ["gold", "model", "reviewed"]

        for src in sources:
            base_path = self._get_base_path(src)
            if not base_path.exists():
                continue

            for json_file in base_path.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = orjson.loads(f.read())
                    annotation = Annotation.model_validate(data)

                    if video_id is None or annotation.video_id == video_id:
                        yield annotation

                except (orjson.JSONDecodeError, OSError, PermissionError) as e:
                    logger.warning("failed_to_load_annotation", file=str(json_file), error=str(e))

    def delete(self, annotation_id: str, source: str = "model") -> bool:
        """Delete an annotation from storage."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete from read-only backend")

        storage_path = self._get_storage_path(annotation_id, source)

        if not storage_path.exists():
            # Try other sources
            for src in ["model", "gold", "reviewed"]:
                storage_path = self._get_storage_path(annotation_id, src)
                if storage_path.exists():
                    break
            else:
                return False

        storage_path.unlink()
        logger.debug("annotation_deleted", id=annotation_id)
        return True

    def exists(self, annotation_id: str, source: str = "model") -> bool:
        """Check if an annotation exists."""
        self.ensure_open()

        storage_path = self._get_storage_path(annotation_id, source)
        if storage_path.exists():
            return True

        # Check other sources
        for src in ["model", "gold", "reviewed"]:
            if src != source:
                storage_path = self._get_storage_path(annotation_id, src)
                if storage_path.exists():
                    return True

        return False

    def get_storage_path(self, annotation_id: str, source: str = "model") -> str:
        """Get the storage path for an annotation."""
        return str(self._get_storage_path(annotation_id, source))

    def get_statistics(self) -> BackendStats:
        """Get storage statistics."""
        self.ensure_open()

        stats = BackendStats()
        total_size = 0
        total_count = 0
        by_source: Dict[str, int] = {}

        for source in ["gold", "model", "reviewed"]:
            path = self._get_base_path(source)
            if path.exists():
                json_files = list(path.rglob("*.json"))
                count = len(json_files)
                size = sum(f.stat().st_size for f in json_files)

                by_source[source] = count
                total_count += count
                total_size += size

        stats.total_annotations = total_count
        stats.by_source = by_source
        stats.storage_size_bytes = total_size
        stats.last_modified = datetime.now(timezone.utc)

        return stats

    def create_version(self, name: str) -> Path:
        """Create a versioned snapshot of reviewed annotations."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot create version in read-only backend")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.versions_path / f"{name}_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy reviewed annotations
        count = 0
        for json_file in self.reviewed_path.rglob("*.json"):
            rel_path = json_file.relative_to(self.reviewed_path)
            dest_path = version_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_file, dest_path)
            count += 1

        # Write manifest
        manifest = {
            "name": name,
            "timestamp": timestamp,
            "count": count,
        }
        with open(version_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("version_created", name=name, path=str(version_dir), count=count)
        return version_dir

    def export_to_jsonl(
        self,
        output_path: Path,
        source: str = "reviewed",
        format: str = "llava",
    ) -> int:
        """Export annotations to JSONL file."""
        self.ensure_open()

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

        logger.info("exported_to_jsonl", path=str(output_path), count=count, format=format)
        return count

    def backup(self, destination: Path) -> Path:
        """Create a backup of all annotations."""
        self.ensure_open()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = destination / f"annotations_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        for source in ["gold", "model", "reviewed"]:
            src_path = self._get_base_path(source)
            if src_path.exists():
                dst_path = backup_path / source
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        logger.info("backup_created", path=str(backup_path))
        return backup_path

    def restore(self, source: Path) -> None:
        """Restore annotations from a backup."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot restore to read-only backend")

        for source_name in ["gold", "model", "reviewed"]:
            src_path = source / source_name
            if src_path.exists():
                dst_path = self._get_base_path(source_name)
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        logger.info("backup_restored", source=str(source))
