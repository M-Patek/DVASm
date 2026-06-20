"""Backup and restore utilities for annotation storage.

Provides tools for:
- Creating backups of annotation data and indexes
- Restoring from backups
- Incremental backups
- Backup verification
"""

import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class BackupError(Exception):
    """Error during backup/restore operation."""

    pass


class BackupManager:
    """Manages backups for annotation storage.

    Usage::

        manager = BackupManager(backup_root="/backups/dvas")
        # Create backup
        backup_path = manager.backup(storage_backend, metadata_backend)
        # Restore backup
        manager.restore(backup_path, storage_backend, metadata_backend)
    """

    def __init__(self, backup_root: Path):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        storage_backend=None,
        metadata_backend=None,
        name: Optional[str] = None,
        compress: bool = True,
    ) -> Path:
        """Create a full backup.

        Args:
            storage_backend: Storage backend to backup
            metadata_backend: Metadata backend to backup
            name: Optional backup name (default: timestamp)
            compress: Whether to compress the backup

        Returns:
            Path to backup directory or archive
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"backup_{timestamp}"

        backup_dir = self.backup_root / name
        backup_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }

        # Backup storage
        if storage_backend:
            storage_backup = backup_dir / "storage"
            storage_backup.mkdir(exist_ok=True)

            if hasattr(storage_backend, "backup"):
                storage_path = storage_backend.backup(storage_backup)
                manifest["components"]["storage"] = {
                    "type": "native",
                    "path": str(storage_path.relative_to(backup_dir)),
                }
            else:
                # Fallback: copy files
                self._backup_storage_files(storage_backend, storage_backup)
                manifest["components"]["storage"] = {
                    "type": "file_copy",
                    "path": "storage",
                }

        # Backup metadata
        if metadata_backend:
            metadata_backup = backup_dir / "metadata"
            metadata_backup.mkdir(exist_ok=True)

            if hasattr(metadata_backend, "backup"):
                metadata_path = metadata_backend.backup(metadata_backup)
                manifest["components"]["metadata"] = {
                    "type": "native",
                    "path": str(metadata_path.relative_to(backup_dir)),
                }
            else:
                manifest["components"]["metadata"] = {
                    "type": "none",
                    "note": "Backend does not support backup",
                }

        # Write manifest
        with open(backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Compress if requested
        if compress:
            archive_path = self._compress_backup(backup_dir)
            shutil.rmtree(backup_dir)
            logger.info("backup_created", archive=str(archive_path))
            return archive_path

        logger.info("backup_created", directory=str(backup_dir))
        return backup_dir

    def _backup_storage_files(self, storage_backend, backup_dir: Path) -> None:
        """Backup storage by copying files."""
        if hasattr(storage_backend, "root_path"):
            root = Path(storage_backend.root_path)
            if root.exists():
                for source in ["gold", "model", "reviewed"]:
                    src_path = root / source
                    if src_path.exists():
                        dst_path = backup_dir / source
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    def _compress_backup(self, backup_dir: Path) -> Path:
        """Compress backup directory to tar.gz."""
        archive_path = Path(str(backup_dir) + ".tar.gz")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_dir, arcname=backup_dir.name)

        return archive_path

    def restore_backup(
        self,
        backup_path: Path,
        storage_backend=None,
        metadata_backend=None,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """Restore from a backup.

        Args:
            backup_path: Path to backup (directory or archive)
            storage_backend: Storage backend to restore to
            metadata_backend: Metadata backend to restore to
            verify: Whether to verify backup integrity

        Returns:
            Restore statistics
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise BackupError(f"Backup not found: {backup_path}")

        # Extract if compressed
        if backup_path.suffix == ".gz" or backup_path.suffixes == [".tar", ".gz"]:
            extract_dir = self.backup_root / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(extract_dir)

            # Find the actual backup directory
            backup_dir = next(extract_dir.iterdir())
        else:
            backup_dir = backup_path

        # Load manifest
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            raise BackupError("Backup manifest not found")

        with open(manifest_path) as f:
            manifest = json.load(f)

        if verify:
            self._verify_backup(backup_dir, manifest)

        stats = {"restored_from": str(backup_path), "components": {}}

        # Restore storage
        if storage_backend and "storage" in manifest["components"]:
            component = manifest["components"]["storage"]

            if hasattr(storage_backend, "restore"):
                storage_path = backup_dir / component["path"]
                storage_backend.restore(storage_path)
                stats["components"]["storage"] = "restored"
            else:
                # Fallback: copy files
                storage_backup = backup_dir / "storage"
                self._restore_storage_files(storage_backend, storage_backup)
                stats["components"]["storage"] = "restored_from_files"

        # Restore metadata
        if metadata_backend and "metadata" in manifest["components"]:
            component = manifest["components"]["metadata"]

            if component.get("type") == "native" and hasattr(metadata_backend, "restore"):
                metadata_path = backup_dir / component["path"]
                metadata_backend.restore(metadata_path)
                stats["components"]["metadata"] = "restored"
            else:
                stats["components"]["metadata"] = "skipped"

        logger.info("backup_restored", from_backup=str(backup_path))
        return stats

    def _restore_storage_files(self, storage_backend, backup_dir: Path) -> None:
        """Restore storage by copying files."""
        if hasattr(storage_backend, "root_path"):
            root = Path(storage_backend.root_path)
            for source in ["gold", "model", "reviewed"]:
                src_path = backup_dir / source
                if src_path.exists():
                    dst_path = root / source
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    def _verify_backup(self, backup_dir: Path, manifest: Dict) -> None:
        """Verify backup integrity."""
        for component_name, component in manifest.get("components", {}).items():
            if component.get("type") == "none":
                continue

            component_path = backup_dir / component["path"]
            if not component_path.exists():
                raise BackupError(f"Backup component missing: {component_name}")

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        for item in self.backup_root.iterdir():
            if item.suffix == ".gz" or item.is_dir():
                manifest_path = item / "manifest.json" if item.is_dir() else None

                if manifest_path and manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                else:
                    manifest = {"name": item.stem}

                backups.append(
                    {
                        "name": manifest.get("name", item.stem),
                        "path": str(item),
                        "created_at": manifest.get("created_at"),
                        "size_bytes": item.stat().st_size if item.exists() else 0,
                    }
                )

        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_backup(self, name: str) -> bool:
        """Delete a backup."""
        for item in self.backup_root.iterdir():
            if item.stem == name or item.name == name:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                logger.info("backup_deleted", name=name)
                return True

        return False
