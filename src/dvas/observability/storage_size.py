"""Storage size monitoring for DVAS.

Tracks disk usage by storage type with alerting on capacity thresholds.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dvas.observability.collector import get_metrics
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StorageMetrics:
    """Metrics for a single storage location."""

    storage_type: str
    path: str
    size_bytes: int = 0
    file_count: int = 0
    last_updated: float = 0.0


class StorageSizeMonitor:
    """Monitor storage size and disk usage.

    Tracks storage consumption by type with capacity alerting.

    Usage::

        monitor = StorageSizeMonitor(warning_threshold=0.8, critical_threshold=0.95)
        monitor.scan_storage("annotations", Path("/data/annotations"))
        usage = monitor.get_storage_usage("annotations")
    """

    def __init__(
        self,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._storage: Dict[str, StorageMetrics] = {}
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def scan_storage(self, storage_type: str, path: Path) -> StorageMetrics:
        """Scan a storage location and record size.

        Args:
            storage_type: Type of storage (e.g., "annotations", "exports")
            path: Path to scan

        Returns:
            StorageMetrics for the scanned location
        """
        size_bytes = 0
        file_count = 0

        try:
            if path.exists():
                if path.is_file():
                    size_bytes = path.stat().st_size
                    file_count = 1
                elif path.is_dir():
                    for item in path.rglob("*"):
                        if item.is_file():
                            try:
                                size_bytes += item.stat().st_size
                                file_count += 1
                            except (OSError, PermissionError):
                                pass
        except Exception as e:
            logger.error(
                "storage_scan_failed",
                storage_type=storage_type,
                path=str(path),
                error=str(e),
            )

        metrics = StorageMetrics(
            storage_type=storage_type,
            path=str(path),
            size_bytes=size_bytes,
            file_count=file_count,
            last_updated=time.time(),
        )

        with self._lock:
            self._storage[storage_type] = metrics

        # Record in global metrics
        get_metrics().gauge(
            "storage_size_bytes",
            float(size_bytes),
            labels={"storage_type": storage_type},
        )
        get_metrics().gauge(
            "storage_file_count",
            float(file_count),
            labels={"storage_type": storage_type},
        )

        # Check thresholds (using total if available)
        self._check_thresholds(storage_type, size_bytes)

        return metrics

    def record_size(
        self,
        storage_type: str,
        size_bytes: int,
        file_count: int = 0,
        path: str = "",
    ) -> None:
        """Record storage size directly.

        Args:
            storage_type: Type of storage
            size_bytes: Size in bytes
            file_count: Number of files
            path: Storage path
        """
        metrics = StorageMetrics(
            storage_type=storage_type,
            path=path,
            size_bytes=size_bytes,
            file_count=file_count,
            last_updated=time.time(),
        )

        with self._lock:
            self._storage[storage_type] = metrics

        get_metrics().gauge(
            "storage_size_bytes",
            float(size_bytes),
            labels={"storage_type": storage_type},
        )
        get_metrics().gauge(
            "storage_file_count",
            float(file_count),
            labels={"storage_type": storage_type},
        )

        self._check_thresholds(storage_type, size_bytes)

    def _check_thresholds(self, storage_type: str, size_bytes: int) -> None:
        """Check if storage exceeds thresholds.

        For now, uses a simple absolute check. In production,
        this would compare against total disk capacity.
        """
        # Check if we have a capacity limit set for this storage type
        capacity = self._get_capacity(storage_type)
        if capacity > 0:
            utilization = size_bytes / capacity
            if utilization >= self.critical_threshold:
                self._trigger_alert(
                    "storage_critical",
                    {
                        "storage_type": storage_type,
                        "size_bytes": size_bytes,
                        "capacity_bytes": capacity,
                        "utilization": utilization,
                        "severity": "critical",
                    },
                )
            elif utilization >= self.warning_threshold:
                self._trigger_alert(
                    "storage_warning",
                    {
                        "storage_type": storage_type,
                        "size_bytes": size_bytes,
                        "capacity_bytes": capacity,
                        "utilization": utilization,
                        "severity": "warning",
                    },
                )

    def _get_capacity(self, storage_type: str) -> int:
        """Get capacity for a storage type.

        In production, this would query the filesystem.
        Returns 0 if unknown.
        """
        # Default capacities - override in production
        defaults: Dict[str, int] = {
            "annotations": 100 * 1024 * 1024 * 1024,  # 100 GB
            "exports": 50 * 1024 * 1024 * 1024,  # 50 GB
            "models": 20 * 1024 * 1024 * 1024,  # 20 GB
            "temp": 10 * 1024 * 1024 * 1024,  # 10 GB
        }
        return defaults.get(storage_type, 0)

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "storage_alert",
            alert_type=alert_type,
            **details,
        )
        for handler in self._alert_handlers:
            try:
                handler(alert_type, details)
            except Exception as e:
                logger.error("alert_handler_failed", error=str(e))

    def add_alert_handler(
        self, handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)

    def remove_alert_handler(
        self, handler: Callable[[str, Dict[str, Any]], None]
    ) -> bool:
        """Remove an alert handler.

        Returns:
            True if handler was found and removed
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            return True
        return False

    def get_storage_usage(self, storage_type: str) -> Dict[str, Any]:
        """Get usage for a storage type.

        Args:
            storage_type: Type of storage

        Returns:
            Dict with size, file count, and utilization
        """
        with self._lock:
            metrics = self._storage.get(storage_type)

        if not metrics:
            return {
                "storage_type": storage_type,
                "size_bytes": 0,
                "file_count": 0,
                "utilization": 0.0,
                "status": "unknown",
            }

        capacity = self._get_capacity(storage_type)
        utilization = metrics.size_bytes / capacity if capacity > 0 else 0.0

        if utilization >= self.critical_threshold:
            status = "critical"
        elif utilization >= self.warning_threshold:
            status = "warning"
        else:
            status = "healthy"

        return {
            "storage_type": storage_type,
            "path": metrics.path,
            "size_bytes": metrics.size_bytes,
            "size_human": self._format_bytes(metrics.size_bytes),
            "file_count": metrics.file_count,
            "capacity_bytes": capacity,
            "capacity_human": self._format_bytes(capacity) if capacity > 0 else "unknown",
            "utilization": utilization,
            "status": status,
            "last_updated": metrics.last_updated,
        }

    def get_all_storage_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage for all storage types.

        Returns:
            Dict mapping storage types to usage info
        """
        with self._lock:
            types = list(self._storage.keys())
        return {t: self.get_storage_usage(t) for t in types}

    def get_total_usage(self) -> Dict[str, Any]:
        """Get total storage usage across all types.

        Returns:
            Dict with total size and file count
        """
        with self._lock:
            total_bytes = sum(m.size_bytes for m in self._storage.values())
            total_files = sum(m.file_count for m in self._storage.values())

        return {
            "total_bytes": total_bytes,
            "total_files": total_files,
            "total_human": self._format_bytes(total_bytes),
        }

    def get_largest_storage(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the largest storage types.

        Args:
            n: Number of storage types to return

        Returns:
            List of storage usage dicts sorted by size
        """
        all_usage = self.get_all_storage_usage()
        sorted_usage = sorted(
            all_usage.items(), key=lambda x: x[1]["size_bytes"], reverse=True
        )
        return [usage for _, usage in sorted_usage[:n]]

    def get_growth_rate(
        self,
        storage_type: str,
        window_seconds: float = 3600.0,
    ) -> float:
        """Estimate growth rate in bytes per second.

        Args:
            storage_type: Type of storage
            window_seconds: Time window for calculation

        Returns:
            Growth rate in bytes per second
        """
        # This is a simplified implementation
        # In production, would track historical sizes
        with self._lock:
            metrics = self._storage.get(storage_type)
            if not metrics:
                return 0.0

        # Return current size / time since last update as a rough estimate
        elapsed = time.time() - metrics.last_updated
        if elapsed > 0:
            return metrics.size_bytes / elapsed
        return 0.0

    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics.

        Returns:
            Dict with all storage metrics and overall health
        """
        all_usage = self.get_all_storage_usage()
        critical = [
            t for t, u in all_usage.items() if u["status"] == "critical"
        ]
        warning = [
            t for t, u in all_usage.items() if u["status"] == "warning"
        ]

        return {
            "total": self.get_total_usage(),
            "storage_types": all_usage,
            "largest": self.get_largest_storage(),
            "critical": critical,
            "warning": warning,
            "healthy_count": len(all_usage) - len(critical) - len(warning),
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }

    def is_healthy(self, storage_type: Optional[str] = None) -> bool:
        """Check if storage is healthy.

        Args:
            storage_type: Optional specific storage to check

        Returns:
            True if no storage is in critical state
        """
        if storage_type:
            usage = self.get_storage_usage(storage_type)
            return usage["status"] != "critical"

        all_usage = self.get_all_storage_usage()
        return all(u["status"] != "critical" for u in all_usage.values())

    def reset(self, storage_type: Optional[str] = None) -> None:
        """Reset storage data.

        Args:
            storage_type: Optional storage type to reset (all if None)
        """
        with self._lock:
            if storage_type:
                self._storage.pop(storage_type, None)
            else:
                self._storage.clear()
