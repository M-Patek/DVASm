"""Export throughput monitoring for DVAS.

Tracks export performance metrics including throughput, latency,
and success rates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dvas.observability.collector import get_metrics
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExportRecord:
    """A single export operation record."""

    export_format: str
    bytes_written: int
    duration_ms: float
    success: bool
    timestamp: float
    error: Optional[str] = None


class ExportThroughputMonitor:
    """Monitor export throughput and performance.

    Tracks export operations by format with throughput calculation
    and alerting on slow exports.

    Usage::

        monitor = ExportThroughputMonitor()
        monitor.record_export("llava", bytes_written=1024, duration_ms=500)
        throughput = monitor.get_throughput("llava")
    """

    def __init__(
        self,
        slow_threshold_ms: float = 10000.0,
        max_records: int = 10000,
    ) -> None:
        self.slow_threshold_ms = slow_threshold_ms
        self.max_records = max_records
        self._exports: List[ExportRecord] = []
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_export(
        self,
        export_format: str,
        bytes_written: int,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record an export operation.

        Args:
            export_format: Export format (e.g., "llava", "openai")
            bytes_written: Number of bytes written
            duration_ms: Export duration in milliseconds
            success: Whether the export succeeded
            error: Optional error message if failed
        """
        record = ExportRecord(
            export_format=export_format,
            bytes_written=bytes_written,
            duration_ms=duration_ms,
            success=success,
            timestamp=time.time(),
            error=error,
        )

        with self._lock:
            self._exports.append(record)
            if len(self._exports) > self.max_records:
                self._exports = self._exports[-self.max_records :]

        # Record in global metrics
        get_metrics().increment(
            "exports_total",
            labels={"format": export_format, "status": "success" if success else "failed"},
        )
        get_metrics().increment(
            "export_throughput_bytes",
            value=float(bytes_written),
            labels={"format": export_format},
        )
        get_metrics().observe(
            "export_duration_seconds",
            duration_ms / 1000.0,
            labels={"format": export_format},
        )

        # Check for slow export
        if success and duration_ms > self.slow_threshold_ms:
            self._trigger_alert(
                "export_slow",
                {
                    "format": export_format,
                    "duration_ms": duration_ms,
                    "threshold_ms": self.slow_threshold_ms,
                    "bytes_written": bytes_written,
                    "severity": "warning",
                },
            )

        if not success:
            self._trigger_alert(
                "export_failed",
                {
                    "format": export_format,
                    "error": error or "unknown",
                    "severity": "critical",
                },
            )

        logger.info(
            "export_recorded",
            format=export_format,
            bytes_written=bytes_written,
            duration_ms=duration_ms,
            success=success,
        )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "export_alert",
            alert_type=alert_type,
            **details,
        )
        for handler in self._alert_handlers:
            try:
                handler(alert_type, details)
            except Exception as e:
                logger.error("alert_handler_failed", error=str(e))

    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Remove an alert handler.

        Returns:
            True if handler was found and removed
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            return True
        return False

    def get_throughput(
        self,
        export_format: Optional[str] = None,
        window_seconds: float = 300.0,
    ) -> float:
        """Get export throughput in bytes per second.

        Args:
            export_format: Optional format filter
            window_seconds: Time window for calculation

        Returns:
            Throughput in bytes per second
        """
        cutoff = time.time() - window_seconds
        total_bytes = 0
        total_time = 0.0

        with self._lock:
            for record in self._exports:
                if record.timestamp >= cutoff and record.success:
                    if export_format is None or record.export_format == export_format:
                        total_bytes += record.bytes_written
                        total_time += record.duration_ms / 1000.0

        if total_time == 0:
            return 0.0
        return total_bytes / total_time

    def get_export_stats(
        self,
        export_format: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get export statistics.

        Args:
            export_format: Optional format filter
            window_seconds: Optional time window

        Returns:
            Dict with count, success rate, avg duration, throughput
        """
        cutoff = time.time() - window_seconds if window_seconds else 0

        with self._lock:
            records = [
                r
                for r in self._exports
                if r.timestamp >= cutoff
                and (export_format is None or r.export_format == export_format)
            ]

        if not records:
            return {
                "format": export_format or "all",
                "count": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "total_bytes": 0,
                "throughput_bps": 0.0,
            }

        successful = [r for r in records if r.success]
        durations = [r.duration_ms for r in records]
        total_bytes = sum(r.bytes_written for r in successful)
        total_duration_sec = sum(durations) / 1000.0

        return {
            "format": export_format or "all",
            "count": len(records),
            "success_count": len(successful),
            "failure_count": len(records) - len(successful),
            "success_rate": len(successful) / len(records),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_bytes": total_bytes,
            "throughput_bps": total_bytes / max(total_duration_sec, 0.001),
        }

    def get_format_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare export performance across formats.

        Returns:
            Dict mapping format names to performance stats
        """
        with self._lock:
            formats: set[str] = set(r.export_format for r in self._exports)

        return {fmt: self.get_export_stats(export_format=fmt) for fmt in formats}

    def get_slow_exports(
        self,
        threshold_ms: Optional[float] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the slowest exports.

        Args:
            threshold_ms: Optional duration threshold
            n: Maximum number of exports to return

        Returns:
            List of slow export records
        """
        threshold = threshold_ms or self.slow_threshold_ms
        with self._lock:
            slow = [
                {
                    "format": r.export_format,
                    "duration_ms": r.duration_ms,
                    "bytes_written": r.bytes_written,
                    "timestamp": r.timestamp,
                }
                for r in self._exports
                if r.duration_ms > threshold and r.success
            ]
        return sorted(slow, key=lambda x: x["duration_ms"], reverse=True)[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive export statistics.

        Returns:
            Dict with overall stats and format breakdown
        """
        return {
            "total_exports": len(self._exports),
            "overall": self.get_export_stats(),
            "by_format": self.get_format_comparison(),
            "slow_exports": self.get_slow_exports(n=5),
            "slow_threshold_ms": self.slow_threshold_ms,
        }

    def is_healthy(self, export_format: Optional[str] = None) -> bool:
        """Check if exports are healthy.

        Args:
            export_format: Optional format to check

        Returns:
            True if success rate is above 95%
        """
        stats = self.get_export_stats(export_format, window_seconds=300)
        if stats["count"] == 0:
            return True
        return stats["success_rate"] >= 0.95

    def reset(self) -> None:
        """Reset all export data."""
        with self._lock:
            self._exports.clear()
