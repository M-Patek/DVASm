"""Task queue depth monitoring for DVAS.

Tracks queue health metrics with depth, throughput, and latency monitoring.
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
class QueueMetrics:
    """Metrics for a single queue."""

    queue_name: str
    depth: int = 0
    max_depth: int = 0
    processed_count: int = 0
    failed_count: int = 0
    avg_wait_time_ms: float = 0.0
    last_updated: float = 0.0


class TaskQueueMonitor:
    """Monitor task queue health.

    Tracks queue depth, processing rates, and wait times with
    alerting on queue depth thresholds.

    Usage::

        monitor = TaskQueueMonitor()
        monitor.record_depth("annotation_queue", 150)
        monitor.record_task_completed("annotation_queue", wait_time_ms=5000)
        health = monitor.get_queue_health("annotation_queue")
    """

    def __init__(
        self,
        depth_threshold: int = 100,
        critical_depth: int = 500,
        max_wait_time_ms: float = 30000.0,
    ) -> None:
        self.depth_threshold = depth_threshold
        self.critical_depth = critical_depth
        self.max_wait_time_ms = max_wait_time_ms
        self._queues: Dict[str, QueueMetrics] = {}
        self._wait_times: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_depth(self, queue_name: str, depth: int) -> None:
        """Record current queue depth.

        Args:
            queue_name: Name of the queue
            depth: Current number of items in queue
        """
        with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = QueueMetrics(queue_name=queue_name)

            self._queues[queue_name].depth = depth
            self._queues[queue_name].max_depth = max(
                self._queues[queue_name].max_depth, depth
            )
            self._queues[queue_name].last_updated = time.time()

        # Record in global metrics
        get_metrics().gauge(
            "task_queue_depth",
            float(depth),
            labels={"queue_name": queue_name},
        )

        # Check thresholds
        if depth >= self.critical_depth:
            self._trigger_alert(
                "queue_depth_critical",
                {
                    "queue_name": queue_name,
                    "depth": depth,
                    "threshold": self.critical_depth,
                    "severity": "critical",
                },
            )
        elif depth >= self.depth_threshold:
            self._trigger_alert(
                "queue_depth_warning",
                {
                    "queue_name": queue_name,
                    "depth": depth,
                    "threshold": self.depth_threshold,
                    "severity": "warning",
                },
            )

    def record_task_enqueued(self, queue_name: str) -> None:
        """Record a task being added to the queue.

        Args:
            queue_name: Name of the queue
        """
        get_metrics().increment(
            "tasks_enqueued_total",
            labels={"queue_name": queue_name},
        )

    def record_task_started(self, queue_name: str) -> None:
        """Record a task being picked up for processing.

        Args:
            queue_name: Name of the queue
        """
        get_metrics().increment(
            "tasks_started_total",
            labels={"queue_name": queue_name},
        )

    def record_task_completed(
        self,
        queue_name: str,
        wait_time_ms: float = 0.0,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Record a completed task.

        Args:
            queue_name: Name of the queue
            wait_time_ms: Time task waited in queue
            processing_time_ms: Time spent processing
        """
        with self._lock:
            if queue_name in self._queues:
                self._queues[queue_name].processed_count += 1

            if queue_name not in self._wait_times:
                self._wait_times[queue_name] = []
            self._wait_times[queue_name].append(wait_time_ms)
            # Keep last 1000 wait times
            if len(self._wait_times[queue_name]) > 1000:
                self._wait_times[queue_name] = self._wait_times[queue_name][-1000:]

        get_metrics().increment(
            "tasks_completed_total",
            labels={"queue_name": queue_name},
        )
        get_metrics().observe(
            "task_wait_time_seconds",
            wait_time_ms / 1000.0,
            labels={"queue_name": queue_name},
        )
        get_metrics().observe(
            "task_processing_time_seconds",
            processing_time_ms / 1000.0,
            labels={"queue_name": queue_name},
        )

        # Check wait time threshold
        if wait_time_ms > self.max_wait_time_ms:
            self._trigger_alert(
                "queue_wait_time_exceeded",
                {
                    "queue_name": queue_name,
                    "wait_time_ms": wait_time_ms,
                    "threshold_ms": self.max_wait_time_ms,
                    "severity": "warning",
                },
            )

    def record_task_failed(self, queue_name: str, error: str) -> None:
        """Record a failed task.

        Args:
            queue_name: Name of the queue
            error: Error message or type
        """
        with self._lock:
            if queue_name in self._queues:
                self._queues[queue_name].failed_count += 1

        get_metrics().increment(
            "tasks_failed_total",
            labels={"queue_name": queue_name, "error": error},
        )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "queue_alert",
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

    def get_queue_health(self, queue_name: str) -> Dict[str, Any]:
        """Get health status for a queue.

        Args:
            queue_name: Name of the queue

        Returns:
            Dict with depth, throughput, and health status
        """
        with self._lock:
            metrics = self._queues.get(queue_name)
            if not metrics:
                return {
                    "queue_name": queue_name,
                    "status": "unknown",
                    "depth": 0,
                    "max_depth": 0,
                }

            depth = metrics.depth
            wait_times = self._wait_times.get(queue_name, [])
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

            if depth >= self.critical_depth:
                status = "critical"
            elif depth >= self.depth_threshold:
                status = "warning"
            elif avg_wait > self.max_wait_time_ms:
                status = "degraded"
            else:
                status = "healthy"

            return {
                "queue_name": queue_name,
                "status": status,
                "depth": depth,
                "max_depth": metrics.max_depth,
                "processed_count": metrics.processed_count,
                "failed_count": metrics.failed_count,
                "avg_wait_time_ms": avg_wait,
                "last_updated": metrics.last_updated,
            }

    def get_all_queue_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all queues.

        Returns:
            Dict mapping queue names to health status
        """
        with self._lock:
            queues = list(self._queues.keys())
        return {name: self.get_queue_health(name) for name in queues}

    def get_throughput(self, queue_name: str, window_seconds: float = 300.0) -> float:
        """Get task throughput for a queue.

        Args:
            queue_name: Name of the queue
            window_seconds: Time window for throughput calculation

        Returns:
            Tasks per second
        """
        with self._lock:
            metrics = self._queues.get(queue_name)
            if not metrics or not metrics.processed_count:
                return 0.0

        # Approximate based on total processed / uptime
        # In production, this would use time-bucketed counters
        return metrics.processed_count / max(window_seconds, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics.

        Returns:
            Dict with all queue metrics and overall health
        """
        all_health = self.get_all_queue_health()
        total_depth = sum(h["depth"] for h in all_health.values())
        critical_queues = [
            name for name, h in all_health.items() if h["status"] == "critical"
        ]
        warning_queues = [
            name for name, h in all_health.items() if h["status"] == "warning"
        ]

        return {
            "total_queues": len(all_health),
            "total_depth": total_depth,
            "healthy_queues": len(all_health) - len(critical_queues) - len(warning_queues),
            "warning_queues": warning_queues,
            "critical_queues": critical_queues,
            "queue_health": all_health,
            "depth_threshold": self.depth_threshold,
            "critical_depth": self.critical_depth,
        }

    def is_healthy(self, queue_name: Optional[str] = None) -> bool:
        """Check if queues are healthy.

        Args:
            queue_name: Optional specific queue to check (all if None)

        Returns:
            True if no queues are in critical state
        """
        if queue_name:
            health = self.get_queue_health(queue_name)
            return health["status"] not in ("critical", "warning")

        all_health = self.get_all_queue_health()
        return all(h["status"] != "critical" for h in all_health.values())

    def reset(self, queue_name: Optional[str] = None) -> None:
        """Reset queue data.

        Args:
            queue_name: Optional queue to reset (all if None)
        """
        with self._lock:
            if queue_name:
                self._queues.pop(queue_name, None)
                self._wait_times.pop(queue_name, None)
            else:
                self._queues.clear()
                self._wait_times.clear()
