"""Student fallback rate monitoring for DVAS.

Tracks and analyzes student model fallback rates with alerting.
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
class FallbackRecord:
    """A single fallback event record."""

    reason: str
    student_model: str
    teacher_model: str
    timestamp: float
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StudentFallbackMonitor:
    """Monitor student model fallback rates.

    Tracks when student models fall back to teacher models,
    analyzes reasons, and alerts on high fallback rates.

    Usage::

        monitor = StudentFallbackMonitor(fallback_threshold=0.1)
        monitor.record_fallback("low_confidence", "student_v1", "gpt-5.5")
        rate = monitor.get_fallback_rate()
    """

    def __init__(
        self,
        fallback_threshold: float = 0.1,
        critical_threshold: float = 0.3,
        max_records: int = 10000,
    ) -> None:
        self.fallback_threshold = fallback_threshold
        self.critical_threshold = critical_threshold
        self.max_records = max_records
        self._fallbacks: List[FallbackRecord] = []
        self._total_student_calls: int = 0
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_fallback(
        self,
        reason: str,
        student_model: str,
        teacher_model: str,
        video_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a fallback event.

        Args:
            reason: Reason for fallback (e.g., "low_confidence", "error")
            student_model: Name of the student model
            teacher_model: Name of the teacher model used as fallback
            video_id: Optional associated video ID
            metadata: Optional additional metadata
        """
        record = FallbackRecord(
            reason=reason,
            student_model=student_model,
            teacher_model=teacher_model,
            timestamp=time.time(),
            video_id=video_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._fallbacks.append(record)
            if len(self._fallbacks) > self.max_records:
                self._fallbacks = self._fallbacks[-self.max_records :]

        # Record in global metrics
        get_metrics().increment(
            "student_fallback_total",
            labels={
                "reason": reason,
                "student_model": student_model,
                "teacher_model": teacher_model,
            },
        )

        logger.info(
            "student_fallback",
            reason=reason,
            student_model=student_model,
            teacher_model=teacher_model,
            video_id=video_id,
        )

        # Check threshold
        self._check_threshold(student_model)

    def record_student_call(self, student_model: str) -> None:
        """Record a student model call (non-fallback).

        Args:
            student_model: Name of the student model
        """
        with self._lock:
            self._total_student_calls += 1

        get_metrics().increment(
            "student_calls_total",
            labels={"student_model": student_model},
        )

    def _check_threshold(self, student_model: str) -> None:
        """Check if fallback rate exceeds threshold."""
        rate = self.get_fallback_rate(student_model, window_seconds=300)
        if rate >= self.critical_threshold:
            self._trigger_alert(
                "fallback_rate_critical",
                {
                    "student_model": student_model,
                    "fallback_rate": rate,
                    "threshold": self.critical_threshold,
                    "severity": "critical",
                },
            )
        elif rate >= self.fallback_threshold:
            self._trigger_alert(
                "fallback_rate_warning",
                {
                    "student_model": student_model,
                    "fallback_rate": rate,
                    "threshold": self.fallback_threshold,
                    "severity": "warning",
                },
            )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "fallback_alert",
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

    def get_fallback_rate(
        self,
        student_model: Optional[str] = None,
        window_seconds: float = 3600.0,
    ) -> float:
        """Get fallback rate for a student model.

        Args:
            student_model: Optional student model filter
            window_seconds: Time window for calculation

        Returns:
            Fallback rate (0.0 to 1.0)
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            fallbacks = [
                f for f in self._fallbacks
                if f.timestamp >= cutoff
                and (student_model is None or f.student_model == student_model)
            ]
            total_calls = self._total_student_calls + len(self._fallbacks)
            if total_calls == 0:
                return 0.0
            return len(fallbacks) / total_calls

    def get_fallback_by_reason(
        self,
        student_model: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, int]:
        """Get fallback counts by reason.

        Args:
            student_model: Optional student model filter
            window_seconds: Optional time window

        Returns:
            Dict mapping reasons to fallback counts
        """
        cutoff = time.time() - window_seconds if window_seconds else 0
        counts: Dict[str, int] = {}
        with self._lock:
            for fallback in self._fallbacks:
                if fallback.timestamp >= cutoff:
                    if student_model is None or fallback.student_model == student_model:
                        counts[fallback.reason] = counts.get(fallback.reason, 0) + 1
        return counts

    def get_fallback_trends(
        self,
        interval_seconds: float = 3600.0,
        student_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get fallback trends over time.

        Args:
            interval_seconds: Interval for bucketing
            student_model: Optional student model filter

        Returns:
            List of trend data points
        """
        with self._lock:
            records = [
                f for f in self._fallbacks
                if student_model is None or f.student_model == student_model
            ]
            if not records:
                return []

            min_time = min(f.timestamp for f in records)
            buckets: Dict[int, List[FallbackRecord]] = {}
            for record in records:
                bucket = int((record.timestamp - min_time) / interval_seconds)
                if bucket not in buckets:
                    buckets[bucket] = []
                buckets[bucket].append(record)

            trends = []
            for bucket in sorted(buckets.keys()):
                bucket_records = buckets[bucket]
                reason_counts: Dict[str, int] = {}
                for r in bucket_records:
                    reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

                trends.append({
                    "timestamp": min_time + bucket * interval_seconds,
                    "count": len(bucket_records),
                    "reason_breakdown": reason_counts,
                })

            return trends

    def get_stats(self, student_model: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive fallback statistics.

        Args:
            student_model: Optional student model filter

        Returns:
            Dict with rates, trends, and breakdowns
        """
        return {
            "total_fallbacks": len(self._fallbacks),
            "fallback_rate_5m": self.get_fallback_rate(student_model, 300),
            "fallback_rate_1h": self.get_fallback_rate(student_model, 3600),
            "fallback_rate_24h": self.get_fallback_rate(student_model, 86400),
            "by_reason": self.get_fallback_by_reason(student_model),
            "trends": self.get_fallback_trends(student_model=student_model),
            "threshold": self.fallback_threshold,
            "critical_threshold": self.critical_threshold,
        }

    def get_top_fallback_reasons(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most common fallback reasons.

        Args:
            n: Number of reasons to return

        Returns:
            List of dicts with reason and count
        """
        counts = self.get_fallback_by_reason()
        sorted_reasons = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"reason": reason, "count": count}
            for reason, count in sorted_reasons[:n]
        ]

    def is_healthy(self, student_model: Optional[str] = None) -> bool:
        """Check if fallback rate is healthy.

        Args:
            student_model: Optional student model to check

        Returns:
            True if fallback rate is below threshold
        """
        return self.get_fallback_rate(student_model, window_seconds=300) <= self.fallback_threshold

    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare fallback rates across student models.

        Returns:
            Dict mapping student models to fallback statistics
        """
        with self._lock:
            models: set[str] = set(f.student_model for f in self._fallbacks)

        result: Dict[str, Dict[str, Any]] = {}
        for model in models:
            result[model] = {
                "fallback_rate_1h": self.get_fallback_rate(model, 3600),
                "fallback_rate_24h": self.get_fallback_rate(model, 86400),
                "by_reason": self.get_fallback_by_reason(model),
            }
        return result

    def reset(self, student_model: Optional[str] = None) -> None:
        """Reset fallback data.

        Args:
            student_model: Optional model to reset (all if None)
        """
        with self._lock:
            if student_model:
                self._fallbacks = [
                    f for f in self._fallbacks if f.student_model != student_model
                ]
            else:
                self._fallbacks.clear()
                self._total_student_calls = 0
