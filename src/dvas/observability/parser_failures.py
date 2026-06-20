"""Parser failure monitoring for DVAS.

Tracks and alerts on parser failures with categorization
and trend analysis.
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
class ParserFailure:
    """A single parser failure record."""

    parser_type: str
    error_type: str
    timestamp: float
    video_id: Optional[str] = None
    raw_text_preview: Optional[str] = None


class ParserFailureMonitor:
    """Monitor parser failures with alerting and trend analysis.

    Tracks failures by parser type and error category, triggering
    alerts when failure rates exceed thresholds.

    Usage::

        monitor = ParserFailureMonitor(failure_threshold=0.1)
        monitor.record_failure("json_parser", "json_decode_error", video_id="vid_001")
        trends = monitor.get_failure_trends()
    """

    def __init__(
        self,
        failure_threshold: float = 0.05,
        alert_window_seconds: float = 300.0,
        max_records: int = 10000,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.alert_window_seconds = alert_window_seconds
        self.max_records = max_records
        self._failures: List[ParserFailure] = []
        self._total_parsed: int = 0
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_failure(
        self,
        parser_type: str,
        error_type: str,
        video_id: Optional[str] = None,
        raw_text_preview: Optional[str] = None,
    ) -> None:
        """Record a parser failure.

        Args:
            parser_type: Type of parser that failed (e.g., "json_parser")
            error_type: Category of error (e.g., "json_decode_error")
            video_id: Optional associated video ID
            raw_text_preview: Optional preview of the text that failed to parse
        """
        failure = ParserFailure(
            parser_type=parser_type,
            error_type=error_type,
            timestamp=time.time(),
            video_id=video_id,
            raw_text_preview=raw_text_preview,
        )

        with self._lock:
            self._failures.append(failure)
            if len(self._failures) > self.max_records:
                self._failures = self._failures[-self.max_records :]

        # Record in global metrics
        get_metrics().increment(
            "parser_failures_total",
            labels={"parser_type": parser_type, "error_type": error_type},
        )

        logger.warning(
            "parser_failure",
            parser_type=parser_type,
            error_type=error_type,
            video_id=video_id,
        )

        # Check threshold
        self._check_threshold(parser_type)

    def record_success(self, parser_type: str) -> None:
        """Record a successful parse.

        Args:
            parser_type: Type of parser that succeeded
        """
        with self._lock:
            self._total_parsed += 1

        get_metrics().increment(
            "parser_success_total",
            labels={"parser_type": parser_type},
        )

    def _check_threshold(self, parser_type: str) -> None:
        """Check if failure rate exceeds threshold."""
        rate = self.get_failure_rate(parser_type, window_seconds=self.alert_window_seconds)
        if rate > self.failure_threshold:
            self._trigger_alert(
                "parser_failure_rate_exceeded",
                {
                    "parser_type": parser_type,
                    "failure_rate": rate,
                    "threshold": self.failure_threshold,
                    "window_seconds": self.alert_window_seconds,
                    "severity": "warning" if rate < 0.2 else "critical",
                },
            )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "parser_failure_alert",
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

    def get_failure_rate(
        self,
        parser_type: Optional[str] = None,
        window_seconds: float = 300.0,
    ) -> float:
        """Get failure rate for a parser type.

        Args:
            parser_type: Optional parser type filter
            window_seconds: Time window to consider

        Returns:
            Failure rate (0.0 to 1.0)
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            failures = [
                f for f in self._failures
                if f.timestamp >= cutoff
                and (parser_type is None or f.parser_type == parser_type)
            ]
            total = self._total_parsed + len(self._failures)
            if total == 0:
                return 0.0
            return len(failures) / total

    def get_failure_counts(
        self,
        parser_type: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, int]:
        """Get failure counts by error type.

        Args:
            parser_type: Optional parser type filter
            window_seconds: Optional time window

        Returns:
            Dict mapping error types to failure counts
        """
        cutoff = time.time() - window_seconds if window_seconds else 0
        counts: Dict[str, int] = {}
        with self._lock:
            for failure in self._failures:
                if failure.timestamp >= cutoff:
                    if parser_type is None or failure.parser_type == parser_type:
                        counts[failure.error_type] = counts.get(failure.error_type, 0) + 1
        return counts

    def get_failure_trends(self, interval_seconds: float = 3600.0) -> List[Dict[str, Any]]:
        """Get failure trends over time intervals.

        Args:
            interval_seconds: Interval duration for bucketing

        Returns:
            List of trend data points
        """
        _ = time.time()  # now captured for potential future use
        with self._lock:
            if not self._failures:
                return []

            # Find time range
            min_time = min(f.timestamp for f in self._failures)
            _ = max(f.timestamp for f in self._failures)  # max_time for range context

            # Create buckets
            buckets: Dict[int, List[ParserFailure]] = {}
            for failure in self._failures:
                bucket = int((failure.timestamp - min_time) / interval_seconds)
                if bucket not in buckets:
                    buckets[bucket] = []
                buckets[bucket].append(failure)

            trends = []
            for bucket in sorted(buckets.keys()):
                bucket_time = min_time + bucket * interval_seconds
                failures = buckets[bucket]
                error_counts: Dict[str, int] = {}
                for f in failures:
                    error_counts[f.error_type] = error_counts.get(f.error_type, 0) + 1

                trends.append({
                    "timestamp": bucket_time,
                    "count": len(failures),
                    "error_breakdown": error_counts,
                })

            return trends

    def get_most_common_errors(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most common error types.

        Args:
            n: Number of error types to return

        Returns:
            List of dicts with error type and count
        """
        counts = self.get_failure_counts()
        sorted_errors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"error_type": error, "count": count}
            for error, count in sorted_errors[:n]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive parser failure statistics.

        Returns:
            Dict with total failures, rates, trends, and top errors
        """
        with self._lock:
            total_failures = len(self._failures)

        return {
            "total_failures": total_failures,
            "total_parsed": self._total_parsed,
            "failure_rate_5m": self.get_failure_rate(window_seconds=300),
            "failure_rate_1h": self.get_failure_rate(window_seconds=3600),
            "failure_rate_24h": self.get_failure_rate(window_seconds=86400),
            "threshold": self.failure_threshold,
            "top_errors": self.get_most_common_errors(),
            "by_parser": self._get_by_parser(),
        }

    def _get_by_parser(self) -> Dict[str, Dict[str, Any]]:
        """Get stats grouped by parser type."""
        result: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for failure in self._failures:
                pt = failure.parser_type
                if pt not in result:
                    result[pt] = {"count": 0, "errors": {}}
                result[pt]["count"] += 1
                result[pt]["errors"][failure.error_type] = (
                    result[pt]["errors"].get(failure.error_type, 0) + 1
                )
        return result

    def is_healthy(self, parser_type: Optional[str] = None) -> bool:
        """Check if parser failure rate is within acceptable bounds.

        Args:
            parser_type: Optional parser type to check

        Returns:
            True if failure rate is below threshold
        """
        return self.get_failure_rate(parser_type, window_seconds=300) <= self.failure_threshold

    def reset(self) -> None:
        """Reset all failure data."""
        with self._lock:
            self._failures.clear()
            self._total_parsed = 0
