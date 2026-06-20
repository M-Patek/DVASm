"""Annotation quality monitoring for DVAS.

Tracks annotation quality trends over time with scoring
and threshold-based alerting.
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
class QualityScore:
    """A single quality score record."""

    video_id: str
    score: float
    timestamp: float
    model_name: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None


class AnnotationQualityMonitor:
    """Monitor annotation quality trends.

    Tracks quality scores over time, detects trends, and alerts
    when quality drops below thresholds.

    Usage::

        monitor = AnnotationQualityMonitor(min_score=0.7)
        monitor.record_score("vid_001", 0.85, model_name="gpt-5.5")
        trends = monitor.get_quality_trends()
    """

    def __init__(
        self,
        min_score: float = 0.7,
        alert_threshold: float = 0.6,
        max_records: int = 10000,
    ) -> None:
        self.min_score = min_score
        self.alert_threshold = alert_threshold
        self.max_records = max_records
        self._scores: List[QualityScore] = []
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_score(
        self,
        video_id: str,
        score: float,
        model_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a quality score.

        Args:
            video_id: Video identifier
            score: Quality score (0.0 to 1.0)
            model_name: Model that generated the annotation
            metadata: Optional additional metadata
        """
        entry = QualityScore(
            video_id=video_id,
            score=max(0.0, min(1.0, score)),
            timestamp=time.time(),
            model_name=model_name,
            metadata=metadata or {},
        )

        with self._lock:
            self._scores.append(entry)
            if len(self._scores) > self.max_records:
                self._scores = self._scores[-self.max_records :]

        # Record in global metrics
        get_metrics().gauge(
            "annotation_quality_score",
            score,
            labels={"model": model_name, "video_id": video_id},
        )
        get_metrics().increment(
            "annotations_scored_total",
            labels={"model": model_name},
        )

        # Check threshold
        if score < self.alert_threshold:
            self._trigger_alert(
                "quality_below_threshold",
                {
                    "video_id": video_id,
                    "score": score,
                    "threshold": self.alert_threshold,
                    "model": model_name,
                    "severity": "warning",
                },
            )

        logger.info(
            "quality_score_recorded",
            video_id=video_id,
            score=score,
            model=model_name,
        )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "quality_alert",
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

    def get_average_score(
        self,
        model_name: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> float:
        """Get average quality score.

        Args:
            model_name: Optional model filter
            window_seconds: Optional time window

        Returns:
            Average score (0.0 to 1.0)
        """
        scores = self._get_scores(model_name, window_seconds)
        if not scores:
            return 0.0
        return sum(s.score for s in scores) / len(scores)

    def get_score_distribution(
        self,
        model_name: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, int]:
        """Get quality score distribution in buckets.

        Args:
            model_name: Optional model filter
            window_seconds: Optional time window

        Returns:
            Dict mapping bucket names to counts
        """
        scores = self._get_scores(model_name, window_seconds)
        buckets = {
            "excellent (0.9-1.0)": 0,
            "good (0.8-0.9)": 0,
            "acceptable (0.7-0.8)": 0,
            "poor (0.6-0.7)": 0,
            "bad (0.0-0.6)": 0,
        }
        for s in scores:
            if s.score >= 0.9:
                buckets["excellent (0.9-1.0)"] += 1
            elif s.score >= 0.8:
                buckets["good (0.8-0.9)"] += 1
            elif s.score >= 0.7:
                buckets["acceptable (0.7-0.8)"] += 1
            elif s.score >= 0.6:
                buckets["poor (0.6-0.7)"] += 1
            else:
                buckets["bad (0.0-0.6)"] += 1
        return buckets

    def get_quality_trends(
        self,
        interval_seconds: float = 3600.0,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get quality trends over time intervals.

        Args:
            interval_seconds: Interval duration for bucketing
            model_name: Optional model filter

        Returns:
            List of trend data points with avg, min, max scores
        """
        scores = self._get_scores(model_name)
        if not scores:
            return []

        min_time = min(s.timestamp for s in scores)
        buckets: Dict[int, List[QualityScore]] = {}
        for s in scores:
            bucket = int((s.timestamp - min_time) / interval_seconds)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(s)

        trends = []
        for bucket in sorted(buckets.keys()):
            bucket_scores = [s.score for s in buckets[bucket]]
            trends.append({
                "timestamp": min_time + bucket * interval_seconds,
                "count": len(bucket_scores),
                "avg_score": sum(bucket_scores) / len(bucket_scores),
                "min_score": min(bucket_scores),
                "max_score": max(bucket_scores),
                "p50_score": sorted(bucket_scores)[len(bucket_scores) // 2],
            })

        return trends

    def get_low_quality_videos(
        self,
        threshold: Optional[float] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get videos with lowest quality scores.

        Args:
            threshold: Optional score threshold
            n: Maximum number of videos to return

        Returns:
            List of video quality records
        """
        threshold = threshold or self.alert_threshold
        with self._lock:
            low_scores = [
                {
                    "video_id": s.video_id,
                    "score": s.score,
                    "model": s.model_name,
                    "timestamp": s.timestamp,
                }
                for s in self._scores
                if s.score < threshold
            ]
        return sorted(low_scores, key=lambda x: x["score"])[:n]

    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare quality scores across models.

        Returns:
            Dict mapping model names to quality statistics
        """
        with self._lock:
            models: set[str] = set(s.model_name for s in self._scores)

        result: Dict[str, Dict[str, Any]] = {}
        for model in models:
            scores = self._get_scores(model)
            if scores:
                values = [s.score for s in scores]
                result[model] = {
                    "count": len(values),
                    "avg_score": sum(values) / len(values),
                    "min_score": min(values),
                    "max_score": max(values),
                }
        return result

    def _get_scores(
        self,
        model_name: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> List[QualityScore]:
        """Get filtered scores."""
        cutoff = time.time() - window_seconds if window_seconds else 0
        with self._lock:
            return [
                s for s in self._scores
                if s.timestamp >= cutoff
                and (model_name is None or s.model_name == model_name)
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics.

        Returns:
            Dict with overall stats, trends, and distributions
        """
        return {
            "total_scored": len(self._scores),
            "average_score": self.get_average_score(),
            "score_distribution": self.get_score_distribution(),
            "trends_1h": self.get_quality_trends(interval_seconds=3600),
            "model_comparison": self.get_model_comparison(),
            "low_quality_count": len(self.get_low_quality_videos()),
            "min_score_threshold": self.min_score,
            "alert_threshold": self.alert_threshold,
        }

    def is_quality_acceptable(self, model_name: Optional[str] = None) -> bool:
        """Check if quality is above minimum threshold.

        Args:
            model_name: Optional model to check

        Returns:
            True if average score is above minimum
        """
        return self.get_average_score(model_name) >= self.min_score

    def detect_degradation(
        self,
        recent_window: float = 3600.0,
        baseline_window: float = 86400.0,
    ) -> Optional[Dict[str, Any]]:
        """Detect quality degradation by comparing recent vs baseline.

        Args:
            recent_window: Recent time window in seconds
            baseline_window: Baseline time window in seconds

        Returns:
            Degradation info if detected, None otherwise
        """
        recent = self.get_average_score(window_seconds=recent_window)
        baseline = self.get_average_score(window_seconds=baseline_window)

        if baseline == 0:
            return None

        change = (recent - baseline) / baseline
        if change < -0.1:  # 10% degradation
            return {
                "degraded": True,
                "recent_avg": recent,
                "baseline_avg": baseline,
                "change_percent": change * 100,
                "severity": "critical" if change < -0.25 else "warning",
            }
        return None

    def reset(self) -> None:
        """Reset all quality data."""
        with self._lock:
            self._scores.clear()
