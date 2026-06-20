"""Reviewer metrics for tracking reviewer performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewSession:
    """A single review session."""
    session_id: str
    reviewer_id: str
    annotation_id: str
    start_time: str
    end_time: Optional[str] = None
    duration_min: Optional[float] = None
    agreement: Optional[bool] = None
    accuracy: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reviewer_id": self.reviewer_id,
            "annotation_id": self.annotation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_min": self.duration_min,
            "agreement": self.agreement,
            "accuracy": self.accuracy,
            "notes": self.notes,
        }


@dataclass
class ReviewerPerformance:
    """Performance metrics for a single reviewer."""
    reviewer_id: str
    name: str
    total_reviews: int = 0
    completed_reviews: int = 0
    agreement_count: int = 0
    total_accuracy: float = 0.0
    total_duration_min: float = 0.0
    avg_review_time_min: float = 0.0
    throughput_per_day: float = 0.0
    agreement_rate: float = 0.0
    accuracy_rate: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "name": self.name,
            "total_reviews": self.total_reviews,
            "completed_reviews": self.completed_reviews,
            "agreement_count": self.agreement_count,
            "total_accuracy": round(self.total_accuracy, 4),
            "avg_review_time_min": round(self.avg_review_time_min, 2),
            "throughput_per_day": round(self.throughput_per_day, 2),
            "agreement_rate": round(self.agreement_rate, 4),
            "accuracy_rate": round(self.accuracy_rate, 4),
            "rank": self.rank,
        }


class ReviewerMetrics:
    """Metrics tracking for reviewer performance."""

    def __init__(self):
        self._sessions: Dict[str, ReviewSession] = {}
        self._reviewer_sessions: Dict[str, List[str]] = {}

    def record_session(self, session: ReviewSession) -> None:
        self._sessions[session.session_id] = session
        if session.reviewer_id not in self._reviewer_sessions:
            self._reviewer_sessions[session.reviewer_id] = []
        self._reviewer_sessions[session.reviewer_id].append(session.session_id)
        logger.info("session_recorded", session_id=session.session_id, reviewer_id=session.reviewer_id)

    def complete_session(self, session_id: str, end_time: str, agreement: Optional[bool] = None, accuracy: Optional[float] = None) -> Optional[ReviewSession]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.end_time = end_time
        try:
            from datetime import datetime
            start = datetime.fromisoformat(session.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            session.duration_min = (end - start).total_seconds() / 60.0
        except (ValueError, AttributeError):
            session.duration_min = None
        session.agreement = agreement
        session.accuracy = accuracy
        logger.info("session_completed", session_id=session_id, agreement=agreement, accuracy=accuracy)
        return session

    def get_reviewer_stats(self, reviewer_id: str, name: str = "") -> ReviewerPerformance:
        session_ids = self._reviewer_sessions.get(reviewer_id, [])
        sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        completed = [s for s in sessions if s.end_time is not None]
        agreements = [s for s in completed if s.agreement is True]
        accuracies = [s.accuracy for s in completed if s.accuracy is not None]
        total_duration = sum(s.duration_min or 0 for s in completed)
        avg_time = total_duration / len(completed) if completed else 0.0
        throughput = 0.0
        if total_duration > 0:
            throughput = (len(completed) / total_duration) * 480
        return ReviewerPerformance(
            reviewer_id=reviewer_id,
            name=name or reviewer_id,
            total_reviews=len(sessions),
            completed_reviews=len(completed),
            agreement_count=len(agreements),
            total_accuracy=sum(accuracies) if accuracies else 0.0,
            total_duration_min=total_duration,
            avg_review_time_min=avg_time,
            throughput_per_day=throughput,
            agreement_rate=len(agreements) / len(completed) if completed else 0.0,
            accuracy_rate=sum(accuracies) / len(accuracies) if accuracies else 0.0,
        )

    def get_leaderboard(self) -> List[ReviewerPerformance]:
        stats_list = []
        for reviewer_id in self._reviewer_sessions:
            stats = self.get_reviewer_stats(reviewer_id)
            stats_list.append(stats)
        stats_list.sort(key=lambda s: (s.accuracy_rate, s.agreement_rate, s.throughput_per_day), reverse=True)
        for i, stats in enumerate(stats_list):
            stats.rank = i + 1
        return stats_list

    def get_top_reviewers(self, n: int = 5) -> List[ReviewerPerformance]:
        leaderboard = self.get_leaderboard()
        return leaderboard[:n]

    def get_global_statistics(self) -> Dict[str, Any]:
        all_sessions = list(self._sessions.values())
        completed = [s for s in all_sessions if s.end_time is not None]
        agreements = [s for s in completed if s.agreement is True]
        accuracies = [s.accuracy for s in completed if s.accuracy is not None]
        total_duration = sum(s.duration_min or 0 for s in completed)
        return {
            "total_reviewers": len(self._reviewer_sessions),
            "total_sessions": len(all_sessions),
            "completed_sessions": len(completed),
            "total_agreements": len(agreements),
            "avg_agreement_rate": len(agreements) / len(completed) if completed else 0.0,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "avg_review_time_min": total_duration / len(completed) if completed else 0.0,
            "total_review_time_hours": total_duration / 60.0,
        }

    def get_reviewer_trend(self, reviewer_id: str) -> List[Dict[str, Any]]:
        session_ids = self._reviewer_sessions.get(reviewer_id, [])
        sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        completed = [s for s in sessions if s.end_time is not None]
        completed.sort(key=lambda s: s.start_time)
        trend = []
        cumulative_agreements = 0
        cumulative_count = 0
        for session in completed:
            cumulative_count += 1
            if session.agreement:
                cumulative_agreements += 1
            trend.append({
                "session_id": session.session_id,
                "annotation_id": session.annotation_id,
                "start_time": session.start_time,
                "agreement": session.agreement,
                "accuracy": session.accuracy,
                "cumulative_agreement_rate": cumulative_agreements / cumulative_count if cumulative_count > 0 else 0.0,
            })
        return trend

    def get_comparison(self, reviewer_ids: List[str]) -> Dict[str, Any]:
        comparison = {
            "reviewers": [],
            "metrics": ["total_reviews", "agreement_rate", "accuracy_rate", "avg_time_min", "throughput"],
        }
        for rid in reviewer_ids:
            stats = self.get_reviewer_stats(rid)
            comparison["reviewers"].append({
                "reviewer_id": rid,
                "name": stats.name,
                "total_reviews": stats.total_reviews,
                "agreement_rate": round(stats.agreement_rate, 4),
                "accuracy_rate": round(stats.accuracy_rate, 4),
                "avg_time_min": round(stats.avg_review_time_min, 2),
                "throughput": round(stats.throughput_per_day, 2),
            })
        return comparison
