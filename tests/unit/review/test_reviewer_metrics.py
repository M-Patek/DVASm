"""Tests for reviewer metrics calculation."""

import pytest

from dvas.review.reviewer_metrics import (
    ReviewerMetrics,
    ReviewSession,
)


class TestReviewerMetrics:
    """Test suite for ReviewerMetrics."""

    def test_record_session(self):
        """Test recording a review session."""
        metrics = ReviewerMetrics()
        session = ReviewSession(
            session_id="s1",
            reviewer_id="rev1",
            annotation_id="ann1",
            start_time="2024-01-01T10:00:00",
        )
        metrics.record_session(session)

        assert len(metrics._sessions) == 1
        assert "s1" in metrics._sessions

    def test_complete_session(self):
        """Test completing a review session."""
        metrics = ReviewerMetrics()
        session = ReviewSession(
            session_id="s1",
            reviewer_id="rev1",
            annotation_id="ann1",
            start_time="2024-01-01T10:00:00",
        )
        metrics.record_session(session)

        result = metrics.complete_session(
            "s1",
            end_time="2024-01-01T10:15:00",
            agreement=True,
            accuracy=0.85,
        )
        assert result is not None
        assert result.end_time == "2024-01-01T10:15:00"
        assert result.agreement is True
        assert result.accuracy == pytest.approx(0.85, abs=0.01)
        assert result.duration_min is not None

    def test_complete_nonexistent_session(self):
        """Test completing a non-existent session."""
        metrics = ReviewerMetrics()
        result = metrics.complete_session("nonexistent", "2024-01-01T10:00:00")
        assert result is None

    def test_get_reviewer_stats(self):
        """Test reviewer statistics calculation."""
        metrics = ReviewerMetrics()

        # Record sessions for reviewer rev1
        for i in range(5):
            session = ReviewSession(
                session_id=f"s{i}",
                reviewer_id="rev1",
                annotation_id=f"ann{i}",
                start_time=f"2024-01-0{i + 1}T10:00:00",
            )
            metrics.record_session(session)
            metrics.complete_session(
                f"s{i}",
                end_time=f"2024-01-0{i + 1}T10:15:00",
                agreement=(i < 4),  # 4 agreements, 1 disagreement
                accuracy=0.8 + i * 0.02,
            )

        stats = metrics.get_reviewer_stats("rev1", name="Alice")
        assert stats.reviewer_id == "rev1"
        assert stats.name == "Alice"
        assert stats.total_reviews == 5
        assert stats.completed_reviews == 5
        assert stats.agreement_count == 4
        assert stats.agreement_rate == pytest.approx(4 / 5, abs=0.01)
        assert stats.accuracy_rate > 0

    def test_get_reviewer_stats_empty(self):
        """Test reviewer stats with no sessions."""
        metrics = ReviewerMetrics()
        stats = metrics.get_reviewer_stats("rev1", name="Alice")

        assert stats.total_reviews == 0
        assert stats.completed_reviews == 0
        assert stats.agreement_rate == 0.0
        assert stats.accuracy_rate == 0.0
        assert stats.avg_review_time_min == 0.0

    def test_get_leaderboard(self):
        """Test leaderboard generation."""
        metrics = ReviewerMetrics()

        # Reviewer 1: high accuracy
        for i in range(3):
            session = ReviewSession(
                session_id=f"r1_s{i}",
                reviewer_id="rev1",
                annotation_id=f"ann{i}",
                start_time=f"2024-01-0{i + 1}T10:00:00",
            )
            metrics.record_session(session)
            metrics.complete_session(
                f"r1_s{i}",
                end_time=f"2024-01-0{i + 1}T10:10:00",
                agreement=True,
                accuracy=0.95,
            )

        # Reviewer 2: lower accuracy
        for i in range(3):
            session = ReviewSession(
                session_id=f"r2_s{i}",
                reviewer_id="rev2",
                annotation_id=f"ann{i}",
                start_time=f"2024-01-0{i + 1}T10:00:00",
            )
            metrics.record_session(session)
            metrics.complete_session(
                f"r2_s{i}",
                end_time=f"2024-01-0{i + 1}T10:10:00",
                agreement=True,
                accuracy=0.7,
            )

        leaderboard = metrics.get_leaderboard()
        assert len(leaderboard) == 2
        # rev1 should be ranked higher due to higher accuracy
        assert leaderboard[0].reviewer_id == "rev1"
        assert leaderboard[0].rank == 1
        assert leaderboard[1].reviewer_id == "rev2"
        assert leaderboard[1].rank == 2

    def test_get_top_reviewers(self):
        """Test getting top N reviewers."""
        metrics = ReviewerMetrics()

        for rid in ["rev1", "rev2", "rev3", "rev4"]:
            session = ReviewSession(
                session_id=f"s_{rid}",
                reviewer_id=rid,
                annotation_id="ann1",
                start_time="2024-01-01T10:00:00",
            )
            metrics.record_session(session)
            metrics.complete_session(
                f"s_{rid}",
                end_time="2024-01-01T10:10:00",
                agreement=True,
                accuracy=0.5 + hash(rid) % 50 / 100,  # Random-ish accuracy
            )

        top = metrics.get_top_reviewers(n=2)
        assert len(top) == 2
        assert top[0].rank == 1
        assert top[1].rank == 2

    def test_get_global_statistics(self):
        """Test global statistics."""
        metrics = ReviewerMetrics()

        for rid in ["rev1", "rev2"]:
            for i in range(3):
                session = ReviewSession(
                    session_id=f"s_{rid}_{i}",
                    reviewer_id=rid,
                    annotation_id=f"ann{i}",
                    start_time=f"2024-01-0{i + 1}T10:00:00",
                )
                metrics.record_session(session)
                metrics.complete_session(
                    f"s_{rid}_{i}",
                    end_time=f"2024-01-0{i + 1}T10:15:00",
                    agreement=(i < 2),
                    accuracy=0.8,
                )

        stats = metrics.get_global_statistics()
        assert stats["total_reviewers"] == 2
        assert stats["total_sessions"] == 6
        assert stats["completed_sessions"] == 6
        assert stats["total_agreements"] == 4
        assert stats["avg_agreement_rate"] == pytest.approx(4 / 6, abs=0.01)
        assert stats["avg_accuracy"] == pytest.approx(0.8, abs=0.01)

    def test_get_reviewer_trend(self):
        """Test reviewer trend data."""
        metrics = ReviewerMetrics()

        for i in range(5):
            session = ReviewSession(
                session_id=f"s{i}",
                reviewer_id="rev1",
                annotation_id=f"ann{i}",
                start_time=f"2024-01-0{i + 1}T10:00:00",
            )
            metrics.record_session(session)
            metrics.complete_session(
                f"s{i}",
                end_time=f"2024-01-0{i + 1}T10:10:00",
                agreement=(i % 2 == 0),  # alternating agreement
                accuracy=0.8,
            )

        trend = metrics.get_reviewer_trend("rev1")
        assert len(trend) == 5
        # Check cumulative agreement rate
        assert trend[0]["cumulative_agreement_rate"] == 1.0  # First session: agreed
        assert trend[1]["cumulative_agreement_rate"] == 0.5  # 1/2 agreed

    def test_get_comparison(self):
        """Test reviewer comparison."""
        metrics = ReviewerMetrics()

        for rid in ["rev1", "rev2"]:
            for i in range(3):
                session = ReviewSession(
                    session_id=f"s_{rid}_{i}",
                    reviewer_id=rid,
                    annotation_id=f"ann{i}",
                    start_time=f"2024-01-0{i + 1}T10:00:00",
                )
                metrics.record_session(session)
                metrics.complete_session(
                    f"s_{rid}_{i}",
                    end_time=f"2024-01-0{i + 1}T10:10:00",
                    agreement=True,
                    accuracy=0.9 if rid == "rev1" else 0.7,
                )

        comparison = metrics.get_comparison(["rev1", "rev2"])
        assert len(comparison["reviewers"]) == 2
        rev1_data = next(r for r in comparison["reviewers"] if r["reviewer_id"] == "rev1")
        rev2_data = next(r for r in comparison["reviewers"] if r["reviewer_id"] == "rev2")
        assert rev1_data["accuracy_rate"] > rev2_data["accuracy_rate"]

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = ReviewerMetrics()

        # Create sessions with known durations
        session = ReviewSession(
            session_id="s1",
            reviewer_id="rev1",
            annotation_id="ann1",
            start_time="2024-01-01T10:00:00",
        )
        metrics.record_session(session)
        metrics.complete_session(
            "s1",
            end_time="2024-01-01T10:30:00",  # 30 minutes
            agreement=True,
            accuracy=0.9,
        )

        stats = metrics.get_reviewer_stats("rev1")
        # 1 review in 30 min -> 2 per hour -> 16 per 8-hour day
        assert stats.throughput_per_day > 0

    def test_session_duration_calculation(self):
        """Test session duration calculation."""
        metrics = ReviewerMetrics()

        session = ReviewSession(
            session_id="s1",
            reviewer_id="rev1",
            annotation_id="ann1",
            start_time="2024-01-01T10:00:00",
        )
        metrics.record_session(session)
        result = metrics.complete_session(
            "s1",
            end_time="2024-01-01T10:15:00",
            agreement=True,
        )

        assert result.duration_min is not None
        assert result.duration_min == pytest.approx(15.0, abs=0.1)
