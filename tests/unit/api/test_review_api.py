"""Tests for review API endpoints."""

import pytest
from fastapi.testclient import TestClient

from dvas.api.review_api import router as review_router


class TestReviewAPI:
    """Test review API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(review_router, prefix="/api/v1")
        return TestClient(app)

    def test_get_review_queue_empty(self, client):
        """Test getting empty review queue."""
        response = client.get("/api/v1/review/queue")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_get_review_queue_with_filter(self, client):
        """Test getting review queue with status filter."""
        response = client.get("/api/v1/review/queue?status_filter=pending_review")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_add_to_review_queue_not_found(self, client):
        """Test adding non-existent annotation to review queue."""
        response = client.post("/api/v1/review/queue?annotation_id=nonexistent&priority=5")
        assert response.status_code == 404

    def test_assign_review_not_found(self, client):
        """Test assigning non-existent review."""
        response = client.post(
            "/api/v1/review/assign",
            json={"annotation_id": "nonexistent", "reviewer_id": "reviewer_1"},
        )
        assert response.status_code == 404

    def test_assign_review(self, client):
        """Test assigning a review."""
        # Note: This requires an annotation to exist in the store
        # Since we don't have real annotations, we test the error case
        response = client.post(
            "/api/v1/review/assign",
            json={"annotation_id": "test_ann", "reviewer_id": "reviewer_1"},
        )
        # Will fail because annotation not in queue
        assert response.status_code in [404, 409]

    def test_submit_review_decision_not_found(self, client):
        """Test submitting decision for non-existent review."""
        response = client.post(
            "/api/v1/review/decision",
            json={
                "annotation_id": "nonexistent",
                "decision": "approve",
                "reviewer_id": "reviewer_1",
            },
        )
        assert response.status_code == 404

    def test_get_review_stats(self, client):
        """Test getting review statistics."""
        response = client.get("/api/v1/review/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_pending" in data
        assert "total_approved" in data
        assert "total_rejected" in data
        assert "total_unassigned" in data
        assert "reviewer_breakdown" in data

    def test_get_review_item_not_found(self, client):
        """Test getting non-existent review item."""
        response = client.get("/api/v1/review/nonexistent")
        assert response.status_code == 404

    def test_review_decision_validation(self, client):
        """Test review decision validation."""
        # Invalid decision value
        response = client.post(
            "/api/v1/review/decision",
            json={
                "annotation_id": "test_ann",
                "decision": "invalid",
                "reviewer_id": "reviewer_1",
            },
        )
        # Will fail because annotation not in queue, but tests the schema
        assert response.status_code in [400, 404]

    def test_review_models(self):
        """Test review pydantic models."""
        from dvas.api.review_api import ReviewDecision, ReviewAssignment, ReviewStats

        assignment = ReviewAssignment(
            annotation_id="ann_123",
            reviewer_id="reviewer_1",
        )
        assert assignment.annotation_id == "ann_123"
        assert assignment.reviewer_id == "reviewer_1"

        decision = ReviewDecision(
            annotation_id="ann_123",
            decision="approve",
            reviewer_id="reviewer_1",
            comments="Looks good",
        )
        assert decision.decision == "approve"
        assert decision.comments == "Looks good"

        stats = ReviewStats(
            total_pending=5,
            total_approved=10,
            total_rejected=2,
            total_unassigned=3,
            average_review_time_seconds=None,
            reviewer_breakdown={},
        )
        assert stats.total_pending == 5
        assert stats.total_approved == 10

    def test_review_queue_item_model(self):
        """Test ReviewQueueItem model."""
        from dvas.api.review_api import ReviewQueueItem

        item = ReviewQueueItem(
            annotation_id="ann_123",
            video_id="vid_456",
            quality_score=0.85,
            status="pending_review",
            assigned_to=None,
            submitted_by="model",
            created_at="2024-01-01T00:00:00+00:00",
            priority=3,
        )
        assert item.annotation_id == "ann_123"
        assert item.status == "pending_review"
        assert item.priority == 3
