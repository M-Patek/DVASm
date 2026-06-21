"""Tests for batch API endpoints."""

import pytest
from fastapi.testclient import TestClient

from dvas.api.batch_api import router as batch_router


class TestBatchAPI:
    """Test batch API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(batch_router, prefix="/api/v1")
        return TestClient(app)

    def test_create_batch_job(self, client):
        """Test creating a batch job."""
        response = client.post(
            "/api/v1/batch/jobs",
            json={
                "video_ids": ["vid_1", "vid_2", "vid_3"],
                "teacher_model": "gpt-5.5",
                "num_frames": 16,
                "priority": 5,
                "max_retries": 3,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "batch_id" in data
        assert data["status"] == "pending"
        assert data["total_videos"] == 3
        assert "task_ids" in data
        assert len(data["task_ids"]) == 3

    def test_get_batch_status(self, client):
        """Test getting batch status."""
        # Create a batch job first
        create_response = client.post(
            "/api/v1/batch/jobs",
            json={"video_ids": ["vid_1"], "priority": 5},
        )
        batch_id = create_response.json()["batch_id"]

        response = client.get(f"/api/v1/batch/jobs/{batch_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        assert "status" in data
        assert "total_videos" in data

    def test_get_batch_status_not_found(self, client):
        """Test getting status for non-existent batch."""
        response = client.get("/api/v1/batch/jobs/nonexistent")
        assert response.status_code == 404

    def test_get_batch_results(self, client):
        """Test getting batch results."""
        # Create a batch job
        create_response = client.post(
            "/api/v1/batch/jobs",
            json={"video_ids": ["vid_1"], "priority": 5},
        )
        batch_id = create_response.json()["batch_id"]

        response = client.get(f"/api/v1/batch/jobs/{batch_id}/results")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        assert "results" in data
        assert "errors" in data

    def test_get_batch_progress(self, client):
        """Test getting batch progress."""
        # Create a batch job
        create_response = client.post(
            "/api/v1/batch/jobs",
            json={"video_ids": ["vid_1", "vid_2"], "priority": 5},
        )
        batch_id = create_response.json()["batch_id"]

        response = client.get(f"/api/v1/batch/jobs/{batch_id}/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        assert "progress" in data
        assert "total" in data
        assert data["total"] == 2

    def test_cancel_batch_job(self, client):
        """Test cancelling a batch job."""
        # Create a batch job
        create_response = client.post(
            "/api/v1/batch/jobs",
            json={"video_ids": ["vid_1"], "priority": 5},
        )
        batch_id = create_response.json()["batch_id"]

        response = client.post(f"/api/v1/batch/jobs/{batch_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        assert "cancelled" in data

    def test_cancel_completed_batch(self, client):
        """Test cancelling a completed batch job."""
        # Create a batch job
        create_response = client.post(
            "/api/v1/batch/jobs",
            json={"video_ids": ["vid_1"], "priority": 5},
        )
        batch_id = create_response.json()["batch_id"]

        # Cancel once
        client.post(f"/api/v1/batch/jobs/{batch_id}/cancel")

        # Try to cancel again (should fail)
        response = client.post(f"/api/v1/batch/jobs/{batch_id}/cancel")
        assert response.status_code == 400

    def test_cancel_nonexistent_batch(self, client):
        """Test cancelling non-existent batch."""
        response = client.post("/api/v1/batch/jobs/nonexistent/cancel")
        assert response.status_code == 404

    def test_list_batch_jobs(self, client):
        """Test listing batch jobs."""
        # Create a few batch jobs
        for i in range(3):
            client.post(
                "/api/v1/batch/jobs",
                json={"video_ids": [f"vid_{i}"], "priority": 5},
            )

        response = client.get("/api/v1/batch/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["total"] >= 3
        assert "offset" in data
        assert "limit" in data

    def test_list_batch_jobs_with_pagination(self, client):
        """Test listing with pagination."""
        # Create a few batch jobs
        for i in range(5):
            client.post(
                "/api/v1/batch/jobs",
                json={"video_ids": [f"vid_{i}"], "priority": 5},
            )

        response = client.get("/api/v1/batch/jobs?offset=0&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["offset"] == 0
        assert data["limit"] == 2

    def test_batch_job_models(self):
        """Test batch job pydantic models."""
        from dvas.api.batch_api import BatchJobRequest, BatchJobStatus

        request = BatchJobRequest(
            video_ids=["vid_1", "vid_2"],
            teacher_model="gpt-5.5",
            num_frames=16,
            priority=5,
            max_retries=3,
        )
        assert request.video_ids == ["vid_1", "vid_2"]
        assert request.teacher_model == "gpt-5.5"
        assert request.num_frames == 16
        assert request.priority == 5

        status = BatchJobStatus(
            batch_id="batch_123",
            status="pending",
            total_videos=2,
            completed=0,
            failed=0,
            pending=2,
            processing=0,
            progress=0.0,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        assert status.batch_id == "batch_123"
        assert status.total_videos == 2
