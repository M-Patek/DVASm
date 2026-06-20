"""Tests for dataset API endpoints."""

import pytest
from fastapi.testclient import TestClient

from dvas.api.dataset_api import router as dataset_router


class TestDatasetAPI:
    """Test dataset API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(dataset_router, prefix="/api/v1")
        return TestClient(app)

    def test_create_dataset(self, client):
        """Test creating a dataset."""
        response = client.post(
            "/api/v1/datasets",
            json={
                "name": "Test Dataset",
                "description": "A test dataset",
                "source": "model",
                "tags": ["test", "demo"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Dataset"
        assert data["source"] == "model"
        assert data["tags"] == ["test", "demo"]

    def test_get_dataset(self, client):
        """Test getting a dataset."""
        # Create first
        create_response = client.post(
            "/api/v1/datasets",
            json={"name": "Test Dataset", "source": "model"},
        )
        dataset_id = create_response.json()["id"]

        response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"

    def test_get_dataset_not_found(self, client):
        """Test getting non-existent dataset."""
        response = client.get("/api/v1/datasets/nonexistent")
        assert response.status_code == 404

    def test_update_dataset(self, client):
        """Test updating a dataset."""
        # Create first
        create_response = client.post(
            "/api/v1/datasets",
            json={"name": "Original Name", "source": "model"},
        )
        dataset_id = create_response.json()["id"]

        # Update
        response = client.put(
            f"/api/v1/datasets/{dataset_id}",
            json={"name": "Updated Name", "tags": ["updated"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert "updated" in data["tags"]

    def test_update_dataset_not_found(self, client):
        """Test updating non-existent dataset."""
        response = client.put(
            "/api/v1/datasets/nonexistent",
            json={"name": "Updated"},
        )
        assert response.status_code == 404

    def test_delete_dataset(self, client):
        """Test deleting a dataset."""
        # Create first
        create_response = client.post(
            "/api/v1/datasets",
            json={"name": "To Delete", "source": "model"},
        )
        dataset_id = create_response.json()["id"]

        # Delete
        response = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        assert "deleted" in response.json()["message"]

        # Verify deleted
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 404

    def test_delete_dataset_not_found(self, client):
        """Test deleting non-existent dataset."""
        response = client.delete("/api/v1/datasets/nonexistent")
        assert response.status_code == 404

    def test_list_datasets(self, client):
        """Test listing datasets."""
        # Create a few datasets
        for i in range(3):
            client.post(
                "/api/v1/datasets",
                json={"name": f"Dataset {i}", "source": "model"},
            )

        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert data["total"] >= 3
        assert "offset" in data
        assert "limit" in data

    def test_list_datasets_with_filter(self, client):
        """Test listing with source filter."""
        # Create datasets with different sources
        client.post(
            "/api/v1/datasets",
            json={"name": "Gold Dataset", "source": "gold"},
        )
        client.post(
            "/api/v1/datasets",
            json={"name": "Model Dataset", "source": "model"},
        )

        response = client.get("/api/v1/datasets?source=gold")
        assert response.status_code == 200
        data = response.json()
        assert all(d["source"] == "gold" for d in data["datasets"])

    def test_list_datasets_with_tag_filter(self, client):
        """Test listing with tag filter."""
        # Create datasets with different tags
        client.post(
            "/api/v1/datasets",
            json={"name": "Tagged Dataset", "source": "model", "tags": ["important"]},
        )
        client.post(
            "/api/v1/datasets",
            json={"name": "Untagged Dataset", "source": "model", "tags": []},
        )

        response = client.get("/api/v1/datasets?tag=important")
        assert response.status_code == 200
        data = response.json()
        assert all("important" in d.get("tags", []) for d in data["datasets"])

    def test_list_datasets_pagination(self, client):
        """Test listing with pagination."""
        # Create multiple datasets
        for i in range(5):
            client.post(
                "/api/v1/datasets",
                json={"name": f"Dataset {i}", "source": "model"},
            )

        response = client.get("/api/v1/datasets?offset=0&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["datasets"]) == 2
        assert data["offset"] == 0
        assert data["limit"] == 2

    def test_get_dataset_statistics(self, client):
        """Test getting dataset statistics."""
        # Create a dataset
        create_response = client.post(
            "/api/v1/datasets",
            json={"name": "Stats Dataset", "source": "model"},
        )
        dataset_id = create_response.json()["id"]

        response = client.get(f"/api/v1/datasets/{dataset_id}/statistics")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "name" in data
        assert "total_annotations" in data
        assert "total_videos" in data
        assert "quality_distribution" in data

    def test_get_dataset_statistics_not_found(self, client):
        """Test getting stats for non-existent dataset."""
        response = client.get("/api/v1/datasets/nonexistent/statistics")
        assert response.status_code == 404

    def test_get_dataset_annotations(self, client):
        """Test getting annotations in a dataset."""
        # Create a dataset
        create_response = client.post(
            "/api/v1/datasets",
            json={"name": "Annotations Dataset", "source": "model"},
        )
        dataset_id = create_response.json()["id"]

        response = client.get(f"/api/v1/datasets/{dataset_id}/annotations")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert "annotations" in data
        assert "total" in data

    def test_dataset_models(self):
        """Test dataset pydantic models."""
        from dvas.api.dataset_api import DatasetCreate, DatasetResponse

        create = DatasetCreate(
            name="Test",
            description="Test dataset",
            source="model",
            tags=["test"],
        )
        assert create.name == "Test"
        assert create.source == "model"

        response = DatasetResponse(
            id="ds_123",
            name="Test",
            description="Test",
            source="model",
            tags=["test"],
            video_count=0,
            annotation_count=0,
            total_duration_seconds=0.0,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        assert response.id == "ds_123"
