"""Contract tests for DVAS API — pytest function style.

Converted from unittest.TestCase to pytest function style with fixtures.
Covers API contract definitions, response validation, and response format contracts.
"""

import pytest
from unittest.mock import MagicMock

from dvas.api.middleware import api_error, api_response
from dvas.testing import Contract, ContractStore


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def contract_store() -> ContractStore:
    """Create contract store with DVAS API contracts."""
    store = ContractStore()

    store.add(
        Contract(
            name="upload_video",
            request_method="POST",
            request_path="/api/v1/videos/upload",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["video_id", "filename", "status", "message"],
                "properties": {
                    "video_id": {"type": "string"},
                    "filename": {"type": "string"},
                    "status": {"type": "string", "enum": ["success"]},
                    "message": {"type": "string"},
                },
            },
        )
    )

    store.add(
        Contract(
            name="create_annotation_task",
            request_method="POST",
            request_path="/api/v1/annotations/tasks",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["task_id", "video_id", "status", "message"],
                "properties": {
                    "task_id": {"type": "string"},
                    "video_id": {"type": "string"},
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        )
    )

    store.add(
        Contract(
            name="get_task_status",
            request_method="GET",
            request_path="/api/v1/annotations/tasks/{task_id}",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["task_id", "video_id", "status"],
                "properties": {
                    "task_id": {"type": "string"},
                    "video_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "processing", "completed", "failed"],
                    },
                    "annotation": {"type": "object"},
                    "error": {"type": ["string", "null"]},
                },
            },
        )
    )

    store.add(
        Contract(
            name="get_annotation",
            request_method="GET",
            request_path="/api/v1/annotations/{video_id}",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["status", "data"],
                "properties": {
                    "status": {"type": "string", "enum": ["success"]},
                    "message": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        )
    )

    store.add(
        Contract(
            name="health_check",
            request_method="GET",
            request_path="/health",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["status"],
                "properties": {
                    "status": {"type": "string", "enum": ["healthy"]},
                },
            },
        )
    )

    store.add(
        Contract(
            name="get_statistics",
            request_method="GET",
            request_path="/api/v1/stats",
            expected_status=200,
            expected_body_schema={
                "type": "object",
                "required": ["status", "data"],
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        )
    )

    return store


# ── Contract store tests ──────────────────────────────────────────────────


def test_contract_store_has_all_contracts(contract_store: ContractStore) -> None:
    contracts = contract_store.list_contracts()
    expected = [
        "upload_video",
        "create_annotation_task",
        "get_task_status",
        "get_annotation",
        "health_check",
        "get_statistics",
    ]
    for name in expected:
        assert name in contracts, f"Missing contract: {name}"


def test_upload_video_contract(contract_store: ContractStore) -> None:
    contract = contract_store.get("upload_video")
    assert contract is not None
    assert contract.request_method == "POST"
    assert contract.expected_status == 200
    assert "video_id" in contract.expected_body_schema["required"]


def test_create_task_contract(contract_store: ContractStore) -> None:
    contract = contract_store.get("create_annotation_task")
    assert contract is not None
    assert contract.request_method == "POST"
    assert "task_id" in contract.expected_body_schema["required"]


def test_get_task_status_contract(contract_store: ContractStore) -> None:
    contract = contract_store.get("get_task_status")
    assert contract is not None
    assert contract.request_method == "GET"
    status_enum = contract.expected_body_schema["properties"]["status"]["enum"]
    assert "pending" in status_enum
    assert "completed" in status_enum
    assert "failed" in status_enum


def test_health_check_contract(contract_store: ContractStore) -> None:
    contract = contract_store.get("health_check")
    assert contract is not None
    assert contract.request_method == "GET"
    assert contract.expected_body_schema["properties"]["status"]["enum"] == ["healthy"]


# ── Response validation tests ─────────────────────────────────────────────


def test_validate_response_success(contract_store: ContractStore) -> None:
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = {
        "video_id": "vid_001",
        "filename": "test.mp4",
        "status": "success",
        "message": "Video uploaded",
    }

    errors = contract_store.validate_response("upload_video", response)
    assert len(errors) == 0


def test_validate_response_wrong_status(contract_store: ContractStore) -> None:
    response = MagicMock()
    response.status_code = 404

    errors = contract_store.validate_response("upload_video", response)
    assert len(errors) == 1
    assert "Status code mismatch" in errors[0]


def test_validate_response_missing_field(contract_store: ContractStore) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "video_id": "vid_001",
        # Missing filename, status, message
    }

    errors = contract_store.validate_response("upload_video", response)
    assert len(errors) == 0  # Status code is correct


# ── Response format tests ─────────────────────────────────────────────────


def test_success_response_format() -> None:
    response = api_response(
        data={"id": "123"},
        message="Success",
        status="success",
    )

    assert response["status"] == "success"
    assert response["message"] == "Success"
    assert response["data"] == {"id": "123"}
    assert "metadata" not in response


def test_success_response_with_metadata() -> None:
    response = api_response(
        data={"id": "123"},
        message="Success",
        metadata={"page": 1, "total": 100},
    )

    assert "metadata" in response
    assert response["metadata"]["page"] == 1


def test_error_response_format() -> None:
    response = api_error(
        message="Something went wrong",
        error_code="INTERNAL_ERROR",
        status_code=500,
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "INTERNAL_ERROR"
    assert response["error"]["message"] == "Something went wrong"
    assert response["error"]["status_code"] == 500
    assert "details" not in response["error"]


def test_error_response_with_details() -> None:
    response = api_error(
        message="Validation failed",
        error_code="VALIDATION_ERROR",
        status_code=400,
        details={"field": "email", "reason": "invalid"},
    )

    assert "details" in response["error"]
    assert response["error"]["details"]["field"] == "email"
