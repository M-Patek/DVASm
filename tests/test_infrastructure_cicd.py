"""Tests for the CI/CD infrastructure module."""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.infrastructure.cicd_manager import (
    Artifact,
    BuildStatus,
    CICDManager,
    DeploymentTarget,
    PipelineRun,
    SecretRef,
)


# ---------------------------------------------------------------------------
# BuildStatus enum
# ---------------------------------------------------------------------------
def test_build_status_values() -> None:
    """BuildStatus enum members have the expected string values."""
    assert BuildStatus.PENDING.value == "pending"
    assert BuildStatus.RUNNING.value == "running"
    assert BuildStatus.SUCCESS.value == "success"
    assert BuildStatus.FAILURE.value == "failure"
    assert BuildStatus.CANCELLED.value == "cancelled"
    assert BuildStatus.SKIPPED.value == "skipped"
    assert BuildStatus.UNKNOWN.value == "unknown"


def test_build_status_is_str_enum() -> None:
    """BuildStatus can be compared directly to strings."""
    assert BuildStatus.SUCCESS == "success"
    assert isinstance(BuildStatus.FAILURE, str)


# ---------------------------------------------------------------------------
# DeploymentTarget enum
# ---------------------------------------------------------------------------
def test_deployment_target_values() -> None:
    """DeploymentTarget enum members have the expected string values."""
    assert DeploymentTarget.DEV.value == "dev"
    assert DeploymentTarget.STAGING.value == "staging"
    assert DeploymentTarget.PRODUCTION.value == "production"


# ---------------------------------------------------------------------------
# SecretRef
# ---------------------------------------------------------------------------
def test_secret_ref_to_dict_defaults() -> None:
    """SecretRef.to_dict() serializes correctly with defaults."""
    secret = SecretRef(name="API_KEY")
    data = secret.to_dict()
    assert data == {
        "name": "API_KEY",
        "env_var": "API_KEY",
        "description": "",
        "required": True,
    }


def test_secret_ref_to_dict_custom() -> None:
    """SecretRef.to_dict() serializes with all custom fields."""
    secret = SecretRef(
        name="DB_PASSWORD",
        value="s3cr3t",
        env_var="DB_PASS",
        description="Database password",
        required=False,
    )
    data = secret.to_dict()
    assert data == {
        "name": "DB_PASSWORD",
        "env_var": "DB_PASS",
        "description": "Database password",
        "required": False,
    }


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------
def test_artifact_to_dict_defaults() -> None:
    """Artifact.to_dict() serializes correctly with defaults."""
    artifact = Artifact(name="coverage", path="coverage.xml")
    data = artifact.to_dict()
    assert data == {
        "name": "coverage",
        "path": "coverage.xml",
        "retention-days": 30,
        "if": "always()",
    }


def test_artifact_to_dict_custom() -> None:
    """Artifact.to_dict() serializes with custom values."""
    artifact = Artifact(
        name="docs",
        path="site/",
        retention_days=7,
        if_condition="github.ref == 'refs/heads/main'",
    )
    data = artifact.to_dict()
    assert data == {
        "name": "docs",
        "path": "site/",
        "retention-days": 7,
        "if": "github.ref == 'refs/heads/main'",
    }


# ---------------------------------------------------------------------------
# PipelineRun
# ---------------------------------------------------------------------------
def test_pipeline_run_to_dict() -> None:
    """PipelineRun.to_dict() serializes all fields."""
    run = PipelineRun(
        run_id="run-123",
        status=BuildStatus.SUCCESS,
        branch="main",
        commit_sha="abc123",
        started_at="2024-01-01T00:00:00Z",
        finished_at="2024-01-01T00:05:00Z",
        duration_s=300,
        triggered_by="user",
        url="https://example.com/run/123",
    )
    data = run.to_dict()
    assert data == {
        "run_id": "run-123",
        "status": "success",
        "branch": "main",
        "commit_sha": "abc123",
        "started_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T00:05:00Z",
        "duration_s": 300,
        "triggered_by": "user",
        "url": "https://example.com/run/123",
    }


def test_pipeline_run_to_dict_defaults() -> None:
    """PipelineRun.to_dict() handles optional defaults."""
    run = PipelineRun(
        run_id="run-456",
        status=BuildStatus.PENDING,
        branch="feature/x",
        commit_sha="def456",
        started_at="2024-01-02T00:00:00Z",
    )
    data = run.to_dict()
    assert data["triggered_by"] == "push"
    assert data["finished_at"] is None
    assert data["duration_s"] is None
    assert data["url"] is None


# ---------------------------------------------------------------------------
# CICDManager
# ---------------------------------------------------------------------------
class TestCICDManager:
    """Tests for the CICDManager class."""

    @pytest.fixture
    def manager(self) -> CICDManager:
        """Return a fresh CICDManager instance."""
        return CICDManager()

    # -- secrets -----------------------------------------------------------

    def test_add_secret(self, manager: CICDManager) -> None:
        """add_secret() appends a secret and returns self."""
        secret = SecretRef(name="TOKEN", env_var="TOKEN")
        result = manager.add_secret(secret)
        assert result is manager
        assert len(manager.get_secrets()) == 1
        assert manager.get_secrets()[0]["name"] == "TOKEN"

    def test_get_secrets_empty(self, manager: CICDManager) -> None:
        """get_secrets() returns an empty list initially."""
        assert manager.get_secrets() == []

    def test_validate_secrets_all_present(self, manager: CICDManager) -> None:
        """validate_secrets() passes when required secrets have values."""
        manager.add_secret(SecretRef(name="A", value="val"))
        manager.add_secret(SecretRef(name="B", value="val", required=False))
        result = manager.validate_secrets()
        assert result["valid"] is True
        assert result["missing"] == []
        assert result["total"] == 2

    def test_validate_secrets_missing(self, manager: CICDManager) -> None:
        """validate_secrets() reports missing required secrets."""
        manager.add_secret(SecretRef(name="MISSING"))
        result = manager.validate_secrets()
        assert result["valid"] is False
        assert "MISSING" in result["missing"]
        assert result["total"] == 1

    # -- artifacts ---------------------------------------------------------

    def test_add_artifact(self, manager: CICDManager) -> None:
        """add_artifact() appends an artifact and returns self."""
        art = Artifact(name="reports", path="htmlcov/")
        result = manager.add_artifact(art)
        assert result is manager
        assert len(manager.get_artifacts()) == 1
        assert manager.get_artifacts()[0]["name"] == "reports"

    def test_get_artifacts_empty(self, manager: CICDManager) -> None:
        """get_artifacts() returns an empty list initially."""
        assert manager.get_artifacts() == []

    # -- runs --------------------------------------------------------------

    def test_record_run(self, manager: CICDManager) -> None:
        """record_run() stores a pipeline run."""
        run = PipelineRun(
            run_id="r1",
            status=BuildStatus.RUNNING,
            branch="main",
            commit_sha="abc",
            started_at="2024-01-01T00:00:00Z",
        )
        manager.record_run(run)
        assert len(manager) == 1
        retrieved = manager.get_pipeline_status("r1")
        assert retrieved is not None
        assert retrieved.run_id == "r1"

    def test_get_pipeline_status_missing(self, manager: CICDManager) -> None:
        """get_pipeline_status() returns None for unknown run IDs."""
        assert manager.get_pipeline_status("no-such-id") is None

    def test_list_pipeline_runs_no_filter(self, manager: CICDManager) -> None:
        """list_pipeline_runs() returns all runs without filters."""
        r1 = PipelineRun("r1", BuildStatus.SUCCESS, "main", "a", "t1")
        r2 = PipelineRun("r2", BuildStatus.FAILURE, "dev", "b", "t2")
        manager.record_run(r1)
        manager.record_run(r2)
        assert len(manager.list_pipeline_runs()) == 2

    def test_list_pipeline_runs_filter_branch(self, manager: CICDManager) -> None:
        """list_pipeline_runs() filters by branch."""
        manager.record_run(PipelineRun("r1", BuildStatus.SUCCESS, "main", "a", "t1"))
        manager.record_run(PipelineRun("r2", BuildStatus.FAILURE, "dev", "b", "t2"))
        manager.record_run(PipelineRun("r3", BuildStatus.SUCCESS, "main", "c", "t3"))
        assert len(manager.list_pipeline_runs(branch="main")) == 2
        assert len(manager.list_pipeline_runs(branch="dev")) == 1

    def test_list_pipeline_runs_filter_status(self, manager: CICDManager) -> None:
        """list_pipeline_runs() filters by status."""
        manager.record_run(PipelineRun("r1", BuildStatus.SUCCESS, "main", "a", "t1"))
        manager.record_run(PipelineRun("r2", BuildStatus.FAILURE, "main", "b", "t2"))
        assert len(manager.list_pipeline_runs(status=BuildStatus.SUCCESS)) == 1

    # -- GitHub Actions generation -----------------------------------------

    def test_generate_github_actions(self, manager: CICDManager) -> None:
        """generate_github_actions() writes a JSON workflow file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / ".github" / "workflows" / "ci.yml"
            path = manager.generate_github_actions(out)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["name"] == "DVAS CI"
            assert "jobs" in data
            assert "test" in data["jobs"]
            assert "build" in data["jobs"]

    def test_generate_github_actions_with_artifacts(self, manager: CICDManager) -> None:
        """generate_github_actions() includes artifact upload steps."""
        manager.add_artifact(Artifact(name="cov", path="coverage.xml"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ci.yml"
            manager.generate_github_actions(out)
            data = json.loads(out.read_text())
            steps = data["jobs"]["test"]["steps"]
            artifact_steps = [s for s in steps if "upload" in s.get("name", "").lower()]
            assert len(artifact_steps) == 1

    # -- GitLab CI generation ----------------------------------------------

    def test_generate_gitlab_ci(self, manager: CICDManager) -> None:
        """generate_gitlab_ci() writes a YAML configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / ".gitlab-ci.yml"
            path = manager.generate_gitlab_ci(out)
            assert path.exists()
            text = path.read_text()
            assert "GitLab CI configuration for DVAS" in text
            assert "stages:" in text
            assert "test:" in text
            assert "build:" in text
            assert "deploy:" in text

    # -- triggering --------------------------------------------------------

    def test_trigger_build_no_token(self, manager: CICDManager) -> None:
        """trigger_build() returns empty string when token is not set."""
        assert manager.trigger_build("main") == ""

    def test_trigger_build_with_token(self, manager: CICDManager) -> None:
        """trigger_build() returns a run ID when token is set."""
        manager.set_github_token("ghp_123")
        run_id = manager.trigger_build("main")
        assert len(run_id) == 8

    def test_trigger_deployment(self, manager: CICDManager) -> None:
        """trigger_deployment() returns a deployment ID."""
        dep_id = manager.trigger_deployment(DeploymentTarget.STAGING, image_tag="v1.0")
        assert len(dep_id) == 8

    # -- step generation ---------------------------------------------------

    def test_generate_docker_build_step(self, manager: CICDManager) -> None:
        """generate_docker_build_step() returns a valid GitHub Actions step."""
        step = manager.generate_docker_build_step(image_name="dvas", tag="latest")
        assert step["name"] == "Build and push Docker image"
        assert "docker/build-push-action@v5" in step["uses"]
        assert step["with"]["push"] is True
        assert step["with"]["tags"] == "ghcr.io/dvas:latest"

    def test_generate_deploy_step(self, manager: CICDManager) -> None:
        """generate_deploy_step() returns a valid deploy step."""
        step = manager.generate_deploy_step(DeploymentTarget.PRODUCTION, namespace="prod")
        assert step["name"] == "Deploy to production"
        assert "kubectl set image" in step["run"]
        assert "-n prod" in step["run"]

    # -- dunder ------------------------------------------------------------

    def test_len(self, manager: CICDManager) -> None:
        """__len__() returns the number of recorded runs."""
        assert len(manager) == 0
        manager.record_run(PipelineRun("r1", BuildStatus.PENDING, "b", "sha", "t"))
        assert len(manager) == 1
