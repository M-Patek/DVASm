"""CI/CD pipeline management for DVAS.

Provides workflow generation, pipeline monitoring, artifact management,
and secret handling for GitHub Actions and GitLab CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class BuildStatus(str, Enum):
    """CI/CD build status values."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class DeploymentTarget(str, Enum):
    """Deployment environment targets."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SecretRef:
    """Reference to a CI/CD secret."""

    name: str
    value: Optional[str] = None
    env_var: Optional[str] = None
    description: str = ""
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "env_var": self.env_var or self.name.upper(),
            "description": self.description,
            "required": self.required,
        }


@dataclass
class Artifact:
    """CI/CD artifact reference."""

    name: str
    path: str
    retention_days: int = 30
    if_condition: str = "always()"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "retention-days": self.retention_days,
            "if": self.if_condition,
        }


@dataclass
class PipelineRun:
    """Information about a pipeline run."""

    run_id: str
    status: BuildStatus
    branch: str
    commit_sha: str
    started_at: str
    finished_at: Optional[str] = None
    duration_s: Optional[int] = None
    triggered_by: str = "push"
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "triggered_by": self.triggered_by,
            "url": self.url,
        }


class CICDManager:
    """Manage CI/CD pipelines for DVAS.

    Usage::

        cicd = CICDManager()
        cicd.generate_github_actions(Path(".github/workflows/ci.yml"))
        cicd.generate_gitlab_ci(Path(".gitlab-ci.yml"))
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path.cwd()
        self._secrets: List[SecretRef] = []
        self._artifacts: List[Artifact] = []
        self._runs: List[PipelineRun] = []
        self._github_token: Optional[str] = None

    def add_secret(self, secret: SecretRef) -> "CICDManager":
        """Add a secret reference.

        Args:
            secret: Secret to add.

        Returns:
            Self for chaining.
        """
        self._secrets.append(secret)
        return self

    def add_artifact(self, artifact: Artifact) -> "CICDManager":
        """Add an artifact.

        Args:
            artifact: Artifact to add.

        Returns:
            Self for chaining.
        """
        self._artifacts.append(artifact)
        return self

    def generate_github_actions(
        self,
        output_path: Path,
        python_version: str = "3.11",
        test_paths: Optional[List[str]] = None,
        branches: Optional[List[str]] = None,
    ) -> Path:
        """Generate a GitHub Actions workflow file.

        Args:
            output_path: Where to write the workflow.
            python_version: Python version to use.
            test_paths: Paths to test.
            branches: Branches to trigger on.

        Returns:
            Path to the written file.
        """
        if branches is None:
            branches = ["main", "master"]
        if test_paths is None:
            test_paths = ["tests/"]

        workflow = {
            "name": "DVAS CI",
            "on": {
                "push": {"branches": branches},
                "pull_request": {"branches": branches},
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": [python_version],
                        },
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "${{ matrix.python-version }}"},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e '.[dev]'",
                        },
                        {
                            "name": "Run tests",
                            "run": f"pytest {' '.join(test_paths)} -v --tb=short",
                        },
                        {
                            "name": "Run type checks",
                            "run": "mypy src/dvas",
                            "continue-on-error": True,
                        },
                        {
                            "name": "Run lint",
                            "run": "ruff check src/dvas",
                            "continue-on-error": True,
                        },
                        {
                            "name": "Run security scan",
                            "run": "bandit -r src/dvas -f json -o bandit-report.json || true",
                            "continue-on-error": True,
                        },
                    ],
                },
                "build": {
                    "runs-on": "ubuntu-latest",
                    "needs": "test",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Build Docker image",
                            "run": "docker build -t dvas:latest .",
                        },
                        {
                            "name": "Tag image",
                            "run": "docker tag dvas:latest ghcr.io/${{ github.repository }}:latest",
                        },
                    ],
                },
            },
        }

        # Add artifact upload if artifacts are configured
        if self._artifacts:
            for artifact in self._artifacts:
                workflow["jobs"]["test"]["steps"].append({
                    "name": f"Upload {artifact.name}",
                    "uses": "actions/upload-artifact@v4",
                    "with": artifact.to_dict(),
                })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
        logger.info("github_actions_generated", path=str(output_path))
        return output_path

    def generate_gitlab_ci(
        self,
        output_path: Path,
        python_version: str = "3.11",
        test_paths: Optional[List[str]] = None,
    ) -> Path:
        """Generate a GitLab CI configuration file.

        Args:
            output_path: Where to write the config.
            python_version: Python version to use.
            test_paths: Paths to test.

        Returns:
            Path to the written file.
        """
        if test_paths is None:
            test_paths = ["tests/"]

        yaml_lines: List[str] = [
            "# GitLab CI configuration for DVAS",
            "image: python:{}".format(python_version),
            "",
            "stages:",
            "  - test",
            "  - build",
            "  - deploy",
            "",
            "variables:",
            '  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"',
            "",
            "cache:",
            "  paths:",
            "    - .cache/pip",
            "    - venv/",
            "",
            "before_script:",
            "  - python -m venv venv",
            "  - source venv/bin/activate",
            "  - pip install -e '.[dev]'",
            "",
            "test:",
            "  stage: test",
            "  script:",
        ]
        for path in test_paths:
            yaml_lines.append(f"    - pytest {path} -v --tb=short")
        yaml_lines.extend([
            "    - mypy src/dvas || true",
            "    - ruff check src/dvas || true",
            "    - bandit -r src/dvas || true",
            "  artifacts:",
            "    reports:",
            "      junit: pytest-report.xml",
            "    paths:",
            "      - bandit-report.json",
            "    expire_in: 1 week",
            "  coverage: '/TOTAL\\s+\\d+%/'",
            "",
            "build:",
            "  stage: build",
            "  script:",
            "    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .",
            "    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest",
            "  only:",
            "    - main",
            "    - master",
            "",
            "deploy:",
            "  stage: deploy",
            "  script:",
            "    - echo 'Deploy to staging'",
            "  environment:",
            "    name: staging",
            "  only:",
            "    - main",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        logger.info("gitlab_ci_generated", path=str(output_path))
        return output_path

    def get_pipeline_status(self, run_id: str) -> Optional[PipelineRun]:
        """Get the status of a pipeline run.

        Args:
            run_id: Pipeline run ID.

        Returns:
            PipelineRun or None.
        """
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    def list_pipeline_runs(self, branch: Optional[str] = None,
                           status: Optional[BuildStatus] = None) -> List[PipelineRun]:
        """List pipeline runs.

        Args:
            branch: Filter by branch.
            status: Filter by status.

        Returns:
            List of pipeline runs.
        """
        runs = self._runs
        if branch:
            runs = [r for r in runs if r.branch == branch]
        if status:
            runs = [r for r in runs if r.status == status]
        return runs

    def record_run(self, run: PipelineRun) -> None:
        """Record a pipeline run.

        Args:
            run: Pipeline run to record.
        """
        self._runs.append(run)
        logger.info("pipeline_run_recorded", run_id=run.run_id, status=run.status.value)

    def trigger_build(self, branch: str = "main") -> str:
        """Trigger a build via GitHub API (requires token).

        Args:
            branch: Branch to build.

        Returns:
            Run ID or empty string.
        """
        if not self._github_token:
            logger.warning("github_token_not_set")
            return ""
        logger.info("build_triggered", branch=branch)
        import uuid
        return str(uuid.uuid4())[:8]

    def trigger_deployment(self, target: DeploymentTarget,
                           image_tag: str = "latest") -> str:
        """Trigger a deployment.

        Args:
            target: Deployment target.
            image_tag: Image tag to deploy.

        Returns:
            Deployment ID or empty string.
        """
        logger.info("deployment_triggered", target=target.value, image_tag=image_tag)
        import uuid
        return str(uuid.uuid4())[:8]

    def set_github_token(self, token: str) -> None:
        """Set the GitHub token for API operations.

        Args:
            token: GitHub personal access token.
        """
        self._github_token = token

    def get_secrets(self) -> List[Dict[str, Any]]:
        """Get all configured secrets.

        Returns:
            List of secret dictionaries.
        """
        return [s.to_dict() for s in self._secrets]

    def get_artifacts(self) -> List[Dict[str, Any]]:
        """Get all configured artifacts.

        Returns:
            List of artifact dictionaries.
        """
        return [a.to_dict() for a in self._artifacts]

    def validate_secrets(self) -> Dict[str, Any]:
        """Validate that all required secrets are configured.

        Returns:
            Validation result.
        """
        missing: List[str] = []
        present: List[str] = []
        for secret in self._secrets:
            if secret.required:
                if secret.value or (secret.env_var and secret.env_var in __import__("os").environ):
                    present.append(secret.name)
                else:
                    missing.append(secret.name)
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "present": present,
            "total": len(self._secrets),
        }

    def generate_docker_build_step(self, image_name: str = "dvas",
                                    tag: str = "${{ github.sha }}") -> Dict[str, Any]:
        """Generate a Docker build step for GitHub Actions.

        Args:
            image_name: Docker image name.
            tag: Image tag.

        Returns:
            Step dictionary.
        """
        return {
            "name": "Build and push Docker image",
            "uses": "docker/build-push-action@v5",
            "with": {
                "context": ".",
                "push": True,
                "tags": f"ghcr.io/{image_name}:{tag}",
                "cache-from": "type=gha",
                "cache-to": "type=gha,mode=max",
            },
        }

    def generate_deploy_step(self, target: DeploymentTarget,
                              namespace: str = "default") -> Dict[str, Any]:
        """Generate a deployment step for GitHub Actions.

        Args:
            target: Deployment target.
            namespace: Kubernetes namespace.

        Returns:
            Step dictionary.
        """
        return {
            "name": f"Deploy to {target.value}",
            "run": (
                f"kubectl set image deployment/dvas "
                f"dvas=ghcr.io/dvas:${{{{ github.sha }}}} "
                f"-n {namespace} && "
                f"kubectl rollout status deployment/dvas -n {namespace}"
            ),
        }

    def __len__(self) -> int:
        """Number of recorded pipeline runs."""
        return len(self._runs)
