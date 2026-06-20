"""Tests for DVAS infrastructure CLI entry point.

Tests for __main__.py covering all subcommands:
dockerfile, k8s-deploy, monitoring, terraform-init.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dvas.infrastructure.__main__ import (
    cmd_dockerfile,
    cmd_k8s_deploy,
    cmd_monitoring,
    cmd_terraform_init,
    main,
)


class TestCmdDockerfile:
    """Test dockerfile command."""

    def test_basic_dockerfile(self):
        """Test generating a basic Dockerfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Dockerfile"
            args = MagicMock()
            args.output = str(output)
            args.base_image = "python:3.11-slim"
            args.cuda = False
            args.cuda_version = "12.1"

            result = cmd_dockerfile(args)
            assert result == 0
            assert output.exists()
            content = output.read_text()
            assert "FROM python:3.11-slim" in content
            assert "ffmpeg" in content
            assert "8000" in content
            assert "HEALTHCHECK" in content

    def test_cuda_dockerfile(self):
        """Test generating a CUDA-enabled Dockerfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Dockerfile"
            args = MagicMock()
            args.output = str(output)
            args.base_image = "python:3.11-slim"
            args.cuda = True
            args.cuda_version = "12.1"

            result = cmd_dockerfile(args)
            assert result == 0
            content = output.read_text()
            assert "nvidia/cuda" in content
            assert "NVIDIA_VISIBLE_DEVICES" in content


class TestCmdK8sDeploy:
    """Test k8s-deploy command."""

    def test_basic_k8s_deploy(self):
        """Test generating a basic K8s deployment manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "deployment.json"
            args = MagicMock()
            args.name = "dvas-api"
            args.namespace = "default"
            args.replicas = 3
            args.image = "dvas:latest"
            args.output = str(output)

            result = cmd_k8s_deploy(args)
            assert result == 0
            assert output.exists()
            data = json.loads(output.read_text())
            assert data["kind"] == "Deployment"
            assert data["metadata"]["name"] == "dvas-api"
            assert data["metadata"]["namespace"] == "default"
            assert data["spec"]["replicas"] == 3

    def test_k8s_deploy_custom_values(self):
        """Test generating K8s manifest with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"
            args = MagicMock()
            args.name = "my-app"
            args.namespace = "production"
            args.replicas = 5
            args.image = "my-image:v2"
            args.output = str(output)

            result = cmd_k8s_deploy(args)
            assert result == 0
            data = json.loads(output.read_text())
            assert data["metadata"]["name"] == "my-app"
            assert data["spec"]["replicas"] == 5


class TestCmdMonitoring:
    """Test monitoring command."""

    def test_export_monitoring(self):
        """Test exporting monitoring configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "monitoring"
            args = MagicMock()
            args.output = str(output_dir)

            result = cmd_monitoring(args)
            assert result == 0
            # Check that config files were created
            assert output_dir.exists()
            # MonitoringStack.export_configs creates multiple files
            files = list(output_dir.iterdir())
            assert len(files) > 0


class TestCmdTerraformInit:
    """Test terraform-init command."""

    @patch("dvas.infrastructure.__main__.TerraformManager")
    def test_terraform_init_success(self, mock_tf_class: MagicMock):
        """Test successful terraform init."""
        mock_tf = MagicMock()
        mock_tf.init.return_value = MagicMock(returncode=0, stderr="")
        mock_tf_class.return_value = mock_tf

        args = MagicMock()
        args.dir = "/tmp/terraform"
        args.upgrade = False

        result = cmd_terraform_init(args)
        assert result == 0
        mock_tf.init.assert_called_once_with(upgrade=False)

    @patch("dvas.infrastructure.__main__.TerraformManager")
    def test_terraform_init_failure(self, mock_tf_class: MagicMock):
        """Test failed terraform init."""
        mock_tf = MagicMock()
        mock_tf.init.return_value = MagicMock(returncode=1, stderr="error message")
        mock_tf_class.return_value = mock_tf

        args = MagicMock()
        args.dir = "/tmp/terraform"
        args.upgrade = True

        result = cmd_terraform_init(args)
        assert result == 1
        mock_tf.init.assert_called_once_with(upgrade=True)


class TestMain:
    """Test main CLI entry point."""

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture):
        """Test running without arguments prints help."""
        result = main([])
        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "usage:" in captured.err

    def test_dockerfile_command(self):
        """Test main with dockerfile command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Dockerfile"
            result = main(["dockerfile", "--output", str(output)])
            assert result == 0
            assert output.exists()

    def test_dockerfile_with_cuda(self):
        """Test main with dockerfile --cuda."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Dockerfile"
            result = main([
                "dockerfile",
                "--output", str(output),
                "--cuda",
                "--cuda-version", "12.1",
            ])
            assert result == 0
            content = output.read_text()
            assert "nvidia/cuda" in content

    def test_dockerfile_custom_base_image(self):
        """Test main with custom base image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Dockerfile"
            result = main([
                "dockerfile",
                "--output", str(output),
                "--base-image", "python:3.10-slim",
            ])
            assert result == 0
            content = output.read_text()
            assert "FROM python:3.10-slim" in content

    def test_k8s_deploy_command(self):
        """Test main with k8s-deploy command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "deploy.json"
            result = main([
                "k8s-deploy",
                "--output", str(output),
                "--name", "test-app",
                "--replicas", "2",
            ])
            assert result == 0
            data = json.loads(output.read_text())
            assert data["metadata"]["name"] == "test-app"
            assert data["spec"]["replicas"] == 2

    def test_k8s_deploy_defaults(self):
        """Test k8s-deploy with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "deploy.json"
            result = main(["k8s-deploy", "--output", str(output)])
            assert result == 0
            data = json.loads(output.read_text())
            assert data["metadata"]["name"] == "dvas-api"
            assert data["spec"]["replicas"] == 1

    def test_monitoring_command(self):
        """Test main with monitoring command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "mon"
            result = main(["monitoring", "--output", str(output_dir)])
            assert result == 0
            assert output_dir.exists()

    @patch("dvas.infrastructure.__main__.TerraformManager")
    def test_terraform_init_command(self, mock_tf_class: MagicMock):
        """Test main with terraform-init command."""
        mock_tf = MagicMock()
        mock_tf.init.return_value = MagicMock(returncode=0, stderr="")
        mock_tf_class.return_value = mock_tf

        result = main(["terraform-init", "--dir", "/tmp/tf"])
        assert result == 0

    @patch("dvas.infrastructure.__main__.TerraformManager")
    def test_terraform_init_upgrade(self, mock_tf_class: MagicMock):
        """Test main with terraform-init --upgrade."""
        mock_tf = MagicMock()
        mock_tf.init.return_value = MagicMock(returncode=0, stderr="")
        mock_tf_class.return_value = mock_tf

        result = main(["terraform-init", "--dir", "/tmp/tf", "--upgrade"])
        assert result == 0
        mock_tf.init.assert_called_once_with(upgrade=True)

    def test_unknown_command(self):
        """Test main with unknown command falls through to help."""
        # argparse will error on unknown subcommands
        with pytest.raises(SystemExit):
            main(["unknown-command"])
