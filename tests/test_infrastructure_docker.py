"""Tests for Docker container management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dvas.infrastructure.docker_manager import (
    ContainerConfig,
    ContainerStatus,
    DockerfileBuilder,
    DockerManager,
    ImageBuildResult,
    ResourceLimits,
)


class TestResourceLimits:
    """Test resource limit configuration."""

    def test_basic_limits(self):
        limits = ResourceLimits(cpu_count=4, memory_mb=8192)
        args = limits.to_docker_args()
        assert "--cpus" in args
        assert "4" in args
        assert "--memory" in args
        assert "8192m" in args

    def test_gpu_limits(self):
        limits = ResourceLimits(gpu_count=1)
        args = limits.to_docker_args()
        assert "--gpus" in args

    def test_gpu_ids(self):
        limits = ResourceLimits(gpu_ids=["0", "1"])
        args = limits.to_docker_args()
        assert any("device=0,1" in a for a in args)

    def test_to_dict(self):
        limits = ResourceLimits(cpu_count=2, memory_mb=4096)
        d = limits.to_dict()
        assert d["cpu_count"] == 2
        assert d["memory_mb"] == 4096


class TestContainerConfig:
    """Test container configuration."""

    def test_basic_config(self):
        config = ContainerConfig(
            name="test-container",
            image="python:3.11",
            ports={"8080": "80"},
            env={"FOO": "bar"},
        )
        args = config.to_run_args()
        assert "-d" in args
        assert "--name" in args
        assert "test-container" in args
        assert "-p" in args
        assert "8080:80" in args
        assert "-e" in args
        assert "FOO=bar" in args

    def test_with_command(self):
        config = ContainerConfig(
            name="test",
            image="python:3.11",
            command=["python", "-m", "http.server"],
        )
        args = config.to_run_args()
        assert "python" in args
        assert "-m" in args

    def test_with_health_check(self):
        config = ContainerConfig(
            name="test",
            image="nginx",
            health_check={"command": "curl -f http://localhost || exit 1", "interval": "30s"},
        )
        args = config.to_run_args()
        assert "--health-cmd" in args

    def test_with_resource_limits(self):
        config = ContainerConfig(
            name="test",
            image="dvas",
            resource_limits=ResourceLimits(cpu_count=2, memory_mb=4096),
        )
        args = config.to_run_args()
        assert "--cpus" in args


class TestDockerfileBuilder:
    """Test Dockerfile generation."""

    def test_basic_dockerfile(self):
        builder = DockerfileBuilder(base_image="python:3.11-slim")
        builder.copy(".", "/app")
        builder.run("pip install -r requirements.txt")
        builder.set_cmd(["python", "app.py"])
        dockerfile = builder.build()
        assert "FROM python:3.11-slim" in dockerfile
        assert "COPY . /app" in dockerfile
        assert "pip install -r requirements.txt" in dockerfile
        assert 'CMD ["python", "app.py"]' in dockerfile

    def test_cuda_dockerfile(self):
        builder = DockerfileBuilder()
        builder.enable_cuda("12.1")
        dockerfile = builder.build()
        assert "nvidia/cuda:12.1" in dockerfile
        assert "NVIDIA_VISIBLE_DEVICES" in dockerfile

    def test_health_check(self):
        builder = DockerfileBuilder()
        builder.set_health_check("curl -f http://localhost/health || exit 1")
        dockerfile = builder.build()
        assert "HEALTHCHECK" in dockerfile
        assert "curl -f http://localhost/health || exit 1" in dockerfile

    def test_expose_port(self):
        builder = DockerfileBuilder()
        builder.expose_port(8000)
        dockerfile = builder.build()
        assert "EXPOSE 8000" in dockerfile

    def test_write_dockerfile(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DockerfileBuilder()
            builder.set_cmd(["python", "-m", "http.server"])
            output = Path(tmpdir) / "Dockerfile"
            builder.write(output)
            assert output.exists()
            assert "FROM python:3.11-slim" in output.read_text()

    def test_chaining(self):
        builder = DockerfileBuilder()
        builder.add_system_package("curl").add_pip_package("requests").set_env("FOO", "bar")
        dockerfile = builder.build()
        assert "curl" in dockerfile
        assert "requests" in dockerfile
        assert "FOO=bar" in dockerfile


class TestDockerManager:
    """Test Docker manager operations (mocked)."""

    @pytest.fixture
    def manager(self):
        return DockerManager()

    def test_init(self, manager):
        assert manager.docker_cmd == "docker"
        assert len(manager) == 0

    def test_build_image(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Build successful\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            result = manager.build_image(Path("/tmp"), "test:latest")
            assert result.success is True
            assert result.image_tag == "test:latest"

    def test_build_image_failure(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Build failed"

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            result = manager.build_image(Path("/tmp"), "test:latest")
            assert result.success is False
            assert result.error == "Build failed"

    def test_tag_image(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.tag_image("source", "target") is True

    def test_run_container(self, manager):
        config = ContainerConfig(name="test", image="nginx")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123def456\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            cid = manager.run_container(config)
            assert cid is not None
            assert "test" in manager

    def test_stop_container(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.stop_container("test") is True

    def test_remove_container(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.remove_container("test") is True

    def test_get_container_status(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "running\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            status = manager.get_container_status("test")
            assert status == ContainerStatus.RUNNING

    def test_get_container_status_unknown(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Not found"

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            status = manager.get_container_status("test")
            assert status == ContainerStatus.UNKNOWN

    def test_list_containers(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"
        mock_result.stderr = ""

        inspect_result = MagicMock()
        inspect_result.returncode = 0
        inspect_result.stdout = "/test|nginx|running|true\n"
        inspect_result.stderr = ""

        with patch.object(manager, "_run_cmd", side_effect=[mock_result, inspect_result]):
            containers = manager.list_containers()
            assert len(containers) == 1
            assert containers[0]["name"] == "test"

    def test_get_container_logs(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "line1\nline2\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            logs = manager.get_container_logs("test", tail=10)
            assert len(logs) == 2

    def test_generate_dockerfile(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DockerManager()
            output = Path(tmpdir) / "Dockerfile"
            path = manager.generate_dockerfile(output)
            assert path.exists()
            content = path.read_text()
            assert "FROM python:3.11-slim" in content
            assert "EXPOSE 8000" in content

    def test_generate_dockerfile_cuda(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DockerManager()
            output = Path(tmpdir) / "Dockerfile"
            path = manager.generate_dockerfile(output, cuda_enabled=True)
            assert path.exists()
            content = path.read_text()
            assert "nvidia/cuda" in content

    def test_contains(self, manager):
        config = ContainerConfig(name="test", image="nginx")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            manager.run_container(config)
            assert "test" in manager
            assert "other" not in manager

    def test_len(self, manager):
        assert len(manager) == 0
