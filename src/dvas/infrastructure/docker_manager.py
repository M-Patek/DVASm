"""Docker container management for DVAS.

Provides container lifecycle management, image building, Dockerfile generation
with GPU support, and resource allocation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class ContainerStatus(str, Enum):
    """Container lifecycle statuses."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class ResourceLimits:
    """Resource constraints for a container."""

    cpu_count: Optional[int] = None
    cpu_shares: Optional[int] = None
    memory_mb: Optional[int] = None
    memory_swap_mb: Optional[int] = None
    gpu_count: Optional[int] = None
    gpu_ids: Optional[List[str]] = None

    def to_docker_args(self) -> List[str]:
        """Convert to Docker CLI arguments."""
        args: List[str] = []
        if self.cpu_count is not None:
            args.extend(["--cpus", str(self.cpu_count)])
        if self.memory_mb is not None:
            args.extend(["--memory", f"{self.memory_mb}m"])
        if self.memory_swap_mb is not None:
            args.extend(["--memory-swap", f"{self.memory_swap_mb}m"])
        if self.gpu_count is not None:
            args.extend(["--gpus", f"all,capabilities=compute,utility"])
        if self.gpu_ids is not None:
            gpus = ",".join(self.gpu_ids)
            args.extend(["--gpus", f"\"device={gpus}\""])
        return args

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cpu_count": self.cpu_count,
            "cpu_shares": self.cpu_shares,
            "memory_mb": self.memory_mb,
            "memory_swap_mb": self.memory_swap_mb,
            "gpu_count": self.gpu_count,
            "gpu_ids": self.gpu_ids,
        }


@dataclass
class ContainerConfig:
    """Configuration for a Docker container."""

    name: str
    image: str
    command: Optional[List[str]] = None
    ports: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    network: Optional[str] = None
    working_dir: Optional[str] = None
    user: Optional[str] = None
    restart_policy: Optional[str] = None
    health_check: Optional[Dict[str, Any]] = None
    resource_limits: Optional[ResourceLimits] = None
    detach: bool = True
    remove_on_exit: bool = False

    def to_run_args(self) -> List[str]:
        """Convert to Docker run CLI arguments."""
        args: List[str] = ["run"]
        if self.detach:
            args.append("-d")
        if self.remove_on_exit:
            args.append("--rm")
        if self.name:
            args.extend(["--name", self.name])
        for host, container in self.ports.items():
            args.extend(["-p", f"{host}:{container}"])
        for host, container in self.volumes.items():
            args.extend(["-v", f"{host}:{container}"])
        for key, value in self.env.items():
            args.extend(["-e", f"{key}={value}"])
        for key, value in self.labels.items():
            args.extend(["-l", f"{key}={value}"])
        if self.network:
            args.extend(["--network", self.network])
        if self.working_dir:
            args.extend(["--workdir", self.working_dir])
        if self.user:
            args.extend(["--user", self.user])
        if self.restart_policy:
            args.extend(["--restart", self.restart_policy])
        if self.health_check:
            cmd = self.health_check.get("command", "")
            interval = self.health_check.get("interval", "30s")
            timeout = self.health_check.get("timeout", "10s")
            retries = self.health_check.get("retries", 3)
            args.extend([
                "--health-cmd", cmd,
                "--health-interval", interval,
                "--health-timeout", timeout,
                "--health-retries", str(retries),
            ])
        if self.resource_limits:
            args.extend(self.resource_limits.to_docker_args())
        args.append(self.image)
        if self.command:
            args.extend(self.command)
        return args


@dataclass
class ImageBuildResult:
    """Result of a Docker image build."""

    image_tag: str
    build_time_s: float
    success: bool
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    size_mb: Optional[float] = None


class DockerfileBuilder:
    """Generate Dockerfiles with GPU and dependency support."""

    def __init__(self, base_image: str = "python:3.11-slim") -> None:
        self.base_image = base_image
        self._lines: List[str] = []
        self._packages: List[str] = []
        self._pip_packages: List[str] = []
        self._env_vars: Dict[str, str] = {}
        self._copies: List[tuple[str, str]] = []
        self._commands: List[str] = []
        self._entrypoint: Optional[List[str]] = None
        self._cmd: Optional[List[str]] = None
        self._workdir: Optional[str] = None
        self._exposed_ports: List[int] = []
        self._health_check: Optional[Dict[str, Any]] = None

    def set_base(self, image: str) -> "DockerfileBuilder":
        """Set the base image."""
        self.base_image = image
        return self

    def add_system_package(self, package: str) -> "DockerfileBuilder":
        """Add a system package to install."""
        self._packages.append(package)
        return self

    def add_pip_package(self, package: str) -> "DockerfileBuilder":
        """Add a pip package to install."""
        self._pip_packages.append(package)
        return self

    def set_env(self, key: str, value: str) -> "DockerfileBuilder":
        """Set an environment variable."""
        self._env_vars[key] = value
        return self

    def copy(self, src: str, dst: str) -> "DockerfileBuilder":
        """Add a COPY instruction."""
        self._copies.append((src, dst))
        return self

    def run(self, command: str) -> "DockerfileBuilder":
        """Add a RUN instruction."""
        self._commands.append(command)
        return self

    def set_entrypoint(self, entrypoint: List[str]) -> "DockerfileBuilder":
        """Set the ENTRYPOINT."""
        self._entrypoint = entrypoint
        return self

    def set_cmd(self, cmd: List[str]) -> "DockerfileBuilder":
        """Set the CMD."""
        self._cmd = cmd
        return self

    def set_workdir(self, workdir: str) -> "DockerfileBuilder":
        """Set the WORKDIR."""
        self._workdir = workdir
        return self

    def expose_port(self, port: int) -> "DockerfileBuilder":
        """Expose a port."""
        self._exposed_ports.append(port)
        return self

    def set_health_check(self, command: str, interval: str = "30s",
                         timeout: str = "10s", retries: int = 3,
                         start_period: str = "5s") -> "DockerfileBuilder":
        """Set a health check."""
        self._health_check = {
            "command": command,
            "interval": interval,
            "timeout": timeout,
            "retries": retries,
            "start_period": start_period,
        }
        return self

    def enable_cuda(self, cuda_version: str = "12.1") -> "DockerfileBuilder":
        """Switch to a CUDA-enabled base image."""
        self.base_image = f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"
        self._packages.extend(["python3", "python3-pip"])
        self.set_env("NVIDIA_VISIBLE_DEVICES", "all")
        self.set_env("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
        return self

    def build(self) -> str:
        """Generate the Dockerfile content."""
        lines: List[str] = [f"FROM {self.base_image}"]

        if self._env_vars:
            for key, value in self._env_vars.items():
                lines.append(f"ENV {key}={value}")

        if self._packages:
            pkg_list = " ".join(sorted(set(self._packages)))
            lines.append(f"RUN apt-get update && apt-get install -y {pkg_list} && rm -rf /var/lib/apt/lists/*")

        if self._copies:
            for src, dst in self._copies:
                lines.append(f"COPY {src} {dst}")

        if self._pip_packages:
            pkg_list = " ".join(f"'{p}'" for p in self._pip_packages)
            lines.append(f"RUN pip install --no-cache-dir {pkg_list}")

        for cmd in self._commands:
            lines.append(f"RUN {cmd}")

        if self._workdir:
            lines.append(f"WORKDIR {self._workdir}")

        for port in self._exposed_ports:
            lines.append(f"EXPOSE {port}")

        if self._health_check:
            hc = self._health_check
            lines.append(
                f'HEALTHCHECK --interval={hc["interval"]} '
                f'--timeout={hc["timeout"]} --retries={hc["retries"]} '
                f'--start-period={hc["start_period"]} '
                f'CMD {hc["command"]}'
            )

        if self._entrypoint:
            ep = json.dumps(self._entrypoint)
            lines.append(f"ENTRYPOINT {ep}")

        if self._cmd:
            cmd = json.dumps(self._cmd)
            lines.append(f"CMD {cmd}")

        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        """Write the Dockerfile to disk."""
        path.write_text(self.build(), encoding="utf-8")
        logger.info("dockerfile_written", path=str(path))


class DockerManager:
    """Manage Docker containers and images for DVAS.

    Usage::

        manager = DockerManager()
        config = ContainerConfig(
            name="dvas-api",
            image="dvas:latest",
            ports={"8000": "8000"},
            env={"DVAS_ENV": "production"},
        )
        manager.run_container(config)
    """

    def __init__(self, docker_cmd: str = "docker") -> None:
        self.docker_cmd = docker_cmd
        self._containers: Dict[str, Dict[str, Any]] = {}

    def _run_cmd(self, args: List[str], capture: bool = True) -> subprocess.CompletedProcess:
        """Execute a Docker CLI command."""
        cmd = [self.docker_cmd] + args
        logger.debug("docker_cmd", command=" ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=capture,
            check=False,
        )

    def build_image(
        self,
        context_path: Path,
        tag: str,
        dockerfile: str = "Dockerfile",
        build_args: Optional[Dict[str, str]] = None,
        no_cache: bool = False,
    ) -> ImageBuildResult:
        """Build a Docker image.

        Args:
            context_path: Build context directory.
            tag: Image tag.
            dockerfile: Dockerfile name.
            build_args: Build arguments.
            no_cache: Disable cache.

        Returns:
            ImageBuildResult with build status.
        """
        import time

        start = time.time()
        args = ["build", "-t", tag, "-f", str(context_path / dockerfile)]
        if no_cache:
            args.append("--no-cache")
        if build_args:
            for key, value in build_args.items():
                args.extend(["--build-arg", f"{key}={value}"])
        args.append(str(context_path))

        result = self._run_cmd(args)
        elapsed = time.time() - start

        logs = result.stdout.splitlines() if result.stdout else []
        success = result.returncode == 0

        size_mb: Optional[float] = None
        if success:
            inspect = self._run_cmd(["inspect", "-f", "{{.Size}}", tag])
            if inspect.returncode == 0 and inspect.stdout:
                try:
                    size_bytes = int(inspect.stdout.strip())
                    size_mb = round(size_bytes / (1024 * 1024), 2)
                except ValueError:
                    pass

        return ImageBuildResult(
            image_tag=tag,
            build_time_s=round(elapsed, 2),
            success=success,
            logs=logs,
            error=result.stderr if not success else None,
            size_mb=size_mb,
        )

    def tag_image(self, source_tag: str, target_tag: str) -> bool:
        """Tag an existing image.

        Args:
            source_tag: Source image tag.
            target_tag: Target image tag.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["tag", source_tag, target_tag])
        if result.returncode == 0:
            logger.info("image_tagged", source=source_tag, target=target_tag)
            return True
        logger.error("image_tag_failed", source=source_tag, error=result.stderr)
        return False

    def push_image(self, tag: str) -> bool:
        """Push an image to a registry.

        Args:
            tag: Image tag to push.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["push", tag])
        return result.returncode == 0

    def pull_image(self, tag: str) -> bool:
        """Pull an image from a registry.

        Args:
            tag: Image tag to pull.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["pull", tag])
        return result.returncode == 0

    def run_container(self, config: ContainerConfig) -> Optional[str]:
        """Run a container from configuration.

        Args:
            config: Container configuration.

        Returns:
            Container ID if successful.
        """
        args = config.to_run_args()
        args.append(config.image)
        if config.command:
            args.extend(config.command)

        result = self._run_cmd(args)
        if result.returncode == 0 and result.stdout:
            container_id = result.stdout.strip()
            self._containers[config.name] = {
                "id": container_id,
                "config": config,
                "status": ContainerStatus.RUNNING,
            }
            logger.info("container_started", name=config.name, id=container_id[:12])
            return container_id
        logger.error("container_start_failed", name=config.name, error=result.stderr)
        return None

    def stop_container(self, name_or_id: str, timeout: int = 10) -> bool:
        """Stop a running container.

        Args:
            name_or_id: Container name or ID.
            timeout: Seconds to wait before force kill.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["stop", "-t", str(timeout), name_or_id])
        if result.returncode == 0:
            for name, info in self._containers.items():
                if name == name_or_id or info["id"].startswith(name_or_id):
                    info["status"] = ContainerStatus.EXITED
            logger.info("container_stopped", container=name_or_id)
            return True
        return False

    def remove_container(self, name_or_id: str, force: bool = False) -> bool:
        """Remove a container.

        Args:
            name_or_id: Container name or ID.
            force: Force removal.

        Returns:
            True if successful.
        """
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(name_or_id)
        result = self._run_cmd(args)
        if result.returncode == 0:
            self._containers.pop(name_or_id, None)
            logger.info("container_removed", container=name_or_id)
            return True
        return False

    def get_container_status(self, name_or_id: str) -> ContainerStatus:
        """Get the status of a container.

        Args:
            name_or_id: Container name or ID.

        Returns:
            ContainerStatus.
        """
        result = self._run_cmd(
            ["inspect", "-f", "{{.State.Status}}", name_or_id]
        )
        if result.returncode == 0 and result.stdout:
            status_str = result.stdout.strip().lower()
            try:
                return ContainerStatus(status_str)
            except ValueError:
                return ContainerStatus.UNKNOWN
        return ContainerStatus.UNKNOWN

    def list_containers(self, all_containers: bool = False) -> List[Dict[str, Any]]:
        """List containers.

        Args:
            all_containers: Include stopped containers.

        Returns:
            List of container info dictionaries.
        """
        args = ["ps", "-q"]
        if all_containers:
            args.append("-a")
        result = self._run_cmd(args)
        if result.returncode != 0 or not result.stdout:
            return []

        ids = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        containers: List[Dict[str, Any]] = []
        for cid in ids:
            inspect = self._run_cmd(
                ["inspect", "-f",
                 "{{.Name}}|{{.Config.Image}}|{{.State.Status}}|{{.State.Running}}",
                 cid]
            )
            if inspect.returncode == 0 and inspect.stdout:
                parts = inspect.stdout.strip().split("|")
                if len(parts) >= 4:
                    containers.append({
                        "id": cid[:12],
                        "name": parts[0].lstrip("/"),
                        "image": parts[1],
                        "status": parts[2],
                        "running": parts[3] == "true",
                    })
        return containers

    def get_container_logs(self, name_or_id: str, tail: int = 100,
                           follow: bool = False) -> List[str]:
        """Get container logs.

        Args:
            name_or_id: Container name or ID.
            tail: Number of lines to return.
            follow: Follow log output.

        Returns:
            Log lines.
        """
        args = ["logs", "--tail", str(tail)]
        if follow:
            args.append("-f")
        args.append(name_or_id)
        result = self._run_cmd(args)
        if result.returncode == 0 and result.stdout:
            return result.stdout.splitlines()
        return []

    def exec_in_container(self, name_or_id: str, command: List[str]) -> subprocess.CompletedProcess:
        """Execute a command inside a running container.

        Args:
            name_or_id: Container name or ID.
            command: Command and arguments.

        Returns:
            CompletedProcess result.
        """
        return self._run_cmd(["exec", name_or_id] + command)

    def generate_dockerfile(
        self,
        output_path: Path,
        base_image: str = "python:3.11-slim",
        cuda_enabled: bool = False,
        cuda_version: str = "12.1",
    ) -> Path:
        """Generate a standard DVAS Dockerfile.

        Args:
            output_path: Where to write the Dockerfile.
            base_image: Base image to use.
            cuda_enabled: Enable CUDA support.
            cuda_version: CUDA version.

        Returns:
            Path to the written Dockerfile.
        """
        builder = DockerfileBuilder(base_image=base_image)

        if cuda_enabled:
            builder.enable_cuda(cuda_version)

        builder.add_system_package("ffmpeg")
        builder.add_system_package("libsm6")
        builder.add_system_package("libxext6")
        builder.add_system_package("libgl1")

        builder.copy(".", "/app")
        builder.run("cd /app && pip install -e .")
        builder.set_workdir("/app")
        builder.set_env("PYTHONUNBUFFERED", "1")
        builder.set_env("DVAS_ENV", "docker")
        builder.expose_port(8000)
        builder.set_health_check("curl -f http://localhost:8000/health || exit 1")
        builder.set_cmd(["python", "-m", "dvas"])

        builder.write(output_path)
        return output_path

    def prune_images(self) -> int:
        """Remove unused images.

        Returns:
            Number of images removed.
        """
        result = self._run_cmd(["image", "prune", "-f"])
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.splitlines()
            for line in lines:
                if "Total reclaimed space" in line:
                    logger.info("images_pruned", output=line)
        return 0

    def get_image_digest(self, tag: str) -> Optional[str]:
        """Get the digest of an image.

        Args:
            tag: Image tag.

        Returns:
            Image digest or None.
        """
        result = self._run_cmd(["inspect", "-f", "{{.RepoDigests}}", tag])
        if result.returncode == 0 and result.stdout:
            digests = result.stdout.strip().strip("[]").split()
            if digests:
                return digests[0]
        return None

    def __contains__(self, name: str) -> bool:
        """Check if a container is managed."""
        return name in self._containers

    def __len__(self) -> int:
        """Number of managed containers."""
        return len(self._containers)
