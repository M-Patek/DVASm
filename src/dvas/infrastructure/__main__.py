"""CLI entry point for DVAS infrastructure operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dvas.infrastructure.docker_manager import DockerfileBuilder, DockerManager
from dvas.infrastructure.kubernetes_manager import (
    ContainerSpec,
    DeploymentSpec,
    KubernetesManager,
)
from dvas.infrastructure.monitoring_stack import MonitoringStack
from dvas.infrastructure.terraform_manager import TerraformManager
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_dockerfile(args: argparse.Namespace) -> int:
    """Generate a Dockerfile."""
    builder = DockerfileBuilder(base_image=args.base_image)
    if args.cuda:
        builder.enable_cuda(args.cuda_version)
    builder.add_system_package("ffmpeg")
    builder.add_system_package("libgl1")
    builder.copy(".", "/app")
    builder.run("cd /app && pip install -e .")
    builder.set_env("PYTHONUNBUFFERED", "1")
    builder.set_env("DVAS_ENV", "docker")
    builder.expose_port(8000)
    builder.set_health_check("curl -f http://localhost:8000/health || exit 1")
    builder.set_cmd(["python", "-m", "dvas"])
    builder.write(Path(args.output))
    print(f"Dockerfile written to {args.output}")
    return 0


def cmd_k8s_deploy(args: argparse.Namespace) -> int:
    """Generate a Kubernetes deployment manifest."""
    deployment = DeploymentSpec(
        name=args.name,
        namespace=args.namespace,
        replicas=args.replicas,
        containers=[
            ContainerSpec(
                name="dvas",
                image=args.image,
                ports=[{"containerPort": 8000}],
                liveness_probe=None,
                readiness_probe=None,
            ),
        ],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        __import__("json").dumps(deployment.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"Deployment manifest written to {args.output}")
    return 0


def cmd_monitoring(args: argparse.Namespace) -> int:
    """Export monitoring configurations."""
    stack = MonitoringStack()
    stack.configure_prometheus()
    stack.add_scrape_target("dvas", ["localhost"], port=8000)
    stack.configure_alertmanager()
    stack.create_default_dashboard()
    stack.configure_loki()
    stack.configure_tracing()
    output_dir = Path(args.output)
    stack.export_configs(output_dir)
    print(f"Monitoring configs exported to {output_dir}")
    return 0


def cmd_terraform_init(args: argparse.Namespace) -> int:
    """Initialize Terraform."""
    tf = TerraformManager(Path(args.dir))
    result = tf.init(upgrade=args.upgrade)
    if result.returncode == 0:
        print("Terraform initialized successfully")
        return 0
    print(f"Terraform init failed: {result.stderr}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dvas-infrastructure",
        description="DVAS infrastructure management CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Dockerfile command
    docker_parser = subparsers.add_parser("dockerfile", help="Generate a Dockerfile")
    docker_parser.add_argument("--output", "-o", default="Dockerfile", help="Output path")
    docker_parser.add_argument("--base-image", default="python:3.11-slim", help="Base image")
    docker_parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    docker_parser.add_argument("--cuda-version", default="12.1", help="CUDA version")
    docker_parser.set_defaults(func=cmd_dockerfile)

    # K8s deploy command
    k8s_parser = subparsers.add_parser("k8s-deploy", help="Generate K8s deployment manifest")
    k8s_parser.add_argument("--name", default="dvas-api", help="Deployment name")
    k8s_parser.add_argument("--namespace", default="default", help="Namespace")
    k8s_parser.add_argument("--replicas", type=int, default=1, help="Replica count")
    k8s_parser.add_argument("--image", default="dvas:latest", help="Container image")
    k8s_parser.add_argument("--output", "-o", default="deployment.json", help="Output path")
    k8s_parser.set_defaults(func=cmd_k8s_deploy)

    # Monitoring command
    mon_parser = subparsers.add_parser("monitoring", help="Export monitoring configs")
    mon_parser.add_argument("--output", "-o", default="monitoring", help="Output directory")
    mon_parser.set_defaults(func=cmd_monitoring)

    # Terraform command
    tf_parser = subparsers.add_parser("terraform-init", help="Initialize Terraform")
    tf_parser.add_argument("--dir", default=".", help="Working directory")
    tf_parser.add_argument("--upgrade", action="store_true", help="Upgrade modules")
    tf_parser.set_defaults(func=cmd_terraform_init)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
