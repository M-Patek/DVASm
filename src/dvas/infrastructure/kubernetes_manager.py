"""Kubernetes deployment management for DVAS.

Provides K8s operations for deployment creation, service configuration,
autoscaling, ConfigMap/Secret management, and rolling updates.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DeploymentStrategy(str, Enum):
    """Kubernetes deployment update strategies."""

    ROLLING_UPDATE = "RollingUpdate"
    RECREATE = "Recreate"


@dataclass
class HealthProbe:
    """Health check probe configuration."""

    path: str = "/health"
    port: int = 8000
    initial_delay_seconds: int = 10
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes probe spec."""
        return {
            "httpGet": {
                "path": self.path,
                "port": self.port,
            },
            "initialDelaySeconds": self.initial_delay_seconds,
            "periodSeconds": self.period_seconds,
            "timeoutSeconds": self.timeout_seconds,
            "failureThreshold": self.failure_threshold,
            "successThreshold": self.success_threshold,
        }


@dataclass
class ResourceQuota:
    """Resource quota for a namespace."""

    cpu_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    memory_request: Optional[str] = None
    memory_limit: Optional[str] = None
    pods: Optional[int] = None
    services: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes resource spec."""
        limits: Dict[str, Any] = {}
        if self.cpu_limit:
            limits["cpu"] = self.cpu_limit
        if self.memory_limit:
            limits["memory"] = self.memory_limit
        if self.pods is not None:
            limits["pods"] = str(self.pods)
        if self.services is not None:
            limits["services"] = str(self.services)
        return {"limits": limits}


@dataclass
class ContainerSpec:
    """Container specification for a pod."""

    name: str
    image: str
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    ports: List[Dict[str, Any]] = field(default_factory=list)
    env: List[Dict[str, Any]] = field(default_factory=list)
    env_from: List[Dict[str, Any]] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    volume_mounts: List[Dict[str, Any]] = field(default_factory=list)
    liveness_probe: Optional[HealthProbe] = None
    readiness_probe: Optional[HealthProbe] = None
    startup_probe: Optional[HealthProbe] = None
    image_pull_policy: str = "IfNotPresent"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes container spec."""
        spec: Dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "imagePullPolicy": self.image_pull_policy,
        }
        if self.command:
            spec["command"] = self.command
        if self.args:
            spec["args"] = self.args
        if self.ports:
            spec["ports"] = self.ports
        if self.env:
            spec["env"] = self.env
        if self.env_from:
            spec["envFrom"] = self.env_from
        if self.resources:
            spec["resources"] = self.resources
        if self.volume_mounts:
            spec["volumeMounts"] = self.volume_mounts
        if self.liveness_probe:
            spec["livenessProbe"] = self.liveness_probe.to_dict()
        if self.readiness_probe:
            spec["readinessProbe"] = self.readiness_probe.to_dict()
        if self.startup_probe:
            spec["startupProbe"] = self.startup_probe.to_dict()
        return spec


@dataclass
class DeploymentSpec:
    """Kubernetes deployment specification."""

    name: str
    namespace: str = "default"
    replicas: int = 1
    selector: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    containers: List[ContainerSpec] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    max_surge: str = "25%"
    max_unavailable: str = "25%"
    revision_history_limit: int = 10
    pod_annotations: Dict[str, str] = field(default_factory=dict)
    service_account: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes Deployment manifest."""
        labels = dict(self.labels)
        labels.setdefault("app", self.name)

        template: Dict[str, Any] = {
            "metadata": {
                "labels": labels,
                "annotations": self.pod_annotations,
            },
            "spec": {
                "containers": [c.to_dict() for c in self.containers],
            },
        }
        if self.volumes:
            template["spec"]["volumes"] = self.volumes
        if self.service_account:
            template["spec"]["serviceAccountName"] = self.service_account

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": {
                "replicas": self.replicas,
                "revisionHistoryLimit": self.revision_history_limit,
                "selector": {
                    "matchLabels": self.selector or {"app": self.name},
                },
                "strategy": {
                    "type": self.strategy.value,
                    "rollingUpdate": {
                        "maxSurge": self.max_surge,
                        "maxUnavailable": self.max_unavailable,
                    },
                },
                "template": template,
            },
        }


@dataclass
class ServiceConfig:
    """Kubernetes service configuration."""

    name: str
    namespace: str = "default"
    selector: Dict[str, str] = field(default_factory=dict)
    ports: List[Dict[str, Any]] = field(default_factory=list)
    service_type: str = "ClusterIP"
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "selector": self.selector,
                "ports": self.ports,
                "type": self.service_type,
            },
        }


@dataclass
class ConfigMapRef:
    """Reference to a ConfigMap."""

    name: str
    namespace: str = "default"
    data: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes ConfigMap manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "data": self.data,
        }


@dataclass
class SecretRef:
    """Reference to a Kubernetes Secret."""

    name: str
    namespace: str = "default"
    data: Dict[str, str] = field(default_factory=dict)
    type: str = "Opaque"
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes Secret manifest."""
        import base64

        encoded = {k: base64.b64encode(v.encode()).decode() for k, v in self.data.items()}
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "type": self.type,
            "data": encoded,
        }


@dataclass
class HorizontalPodAutoscaler:
    """HPA configuration."""

    name: str
    namespace: str = "default"
    target_deployment: str = ""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 80
    target_memory_utilization: Optional[int] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes HPA manifest."""
        metrics: List[Dict[str, Any]] = [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": self.target_cpu_utilization,
                    },
                },
            }
        ]
        if self.target_memory_utilization is not None:
            metrics.append(
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": self.target_memory_utilization,
                        },
                    },
                }
            )
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.target_deployment,
                },
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "metrics": metrics,
            },
        }


@dataclass
class NamespaceConfig:
    """Namespace configuration."""

    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Kubernetes Namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
                "annotations": self.annotations,
            },
        }


class KubernetesManager:
    """Manage Kubernetes resources for DVAS.

    Usage::

        k8s = KubernetesManager()
        deployment = DeploymentSpec(
            name="dvas-api",
            namespace="production",
            replicas=3,
            containers=[
                ContainerSpec(name="api", image="dvas/api:latest", ports=[{"containerPort": 8000}]),
            ],
        )
        k8s.apply_deployment(deployment)
    """

    def __init__(self, kubectl_cmd: str = "kubectl") -> None:
        self.kubectl_cmd = kubectl_cmd
        self._deployments: Dict[str, Dict[str, Any]] = {}

    def _run_cmd(
        self, args: List[str], input_text: Optional[str] = None, capture: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute a kubectl command."""
        cmd = [self.kubectl_cmd] + args
        logger.debug("kubectl_cmd", command=" ".join(cmd))
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=capture,
            text=capture,
            check=False,
        )

    def apply_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Apply a Kubernetes manifest.

        Args:
            manifest: Kubernetes resource manifest.

        Returns:
            True if successful.
        """
        yaml_str = json.dumps(manifest)
        result = self._run_cmd(["apply", "-f", "-"], input_text=yaml_str)
        if result.returncode == 0:
            name = manifest.get("metadata", {}).get("name", "unknown")
            kind = manifest.get("kind", "unknown")
            logger.info("manifest_applied", kind=kind, name=name)
            return True
        logger.error("manifest_apply_failed", error=result.stderr)
        return False

    def apply_deployment(self, spec: DeploymentSpec) -> bool:
        """Apply a Deployment manifest.

        Args:
            spec: Deployment specification.

        Returns:
            True if successful.
        """
        key = f"{spec.namespace}/{spec.name}"
        if self.apply_manifest(spec.to_dict()):
            self._deployments[key] = {"spec": spec, "status": "applied"}
            return True
        return False

    def delete_deployment(self, name: str, namespace: str = "default") -> bool:
        """Delete a Deployment.

        Args:
            name: Deployment name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "deployment", name, "-n", namespace])
        if result.returncode == 0:
            key = f"{namespace}/{name}"
            self._deployments.pop(key, None)
            logger.info("deployment_deleted", name=name, namespace=namespace)
            return True
        return False

    def get_deployment_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get deployment status.

        Args:
            name: Deployment name.
            namespace: Namespace.

        Returns:
            Status dictionary.
        """
        result = self._run_cmd(["get", "deployment", name, "-n", namespace, "-o", "json"])
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {}

    def scale_deployment(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """Scale a deployment.

        Args:
            name: Deployment name.
            replicas: Desired replica count.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(
            ["scale", "deployment", name, "--replicas", str(replicas), "-n", namespace]
        )
        if result.returncode == 0:
            key = f"{namespace}/{name}"
            if key in self._deployments:
                self._deployments[key]["status"] = f"scaled_to_{replicas}"
            logger.info("deployment_scaled", name=name, replicas=replicas, namespace=namespace)
            return True
        return False

    def rollout_restart(self, name: str, namespace: str = "default") -> bool:
        """Trigger a rolling restart.

        Args:
            name: Deployment name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["rollout", "restart", "deployment", name, "-n", namespace])
        if result.returncode == 0:
            logger.info("rollout_restarted", name=name, namespace=namespace)
            return True
        return False

    def rollout_undo(
        self, name: str, namespace: str = "default", revision: Optional[int] = None
    ) -> bool:
        """Undo a rollout.

        Args:
            name: Deployment name.
            namespace: Namespace.
            revision: Specific revision to rollback to.

        Returns:
            True if successful.
        """
        args = ["rollout", "undo", "deployment", name, "-n", namespace]
        if revision is not None:
            args.extend(["--to-revision", str(revision)])
        result = self._run_cmd(args)
        if result.returncode == 0:
            logger.info("rollout_undone", name=name, namespace=namespace, revision=revision)
            return True
        return False

    def rollout_status(self, name: str, namespace: str = "default") -> str:
        """Get rollout status.

        Args:
            name: Deployment name.
            namespace: Namespace.

        Returns:
            Status string.
        """
        result = self._run_cmd(
            ["rollout", "status", "deployment", name, "-n", namespace],
            capture=True,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        return "unknown"

    def apply_service(self, config: ServiceConfig) -> bool:
        """Apply a Service manifest.

        Args:
            config: Service configuration.

        Returns:
            True if successful.
        """
        return self.apply_manifest(config.to_dict())

    def delete_service(self, name: str, namespace: str = "default") -> bool:
        """Delete a Service.

        Args:
            name: Service name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "service", name, "-n", namespace])
        return result.returncode == 0

    def apply_configmap(self, config: ConfigMapRef) -> bool:
        """Apply a ConfigMap.

        Args:
            config: ConfigMap reference.

        Returns:
            True if successful.
        """
        return self.apply_manifest(config.to_dict())

    def delete_configmap(self, name: str, namespace: str = "default") -> bool:
        """Delete a ConfigMap.

        Args:
            name: ConfigMap name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "configmap", name, "-n", namespace])
        return result.returncode == 0

    def apply_secret(self, secret: SecretRef) -> bool:
        """Apply a Secret.

        Args:
            secret: Secret reference.

        Returns:
            True if successful.
        """
        return self.apply_manifest(secret.to_dict())

    def delete_secret(self, name: str, namespace: str = "default") -> bool:
        """Delete a Secret.

        Args:
            name: Secret name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "secret", name, "-n", namespace])
        return result.returncode == 0

    def apply_hpa(self, hpa: HorizontalPodAutoscaler) -> bool:
        """Apply a HorizontalPodAutoscaler.

        Args:
            hpa: HPA configuration.

        Returns:
            True if successful.
        """
        return self.apply_manifest(hpa.to_dict())

    def delete_hpa(self, name: str, namespace: str = "default") -> bool:
        """Delete an HPA.

        Args:
            name: HPA name.
            namespace: Namespace.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "hpa", name, "-n", namespace])
        return result.returncode == 0

    def create_namespace(self, config: NamespaceConfig) -> bool:
        """Create a namespace.

        Args:
            config: Namespace configuration.

        Returns:
            True if successful.
        """
        return self.apply_manifest(config.to_dict())

    def delete_namespace(self, name: str) -> bool:
        """Delete a namespace.

        Args:
            name: Namespace name.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["delete", "namespace", name])
        return result.returncode == 0

    def get_pods(
        self, namespace: str = "default", selector: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List pods in a namespace.

        Args:
            namespace: Namespace.
            selector: Label selector.

        Returns:
            List of pod info dictionaries.
        """
        args = ["get", "pods", "-n", namespace, "-o", "json"]
        if selector:
            labels = ",".join(f"{k}={v}" for k, v in selector.items())
            args.extend(["-l", labels])
        result = self._run_cmd(args)
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout)
                return [
                    {
                        "name": p["metadata"]["name"],
                        "status": p["status"]["phase"],
                        "node": p["spec"].get("nodeName"),
                        "ready": all(
                            c.get("ready", False) for c in p["status"].get("containerStatuses", [])
                        ),
                    }
                    for p in data.get("items", [])
                ]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def get_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        container: Optional[str] = None,
        tail: int = 100,
    ) -> List[str]:
        """Get pod logs.

        Args:
            pod_name: Pod name.
            namespace: Namespace.
            container: Container name.
            tail: Number of lines.

        Returns:
            Log lines.
        """
        args = ["logs", pod_name, "-n", namespace, "--tail", str(tail)]
        if container:
            args.extend(["-c", container])
        result = self._run_cmd(args)
        if result.returncode == 0 and result.stdout:
            return result.stdout.splitlines()
        return []

    def exec_in_pod(
        self, pod_name: str, command: List[str], namespace: str = "default"
    ) -> subprocess.CompletedProcess:
        """Execute a command in a pod.

        Args:
            pod_name: Pod name.
            command: Command and arguments.
            namespace: Namespace.

        Returns:
            CompletedProcess result.
        """
        return self._run_cmd(["exec", pod_name, "-n", namespace, "--"] + command)

    def export_manifest(self, spec: Any, path: Path) -> None:
        """Export a manifest to a JSON file.

        Args:
            spec: Resource specification.
            path: Output path.
        """
        if hasattr(spec, "to_dict"):
            manifest = spec.to_dict()
        else:
            manifest = dict(spec)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("manifest_exported", path=str(path), kind=manifest.get("kind", "unknown"))

    def get_resource(self, kind: str, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get a Kubernetes resource.

        Args:
            kind: Resource kind.
            name: Resource name.
            namespace: Namespace.

        Returns:
            Resource dictionary.
        """
        result = self._run_cmd(["get", kind, name, "-n", namespace, "-o", "json"])
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {}

    def list_deployments(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """List deployments in a namespace.

        Args:
            namespace: Namespace.

        Returns:
            List of deployment summaries.
        """
        result = self._run_cmd(["get", "deployments", "-n", namespace, "-o", "json"])
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout)
                return [
                    {
                        "name": d["metadata"]["name"],
                        "replicas": d["spec"].get("replicas", 0),
                        "ready": d["status"].get("readyReplicas", 0),
                        "updated": d["status"].get("updatedReplicas", 0),
                    }
                    for d in data.get("items", [])
                ]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def __contains__(self, key: str) -> bool:
        """Check if a deployment is managed."""
        return key in self._deployments

    def __len__(self) -> int:
        """Number of managed deployments."""
        return len(self._deployments)
