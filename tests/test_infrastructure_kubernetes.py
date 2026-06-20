"""Tests for Kubernetes deployment management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dvas.infrastructure.kubernetes_manager import (
    ConfigMapRef,
    ContainerSpec,
    DeploymentSpec,
    DeploymentStrategy,
    HealthProbe,
    HorizontalPodAutoscaler,
    KubernetesManager,
    NamespaceConfig,
    ResourceQuota,
    SecretRef,
    ServiceConfig,
)


class TestHealthProbe:
    """Test health probe configuration."""

    def test_default_probe(self):
        probe = HealthProbe()
        assert probe.path == "/health"
        assert probe.port == 8000
        assert probe.initial_delay_seconds == 10

    def test_custom_probe(self):
        probe = HealthProbe(
            path="/ready",
            port=8080,
            initial_delay_seconds=5,
            failure_threshold=5,
        )
        d = probe.to_dict()
        assert d["httpGet"]["path"] == "/ready"
        assert d["httpGet"]["port"] == 8080
        assert d["initialDelaySeconds"] == 5
        assert d["failureThreshold"] == 5


class TestResourceQuota:
    """Test resource quota configuration."""

    def test_basic_quota(self):
        quota = ResourceQuota(cpu_limit="4", memory_limit="16Gi")
        d = quota.to_dict()
        assert d["limits"]["cpu"] == "4"
        assert d["limits"]["memory"] == "16Gi"

    def test_pods_limit(self):
        quota = ResourceQuota(pods=10, services=5)
        d = quota.to_dict()
        assert d["limits"]["pods"] == "10"
        assert d["limits"]["services"] == "5"


class TestContainerSpec:
    """Test container specification."""

    def test_basic_spec(self):
        spec = ContainerSpec(name="dvas", image="dvas:latest")
        d = spec.to_dict()
        assert d["name"] == "dvas"
        assert d["image"] == "dvas:latest"
        assert d["imagePullPolicy"] == "IfNotPresent"

    def test_with_ports(self):
        spec = ContainerSpec(
            name="api",
            image="dvas:latest",
            ports=[{"containerPort": 8000}, {"containerPort": 9090}],
        )
        d = spec.to_dict()
        assert len(d["ports"]) == 2

    def test_with_probes(self):
        spec = ContainerSpec(
            name="api",
            image="dvas:latest",
            liveness_probe=HealthProbe(path="/health"),
            readiness_probe=HealthProbe(path="/ready"),
        )
        d = spec.to_dict()
        assert "livenessProbe" in d
        assert "readinessProbe" in d

    def test_with_resources(self):
        spec = ContainerSpec(
            name="api",
            image="dvas:latest",
            resources={
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"},
            },
        )
        d = spec.to_dict()
        assert d["resources"]["limits"]["cpu"] == "2"


class TestDeploymentSpec:
    """Test deployment specification."""

    def test_basic_deployment(self):
        spec = DeploymentSpec(
            name="dvas-api",
            namespace="production",
            replicas=3,
            containers=[
                ContainerSpec(name="api", image="dvas:latest"),
            ],
        )
        d = spec.to_dict()
        assert d["apiVersion"] == "apps/v1"
        assert d["kind"] == "Deployment"
        assert d["metadata"]["name"] == "dvas-api"
        assert d["metadata"]["namespace"] == "production"
        assert d["spec"]["replicas"] == 3

    def test_rolling_update_strategy(self):
        spec = DeploymentSpec(
            name="dvas",
            replicas=3,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            max_surge="50%",
            max_unavailable="0",
        )
        d = spec.to_dict()
        assert d["spec"]["strategy"]["type"] == "RollingUpdate"
        assert d["spec"]["strategy"]["rollingUpdate"]["maxSurge"] == "50%"

    def test_recreate_strategy(self):
        spec = DeploymentSpec(
            name="dvas",
            replicas=1,
            strategy=DeploymentStrategy.RECREATE,
        )
        d = spec.to_dict()
        assert d["spec"]["strategy"]["type"] == "Recreate"

    def test_with_volumes(self):
        spec = DeploymentSpec(
            name="dvas",
            replicas=1,
            containers=[
                ContainerSpec(name="api", image="dvas:latest"),
            ],
            volumes=[{"name": "data", "emptyDir": {}}],
        )
        d = spec.to_dict()
        assert "volumes" in d["spec"]["template"]["spec"]

    def test_selector(self):
        spec = DeploymentSpec(
            name="dvas",
            selector={"app": "dvas", "tier": "api"},
        )
        d = spec.to_dict()
        assert d["spec"]["selector"]["matchLabels"]["app"] == "dvas"
        assert d["spec"]["selector"]["matchLabels"]["tier"] == "api"


class TestServiceConfig:
    """Test service configuration."""

    def test_basic_service(self):
        config = ServiceConfig(
            name="dvas-service",
            selector={"app": "dvas"},
            ports=[{"port": 80, "targetPort": 8000}],
        )
        d = config.to_dict()
        assert d["apiVersion"] == "v1"
        assert d["kind"] == "Service"
        assert d["spec"]["type"] == "ClusterIP"

    def test_load_balancer(self):
        config = ServiceConfig(
            name="dvas-lb",
            service_type="LoadBalancer",
            ports=[{"port": 443, "targetPort": 8000}],
        )
        d = config.to_dict()
        assert d["spec"]["type"] == "LoadBalancer"


class TestConfigMapRef:
    """Test ConfigMap reference."""

    def test_configmap(self):
        config = ConfigMapRef(
            name="dvas-config",
            namespace="production",
            data={"LOG_LEVEL": "INFO", "MAX_WORKERS": "4"},
        )
        d = config.to_dict()
        assert d["kind"] == "ConfigMap"
        assert d["data"]["LOG_LEVEL"] == "INFO"


class TestSecretRef:
    """Test Secret reference."""

    def test_secret(self):
        secret = SecretRef(
            name="api-keys",
            namespace="production",
            data={"API_KEY": "secret123"},
        )
        d = secret.to_dict()
        assert d["kind"] == "Secret"
        assert d["type"] == "Opaque"
        import base64
        assert d["data"]["API_KEY"] == base64.b64encode(b"secret123").decode()


class TestHorizontalPodAutoscaler:
    """Test HPA configuration."""

    def test_basic_hpa(self):
        hpa = HorizontalPodAutoscaler(
            name="dvas-hpa",
            target_deployment="dvas-api",
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=75,
        )
        d = hpa.to_dict()
        assert d["apiVersion"] == "autoscaling/v2"
        assert d["kind"] == "HorizontalPodAutoscaler"
        assert d["spec"]["minReplicas"] == 2
        assert d["spec"]["maxReplicas"] == 10
        assert d["spec"]["metrics"][0]["resource"]["name"] == "cpu"

    def test_memory_metric(self):
        hpa = HorizontalPodAutoscaler(
            name="dvas-hpa",
            target_deployment="dvas-api",
            target_cpu_utilization=80,
            target_memory_utilization=70,
        )
        d = hpa.to_dict()
        assert len(d["spec"]["metrics"]) == 2


class TestNamespaceConfig:
    """Test namespace configuration."""

    def test_namespace(self):
        config = NamespaceConfig(
            name="dvas-production",
            labels={"env": "production"},
        )
        d = config.to_dict()
        assert d["kind"] == "Namespace"
        assert d["metadata"]["name"] == "dvas-production"


class TestKubernetesManager:
    """Test Kubernetes manager (mocked)."""

    @pytest.fixture
    def manager(self):
        return KubernetesManager()

    def test_init(self, manager):
        assert manager.kubectl_cmd == "kubectl"
        assert len(manager) == 0

    def test_apply_manifest(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "deployment.apps/dvas created\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "dvas"},
            }
            assert manager.apply_manifest(manifest) is True

    def test_apply_deployment(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            spec = DeploymentSpec(
                name="dvas",
                containers=[ContainerSpec(name="api", image="dvas:latest")],
            )
            assert manager.apply_deployment(spec) is True
            assert "default/dvas" in manager

    def test_delete_deployment(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.delete_deployment("dvas") is True

    def test_scale_deployment(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.scale_deployment("dvas", 5) is True

    def test_rollout_restart(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.rollout_restart("dvas") is True

    def test_rollout_undo(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            assert manager.rollout_undo("dvas") is True
            assert manager.rollout_undo("dvas", revision=3) is True

    def test_rollout_status(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "deployment \"dvas\" successfully rolled out\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            status = manager.rollout_status("dvas")
            assert "successfully" in status

    def test_apply_service(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            config = ServiceConfig(name="dvas-svc", selector={"app": "dvas"})
            assert manager.apply_service(config) is True

    def test_apply_configmap(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            config = ConfigMapRef(name="cfg", data={"key": "value"})
            assert manager.apply_configmap(config) is True

    def test_apply_secret(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            secret = SecretRef(name="sec", data={"key": "val"})
            assert manager.apply_secret(secret) is True

    def test_apply_hpa(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            hpa = HorizontalPodAutoscaler(name="hpa", target_deployment="dvas")
            assert manager.apply_hpa(hpa) is True

    def test_create_namespace(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            config = NamespaceConfig(name="dvas-ns")
            assert manager.create_namespace(config) is True

    def test_get_pods(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"items":[{"metadata":{"name":"pod-1"},"spec":{"nodeName":"node-1"},"status":{"phase":"Running","containerStatuses":[{"ready":true}]}}]}'
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            pods = manager.get_pods()
            assert len(pods) == 1
            assert pods[0]["name"] == "pod-1"
            assert pods[0]["status"] == "Running"
            assert pods[0]["ready"] is True

    def test_get_logs(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "log line 1\nlog line 2\n"
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            logs = manager.get_logs("pod-1", tail=10)
            assert len(logs) == 2

    def test_get_deployment_status(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"metadata":{"name":"dvas"},"spec":{"replicas":3},"status":{"readyReplicas":3}}'
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            status = manager.get_deployment_status("dvas")
            assert status["spec"]["replicas"] == 3

    def test_list_deployments(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"items":[{"metadata":{"name":"dvas"},"spec":{"replicas":3},"status":{"readyReplicas":3,"updatedReplicas":3}}]}'
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            deployments = manager.list_deployments()
            assert len(deployments) == 1
            assert deployments[0]["name"] == "dvas"
            assert deployments[0]["replicas"] == 3

    def test_export_manifest(self, manager):
        import tempfile
        spec = DeploymentSpec(
            name="dvas",
            containers=[ContainerSpec(name="api", image="dvas:latest")],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "deployment.json"
            manager.export_manifest(spec, output)
            assert output.exists()
            data = __import__("json").loads(output.read_text())
            assert data["kind"] == "Deployment"

    def test_get_resource(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"metadata":{"name":"dvas"}}'
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            resource = manager.get_resource("deployment", "dvas")
            assert resource["metadata"]["name"] == "dvas"

    def test_contains(self, manager):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(manager, "_run_cmd", return_value=mock_result):
            spec = DeploymentSpec(
                name="dvas",
                containers=[ContainerSpec(name="api", image="dvas:latest")],
            )
            manager.apply_deployment(spec)
            assert "default/dvas" in manager
