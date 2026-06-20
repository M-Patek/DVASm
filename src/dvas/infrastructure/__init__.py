"""Infrastructure management for DVAS.

Provides deployment and operations tooling for Docker, Kubernetes,
Terraform, CI/CD pipelines, and monitoring infrastructure.
"""

from dvas.infrastructure.cicd_manager import (
    Artifact,
    BuildStatus,
    CICDManager,
    DeploymentTarget,
    PipelineRun,
    SecretRef,
)
from dvas.infrastructure.docker_manager import (
    ContainerConfig,
    ContainerStatus,
    DockerManager,
    DockerfileBuilder,
    ImageBuildResult,
    ResourceLimits,
)
from dvas.infrastructure.kubernetes_manager import (
    ConfigMapRef,
    ContainerSpec,
    DeploymentSpec,
    HealthProbe,
    HorizontalPodAutoscaler,
    KubernetesManager,
    NamespaceConfig,
    ResourceQuota,
    SecretRef,
    ServiceConfig,
)
from dvas.infrastructure.monitoring_stack import (
    AlertRule,
    AlertmanagerConfig,
    DashboardConfig,
    GrafanaDashboard,
    LokiConfig,
    MonitoringStack,
    PrometheusConfig,
    TracingConfig,
)
from dvas.infrastructure.terraform_manager import (
    TerraformManager,
    TerraformModule,
    TerraformOutput,
    TerraformProvider,
    TerraformState,
    TerraformVariable,
    TerraformWorkspace,
)

__all__ = [
    # Docker
    "DockerManager",
    "DockerfileBuilder",
    "ContainerConfig",
    "ContainerStatus",
    "ImageBuildResult",
    "ResourceLimits",
    # Kubernetes
    "KubernetesManager",
    "DeploymentSpec",
    "ServiceConfig",
    "ConfigMapRef",
    "ContainerSpec",
    "HealthProbe",
    "HorizontalPodAutoscaler",
    "NamespaceConfig",
    "ResourceQuota",
    # Terraform
    "TerraformManager",
    "TerraformModule",
    "TerraformOutput",
    "TerraformProvider",
    "TerraformState",
    "TerraformVariable",
    "TerraformWorkspace",
    # CI/CD
    "CICDManager",
    "Artifact",
    "BuildStatus",
    "DeploymentTarget",
    "PipelineRun",
    "SecretRef",
    # Monitoring
    "MonitoringStack",
    "PrometheusConfig",
    "GrafanaDashboard",
    "AlertmanagerConfig",
    "LokiConfig",
    "DashboardConfig",
    "AlertRule",
    "TracingConfig",
]
