"""Monitoring infrastructure setup for DVAS.

Provides configuration and deployment helpers for Prometheus, Grafana,
Alertmanager, Loki, and Jaeger distributed tracing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PrometheusConfig:
    """Prometheus scrape configuration."""

    scrape_interval: str = "15s"
    evaluation_interval: str = "15s"
    targets: List[Dict[str, Any]] = field(default_factory=list)
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)
    retention_days: int = 15
    storage_path: str = "/prometheus"

    def to_yaml(self) -> str:
        """Generate Prometheus configuration YAML."""
        lines: List[str] = [
            "global:",
            f"  scrape_interval: {self.scrape_interval}",
            f"  evaluation_interval: {self.evaluation_interval}",
            "",
            "scrape_configs:",
        ]
        for target in self.targets:
            lines.append(f"  - job_name: '{target.get('job_name', 'default')}'")
            lines.append("    static_configs:")
            for static in target.get("static_configs", []):
                targets = static.get("targets", [])
                lines.append(f"      - targets: {json.dumps(targets)}")
        lines.append("")
        if self.alert_rules:
            lines.append("rule_files:")
            for rule in self.alert_rules:
                lines.append(f"  - '{rule.get('file', 'alerts.yml')}'")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "global": {
                "scrape_interval": self.scrape_interval,
                "evaluation_interval": self.evaluation_interval,
            },
            "scrape_configs": self.targets,
            "rule_files": [r.get("file", "alerts.yml") for r in self.alert_rules],
            "retention_days": self.retention_days,
            "storage_path": self.storage_path,
        }


@dataclass
class AlertRule:
    """Prometheus alerting rule."""

    name: str
    expr: str
    duration: str = "5m"
    severity: str = "warning"
    summary: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to alert rule dictionary."""
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.duration,
            "labels": {
                "severity": self.severity,
                **self.labels,
            },
            "annotations": {
                "summary": self.summary or f"Alert: {self.name}",
                "description": self.description or f"{self.name} triggered",
                **self.annotations,
            },
        }

    def to_yaml(self) -> str:
        """Generate YAML alert rule."""
        return json.dumps({"groups": [{"name": self.name, "rules": [self.to_dict()]}]}, indent=2)


@dataclass
class AlertmanagerConfig:
    """Alertmanager configuration."""

    receivers: List[Dict[str, Any]] = field(default_factory=list)
    routes: List[Dict[str, Any]] = field(default_factory=list)
    inhibit_rules: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "global": {
                "resolve_timeout": "5m",
            },
            "route": {
                "group_by": ["alertname"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "12h",
                "receiver": "default",
                "routes": self.routes,
            },
            "receivers": self.receivers,
            "inhibit_rules": self.inhibit_rules,
        }

    def add_email_receiver(self, name: str, to: str, from_addr: str = "alerts@dvas.local",
                           smarthost: str = "localhost:25") -> "AlertmanagerConfig":
        """Add an email receiver.

        Args:
            name: Receiver name.
            to: Recipient email.
            from_addr: Sender email.
            smarthost: SMTP server.

        Returns:
            Self for chaining.
        """
        self.receivers.append({
            "name": name,
            "email_configs": [{
                "to": to,
                "from": from_addr,
                "smarthost": smarthost,
                "headers": {"Subject": "DVAS Alert"},
            }],
        })
        return self

    def add_slack_receiver(self, name: str, webhook_url: str,
                           channel: str = "#alerts") -> "AlertmanagerConfig":
        """Add a Slack receiver.

        Args:
            name: Receiver name.
            webhook_url: Slack webhook URL.
            channel: Slack channel.

        Returns:
            Self for chaining.
        """
        self.receivers.append({
            "name": name,
            "slack_configs": [{
                "api_url": webhook_url,
                "channel": channel,
                "title": "DVAS Alert",
                "text": "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}",
            }],
        })
        return self


@dataclass
class DashboardConfig:
    """Grafana dashboard panel configuration."""

    title: str
    query: str
    datasource: str = "prometheus"
    panel_type: str = "timeseries"
    unit: str = ""
    width: int = 12
    height: int = 8
    targets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to panel dictionary."""
        return {
            "title": self.title,
            "type": self.panel_type,
            "datasource": self.datasource,
            "gridPos": {"w": self.width, "h": self.height},
            "targets": self.targets or [{"expr": self.query, "refId": "A"}],
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                },
            },
        }


@dataclass
class GrafanaDashboard:
    """Grafana dashboard configuration."""

    title: str
    uid: str
    panels: List[DashboardConfig] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refresh: str = "30s"
    timezone: str = "utc"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Grafana dashboard JSON."""
        return {
            "dashboard": {
                "title": self.title,
                "uid": self.uid,
                "tags": self.tags,
                "timezone": self.timezone,
                "refresh": self.refresh,
                "panels": [p.to_dict() for p in self.panels],
            },
            "overwrite": True,
        }

    def add_panel(self, panel: DashboardConfig) -> "GrafanaDashboard":
        """Add a panel to the dashboard.

        Args:
            panel: Panel configuration.

        Returns:
            Self for chaining.
        """
        self.panels.append(panel)
        return self

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LokiConfig:
    """Loki log aggregation configuration."""

    retention_period: str = "720h"
    chunk_size: int = 262144
    max_chunks_age: str = "168h"
    storage_path: str = "/loki"
    labels: List[str] = field(default_factory=lambda: ["job", "level", "service"])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "limits_config": {
                "retention_period": self.retention_period,
            },
            "chunk_store_config": {
                "max_look_back_period": self.max_chunks_age,
            },
            "table_manager": {
                "retention_deletes_enabled": True,
                "retention_period": self.retention_period,
            },
            "storage_config": {
                "filesystem": {
                    "directory": self.storage_path,
                },
            },
        }


@dataclass
class TracingConfig:
    """Distributed tracing (Jaeger) configuration."""

    service_name: str = "dvas"
    agent_host: str = "jaeger-agent"
    agent_port: int = 6831
    sampler_type: str = "probabilistic"
    sampler_param: float = 1.0
    max_operations: int = 2000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "service_name": self.service_name,
            "agent": {
                "host": self.agent_host,
                "port": self.agent_port,
            },
            "sampler": {
                "type": self.sampler_type,
                "param": self.sampler_param,
            },
            "reporter": {
                "max_operations": self.max_operations,
            },
        }

    def to_env(self) -> Dict[str, str]:
        """Export as environment variables."""
        return {
            "JAEGER_SERVICE_NAME": self.service_name,
            "JAEGER_AGENT_HOST": self.agent_host,
            "JAEGER_AGENT_PORT": str(self.agent_port),
            "JAEGER_SAMPLER_TYPE": self.sampler_type,
            "JAEGER_SAMPLER_PARAM": str(self.sampler_param),
        }


class MonitoringStack:
    """Deploy and configure monitoring infrastructure for DVAS.

    Usage::

        stack = MonitoringStack()
        stack.configure_prometheus(scrape_interval="15s")
        stack.add_alert_rule("high_latency", "latency_p95 > 1000", severity="critical")
        stack.export_configs(Path("./monitoring"))
    """

    def __init__(self) -> None:
        self._prometheus: Optional[PrometheusConfig] = None
        self._alertmanager: Optional[AlertmanagerConfig] = None
        self._grafana: List[GrafanaDashboard] = []
        self._loki: Optional[LokiConfig] = None
        self._tracing: Optional[TracingConfig] = None
        self._alert_rules: List[AlertRule] = []

    def configure_prometheus(
        self,
        scrape_interval: str = "15s",
        retention_days: int = 15,
        storage_path: str = "/prometheus",
    ) -> "MonitoringStack":
        """Configure Prometheus.

        Args:
            scrape_interval: Scrape interval.
            retention_days: Data retention.
            storage_path: Storage path.

        Returns:
            Self for chaining.
        """
        self._prometheus = PrometheusConfig(
            scrape_interval=scrape_interval,
            retention_days=retention_days,
            storage_path=storage_path,
        )
        logger.info("prometheus_configured", interval=scrape_interval)
        return self

    def add_scrape_target(self, job_name: str, targets: List[str],
                         port: int = 8000,
                         metrics_path: str = "/metrics") -> "MonitoringStack":
        """Add a scrape target to Prometheus.

        Args:
            job_name: Job name.
            targets: Target hostnames.
            port: Target port.
            metrics_path: Metrics path.

        Returns:
            Self for chaining.
        """
        if self._prometheus is None:
            self.configure_prometheus()

        self._prometheus.targets.append({
            "job_name": job_name,
            "static_configs": [{"targets": [f"{t}:{port}" for t in targets]}],
            "metrics_path": metrics_path,
        })
        logger.info("scrape_target_added", job=job_name, targets=targets)
        return self

    def add_alert_rule(self, name: str, expr: str, duration: str = "5m",
                       severity: str = "warning", summary: str = "",
                       description: str = "") -> "MonitoringStack":
        """Add an alert rule.

        Args:
            name: Rule name.
            expr: PromQL expression.
            duration: Duration before firing.
            severity: Alert severity.
            summary: Summary text.
            description: Description text.

        Returns:
            Self for chaining.
        """
        rule = AlertRule(
            name=name,
            expr=expr,
            duration=duration,
            severity=severity,
            summary=summary,
            description=description,
        )
        self._alert_rules.append(rule)
        if self._prometheus:
            self._prometheus.alert_rules.append({"file": f"{name}.yml"})
        logger.info("alert_rule_added", name=name, severity=severity)
        return self

    def configure_alertmanager(self) -> "MonitoringStack":
        """Configure Alertmanager.

        Returns:
            Self for chaining.
        """
        self._alertmanager = AlertmanagerConfig()
        logger.info("alertmanager_configured")
        return self

    def add_grafana_dashboard(self, dashboard: GrafanaDashboard) -> "MonitoringStack":
        """Add a Grafana dashboard.

        Args:
            dashboard: Dashboard configuration.

        Returns:
            Self for chaining.
        """
        self._grafana.append(dashboard)
        logger.info("dashboard_added", title=dashboard.title, uid=dashboard.uid)
        return self

    def create_default_dashboard(self) -> GrafanaDashboard:
        """Create the default DVAS monitoring dashboard.

        Returns:
            Dashboard configuration.
        """
        dashboard = GrafanaDashboard(
            title="DVAS Overview",
            uid="dvas-overview",
            tags=["dvas", "overview"],
        )
        dashboard.add_panel(DashboardConfig(
            title="Request Rate",
            query='rate(http_requests_total{job="dvas"}[5m])',
            unit="reqps",
        ))
        dashboard.add_panel(DashboardConfig(
            title="Latency (p95)",
            query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            unit="s",
        ))
        dashboard.add_panel(DashboardConfig(
            title="Error Rate",
            query='rate(http_requests_total{job="dvas",status=~"5.."}[5m])',
            unit="percentunit",
        ))
        dashboard.add_panel(DashboardConfig(
            title="GPU Utilization",
            query='nvidia_gpu_utilization_gpu{job="dvas"}',
            unit="percent",
        ))
        dashboard.add_panel(DashboardConfig(
            title="Queue Depth",
            query='dvas_queue_depth{job="dvas"}',
            unit="short",
        ))
        self.add_grafana_dashboard(dashboard)
        return dashboard

    def configure_loki(self, retention_period: str = "720h",
                       storage_path: str = "/loki") -> "MonitoringStack":
        """Configure Loki for log aggregation.

        Args:
            retention_period: Log retention.
            storage_path: Storage path.

        Returns:
            Self for chaining.
        """
        self._loki = LokiConfig(
            retention_period=retention_period,
            storage_path=storage_path,
        )
        logger.info("loki_configured", retention=retention_period)
        return self

    def configure_tracing(self, service_name: str = "dvas",
                          agent_host: str = "jaeger-agent") -> "MonitoringStack":
        """Configure distributed tracing.

        Args:
            service_name: Service name.
            agent_host: Jaeger agent host.

        Returns:
            Self for chaining.
        """
        self._tracing = TracingConfig(
            service_name=service_name,
            agent_host=agent_host,
        )
        logger.info("tracing_configured", service=service_name)
        return self

    def export_configs(self, output_dir: Path) -> None:
        """Export all monitoring configurations to files.

        Args:
            output_dir: Directory to write configs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._prometheus:
            prom_path = output_dir / "prometheus.yml"
            prom_path.write_text(self._prometheus.to_yaml(), encoding="utf-8")
            logger.info("prometheus_config_exported", path=str(prom_path))

        if self._alert_rules:
            rules_dir = output_dir / "rules"
            rules_dir.mkdir(exist_ok=True)
            for rule in self._alert_rules:
                rule_path = rules_dir / f"{rule.name}.yml"
                rule_path.write_text(rule.to_yaml(), encoding="utf-8")

        if self._alertmanager:
            am_path = output_dir / "alertmanager.yml"
            am_path.write_text(
                json.dumps(self._alertmanager.to_dict(), indent=2),
                encoding="utf-8",
            )

        for dashboard in self._grafana:
            dash_path = output_dir / f"dashboard-{dashboard.uid}.json"
            dash_path.write_text(dashboard.to_json(), encoding="utf-8")

        if self._loki:
            loki_path = output_dir / "loki.yml"
            loki_path.write_text(
                json.dumps(self._loki.to_dict(), indent=2),
                encoding="utf-8",
            )

        if self._tracing:
            trace_path = output_dir / "tracing.yml"
            trace_path.write_text(
                json.dumps(self._tracing.to_dict(), indent=2),
                encoding="utf-8",
            )

        logger.info("configs_exported", dir=str(output_dir))

    def get_prometheus_config(self) -> Optional[PrometheusConfig]:
        """Get Prometheus configuration."""
        return self._prometheus

    def get_alertmanager_config(self) -> Optional[AlertmanagerConfig]:
        """Get Alertmanager configuration."""
        return self._alertmanager

    def get_grafana_dashboards(self) -> List[GrafanaDashboard]:
        """Get all Grafana dashboards."""
        return self._grafana.copy()

    def get_loki_config(self) -> Optional[LokiConfig]:
        """Get Loki configuration."""
        return self._loki

    def get_tracing_config(self) -> Optional[TracingConfig]:
        """Get tracing configuration."""
        return self._tracing

    def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return self._alert_rules.copy()

    def __len__(self) -> int:
        """Total number of configured components."""
        count = 0
        if self._prometheus:
            count += 1
        if self._alertmanager:
            count += 1
        count += len(self._grafana)
        if self._loki:
            count += 1
        if self._tracing:
            count += 1
        return count
