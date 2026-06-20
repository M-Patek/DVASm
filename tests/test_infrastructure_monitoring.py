"""Tests for the monitoring infrastructure module."""

import json
import tempfile
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# PrometheusConfig
# ---------------------------------------------------------------------------
def test_prometheus_config_defaults() -> None:
    """PrometheusConfig uses sensible defaults."""
    config = PrometheusConfig()
    assert config.scrape_interval == "15s"
    assert config.evaluation_interval == "15s"
    assert config.retention_days == 15
    assert config.storage_path == "/prometheus"
    assert config.targets == []
    assert config.alert_rules == []


def test_prometheus_config_to_yaml_basic() -> None:
    """to_yaml() produces a valid YAML string with global settings."""
    config = PrometheusConfig(scrape_interval="30s", evaluation_interval="1m")
    yaml = config.to_yaml()
    assert "scrape_interval: 30s" in yaml
    assert "evaluation_interval: 1m" in yaml
    assert "scrape_configs:" in yaml


def test_prometheus_config_to_yaml_with_targets() -> None:
    """to_yaml() includes configured targets."""
    config = PrometheusConfig(targets=[{"job_name": "app", "static_configs": [{"targets": ["localhost:9090"]}]}])
    yaml = config.to_yaml()
    assert "job_name: 'app'" in yaml


def test_prometheus_config_to_dict() -> None:
    """to_dict() returns the expected dictionary."""
    config = PrometheusConfig(scrape_interval="10s", retention_days=7, storage_path="/tmp/prom")
    data = config.to_dict()
    assert data["global"]["scrape_interval"] == "10s"
    assert data["retention_days"] == 7
    assert data["storage_path"] == "/tmp/prom"


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------
def test_alert_rule_to_dict() -> None:
    """AlertRule.to_dict() returns a Prometheus-compatible alert rule."""
    rule = AlertRule(
        name="HighErrorRate",
        expr="rate(errors[5m]) > 0.1",
        duration="10m",
        severity="critical",
        summary="High error rate detected",
        description="Error rate is above threshold",
    )
    data = rule.to_dict()
    assert data["alert"] == "HighErrorRate"
    assert data["expr"] == "rate(errors[5m]) > 0.1"
    assert data["for"] == "10m"
    assert data["labels"]["severity"] == "critical"
    assert data["annotations"]["summary"] == "High error rate detected"
    assert data["annotations"]["description"] == "Error rate is above threshold"


def test_alert_rule_to_dict_defaults() -> None:
    """AlertRule.to_dict() fills in default summary and description."""
    rule = AlertRule(name="TestAlert", expr="up == 0")
    data = rule.to_dict()
    assert data["annotations"]["summary"] == "Alert: TestAlert"
    assert data["annotations"]["description"] == "TestAlert triggered"


def test_alert_rule_labels_and_annotations() -> None:
    """AlertRule includes custom labels and annotations."""
    rule = AlertRule(
        name="Alert",
        expr="1",
        labels={"team": "platform"},
        annotations={"dashboard": "http://grafana/alert"},
    )
    data = rule.to_dict()
    assert data["labels"]["team"] == "platform"
    assert data["annotations"]["dashboard"] == "http://grafana/alert"


def test_alert_rule_to_yaml() -> None:
    """AlertRule.to_yaml() produces a JSON-formatted groups structure."""
    rule = AlertRule(name="Alert", expr="1")
    yaml_text = rule.to_yaml()
    parsed = json.loads(yaml_text)
    assert "groups" in parsed
    assert len(parsed["groups"]) == 1
    assert parsed["groups"][0]["name"] == "Alert"


# ---------------------------------------------------------------------------
# AlertmanagerConfig
# ---------------------------------------------------------------------------
def test_alertmanager_config_defaults() -> None:
    """AlertmanagerConfig has default empty lists."""
    config = AlertmanagerConfig()
    assert config.to_dict()["receivers"] == []
    assert config.to_dict()["route"]["receiver"] == "default"


def test_alertmanager_add_email_receiver() -> None:
    """add_email_receiver() appends an email receiver and returns self."""
    config = AlertmanagerConfig()
    result = config.add_email_receiver("team-email", "team@example.com")
    assert result is config
    receivers = config.to_dict()["receivers"]
    assert len(receivers) == 1
    assert receivers[0]["name"] == "team-email"
    assert receivers[0]["email_configs"][0]["to"] == "team@example.com"


def test_alertmanager_add_slack_receiver() -> None:
    """add_slack_receiver() appends a Slack receiver and returns self."""
    config = AlertmanagerConfig()
    result = config.add_slack_receiver("team-slack", "https://hooks.slack.com/test", channel="#alerts")
    assert result is config
    receivers = config.to_dict()["receivers"]
    assert len(receivers) == 1
    assert receivers[0]["name"] == "team-slack"
    assert receivers[0]["slack_configs"][0]["channel"] == "#alerts"


def test_alertmanager_chained_receivers() -> None:
    """Receivers can be added in a chain."""
    config = AlertmanagerConfig()
    config.add_email_receiver("email", "a@b.com").add_slack_receiver("slack", "https://hooks.example.com")
    assert len(config.to_dict()["receivers"]) == 2


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------
def test_dashboard_config_defaults() -> None:
    """DashboardConfig uses sensible defaults."""
    panel = DashboardConfig(title="CPU", query="cpu_usage")
    data = panel.to_dict()
    assert data["title"] == "CPU"
    assert data["type"] == "timeseries"
    assert data["datasource"] == "prometheus"
    assert data["gridPos"] == {"w": 12, "h": 8}
    assert data["fieldConfig"]["defaults"]["unit"] == ""


def test_dashboard_config_targets_fallback() -> None:
    """DashboardConfig falls back to a query-derived target when targets is empty."""
    panel = DashboardConfig(title="Memory", query="mem_usage")
    data = panel.to_dict()
    assert data["targets"] == [{"expr": "mem_usage", "refId": "A"}]


def test_dashboard_config_custom_targets() -> None:
    """DashboardConfig uses provided targets."""
    panel = DashboardConfig(title="Disk", query="", targets=[{"expr": "disk", "refId": "B"}])
    data = panel.to_dict()
    assert data["targets"] == [{"expr": "disk", "refId": "B"}]


# ---------------------------------------------------------------------------
# GrafanaDashboard
# ---------------------------------------------------------------------------
def test_grafana_dashboard_to_dict() -> None:
    """GrafanaDashboard.to_dict() produces a valid dashboard payload."""
    dash = GrafanaDashboard(title="Test", uid="test-01")
    data = dash.to_dict()
    assert data["dashboard"]["title"] == "Test"
    assert data["dashboard"]["uid"] == "test-01"
    assert data["overwrite"] is True
    assert data["dashboard"]["panels"] == []


def test_grafana_dashboard_add_panel() -> None:
    """add_panel() appends a panel and returns self."""
    dash = GrafanaDashboard(title="Test", uid="test-02")
    panel = DashboardConfig(title="Panel1", query="up")
    result = dash.add_panel(panel)
    assert result is dash
    assert len(dash.panels) == 1
    assert dash.to_dict()["dashboard"]["panels"][0]["title"] == "Panel1"


def test_grafana_dashboard_to_json() -> None:
    """to_json() returns a JSON string that round-trips."""
    dash = GrafanaDashboard(title="JSON Test", uid="json-01")
    dash.add_panel(DashboardConfig(title="P", query="1"))
    json_text = dash.to_json()
    parsed = json.loads(json_text)
    assert parsed["dashboard"]["title"] == "JSON Test"
    assert len(parsed["dashboard"]["panels"]) == 1


# ---------------------------------------------------------------------------
# LokiConfig
# ---------------------------------------------------------------------------
def test_loki_config_defaults() -> None:
    """LokiConfig uses sensible defaults."""
    config = LokiConfig()
    assert config.retention_period == "720h"
    assert config.chunk_size == 262144
    assert config.max_chunks_age == "168h"
    assert config.storage_path == "/loki"
    assert config.labels == ["job", "level", "service"]


def test_loki_config_to_dict() -> None:
    """LokiConfig.to_dict() returns the expected structure."""
    config = LokiConfig(retention_period="48h", storage_path="/var/loki")
    data = config.to_dict()
    assert data["limits_config"]["retention_period"] == "48h"
    assert data["storage_config"]["filesystem"]["directory"] == "/var/loki"
    assert data["table_manager"]["retention_deletes_enabled"] is True


# ---------------------------------------------------------------------------
# TracingConfig
# ---------------------------------------------------------------------------
def test_tracing_config_defaults() -> None:
    """TracingConfig uses sensible defaults."""
    config = TracingConfig()
    assert config.service_name == "dvas"
    assert config.agent_host == "jaeger-agent"
    assert config.agent_port == 6831
    assert config.sampler_type == "probabilistic"
    assert config.sampler_param == 1.0
    assert config.max_operations == 2000


def test_tracing_config_to_dict() -> None:
    """TracingConfig.to_dict() returns the expected structure."""
    config = TracingConfig(service_name="myapp", agent_host="localhost")
    data = config.to_dict()
    assert data["service_name"] == "myapp"
    assert data["agent"]["host"] == "localhost"
    assert data["sampler"]["type"] == "probabilistic"
    assert data["reporter"]["max_operations"] == 2000


def test_tracing_config_to_env() -> None:
    """TracingConfig.to_env() exports environment variables."""
    config = TracingConfig(service_name="svc", agent_port=14268, sampler_param=0.5)
    env = config.to_env()
    assert env["JAEGER_SERVICE_NAME"] == "svc"
    assert env["JAEGER_AGENT_PORT"] == "14268"
    assert env["JAEGER_SAMPLER_PARAM"] == "0.5"


# ---------------------------------------------------------------------------
# MonitoringStack
# ---------------------------------------------------------------------------
class TestMonitoringStack:
    """Tests for the MonitoringStack orchestrator."""

    @pytest.fixture
    def stack(self) -> MonitoringStack:
        """Return a fresh MonitoringStack instance."""
        return MonitoringStack()

    # -- Prometheus -------------------------------------------------------

    def test_configure_prometheus(self, stack: MonitoringStack) -> None:
        """configure_prometheus() stores a PrometheusConfig."""
        result = stack.configure_prometheus(scrape_interval="30s", retention_days=30)
        assert result is stack
        prom = stack.get_prometheus_config()
        assert prom is not None
        assert prom.scrape_interval == "30s"
        assert prom.retention_days == 30

    def test_add_scrape_target(self, stack: MonitoringStack) -> None:
        """add_scrape_target() appends a target to the Prometheus config."""
        stack.configure_prometheus()
        stack.add_scrape_target("dvas-api", ["api-1", "api-2"], port=8080, metrics_path="/metrics")
        prom = stack.get_prometheus_config()
        assert prom is not None
        assert len(prom.targets) == 1
        assert prom.targets[0]["job_name"] == "dvas-api"

    def test_add_scrape_target_auto_initializes(self, stack: MonitoringStack) -> None:
        """add_scrape_target() auto-initializes Prometheus if not configured."""
        stack.add_scrape_target("job", ["localhost"])
        assert stack.get_prometheus_config() is not None

    # -- alert rules ------------------------------------------------------

    def test_add_alert_rule(self, stack: MonitoringStack) -> None:
        """add_alert_rule() stores an AlertRule and links it to Prometheus."""
        stack.configure_prometheus()
        result = stack.add_alert_rule(
            name="high_cpu",
            expr="cpu > 80",
            duration="2m",
            severity="warning",
        )
        assert result is stack
        assert len(stack.get_alert_rules()) == 1
        assert stack.get_alert_rules()[0].name == "high_cpu"
        prom = stack.get_prometheus_config()
        assert prom is not None
        assert any("high_cpu" in str(r) for r in prom.alert_rules)

    # -- Alertmanager -----------------------------------------------------

    def test_configure_alertmanager(self, stack: MonitoringStack) -> None:
        """configure_alertmanager() stores an AlertmanagerConfig."""
        result = stack.configure_alertmanager()
        assert result is stack
        assert stack.get_alertmanager_config() is not None

    # -- Grafana dashboards -----------------------------------------------

    def test_add_grafana_dashboard(self, stack: MonitoringStack) -> None:
        """add_grafana_dashboard() stores a GrafanaDashboard."""
        dash = GrafanaDashboard(title="Test", uid="test-01")
        result = stack.add_grafana_dashboard(dash)
        assert result is stack
        assert len(stack.get_grafana_dashboards()) == 1

    def test_create_default_dashboard(self, stack: MonitoringStack) -> None:
        """create_default_dashboard() adds a pre-built dashboard."""
        dash = stack.create_default_dashboard()
        assert dash.title == "DVAS Overview"
        assert dash.uid == "dvas-overview"
        assert len(dash.panels) == 5
        assert len(stack.get_grafana_dashboards()) == 1

    # -- Loki -------------------------------------------------------------

    def test_configure_loki(self, stack: MonitoringStack) -> None:
        """configure_loki() stores a LokiConfig."""
        result = stack.configure_loki(retention_period="48h", storage_path="/tmp/loki")
        assert result is stack
        loki = stack.get_loki_config()
        assert loki is not None
        assert loki.retention_period == "48h"
        assert loki.storage_path == "/tmp/loki"

    # -- Tracing ----------------------------------------------------------

    def test_configure_tracing(self, stack: MonitoringStack) -> None:
        """configure_tracing() stores a TracingConfig."""
        result = stack.configure_tracing(service_name="worker", agent_host="agent.local")
        assert result is stack
        tracing = stack.get_tracing_config()
        assert tracing is not None
        assert tracing.service_name == "worker"
        assert tracing.agent_host == "agent.local"

    # -- export -----------------------------------------------------------

    def test_export_configs(self, stack: MonitoringStack) -> None:
        """export_configs() writes all configured files to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "monitoring"
            stack.configure_prometheus().add_scrape_target("app", ["localhost"])
            stack.add_alert_rule("rule", "expr")
            stack.configure_alertmanager()
            am = stack.get_alertmanager_config()
            assert am is not None
            am.add_email_receiver("r", "a@b.com")
            dash = GrafanaDashboard(title="D", uid="d")
            dash.add_panel(DashboardConfig(title="P", query="1"))
            stack.add_grafana_dashboard(dash)
            stack.configure_loki()
            stack.configure_tracing()

            stack.export_configs(out)

            assert (out / "prometheus.yml").exists()
            assert (out / "alertmanager.yml").exists()
            assert (out / "dashboard-d.json").exists()
            assert (out / "loki.yml").exists()
            assert (out / "tracing.yml").exists()
            assert (out / "rules" / "rule.yml").exists()

    # -- __len__ ----------------------------------------------------------

    def test_len_empty(self, stack: MonitoringStack) -> None:
        """__len__() returns 0 for an empty stack."""
        assert len(stack) == 0

    def test_len_with_components(self, stack: MonitoringStack) -> None:
        """__len__() counts configured components."""
        stack.configure_prometheus()
        stack.configure_alertmanager()
        stack.configure_loki()
        stack.configure_tracing()
        dash = GrafanaDashboard(title="D", uid="d")
        stack.add_grafana_dashboard(dash)
        assert len(stack) == 5
