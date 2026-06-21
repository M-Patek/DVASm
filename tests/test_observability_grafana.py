"""Tests for Grafana dashboard definitions."""

import json


from dvas.observability.grafana import (
    DashboardPanel,
    DashboardRow,
    GrafanaDashboard,
    export_dashboard_to_file,
)


class TestDashboardPanel:
    def test_creation(self):
        panel = DashboardPanel(title="Test Panel")
        assert panel.title == "Test Panel"
        assert panel.panel_type == "timeseries"

    def test_to_dict(self):
        panel = DashboardPanel(
            title="Test",
            targets=[{"expr": "up"}],
            unit="ms",
        )
        d = panel.to_dict()
        assert d["title"] == "Test"
        assert d["type"] == "timeseries"
        assert d["targets"][0]["expr"] == "up"
        assert d["fieldConfig"]["defaults"]["unit"] == "ms"


class TestDashboardRow:
    def test_creation(self):
        panel = DashboardPanel(title="Panel")
        row = DashboardRow(title="Row", panels=[panel])
        assert row.title == "Row"
        assert len(row.panels) == 1

    def test_to_dict(self):
        panel = DashboardPanel(title="Panel")
        row = DashboardRow(title="Row", panels=[panel])
        d = row.to_dict()
        assert d["title"] == "Row"
        assert d["type"] == "row"
        assert len(d["panels"]) == 1


class TestGrafanaDashboard:
    def test_creation(self):
        dashboard = GrafanaDashboard(title="Test Dashboard")
        assert dashboard.title == "Test Dashboard"
        assert dashboard.uid == "test-dashboard"

    def test_add_panel(self):
        dashboard = GrafanaDashboard(title="Test")
        panel = DashboardPanel(title="Panel")
        dashboard.add_panel(panel)
        assert len(dashboard.panels) == 1

    def test_add_row(self):
        dashboard = GrafanaDashboard(title="Test")
        row = DashboardRow(title="Row")
        dashboard.add_row(row)
        assert len(dashboard.rows) == 1

    def test_add_variable(self):
        dashboard = GrafanaDashboard(title="Test")
        dashboard.add_variable("datasource", "prometheus")
        d = dashboard.to_dict()
        assert len(d["dashboard"]["templating"]["list"]) == 1

    def test_add_annotation(self):
        dashboard = GrafanaDashboard(title="Test")
        dashboard.add_annotation("deployments", expr="deployments")
        d = dashboard.to_dict()
        assert len(d["dashboard"]["annotations"]["list"]) == 1

    def test_to_dict(self):
        dashboard = GrafanaDashboard(title="Test")
        d = dashboard.to_dict()
        assert d["dashboard"]["title"] == "Test"
        assert d["overwrite"] is True

    def test_to_json(self):
        dashboard = GrafanaDashboard(title="Test")
        json_text = dashboard.to_json()
        parsed = json.loads(json_text)
        assert parsed["dashboard"]["title"] == "Test"

    def test_create_default(self):
        dashboard = GrafanaDashboard.create_default("DVAS Overview")
        d = dashboard.to_dict()
        assert d["dashboard"]["title"] == "DVAS Overview"
        assert d["dashboard"]["uid"] == "dvas-observability"
        # Should have rows with panels
        assert len(d["dashboard"]["panels"]) > 0

    def test_create_teacher_dashboard(self):
        dashboard = GrafanaDashboard.create_teacher_dashboard()
        d = dashboard.to_dict()
        assert d["dashboard"]["title"] == "DVAS Teacher Models"
        assert d["dashboard"]["uid"] == "dvas-teachers"

    def test_create_pipeline_dashboard(self):
        dashboard = GrafanaDashboard.create_pipeline_dashboard()
        d = dashboard.to_dict()
        assert d["dashboard"]["title"] == "DVAS Pipeline"
        assert d["dashboard"]["uid"] == "dvas-pipeline"

    def test_custom_uid(self):
        dashboard = GrafanaDashboard(title="Test", uid="custom-uid")
        d = dashboard.to_dict()
        assert d["dashboard"]["uid"] == "custom-uid"


class TestExportDashboardToFile:
    def test_export(self):
        import tempfile
        import os

        dashboard = GrafanaDashboard(title="Export Test")
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            export_dashboard_to_file(dashboard, path)
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            assert content["dashboard"]["title"] == "Export Test"
        finally:
            os.close(fd)
            os.unlink(path)
