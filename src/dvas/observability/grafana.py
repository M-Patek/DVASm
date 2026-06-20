"""Grafana dashboard definitions for DVAS.

Provides dashboard JSON models and configuration for visualizing
DVAS metrics in Grafana.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DashboardPanel:
    """A single Grafana dashboard panel."""

    title: str
    targets: List[Dict[str, Any]] = field(default_factory=list)
    panel_type: str = "timeseries"
    grid_pos: Dict[str, int] = field(default_factory=lambda: {"h": 8, "w": 12, "x": 0, "y": 0})
    unit: str = ""
    description: str = ""
    datasource: str = "prometheus"
    legend: Dict[str, Any] = field(
        default_factory=lambda: {"show": True, "values": True, "avg": True}
    )
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to Grafana JSON."""
        result: Dict[str, Any] = {
            "title": self.title,
            "type": self.panel_type,
            "gridPos": self.grid_pos,
            "targets": self.targets,
            "datasource": {"type": self.datasource, "uid": "${datasource}"},
            "legend": self.legend,
            "options": self.options,
        }
        if self.unit:
            result["fieldConfig"] = {
                "defaults": {"unit": self.unit, "custom": {"displayMode": "gradient"}}
            }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class DashboardRow:
    """A row of panels in a Grafana dashboard."""

    title: str
    panels: List[DashboardPanel] = field(default_factory=list)
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert row to Grafana JSON."""
        return {
            "title": self.title,
            "type": "row",
            "collapsed": self.collapsed,
            "panels": [p.to_dict() for p in self.panels],
        }


class GrafanaDashboard:
    """Grafana dashboard builder for DVAS observability.

    Builds complete dashboard JSON models for Grafana.

    Usage::

        dashboard = GrafanaDashboard.create_default("DVAS Overview")
        json_text = dashboard.to_json()
    """

    def __init__(
        self,
        title: str,
        uid: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timezone: str = "utc",
        refresh: str = "30s",
    ) -> None:
        self.title = title
        self.uid = uid or title.lower().replace(" ", "-")
        self.tags = tags or ["dvas", "observability"]
        self.timezone = timezone
        self.refresh = refresh
        self.panels: List[DashboardPanel] = []
        self.rows: List[DashboardRow] = []
        self._templating: List[Dict[str, Any]] = []
        self._annotations: List[Dict[str, Any]] = []

    def add_panel(self, panel: DashboardPanel) -> "GrafanaDashboard":
        """Add a panel to the dashboard."""
        self.panels.append(panel)
        return self

    def add_row(self, row: DashboardRow) -> "GrafanaDashboard":
        """Add a row to the dashboard."""
        self.rows.append(row)
        return self

    def add_variable(
        self,
        name: str,
        query: str,
        label: str = "",
        multi: bool = False,
    ) -> "GrafanaDashboard":
        """Add a template variable."""
        self._templating.append(
            {
                "name": name,
                "type": "query",
                "query": query,
                "label": label or name,
                "multi": multi,
                "current": {"text": "All", "value": "$__all"},
            }
        )
        return self

    def add_annotation(
        self,
        name: str,
        datasource: str = "prometheus",
        expr: str = "",
    ) -> "GrafanaDashboard":
        """Add an annotation query."""
        self._annotations.append(
            {
                "name": name,
                "datasource": {"type": datasource, "uid": "${datasource}"},
                "expr": expr,
                "iconColor": "red",
            }
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to Grafana JSON model."""
        all_panels: List[Dict[str, Any]] = []

        # Add standalone panels
        for panel in self.panels:
            all_panels.append(panel.to_dict())

        # Add row panels
        for row in self.rows:
            all_panels.append(row.to_dict())

        return {
            "dashboard": {
                "title": self.title,
                "uid": self.uid,
                "tags": self.tags,
                "timezone": self.timezone,
                "refresh": self.refresh,
                "panels": all_panels,
                "templating": {"list": self._templating},
                "annotations": {"list": self._annotations},
                "schemaVersion": 36,
                "version": 1,
            },
            "overwrite": True,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export dashboard as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def create_default(cls, title: str = "DVAS Observability") -> "GrafanaDashboard":
        """Create a default DVAS observability dashboard.

        Args:
            title: Dashboard title

        Returns:
            Pre-configured dashboard with standard DVAS panels
        """
        dashboard = cls(title=title, uid="dvas-observability")

        # Row 1: System Overview
        dashboard.add_row(
            DashboardRow(
                title="System Overview",
                panels=[
                    DashboardPanel(
                        title="Request Rate",
                        panel_type="stat",
                        grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
                        targets=[
                            {
                                "expr": "rate(dvas_requests_total[5m])",
                                "legendFormat": "{{method}}",
                            }
                        ],
                        unit="reqps",
                    ),
                    DashboardPanel(
                        title="Error Rate",
                        panel_type="stat",
                        grid_pos={"h": 4, "w": 6, "x": 6, "y": 0},
                        targets=[
                            {
                                "expr": "rate(dvas_errors_total[5m])",
                                "legendFormat": "{{type}}",
                            }
                        ],
                        unit="percentunit",
                    ),
                    DashboardPanel(
                        title="Active Connections",
                        panel_type="stat",
                        grid_pos={"h": 4, "w": 6, "x": 12, "y": 0},
                        targets=[
                            {
                                "expr": "dvas_active_connections",
                                "legendFormat": "connections",
                            }
                        ],
                    ),
                    DashboardPanel(
                        title="Uptime",
                        panel_type="stat",
                        grid_pos={"h": 4, "w": 6, "x": 18, "y": 0},
                        targets=[
                            {
                                "expr": "time() - dvas_start_time",
                                "legendFormat": "uptime",
                            }
                        ],
                        unit="dtdurations",
                    ),
                ],
            )
        )

        # Row 2: Teacher Model Metrics
        dashboard.add_row(
            DashboardRow(
                title="Teacher Models",
                panels=[
                    DashboardPanel(
                        title="Teacher Latency",
                        grid_pos={"h": 8, "w": 12, "x": 0, "y": 4},
                        targets=[
                            {
                                "expr": "histogram_quantile(0.95, rate(dvas_teacher_latency_bucket[5m]))",
                                "legendFormat": "p95 {{model}}",
                            },
                            {
                                "expr": "histogram_quantile(0.50, rate(dvas_teacher_latency_bucket[5m]))",
                                "legendFormat": "p50 {{model}}",
                            },
                        ],
                        unit="ms",
                        description="Teacher model response latency percentiles",
                    ),
                    DashboardPanel(
                        title="Teacher Cost",
                        grid_pos={"h": 8, "w": 12, "x": 12, "y": 4},
                        targets=[
                            {
                                "expr": "rate(dvas_teacher_cost_usd[1h])",
                                "legendFormat": "{{model}}",
                            }
                        ],
                        unit="currencyUSD",
                        description="Teacher model API cost per hour",
                    ),
                ],
            )
        )

        # Row 3: Pipeline & Quality
        dashboard.add_row(
            DashboardRow(
                title="Pipeline & Quality",
                panels=[
                    DashboardPanel(
                        title="Parser Failure Rate",
                        grid_pos={"h": 8, "w": 12, "x": 0, "y": 12},
                        targets=[
                            {
                                "expr": "rate(dvas_parser_failures_total[5m])",
                                "legendFormat": "{{parser_type}}",
                            }
                        ],
                        unit="percentunit",
                        description="Rate of parser failures by type",
                    ),
                    DashboardPanel(
                        title="Annotation Quality Score",
                        grid_pos={"h": 8, "w": 12, "x": 12, "y": 12},
                        targets=[
                            {
                                "expr": "dvas_annotation_quality_score",
                                "legendFormat": "{{model}}",
                            }
                        ],
                        unit="percentunit",
                        description="Average annotation quality score over time",
                    ),
                ],
            )
        )

        # Row 4: Queue & Export
        dashboard.add_row(
            DashboardRow(
                title="Queue & Export",
                panels=[
                    DashboardPanel(
                        title="Task Queue Depth",
                        grid_pos={"h": 8, "w": 12, "x": 0, "y": 20},
                        targets=[
                            {
                                "expr": "dvas_task_queue_depth",
                                "legendFormat": "{{queue_name}}",
                            }
                        ],
                        description="Number of tasks waiting in queues",
                    ),
                    DashboardPanel(
                        title="Export Throughput",
                        grid_pos={"h": 8, "w": 12, "x": 12, "y": 20},
                        targets=[
                            {
                                "expr": "rate(dvas_export_throughput_bytes[5m])",
                                "legendFormat": "{{format}}",
                            }
                        ],
                        unit="Bps",
                        description="Data export throughput by format",
                    ),
                ],
            )
        )

        # Row 5: Storage & Student
        dashboard.add_row(
            DashboardRow(
                title="Storage & Student Models",
                panels=[
                    DashboardPanel(
                        title="Storage Size",
                        grid_pos={"h": 8, "w": 12, "x": 0, "y": 28},
                        targets=[
                            {
                                "expr": "dvas_storage_size_bytes",
                                "legendFormat": "{{storage_type}}",
                            }
                        ],
                        unit="bytes",
                        description="Storage usage by type",
                    ),
                    DashboardPanel(
                        title="Student Fallback Rate",
                        grid_pos={"h": 8, "w": 12, "x": 12, "y": 28},
                        targets=[
                            {
                                "expr": "rate(dvas_student_fallback_total[5m])",
                                "legendFormat": "{{reason}}",
                            }
                        ],
                        unit="percentunit",
                        description="Rate of student model fallbacks by reason",
                    ),
                ],
            )
        )

        # Add template variable for datasource
        dashboard.add_variable(
            name="datasource",
            query="prometheus",
            label="Data Source",
        )

        return dashboard

    @classmethod
    def create_teacher_dashboard(cls, title: str = "DVAS Teacher Models") -> "GrafanaDashboard":
        """Create a teacher model focused dashboard.

        Args:
            title: Dashboard title

        Returns:
            Dashboard focused on teacher model metrics
        """
        dashboard = cls(title=title, uid="dvas-teachers")

        dashboard.add_panel(
            DashboardPanel(
                title="Latency by Model",
                grid_pos={"h": 8, "w": 24, "x": 0, "y": 0},
                targets=[
                    {
                        "expr": "histogram_quantile(0.99, rate(dvas_teacher_latency_bucket[5m]))",
                        "legendFormat": "p99 {{model}}",
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(dvas_teacher_latency_bucket[5m]))",
                        "legendFormat": "p95 {{model}}",
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(dvas_teacher_latency_bucket[5m]))",
                        "legendFormat": "p50 {{model}}",
                    },
                ],
                unit="ms",
            )
        )

        dashboard.add_panel(
            DashboardPanel(
                title="Cost per Request",
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 8},
                targets=[
                    {
                        "expr": "dvas_teacher_cost_usd / dvas_teacher_requests_total",
                        "legendFormat": "{{model}}",
                    }
                ],
                unit="currencyUSD",
            )
        )

        dashboard.add_panel(
            DashboardPanel(
                title="Token Usage",
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 8},
                targets=[
                    {
                        "expr": "rate(dvas_teacher_tokens_total[5m])",
                        "legendFormat": "{{direction}}",
                    }
                ],
            )
        )

        dashboard.add_variable(
            name="datasource",
            query="prometheus",
            label="Data Source",
        )

        return dashboard

    @classmethod
    def create_pipeline_dashboard(cls, title: str = "DVAS Pipeline") -> "GrafanaDashboard":
        """Create a pipeline focused dashboard.

        Args:
            title: Dashboard title

        Returns:
            Dashboard focused on pipeline metrics
        """
        dashboard = cls(title=title, uid="dvas-pipeline")

        dashboard.add_panel(
            DashboardPanel(
                title="Pipeline Stages",
                grid_pos={"h": 8, "w": 24, "x": 0, "y": 0},
                targets=[
                    {
                        "expr": "rate(dvas_pipeline_stage_duration_seconds[5m])",
                        "legendFormat": "{{stage}}",
                    }
                ],
                unit="s",
            )
        )

        dashboard.add_panel(
            DashboardPanel(
                title="Parser Failures",
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 8},
                targets=[
                    {
                        "expr": "rate(dvas_parser_failures_total[5m])",
                        "legendFormat": "{{parser_type}}",
                    }
                ],
            )
        )

        dashboard.add_panel(
            DashboardPanel(
                title="Queue Depth",
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 8},
                targets=[
                    {
                        "expr": "dvas_task_queue_depth",
                        "legendFormat": "{{queue_name}}",
                    }
                ],
            )
        )

        dashboard.add_variable(
            name="datasource",
            query="prometheus",
            label="Data Source",
        )

        return dashboard


def export_dashboard_to_file(dashboard: GrafanaDashboard, path: str) -> None:
    """Export a dashboard to a JSON file.

    Args:
        dashboard: Dashboard to export
        path: File path to write
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(dashboard.to_json())
