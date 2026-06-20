"""Observability module for DVAS.

Provides unified metrics collection, distributed tracing, structured logging,
OpenTelemetry integration, Prometheus export, Grafana dashboards, and
monitoring for teacher latency, cost, parser failures, annotation quality,
queue depth, export throughput, storage size, and student fallbacks.
"""

from dvas.observability.alerts import (
    AlertEvent,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    CallbackNotificationChannel,
    LogNotificationChannel,
    NotificationChannel,
)
from dvas.observability.annotation_quality import AnnotationQualityMonitor
from dvas.observability.collector import (
    MetricsCollector,
    get_metrics,
    reset_metrics,
)
from dvas.observability.export_throughput import ExportThroughputMonitor
from dvas.observability.grafana import (
    DashboardPanel,
    DashboardRow,
    GrafanaDashboard,
    export_dashboard_to_file,
)
from dvas.observability.logging import (
    BoundStructuredLogger,
    StructuredLogger,
    get_correlation_id,
    get_request_context,
    get_structured_logger,
    log_execution_time,
    logging_context,
    set_correlation_id,
    set_request_context,
)
from dvas.observability.otel import (
    OTelMetric,
    OTelResource,
    OTelTrace,
    OpenTelemetryCollector,
    OpenTelemetryExporter,
)
from dvas.observability.parser_failures import ParserFailureMonitor
from dvas.observability.prometheus import (
    PrometheusExporter,
    PrometheusHTTPHandler,
    create_prometheus_metrics,
)
from dvas.observability.queue_depth import TaskQueueMonitor
from dvas.observability.storage_size import StorageSizeMonitor
from dvas.observability.student_fallback import StudentFallbackMonitor
from dvas.observability.teacher_cost import CostBudget, TeacherCostMonitor
from dvas.observability.teacher_latency import (
    LatencyThreshold,
    TeacherLatencyMonitor,
    TeacherLatencyTracker,
)
from dvas.observability.tracing import Tracer, get_tracer, trace_span

__all__ = [
    # Alerts
    "AlertEvent",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "CallbackNotificationChannel",
    "LogNotificationChannel",
    "NotificationChannel",
    # Annotation Quality
    "AnnotationQualityMonitor",
    # Collector
    "MetricsCollector",
    "get_metrics",
    "reset_metrics",
    # Export Throughput
    "ExportThroughputMonitor",
    # Grafana
    "DashboardPanel",
    "DashboardRow",
    "GrafanaDashboard",
    "export_dashboard_to_file",
    # Logging
    "BoundStructuredLogger",
    "StructuredLogger",
    "get_correlation_id",
    "get_request_context",
    "get_structured_logger",
    "log_execution_time",
    "logging_context",
    "set_correlation_id",
    "set_request_context",
    # OpenTelemetry
    "OTelMetric",
    "OTelResource",
    "OTelTrace",
    "OpenTelemetryCollector",
    "OpenTelemetryExporter",
    # Parser Failures
    "ParserFailureMonitor",
    # Prometheus
    "PrometheusExporter",
    "PrometheusHTTPHandler",
    "create_prometheus_metrics",
    # Queue Depth
    "TaskQueueMonitor",
    # Storage Size
    "StorageSizeMonitor",
    # Student Fallback
    "StudentFallbackMonitor",
    # Teacher Cost
    "CostBudget",
    "TeacherCostMonitor",
    # Teacher Latency
    "LatencyThreshold",
    "TeacherLatencyMonitor",
    "TeacherLatencyTracker",
    # Tracing
    "Tracer",
    "get_tracer",
    "trace_span",
]
