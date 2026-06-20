"""OpenTelemetry integration for DVAS.

Provides OpenTelemetry-compatible metrics, traces, and logs export.
Implements the OTel data model for interoperability with OTel collectors.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.observability.collector import get_metrics
from dvas.observability.tracing import get_tracer
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class OTelResource:
    """OpenTelemetry resource attributes."""

    def __init__(
        self,
        service_name: str = "dvas",
        service_version: str = "0.1.0",
        deployment_environment: str = "production",
        **attributes: str,
    ) -> None:
        self.service_name = service_name
        self.service_version = service_version
        self.deployment_environment = deployment_environment
        self.attributes = attributes

    def to_dict(self) -> Dict[str, str]:
        """Convert to OTel resource dictionary."""
        result = {
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.deployment_environment,
        }
        result.update(self.attributes)
        return result


@dataclass
class OTelMetric:
    """OpenTelemetry metric data point."""

    name: str
    value: float
    timestamp: int  # nanoseconds
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"

    def to_otlp(self) -> Dict[str, Any]:
        """Convert to OTLP format."""
        return {
            "name": self.name,
            "unit": "1",
            "data_points": [
                {
                    "attributes": [
                        {"key": k, "value": {"stringValue": v}}
                        for k, v in self.labels.items()
                    ],
                    "time_unix_nano": self.timestamp,
                    "value": {"doubleValue": self.value},
                }
            ],
        }


@dataclass
class OTelTrace:
    """OpenTelemetry trace span in OTLP format."""

    trace_id: str
    span_id: str
    name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    status: str = "ok"

    def to_otlp(self) -> Dict[str, Any]:
        """Convert to OTLP format."""
        result: Dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": self.start_time_ns,
            "attributes": [
                {"key": k, "value": {"stringValue": v}}
                for k, v in self.attributes.items()
            ],
            "status": {"code": "STATUS_CODE_OK" if self.status == "ok" else "STATUS_CODE_ERROR"},
        }
        if self.end_time_ns:
            result["endTimeUnixNano"] = self.end_time_ns
        if self.parent_span_id:
            result["parentSpanId"] = self.parent_span_id
        return result


class OpenTelemetryExporter:
    """OpenTelemetry metrics/traces/logs exporter.

    Converts internal DVAS observability data to OpenTelemetry format
    for export to OTel collectors.

    Usage::

        exporter = OpenTelemetryExporter(OTelResource(service_name="dvas"))
        metrics = exporter.export_metrics()
        traces = exporter.export_traces()
    """

    def __init__(self, resource: Optional[OTelResource] = None) -> None:
        self.resource = resource or OTelResource()
        self._lock = threading.Lock()
        self._batch: List[Dict[str, Any]] = []

    def export_metrics(self) -> Dict[str, Any]:
        """Export current metrics in OTel format.

        Returns:
            Dict with resource and metrics in OTLP format
        """
        metrics = get_metrics()
        otel_metrics: List[Dict[str, Any]] = []

        # Export counters
        stats = metrics.get_all_stats()
        now_ns = int(time.time() * 1e9)

        for key, value in stats.get("counters", {}).items():
            name = key.split("{")[0]
            labels = self._parse_labels(key)
            otel_metrics.append(
                {
                    "name": name,
                    "unit": "1",
                    "sum": {
                        "data_points": [
                            {
                                "attributes": [
                                    {"key": k, "value": {"stringValue": v}}
                                    for k, v in labels.items()
                                ],
                                "time_unix_nano": now_ns,
                                "value": {"intValue": int(value)},
                            }
                        ],
                        "aggregation_temporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
                        "is_monotonic": True,
                    },
                }
            )

        # Export gauges
        for key, value in stats.get("gauges", {}).items():
            name = key.split("{")[0]
            labels = self._parse_labels(key)
            otel_metrics.append(
                {
                    "name": name,
                    "unit": "1",
                    "gauge": {
                        "data_points": [
                            {
                                "attributes": [
                                    {"key": k, "value": {"stringValue": v}}
                                    for k, v in labels.items()
                                ],
                                "time_unix_nano": now_ns,
                                "value": {"doubleValue": float(value)},
                            }
                        ]
                    },
                }
            )

        return {
            "resource_metrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": k, "value": {"stringValue": v}}
                            for k, v in self.resource.to_dict().items()
                        ]
                    },
                    "scope_metrics": [
                        {
                            "scope": {"name": "dvas.observability", "version": "0.1.0"},
                            "metrics": otel_metrics,
                        }
                    ],
                }
            ]
        }

    def export_traces(self) -> Dict[str, Any]:
        """Export current traces in OTel format.

        Returns:
            Dict with resource and spans in OTLP format
        """
        tracer = get_tracer()
        spans = tracer.get_spans()
        otel_spans = []

        for span in spans:
            otel_span = OTelTrace(
                trace_id=span.trace_id,
                span_id=span.span_id,
                name=span.name,
                start_time_ns=int(span.start_time * 1e9),
                end_time_ns=int(span.end_time * 1e9) if span.end_time else None,
                parent_span_id=span.parent_id,
                attributes=span.tags,
                status=span.status,
            )
            otel_spans.append(otel_span.to_otlp())

        return {
            "resource_spans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": k, "value": {"stringValue": v}}
                            for k, v in self.resource.to_dict().items()
                        ]
                    },
                    "scope_spans": [
                        {
                            "scope": {"name": "dvas.observability", "version": "0.1.0"},
                            "spans": otel_spans,
                        }
                    ],
                }
            ]
        }

    def export_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export logs in OTel format.

        Args:
            logs: List of log dictionaries

        Returns:
            Dict with resource and logs in OTLP format
        """
        otel_logs = []
        for log in logs:
            otel_logs.append(
                {
                    "time_unix_nano": int(log.get("timestamp", time.time()) * 1e9),
                    "severity_text": log.get("level", "INFO"),
                    "body": {"stringValue": log.get("message", "")},
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in log.items()
                        if k not in ("timestamp", "level", "message")
                    ],
                }
            )

        return {
            "resource_logs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": k, "value": {"stringValue": v}}
                            for k, v in self.resource.to_dict().items()
                        ]
                    },
                    "scope_logs": [
                        {
                            "scope": {"name": "dvas.observability", "version": "0.1.0"},
                            "log_records": otel_logs,
                        }
                    ],
                }
            ]
        }

    def _parse_labels(self, key: str) -> Dict[str, str]:
        """Parse labels from metric key."""
        labels: Dict[str, str] = {}
        if "{" in key and "}" in key:
            label_part = key[key.index("{") + 1 : key.index("}")]
            for pair in label_part.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
        return labels

    def to_json(self) -> str:
        """Export all observability data as JSON."""
        import json

        return json.dumps(
            {
                "metrics": self.export_metrics(),
                "traces": self.export_traces(),
            },
            indent=2,
            default=str,
        )


class OpenTelemetryCollector:
    """Collector that buffers and exports OTel data periodically.

    Usage::

        collector = OpenTelemetryCollector(OTelResource())
        collector.start()
        # ... application runs ...
        collector.stop()
    """

    def __init__(
        self,
        resource: Optional[OTelResource] = None,
        export_interval_seconds: float = 60.0,
    ) -> None:
        self.exporter = OpenTelemetryExporter(resource)
        self.export_interval = export_interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start periodic export."""
        with self._lock:
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._export_loop, daemon=True)
                self._thread.start()
                logger.info("otel_collector_started", interval=self.export_interval)

    def stop(self) -> None:
        """Stop periodic export."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("otel_collector_stopped")

    def _export_loop(self) -> None:
        """Background export loop."""
        while True:
            with self._lock:
                if not self._running:
                    break
            time.sleep(self.export_interval)
            try:
                self._do_export()
            except Exception as e:
                logger.error("otel_export_failed", error=str(e))

    def _do_export(self) -> None:
        """Perform actual export. Override for custom export logic."""
        metrics = self.exporter.export_metrics()
        traces = self.exporter.export_traces()
        logger.debug(
            "otel_exported",
            metric_count=len(metrics.get("resource_metrics", [])),
            trace_count=len(traces.get("resource_spans", [])),
        )

    def force_flush(self) -> Dict[str, Any]:
        """Force immediate export and return data."""
        return {
            "metrics": self.exporter.export_metrics(),
            "traces": self.exporter.export_traces(),
        }
