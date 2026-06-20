"""Tests for OpenTelemetry integration."""

import pytest

from dvas.observability.otel import (
    OTelMetric,
    OTelResource,
    OTelTrace,
    OpenTelemetryCollector,
    OpenTelemetryExporter,
)


class TestOTelResource:
    def test_default_creation(self):
        resource = OTelResource()
        assert resource.service_name == "dvas"
        assert resource.service_version == "0.1.0"

    def test_custom_attributes(self):
        resource = OTelResource(
            service_name="custom",
            deployment_environment="staging",
            custom_attr="value",
        )
        d = resource.to_dict()
        assert d["service.name"] == "custom"
        assert d["deployment.environment"] == "staging"
        assert d["custom_attr"] == "value"


class TestOTelMetric:
    def test_to_otlp(self):
        metric = OTelMetric(
            name="test_metric",
            value=42.0,
            timestamp=1234567890000000000,
            labels={"label1": "value1"},
        )
        otlp = metric.to_otlp()
        assert otlp["name"] == "test_metric"
        assert otlp["data_points"][0]["value"]["doubleValue"] == 42.0


class TestOTelTrace:
    def test_to_otlp(self):
        trace = OTelTrace(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time_ns=1234567890000000000,
            end_time_ns=1234567890000000001,
            attributes={"key": "value"},
        )
        otlp = trace.to_otlp()
        assert otlp["traceId"] == "trace-1"
        assert otlp["name"] == "test"
        assert otlp["status"]["code"] == "STATUS_CODE_OK"

    def test_error_status(self):
        trace = OTelTrace(
            trace_id="trace-1",
            span_id="span-1",
            name="test",
            start_time_ns=1234567890000000000,
            status="error",
        )
        otlp = trace.to_otlp()
        assert otlp["status"]["code"] == "STATUS_CODE_ERROR"


class TestOpenTelemetryExporter:
    @pytest.fixture
    def exporter(self):
        return OpenTelemetryExporter(OTelResource(service_name="test"))

    def test_export_metrics_empty(self, exporter):
        result = exporter.export_metrics()
        assert "resource_metrics" in result
        assert len(result["resource_metrics"]) == 1

    def test_export_traces_empty(self, exporter):
        result = exporter.export_traces()
        assert "resource_spans" in result
        assert len(result["resource_spans"]) == 1

    def test_export_logs(self, exporter):
        logs = [
            {"timestamp": 1234567890.0, "level": "INFO", "message": "test log"},
        ]
        result = exporter.export_logs(logs)
        assert "resource_logs" in result
        assert len(result["resource_logs"][0]["scope_logs"][0]["log_records"]) == 1

    def test_to_json(self, exporter):
        json_text = exporter.to_json()
        assert "metrics" in json_text
        assert "traces" in json_text

    def test_parse_labels(self, exporter):
        labels = exporter._parse_labels('metric{a="1",b="2"}')
        assert labels == {"a": "1", "b": "2"}

    def test_parse_labels_no_labels(self, exporter):
        labels = exporter._parse_labels("metric")
        assert labels == {}


class TestOpenTelemetryCollector:
    def test_creation(self):
        collector = OpenTelemetryCollector()
        assert collector.export_interval == 60.0

    def test_force_flush(self):
        collector = OpenTelemetryCollector()
        result = collector.force_flush()
        assert "metrics" in result
        assert "traces" in result
