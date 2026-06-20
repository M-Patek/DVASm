"""Tests for Prometheus metrics export."""

import pytest

from dvas.observability.collector import MetricsCollector
from dvas.observability.prometheus import (
    PrometheusExporter,
    PrometheusHTTPHandler,
    create_prometheus_metrics,
)


class TestPrometheusExporter:
    @pytest.fixture
    def exporter(self):
        metrics = MetricsCollector()
        return PrometheusExporter(metrics=metrics, namespace="dvas")

    def test_creation(self, exporter):
        assert exporter.namespace == "dvas"
        assert isinstance(exporter.metrics, MetricsCollector)

    def test_export_empty(self, exporter):
        text = exporter.export()
        assert isinstance(text, str)

    def test_export_counters(self, exporter):
        exporter.metrics.increment("requests", labels={"method": "GET"})
        text = exporter.export()
        assert "dvas_requests" in text
        assert "counter" in text

    def test_export_gauges(self, exporter):
        exporter.metrics.gauge("active_users", 100.0)
        text = exporter.export()
        assert "dvas_active_users" in text
        assert "gauge" in text

    def test_export_histograms(self, exporter):
        exporter.metrics.observe("duration", 0.5)
        exporter.metrics.observe("duration", 1.0)
        text = exporter.export()
        assert "dvas_duration" in text
        assert "histogram" in text
        assert "_bucket" in text
        assert "_count" in text

    def test_export_summaries(self, exporter):
        exporter.metrics.summary("response_size", 1024.0)
        exporter.metrics.summary("response_size", 2048.0)
        text = exporter.export()
        assert "dvas_response_size" in text
        assert "summary" in text
        assert "quantile" in text

    def test_get_metric_value(self, exporter):
        exporter.metrics.increment("test_counter")
        assert exporter.get_metric_value("test_counter") == 1.0

    def test_get_metric_value_nonexistent(self, exporter):
        assert exporter.get_metric_value("nonexistent") == 0.0

    def test_custom_collector(self, exporter):
        def custom():
            return "# custom metric\ncustom 42"

        exporter.register_custom_collector(custom)
        text = exporter.export()
        assert "custom" in text
        exporter.unregister_custom_collector(custom)
        text2 = exporter.export()
        assert "custom" not in text2

    def test_full_name(self, exporter):
        assert exporter._full_name("requests") == "dvas_requests"


class TestPrometheusHTTPHandler:
    def test_handle(self):
        handler = PrometheusHTTPHandler()
        text = handler.handle()
        assert isinstance(text, str)

    def test_content_type(self):
        handler = PrometheusHTTPHandler()
        assert "text/plain" in handler.get_content_type()


class TestCreatePrometheusMetrics:
    def test_creation(self):
        exporter = create_prometheus_metrics(namespace="test")
        assert exporter.namespace == "test"

    def test_custom_buckets(self):
        buckets = [0.1, 0.5, 1.0]
        exporter = create_prometheus_metrics(buckets=buckets)
        assert exporter.buckets == buckets
