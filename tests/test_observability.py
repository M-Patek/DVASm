"""Tests for observability and metrics collection."""

import time

import pytest

from dvas.observability.metrics import MetricsCollector


class TestMetricsCollector:
    @pytest.fixture
    def metrics(self):
        return MetricsCollector()

    def test_increment_counter(self, metrics):
        metrics.increment("requests_total")
        assert metrics.get_counter("requests_total") == 1

        metrics.increment("requests_total", value=3)
        assert metrics.get_counter("requests_total") == 4

    def test_increment_with_labels(self, metrics):
        metrics.increment("requests", labels={"method": "GET", "status": "200"})
        metrics.increment("requests", labels={"method": "GET", "status": "200"})
        metrics.increment("requests", labels={"method": "POST", "status": "201"})

        assert metrics.get_counter("requests", labels={"method": "GET", "status": "200"}) == 2
        assert metrics.get_counter("requests", labels={"method": "POST", "status": "201"}) == 1

    def test_decrement_counter(self, metrics):
        metrics.increment("active_connections", value=5)
        metrics.decrement("active_connections", value=2)
        assert metrics.get_counter("active_connections") == 3

    def test_gauge(self, metrics):
        metrics.gauge("memory_usage", 1024.5)
        assert metrics.get_gauge("memory_usage") == 1024.5

        metrics.gauge("memory_usage", 512.0)
        assert metrics.get_gauge("memory_usage") == 512.0

    def test_gauge_with_labels(self, metrics):
        metrics.gauge("cpu_usage", 50.0, labels={"cpu": "0"})
        metrics.gauge("cpu_usage", 60.0, labels={"cpu": "1"})

        assert metrics.get_gauge("cpu_usage", labels={"cpu": "0"}) == 50.0
        assert metrics.get_gauge("cpu_usage", labels={"cpu": "1"}) == 60.0

    def test_observe_histogram(self, metrics):
        metrics.observe("request_duration", 0.5)
        metrics.observe("request_duration", 1.0)
        metrics.observe("request_duration", 0.3)

        hist = metrics.get_histogram("request_duration")
        assert hist["count"] == 3
        assert hist["sum"] == 1.8
        assert hist["avg"] == 0.6
        assert hist["min"] == 0.3
        assert hist["max"] == 1.0

    def test_histogram_with_labels(self, metrics):
        metrics.observe("latency", 0.1, labels={"endpoint": "/api"})
        metrics.observe("latency", 0.2, labels={"endpoint": "/api"})

        hist = metrics.get_histogram("latency", labels={"endpoint": "/api"})
        assert hist["count"] == 2

    def test_empty_histogram(self, metrics):
        hist = metrics.get_histogram("nonexistent")
        assert hist["count"] == 0
        assert hist["sum"] == 0.0
        assert hist["avg"] == 0.0

    def test_summary(self, metrics):
        metrics.summary("response_size", 1024.0)
        metrics.summary("response_size", 2048.0)

        key = metrics._make_key("response_size")
        values = metrics._summaries[key]["values"]
        assert len(values) == 2
        assert 1024.0 in values
        assert 2048.0 in values

    def test_make_key_with_labels(self, metrics):
        key = metrics._make_key("metric", {"a": "1", "b": "2"})
        assert key == 'metric{a="1\",b="2\"}'

    def test_make_key_without_labels(self, metrics):
        key = metrics._make_key("metric")
        assert key == "metric"

    def test_multiple_metrics_types(self, metrics):
        # Counter
        metrics.increment("total_requests")
        # Gauge
        metrics.gauge("active_users", 100.0)
        # Histogram
        metrics.observe("request_time", 0.5)
        # Summary
        metrics.summary("payload_size", 1024.0)

        assert metrics.get_counter("total_requests") == 1
        assert metrics.get_gauge("active_users") == 100.0
        assert metrics.get_histogram("request_time")["count"] == 1

    def test_label_ordering(self, metrics):
        # Labels should be sorted for consistent keys
        key1 = metrics._make_key("metric", {"z": "1", "a": "2"})
        key2 = metrics._make_key("metric", {"a": "2", "z": "1"})
        assert key1 == key2

    def test_get_nonexistent_counter(self, metrics):
        assert metrics.get_counter("nonexistent") == 0

    def test_get_nonexistent_gauge(self, metrics):
        assert metrics.get_gauge("nonexistent") == 0.0
