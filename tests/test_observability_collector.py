"""Tests for unified metrics collector."""

import pytest

from dvas.observability.collector import MetricsCollector, get_metrics, reset_metrics


class TestMetricsCollector:
    @pytest.fixture
    def metrics(self):
        m = MetricsCollector()
        yield m
        m.reset()

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

    def test_histogram_percentiles(self, metrics):
        for i in range(100):
            metrics.observe("latency", float(i))

        hist = metrics.get_histogram("latency")
        assert hist["count"] == 100
        assert hist["p50"] == 50.0
        assert hist["p95"] == 95.0
        assert hist["p99"] == 99.0

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

        stats = metrics.get_summary("response_size")
        assert stats["count"] == 2
        assert stats["sum"] == 3072.0
        assert stats["avg"] == 1536.0

    def test_empty_summary(self, metrics):
        stats = metrics.get_summary("nonexistent")
        assert stats["count"] == 0
        assert stats["avg"] == 0.0

    def test_make_key_with_labels(self, metrics):
        key = metrics._make_key("metric", {"a": "1", "b": "2"})
        assert key == 'metric{a="1",b="2"}'

    def test_make_key_without_labels(self, metrics):
        key = metrics._make_key("metric")
        assert key == "metric"

    def test_multiple_metrics_types(self, metrics):
        metrics.increment("total_requests")
        metrics.gauge("active_users", 100.0)
        metrics.observe("request_time", 0.5)
        metrics.summary("payload_size", 1024.0)

        assert metrics.get_counter("total_requests") == 1
        assert metrics.get_gauge("active_users") == 100.0
        assert metrics.get_histogram("request_time")["count"] == 1
        assert metrics.get_summary("payload_size")["count"] == 1

    def test_label_ordering(self, metrics):
        key1 = metrics._make_key("metric", {"z": "1", "a": "2"})
        key2 = metrics._make_key("metric", {"a": "2", "z": "1"})
        assert key1 == key2

    def test_get_nonexistent_counter(self, metrics):
        assert metrics.get_counter("nonexistent") == 0

    def test_get_nonexistent_gauge(self, metrics):
        assert metrics.get_gauge("nonexistent") == 0.0

    def test_prometheus_export(self, metrics):
        metrics.increment("requests_total", labels={"method": "GET"})
        metrics.gauge("active_users", 100.0)
        metrics.observe("duration", 0.5)

        text = metrics.to_prometheus()
        assert "counter" in text
        assert "gauge" in text
        assert "histogram" in text
        assert "requests_total" in text
        assert "active_users" in text

    def test_all_stats(self, metrics):
        metrics.increment("counter1")
        metrics.gauge("gauge1", 42.0)
        stats = metrics.get_all_stats()
        assert "counters" in stats
        assert "gauges" in stats
        assert stats["counters"]["counter1"] == 1
        assert stats["gauges"]["gauge1"] == 42.0

    def test_metric_names(self, metrics):
        metrics.increment("counter1")
        metrics.gauge("gauge1", 1.0)
        names = metrics.get_metric_names()
        assert "counter1" in names
        assert "gauge1" in names

    def test_reset(self, metrics):
        metrics.increment("counter1")
        metrics.gauge("gauge1", 1.0)
        metrics.reset()
        assert metrics.get_counter("counter1") == 0
        assert metrics.get_gauge("gauge1") == 0.0

    def test_callback_registration(self, metrics):
        def get_value():
            return 42.0

        metrics.register_callback("dynamic_gauge", get_value)
        assert "dynamic_gauge" in metrics._registered_callbacks
        metrics.unregister_callback("dynamic_gauge")
        assert "dynamic_gauge" not in metrics._registered_callbacks

    def test_histogram_capping(self, metrics):
        for i in range(15000):
            metrics.observe("capped", float(i))
        # Should cap at 10000
        hist = metrics.get_histogram("capped")
        assert hist["count"] == 10000

    def test_thread_safety(self, metrics):
        import threading

        def worker():
            for _ in range(100):
                metrics.increment("thread_counter")

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.get_counter("thread_counter") == 1000


class TestGlobalMetrics:
    def test_get_metrics_singleton(self):
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_metrics(self):
        get_metrics().increment("test_metric")
        reset_metrics()
        assert get_metrics().get_counter("test_metric") == 0
