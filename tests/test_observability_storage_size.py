"""Tests for storage size monitoring."""

import os
import tempfile

import pytest

from dvas.observability.storage_size import StorageSizeMonitor


class TestStorageSizeMonitor:
    @pytest.fixture
    def monitor(self):
        return StorageSizeMonitor(
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

    def test_record_size(self, monitor):
        monitor.record_size("annotations", size_bytes=1024, file_count=5, path="/data/annotations")
        usage = monitor.get_storage_usage("annotations")
        assert usage["size_bytes"] == 1024
        assert usage["file_count"] == 5
        assert usage["status"] == "healthy"

    def test_scan_storage_file(self, monitor):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".txt")
        try:
            os.write(fd, b"hello world")
            os.close(fd)
            metrics = monitor.scan_storage("temp", __import__("pathlib").Path(path))
            assert metrics.size_bytes == len("hello world")
            assert metrics.file_count == 1
        finally:
            os.unlink(path)

    def test_scan_storage_directory(self, monitor):
        import tempfile
        import os
        import pathlib
        d = pathlib.Path(tempfile.mkdtemp())
        try:
            (d / "file1.txt").write_text("content1")
            (d / "file2.txt").write_text("content2")
            metrics = monitor.scan_storage("temp", d)
            assert metrics.size_bytes == len("content1") + len("content2")
            assert metrics.file_count == 2
        finally:
            import shutil
            shutil.rmtree(d)

    def test_scan_nonexistent_path(self, monitor):
        metrics = monitor.scan_storage("temp", __import__("pathlib").Path("/nonexistent/path"))
        assert metrics.size_bytes == 0
        assert metrics.file_count == 0

    def test_warning_status(self, monitor):
        # 80% of 100GB default capacity
        monitor.record_size("annotations", size_bytes=int(0.85 * 100 * 1024 * 1024 * 1024))
        usage = monitor.get_storage_usage("annotations")
        assert usage["status"] == "warning"

    def test_critical_status(self, monitor):
        # 96% of 100GB default capacity
        monitor.record_size("annotations", size_bytes=int(0.96 * 100 * 1024 * 1024 * 1024))
        usage = monitor.get_storage_usage("annotations")
        assert usage["status"] == "critical"

    def test_total_usage(self, monitor):
        monitor.record_size("annotations", size_bytes=1024, file_count=5)
        monitor.record_size("exports", size_bytes=2048, file_count=3)
        total = monitor.get_total_usage()
        assert total["total_bytes"] == 3072
        assert total["total_files"] == 8

    def test_largest_storage(self, monitor):
        monitor.record_size("annotations", size_bytes=2048)
        monitor.record_size("exports", size_bytes=1024)
        monitor.record_size("models", size_bytes=512)
        largest = monitor.get_largest_storage(n=2)
        assert len(largest) == 2
        assert largest[0]["storage_type"] == "annotations"

    def test_all_storage_usage(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        monitor.record_size("exports", size_bytes=2048)
        all_usage = monitor.get_all_storage_usage()
        assert "annotations" in all_usage
        assert "exports" in all_usage

    def test_is_healthy(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        assert monitor.is_healthy("annotations") is True

    def test_is_not_healthy(self, monitor):
        monitor.record_size("annotations", size_bytes=int(0.96 * 100 * 1024 * 1024 * 1024))
        assert monitor.is_healthy("annotations") is False

    def test_stats(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        stats = monitor.get_stats()
        assert "total" in stats
        assert "storage_types" in stats
        assert "largest" in stats

    def test_format_bytes(self, monitor):
        assert monitor._format_bytes(512) == "512.0 B"
        assert monitor._format_bytes(1024) == "1.0 KB"
        assert monitor._format_bytes(1024 * 1024) == "1.0 MB"
        assert monitor._format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_reset_storage(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        monitor.reset("annotations")
        usage = monitor.get_storage_usage("annotations")
        assert usage["status"] == "unknown"

    def test_reset_all(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        monitor.reset()
        assert monitor.get_all_storage_usage() == {}

    def test_alert_on_critical(self, monitor):
        alerts = []

        def handler(alert_type, details):
            alerts.append((alert_type, details))

        monitor.add_alert_handler(handler)
        monitor.record_size("annotations", size_bytes=int(0.96 * 100 * 1024 * 1024 * 1024))
        assert len(alerts) > 0
        assert alerts[0][0] == "storage_critical"

    def test_remove_alert_handler(self, monitor):
        def handler(a, d):
            pass

        monitor.add_alert_handler(handler)
        assert monitor.remove_alert_handler(handler) is True
        assert monitor.remove_alert_handler(handler) is False

    def test_growth_rate(self, monitor):
        monitor.record_size("annotations", size_bytes=1024)
        rate = monitor.get_growth_rate("annotations")
        assert rate >= 0
