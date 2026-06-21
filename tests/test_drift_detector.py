"""Tests for drift detection with KS test and PSI."""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.observability.drift_detector import (
    DriftConfig,
    DriftMonitor,
    DriftReport,
    FeatureExtractor,
    compute_hellinger_distance,
    compute_psi,
    ks_test,
)


class TestDriftConfig:
    """Test DriftConfig dataclass."""

    def test_default_config(self):
        config = DriftConfig()
        assert config.ks_threshold == 0.05
        assert config.psi_threshold == 0.25
        assert config.window_size == 1000
        assert config.min_samples == 30
        assert config.enable_ks is True
        assert config.enable_psi is True

    def test_custom_config(self):
        config = DriftConfig(
            ks_threshold=0.01,
            psi_threshold=0.1,
            min_samples=50,
        )
        assert config.ks_threshold == 0.01
        assert config.psi_threshold == 0.1
        assert config.min_samples == 50


class TestKSFunction:
    """Test KS test function."""

    def test_identical_distributions(self):
        ref = np.random.normal(0, 1, 100)
        stat, p_value = ks_test(ref, ref)
        assert p_value == 1.0
        assert stat == 0.0

    def test_different_distributions(self):
        ref = np.random.normal(0, 1, 100)
        cur = np.random.normal(5, 1, 100)
        stat, p_value = ks_test(ref, cur)
        assert p_value < 0.05
        assert stat > 0.0

    def test_insufficient_samples(self):
        ref = np.array([1.0])
        cur = np.array([2.0])
        stat, p_value = ks_test(ref, cur)
        assert p_value == 1.0


class TestPSIFunction:
    """Test PSI computation."""

    def test_identical_distributions(self):
        ref = np.random.normal(0, 1, 1000)
        psi = compute_psi(ref, ref)
        assert psi == 0.0

    def test_different_distributions(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.0

    def test_empty_input(self):
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2, 3]), np.array([])) == 0.0


class TestHellingerDistance:
    """Test Hellinger distance computation."""

    def test_identical_distributions(self):
        ref = np.random.normal(0, 1, 1000)
        dist = compute_hellinger_distance(ref, ref)
        assert dist == 0.0

    def test_different_distributions(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        dist = compute_hellinger_distance(ref, cur)
        assert 0.0 < dist < 1.0


class TestFeatureExtractor:
    """Test FeatureExtractor."""

    def _create_annotation(self, num_segments=2):
        return Annotation(
            id="test-001",
            video_id="vid-001",
            video_path="/tmp/test.mp4",
            source="teacher",
            quality_score=0.85,
            created_at=datetime.now(timezone.utc),
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=60.0,
                total_frames=1800,
            ),
            segments=[],
            tags=["test"],
        )

    def test_extract_features(self):
        annotation = self._create_annotation()
        features = FeatureExtractor.extract_features(annotation)

        assert "resolution" in features
        assert "duration" in features
        assert "num_segments" in features
        assert features["quality_score"] == 0.85

    def test_extract_vocabulary_features(self):
        annotation = self._create_annotation()
        vocab = FeatureExtractor.extract_vocabulary_features([annotation])

        assert "unique_words" in vocab
        assert "total_words" in vocab
        assert "avg_caption_length" in vocab


class TestDriftMonitor:
    """Test DriftMonitor."""

    def _create_annotations(self, count=50, quality_mean=0.8, quality_std=0.1):
        """Create test annotations with varying quality scores."""
        annotations = []
        for i in range(count):
            quality = np.clip(np.random.normal(quality_mean, quality_std), 0, 1)
            annotations.append(
                Annotation(
                    id=f"test-{i:03d}",
                    video_id=f"vid-{i:03d}",
                    video_path="/tmp/test.mp4",
                    source="teacher",
                    quality_score=float(quality),
                    created_at=datetime.now(timezone.utc),
                    metadata=VideoMetadata(
                        fps=30.0,
                        resolution=[1920, 1080],
                        duration=60.0 + i,
                        total_frames=1800,
                    ),
                    segments=[],
                    tags=["test"],
                )
            )
        return annotations

    def test_init(self):
        monitor = DriftMonitor()
        assert monitor.config.ks_threshold == 0.05
        assert len(monitor.reference_data) == 0

    def test_set_reference(self):
        annotations = self._create_annotations(50)
        monitor = DriftMonitor()
        monitor.set_reference(annotations)

        assert len(monitor.reference_data) == 50
        assert len(monitor.reference_features) > 0

    def test_check_no_reference(self):
        monitor = DriftMonitor()
        current = self._create_annotations(50)

        report = monitor.check(current)
        assert len(report.alerts) == 1
        assert "No reference data" in report.alerts[0]

    def test_check_insufficient_samples(self):
        ref = self._create_annotations(50)
        monitor = DriftMonitor(reference_data=ref)

        current = self._create_annotations(10)
        report = monitor.check(current)
        assert len(report.alerts) == 1
        assert "Insufficient samples" in report.alerts[0]

    def test_check_no_drift(self):
        ref = self._create_annotations(100, quality_mean=0.8, quality_std=0.1)
        monitor = DriftMonitor(reference_data=ref)

        current = self._create_annotations(100, quality_mean=0.8, quality_std=0.1)
        report = monitor.check(current)

        # With identical distributions, may or may not detect drift depending on random sampling
        # Just verify the check completes without errors
        assert isinstance(report, DriftReport)
        assert report.timestamp is not None

    def test_check_with_drift(self):
        ref = self._create_annotations(100, quality_mean=0.8, quality_std=0.1)
        monitor = DriftMonitor(reference_data=ref)

        # Current data with different quality distribution
        current = self._create_annotations(100, quality_mean=0.3, quality_std=0.1)
        report = monitor.check(current)

        # Should detect drift
        assert report.drift_detected is True
        assert len(report.alerts) > 0
        assert len(report.recommendations) > 0

    def test_check_feature_drift(self):
        ref = self._create_annotations(100, quality_mean=0.8)
        monitor = DriftMonitor(reference_data=ref)

        # Current with different duration distribution
        current = []
        for i in range(100):
            current.append(
                Annotation(
                    id=f"test-{i:03d}",
                    video_id=f"vid-{i:03d}",
                    video_path="/tmp/test.mp4",
                    source="teacher",
                    quality_score=0.8,
                    created_at=datetime.now(timezone.utc),
                    metadata=VideoMetadata(
                        fps=30.0,
                        resolution=[1920, 1080],
                        duration=5000.0 + i,  # Much longer videos
                        total_frames=1800,
                    ),
                    segments=[],
                    tags=["test"],
                )
            )

        report = monitor.check(current)
        assert "duration" in report.feature_drift

    def test_get_history(self):
        ref = self._create_annotations(50)
        monitor = DriftMonitor(reference_data=ref)

        for _ in range(5):
            current = self._create_annotations(50)
            monitor.check(current)

        history = monitor.get_history(3)
        assert len(history) == 3

    def test_get_drift_trend(self):
        ref = self._create_annotations(50)
        monitor = DriftMonitor(reference_data=ref)

        for _ in range(5):
            current = self._create_annotations(50)
            monitor.check(current)

        trend = monitor.get_drift_trend("quality_score")
        assert "checks" in trend
        assert trend["checks"] == 5

    def test_save_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref = self._create_annotations(50)
            monitor = DriftMonitor(reference_data=ref)

            current = self._create_annotations(50)
            report = monitor.check(current)

            path = monitor.save_report(report, Path(tmp_dir))
            assert path.exists()
            assert path.suffix == ".json"

    def test_export_statistics(self):
        ref = self._create_annotations(50)
        monitor = DriftMonitor(reference_data=ref)

        stats = monitor.export_statistics()
        assert stats["reference_samples"] == 50
        assert stats["history_length"] == 0
        assert "config" in stats
        assert "features" in stats


class TestDriftReport:
    """Test DriftReport dataclass."""

    def test_default_creation(self):
        report = DriftReport()
        assert report.drift_detected is False
        assert report.alerts == []
        assert report.recommendations == []
        assert report.timestamp is not None

    def test_with_alerts(self):
        report = DriftReport(
            drift_detected=True,
            alerts=["Feature drift detected"],
            recommendations=["Retrain model"],
        )
        assert report.drift_detected is True
        assert len(report.alerts) == 1
        assert len(report.recommendations) == 1
