"""Tests for confidence calibration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.student.calibration import (
    CalibrationMetrics,
    ConfidenceCalibrator,
    ConfidenceThresholdOptimizer,
    TemperatureScaler,
)


class TestTemperatureScaler:
    """Test TemperatureScaler."""

    def test_initialization(self):
        """Test default initialization."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0
        assert not scaler.is_fitted

    def test_scale_confidences(self):
        """Test confidence scaling."""
        scaler = TemperatureScaler(temperature=2.0)
        confidences = np.array([0.5, 0.7, 0.9])

        scaled = scaler._scale_confidences(confidences, 2.0)

        # Scaling with T > 1 should move toward 0.5
        assert len(scaled) == len(confidences)
        assert scaled[0] == 0.5  # 0.5 stays 0.5
        assert scaled[1] < confidences[1]  # Should decrease
        assert scaled[2] < confidences[2]  # Should decrease

    def test_fit(self):
        """Test fitting temperature parameter."""
        scaler = TemperatureScaler()

        # Create imbalanced data (overconfident)
        confidences = np.array([0.9, 0.8, 0.95, 0.85, 0.9])
        accuracies = np.array([0, 1, 0, 0, 1])  # Less accurate than confidence suggests

        scaler.fit(confidences, accuracies)

        assert scaler.is_fitted
        assert scaler.temperature != 1.0  # Should have learned something
        assert 0.5 <= scaler.temperature <= 2.0  # Within expected range

    def test_transform(self):
        """Test transforming confidences."""
        scaler = TemperatureScaler(temperature=1.5)
        scaler.is_fitted = True

        confidences = np.array([0.9, 0.7, 0.5])
        scaled = scaler.transform(confidences)

        assert len(scaled) == len(confidences)
        # With T > 1, should move toward 0.5
        assert scaled[0] < confidences[0]
        assert scaled[1] < confidences[1]
        assert scaled[2] == confidences[2]  # 0.5 stays 0.5

    def test_save_and_load(self):
        """Test saving and loading scaler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaler.json"

            # Save
            scaler = TemperatureScaler(temperature=1.3)
            scaler.is_fitted = True
            scaler.save(path)

            assert path.exists()

            # Load
            loaded = TemperatureScaler.load(path)
            assert loaded.temperature == 1.3
            assert loaded.is_fitted


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator."""

    def test_initialization(self):
        """Test default initialization."""
        calibrator = ConfidenceCalibrator()
        assert calibrator.method == "temperature"
        assert not calibrator.is_fitted

    def test_fit(self):
        """Test fitting calibrator."""
        calibrator = ConfidenceCalibrator()

        # Create predictions with varying confidence
        predictions = [
            GenerationResult(
                text="test 1",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,
            ),
            GenerationResult(
                text="test 2",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.8,
            ),
            GenerationResult(
                text="test 3",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.6,
            ),
        ]

        # Ground truth that doesn't exactly match confidence
        ground_truth = ["test 1", "different", "test 3"]

        metrics = calibrator.fit(predictions, ground_truth, similarity_threshold=0.5)

        assert calibrator.is_fitted
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.ece >= 0
        assert len(metrics.bin_accuracies) == len(metrics.bin_confidences)

    def test_calibrate(self):
        """Test calibrating a single prediction."""
        calibrator = ConfidenceCalibrator()

        # Setup fitted calibrator
        calibrator.is_fitted = True
        calibrator.scaler = TemperatureScaler(temperature=1.2)
        calibrator.scaler.is_fitted = True

        prediction = GenerationResult(
            text="test",
            model_type=ModelType.STUDENT_LOCAL,
            status=GenerationStatus.SUCCESS,
            confidence=0.9,
        )

        calibrated = calibrator.calibrate(prediction)

        assert calibrated.confidence != prediction.confidence
        assert calibrated.metadata["raw_confidence"] == prediction.confidence
        assert calibrated.metadata["calibrated"]

    def test_calibrate_not_fitted(self):
        """Test calibrate returns original when not fitted."""
        calibrator = ConfidenceCalibrator()

        prediction = GenerationResult(
            text="test",
            model_type=ModelType.STUDENT_LOCAL,
            status=GenerationStatus.SUCCESS,
            confidence=0.8,
        )

        calibrated = calibrator.calibrate(prediction)

        assert calibrated.confidence == prediction.confidence

    def test_compute_correctness(self):
        """Test correctness computation."""
        calibrator = ConfidenceCalibrator()

        # Exact match
        assert calibrator._compute_correctness("hello world", "hello world", 0.7) == 1

        # Similar (overlap)
        assert calibrator._compute_correctness(
            "hello world", "hello universe", 0.7
        ) == 0

        # High threshold
        assert calibrator._compute_correctness(
            "hello world", "hello world test", 0.9
        ) == 0

    def test_compute_metrics(self):
        """Test metric computation."""
        calibrator = ConfidenceCalibrator()

        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        accuracies = np.array([1, 1, 0, 0])

        metrics = calibrator._compute_metrics(confidences, accuracies, n_bins=2)

        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.ece >= 0
        assert metrics.mce >= 0
        assert metrics.nll >= 0
        assert metrics.brier >= 0
        assert len(metrics.bin_accuracies) == 2
        assert len(metrics.bin_counts) == 2


class TestCalibrationMetrics:
    """Test CalibrationMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = CalibrationMetrics(
            ece=0.05,
            mce=0.1,
            nll=0.5,
            brier=0.2,
            bin_accuracies=[0.8, 0.9],
            bin_confidences=[0.75, 0.95],
            bin_counts=[10, 20],
        )

        data = metrics.to_dict()
        assert data["ece"] == 0.05
        assert data["mce"] == 0.1
        assert data["bin_accuracies"] == [0.8, 0.9]
        assert len(data["bin_counts"]) == 2


class TestConfidenceThresholdOptimizer:
    """Test ConfidenceThresholdOptimizer."""

    def test_fit(self):
        """Test fitting thresholds."""
        optimizer = ConfidenceThresholdOptimizer()

        # Create sample data
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        accuracies = np.array([1, 1, 1, 0, 1, 0, 0])

        results = optimizer.fit(confidences, accuracies)

        assert "max_accuracy_threshold" in results
        assert "max_coverage_threshold" in results
        assert "balanced_threshold" in results
        assert 0 <= results["max_accuracy_threshold"] <= 1

    def test_get_threshold_for_target(self):
        """Test getting threshold for targets."""
        optimizer = ConfidenceThresholdOptimizer()

        # Setup with some data
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        accuracies = np.array([1, 1, 0, 1, 0])
        optimizer.fit(confidences, accuracies)

        # Get for target accuracy
        threshold = optimizer.get_threshold_for_target(target_accuracy=0.8)
        assert 0 <= threshold <= 1

        # Get for max fallback
        threshold = optimizer.get_threshold_for_target(max_fallback_rate=0.5)
        assert 0 <= threshold <= 1

    def test_get_stats_empty(self):
        """Test stats with no data."""
        optimizer = ConfidenceThresholdOptimizer()

        # No data yet
        assert optimizer.optimal_threshold == 0.5

        results = optimizer.fit(
            np.array([0.7, 0.8]),
            np.array([1, 0]),
        )

        assert "balanced_threshold" in results
