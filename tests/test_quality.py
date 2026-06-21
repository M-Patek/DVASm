"""Tests for quality analyzer."""


class TestAnomalyDetector:
    """Test anomaly detection."""

    def test_empty_annotations(self):
        """Test with empty list."""
        from dvas.quality.analyzer import AnomalyDetector

        detector = AnomalyDetector()
        outliers = detector.detect_outliers([])
        assert outliers == []


class TestDataQualityAnalyzer:
    """Test quality analyzer."""

    def test_empty_dataset(self):
        """Test empty dataset metrics."""
        from dvas.quality.analyzer import DatasetQualityMetrics

        metrics = DatasetQualityMetrics(
            total_annotations=0,
            avg_segments_per_video=0,
            avg_caption_length=0,
            vocabulary_size=0,
            verb_diversity=0,
            noun_diversity=0,
            action_balance_score=0,
            temporal_coverage=0,
            qaq_pairs_per_segment=0,
            missing_fields_rate=0,
            outlier_annotations=[],
        )

        assert metrics.total_annotations == 0
