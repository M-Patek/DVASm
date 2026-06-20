"""Tests for annotation consistency benchmark."""

import tempfile

import pytest

from dvas.benchmarks.annotation_consistency import ConsistencyResult, AnnotationConsistencyBenchmark


class TestConsistencyResult:
    """Test ConsistencyResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = ConsistencyResult(
            metric_name="fleiss_kappa",
            score=0.75,
            n_annotations=100,
        )
        assert result.metric_name == "fleiss_kappa"
        assert result.score == 0.75
        assert result.n_annotations == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ConsistencyResult(
            metric_name="krippendorff_alpha",
            score=0.82,
            n_annotations=50,
        )
        data = result.to_dict()
        assert data["metric_name"] == "krippendorff_alpha"
        assert data["score"] == 0.82


class TestAnnotationConsistencyBenchmark:
    """Test AnnotationConsistencyBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AnnotationConsistencyBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "annotation_consistency"
        assert temp_benchmark.results_dir.exists()

    def test_compute_fleiss_kappa(self, temp_benchmark):
        """Test Fleiss' kappa computation."""
        annotations = [
            ["A", "A", "A", "A"],  # High agreement
            ["B", "B", "B", "B"],  # High agreement
            ["A", "B", "A", "B"],  # Low agreement
        ]
        kappa = temp_benchmark.compute_fleiss_kappa(annotations)
        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0

    def test_compute_fleiss_kappa_single_annotator(self, temp_benchmark):
        """Test Fleiss' kappa with single annotator per item."""
        annotations = [["A"], ["B"], ["C"]]
        kappa = temp_benchmark.compute_fleiss_kappa(annotations)
        assert kappa == 0.0

    def test_compute_krippendorff_alpha(self, temp_benchmark):
        """Test Krippendorff's alpha computation."""
        annotations = [
            ["A", "A", "A"],
            ["A", "B", "A"],
            ["B", "B", "B"],
        ]
        alpha = temp_benchmark.compute_krippendorff_alpha(annotations)
        assert isinstance(alpha, float)

    def test_compute_pairwise_iou(self, temp_benchmark):
        """Test pairwise IoU computation."""
        ann1 = ["cut the tomato", "wash the hands"]
        ann2 = ["cut the tomato", "clean the hands"]
        iou = temp_benchmark.compute_pairwise_iou(ann1, ann2)
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_compute_stability_score(self, temp_benchmark):
        """Test stability score computation."""
        run1 = ["cut the tomato", "wash the hands"]
        run2 = ["cut the tomato", "clean the hands"]
        score = temp_benchmark.compute_stability_score(run1, run2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_compute_semantic_consistency(self, temp_benchmark):
        """Test semantic consistency computation."""
        annotations = [
            ["cut the tomato", "slice the tomato"],
            ["wash the hands", "clean the hands"],
        ]
        score = temp_benchmark.compute_semantic_consistency(annotations)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_compute_temporal_consistency(self, temp_benchmark):
        """Test temporal consistency computation."""
        annotations = [
            "cut the tomato",
            "slice the tomato",
            "chop the tomato",
        ]
        score = temp_benchmark.compute_temporal_consistency(annotations)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        annotations = [
            ["A", "A", "A"],
            ["B", "B", "B"],
            ["C", "C", "C"],
        ]
        result = temp_benchmark.run_benchmark("test_model", annotations)
        assert result.benchmark_name == "annotation_consistency"
        assert result.model_id == "test_model"

    def test_compare_annotators(self, temp_benchmark):
        """Test annotator comparison."""
        annotations = [
            ["A", "A"],
            ["B", "B"],
            ["A", "B"],
        ]
        comparison = temp_benchmark.compare_annotators(annotations)
        assert isinstance(comparison, dict)

    def test_empty_annotations(self, temp_benchmark):
        """Test with empty annotations."""
        kappa = temp_benchmark.compute_fleiss_kappa([])
        assert kappa == 0.0
