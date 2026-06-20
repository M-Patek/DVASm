"""Tests for human agreement benchmark."""

import tempfile

import pytest

from dvas.benchmarks.human_agreement import HumanAgreementResult, HumanAgreementBenchmark


class TestHumanAgreementResult:
    """Test HumanAgreementResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = HumanAgreementResult(
            metric_name="cohens_kappa",
            score=0.75,
            n_samples=100,
        )
        assert result.metric_name == "cohens_kappa"
        assert result.score == 0.75
        assert result.n_samples == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = HumanAgreementResult(
            metric_name="percentage_agreement",
            score=0.82,
            n_samples=50,
        )
        data = result.to_dict()
        assert data["metric_name"] == "percentage_agreement"
        assert data["score"] == 0.82


class TestHumanAgreementBenchmark:
    """Test HumanAgreementBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield HumanAgreementBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "human_agreement"
        assert temp_benchmark.results_dir.exists()

    def test_compute_cohens_kappa(self, temp_benchmark):
        """Test Cohen's kappa computation."""
        human = ["A", "A", "B", "B", "A"]
        ai = ["A", "A", "B", "A", "A"]
        kappa = temp_benchmark.compute_cohens_kappa(human, ai)
        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0

    def test_compute_cohens_kappa_mismatched_length(self, temp_benchmark):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            temp_benchmark.compute_cohens_kappa(["A", "B"], ["A"])

    def test_compute_percentage_agreement(self, temp_benchmark):
        """Test percentage agreement computation."""
        human = ["A", "A", "B", "B", "A"]
        ai = ["A", "A", "B", "A", "A"]
        agreement = temp_benchmark.compute_percentage_agreement(human, ai)
        assert isinstance(agreement, float)
        assert 0.0 <= agreement <= 1.0

    def test_compute_pearson_correlation(self, temp_benchmark):
        """Test Pearson correlation computation."""
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        ai = [1.1, 2.1, 2.9, 4.2, 5.0]
        corr = temp_benchmark.compute_pearson_correlation(human, ai)
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_compute_spearman_correlation(self, temp_benchmark):
        """Test Spearman correlation computation."""
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        ai = [1.1, 2.1, 2.9, 4.2, 5.0]
        corr = temp_benchmark.compute_spearman_correlation(human, ai)
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_compute_semantic_similarity(self, temp_benchmark):
        """Test semantic similarity computation."""
        human = ["cut the tomato", "wash the hands"]
        ai = ["slice the tomato", "clean the hands"]
        sim = temp_benchmark.compute_semantic_similarity(human, ai)
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_compute_confusion_matrix(self, temp_benchmark):
        """Test confusion matrix computation."""
        human = ["A", "A", "B", "B", "C"]
        ai = ["A", "A", "B", "A", "C"]
        matrix = temp_benchmark.compute_confusion_matrix(human, ai)
        assert isinstance(matrix, dict)
        assert "A" in matrix

    def test_compute_per_class_agreement(self, temp_benchmark):
        """Test per-class agreement computation."""
        human = ["A", "A", "B", "B", "C"]
        ai = ["A", "A", "B", "A", "C"]
        agreement = temp_benchmark.compute_per_class_agreement(human, ai)
        assert isinstance(agreement, dict)
        assert "A" in agreement
        assert "B" in agreement

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        human = ["A", "A", "B", "B", "A"]
        ai = ["A", "A", "B", "A", "A"]
        result = temp_benchmark.run_benchmark("test_model", human, ai)
        assert result.benchmark_name == "human_agreement"
        assert result.model_id == "test_model"

    def test_analyze_agreement_by_difficulty(self, temp_benchmark):
        """Test difficulty-based agreement analysis."""
        human = ["A", "B", "C"]
        ai = ["A", "B", "A"]
        difficulty = ["easy", "medium", "hard"]
        analysis = temp_benchmark.analyze_agreement_by_difficulty(human, ai, difficulty)
        assert isinstance(analysis, dict)

    def test_empty_labels(self, temp_benchmark):
        """Test with empty labels."""
        kappa = temp_benchmark.compute_cohens_kappa([], [])
        assert kappa == 0.0
