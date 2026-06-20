"""Tests for base benchmark classes."""

import tempfile
from datetime import datetime

import pytest

from dvas.benchmarks.base import BenchmarkResult, BenchmarkSuite, BaseBenchmark


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = BenchmarkResult(
            benchmark_name="epic_kitchens",
            model_id="gpt-4",
            timestamp=datetime.utcnow(),
            metrics={"bleu": 0.80, "rouge": 0.75},
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
        )
        assert result.benchmark_name == "epic_kitchens"
        assert result.model_id == "gpt-4"
        assert result.metrics["bleu"] == 0.80

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.utcnow()
        result = BenchmarkResult(
            benchmark_name="ego4d",
            model_id="claude",
            timestamp=now,
            metrics={"accuracy": 0.75},
            predictions=[],
            references=[],
        )
        data = result.to_dict()
        assert data["benchmark_name"] == "ego4d"
        assert data["timestamp"] == now.isoformat()

    def test_from_dict(self):
        """Test creation from dictionary."""
        now = datetime.utcnow()
        data = {
            "benchmark_name": "test",
            "model_id": "model1",
            "timestamp": now.isoformat(),
            "metrics": {"bleu": 0.80},
            "predictions": ["p1"],
            "references": ["r1"],
            "metadata": {"key": "value"},
        }
        result = BenchmarkResult.from_dict(data)
        assert result.benchmark_name == "test"
        assert result.metadata["key"] == "value"

    def test_get_metric(self):
        """Test metric retrieval."""
        result = BenchmarkResult(
            benchmark_name="test",
            model_id="m1",
            timestamp=datetime.utcnow(),
            metrics={"bleu": 0.80},
            predictions=[],
            references=[],
        )
        assert result.get_metric("bleu") == 0.80
        assert result.get_metric("nonexistent", 0.0) == 0.0


class TestBenchmarkSuite:
    """Test BenchmarkSuite dataclass."""

    def test_creation(self):
        """Test basic creation."""
        suite = BenchmarkSuite(model_id="gpt-4")
        assert suite.model_id == "gpt-4"
        assert suite.results == []

    def test_add_result(self):
        """Test adding results."""
        suite = BenchmarkSuite(model_id="gpt-4")
        result = BenchmarkResult(
            benchmark_name="epic",
            model_id="gpt-4",
            timestamp=datetime.utcnow(),
            metrics={"bleu": 0.80},
            predictions=[],
            references=[],
        )
        suite.add_result(result)
        assert len(suite.results) == 1
        assert "bleu" in suite.aggregated_metrics

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suite = BenchmarkSuite(model_id="gpt-4")
        data = suite.to_dict()
        assert data["model_id"] == "gpt-4"
        assert data["results"] == []


class TestBaseBenchmark:
    """Test BaseBenchmark class."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield BaseBenchmark(tmpdir, "test_benchmark")

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "test_benchmark"
        assert temp_benchmark.results_dir.exists()

    def test_save_and_load_results(self, temp_benchmark):
        """Test saving and loading results."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_id="m1",
            timestamp=datetime.utcnow(),
            metrics={"bleu": 0.80},
            predictions=["p1"],
            references=["r1"],
        )
        path = temp_benchmark._save_result(result)
        assert path.exists()

        loaded = temp_benchmark._load_results("test_benchmark")
        assert len(loaded) == 1
        assert loaded[0].model_id == "m1"

    def test_compute_bleu(self, temp_benchmark):
        """Test BLEU computation."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        score = temp_benchmark.compute_bleu(predictions, references)
        assert isinstance(score, float)

    def test_compute_bleu_empty(self, temp_benchmark):
        """Test BLEU with empty inputs."""
        score = temp_benchmark.compute_bleu([], [])
        assert score == 0.0

    def test_compute_rouge_l(self, temp_benchmark):
        """Test ROUGE-L computation."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        score = temp_benchmark.compute_rouge_l(predictions, references)
        assert isinstance(score, float)

    def test_compute_rouge_l_empty(self, temp_benchmark):
        """Test ROUGE-L with empty inputs."""
        score = temp_benchmark.compute_rouge_l([], [])
        assert score == 0.0

    def test_compute_accuracy(self, temp_benchmark):
        """Test exact match accuracy."""
        predictions = ["A", "B", "C", "D"]
        references = ["A", "B", "C", "X"]
        accuracy = temp_benchmark.compute_accuracy(predictions, references)
        assert accuracy == 0.75

    def test_compute_accuracy_empty(self, temp_benchmark):
        """Test accuracy with empty inputs."""
        accuracy = temp_benchmark.compute_accuracy([], [])
        assert accuracy == 0.0
