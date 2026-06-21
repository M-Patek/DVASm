"""Tests for student leaderboard."""

import tempfile

import pytest

from dvas.benchmarks.student_leaderboard import StudentScore, StudentLeaderboard


class TestStudentScore:
    """Test StudentScore dataclass."""

    def test_creation(self):
        """Test basic creation."""
        score = StudentScore(
            model_id="qwen2-vl-7b",
            model_size=7.0,
            quality_score=82.0,
            latency_ms=150.0,
            throughput=10.0,
            training_cost=500.0,
            memory_mb=8000.0,
        )
        assert score.model_id == "qwen2-vl-7b"
        assert score.model_size == 7.0
        assert score.quality_score == 82.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = StudentScore(
            model_id="llava-13b",
            model_size=13.0,
            quality_score=78.0,
            latency_ms=200.0,
            throughput=8.0,
            training_cost=1000.0,
            memory_mb=16000.0,
        )
        data = score.to_dict()
        assert data["model_id"] == "llava-13b"
        assert data["model_size"] == 13.0


class TestStudentLeaderboard:
    """Test StudentLeaderboard."""

    @pytest.fixture
    def temp_leaderboard(self):
        """Create temporary leaderboard directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StudentLeaderboard(tmpdir)

    def test_init(self, temp_leaderboard):
        """Test initialization."""
        assert temp_leaderboard.name == "student_leaderboard"
        assert temp_leaderboard.results_dir.exists()

    def test_register_student(self, temp_leaderboard):
        """Test student registration."""
        temp_leaderboard.register_student(
            "qwen2-vl-7b",
            model_size=7.0,
            quality_score=82.0,
            latency_ms=150.0,
            throughput=10.0,
            training_cost=500.0,
            memory_mb=8000.0,
        )
        assert "qwen2-vl-7b" in temp_leaderboard._student_data

    def test_compute_size_scores(self, temp_leaderboard):
        """Test size score computation."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        temp_leaderboard.register_student(
            "s2", model_size=13.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        scores = temp_leaderboard.compute_size_scores()
        assert "s1" in scores
        assert "s2" in scores
        assert scores["s1"] >= scores["s2"]  # smaller is better

    def test_compute_latency_scores(self, temp_leaderboard):
        """Test latency score computation."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        temp_leaderboard.register_student(
            "s2", model_size=7.0, quality_score=80.0, latency_ms=200.0, throughput=10.0
        )
        scores = temp_leaderboard.compute_latency_scores()
        assert "s1" in scores
        assert "s2" in scores
        assert scores["s1"] >= scores["s2"]  # faster is better

    def test_compute_throughput_scores(self, temp_leaderboard):
        """Test throughput score computation."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        temp_leaderboard.register_student(
            "s2", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=5.0
        )
        scores = temp_leaderboard.compute_throughput_scores()
        assert "s1" in scores
        assert "s2" in scores
        assert scores["s1"] >= scores["s2"]  # higher is better

    def test_compute_overall_scores(self, temp_leaderboard):
        """Test overall score computation."""
        temp_leaderboard.register_student(
            "s1",
            model_size=7.0,
            quality_score=80.0,
            latency_ms=100.0,
            throughput=10.0,
            memory_mb=8000.0,
        )
        temp_leaderboard.register_student(
            "s2",
            model_size=13.0,
            quality_score=70.0,
            latency_ms=200.0,
            throughput=5.0,
            memory_mb=16000.0,
        )
        quality = {"s1": 80.0, "s2": 70.0}
        size = {"s1": 90.0, "s2": 80.0}
        latency = {"s1": 90.0, "s2": 80.0}
        throughput = {"s1": 90.0, "s2": 80.0}
        memory = {"s1": 90.0, "s2": 80.0}
        scores = temp_leaderboard.compute_overall_scores(quality, size, latency, throughput, memory)
        assert "s1" in scores
        assert "s2" in scores

    def test_compute_rankings(self, temp_leaderboard):
        """Test ranking computation."""
        scores = {"s1": 90.0, "s2": 85.0, "s3": 80.0}
        rankings = temp_leaderboard.compute_rankings(scores)
        assert rankings["s1"] == 1
        assert rankings["s2"] == 2
        assert rankings["s3"] == 3

    def test_generate_leaderboard(self, temp_leaderboard):
        """Test leaderboard generation."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        temp_leaderboard.register_student(
            "s2", model_size=13.0, quality_score=70.0, latency_ms=200.0, throughput=5.0
        )
        leaderboard = temp_leaderboard.generate_leaderboard()
        assert isinstance(leaderboard, list)
        assert len(leaderboard) == 2

    def test_run_benchmark(self, temp_leaderboard):
        """Test full benchmark run."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=80.0, latency_ms=100.0, throughput=10.0
        )
        result = temp_leaderboard.run_benchmark("test_run")
        assert result.benchmark_name == "student_leaderboard"
        assert result.model_id == "test_run"

    def test_get_pareto_optimal_models(self, temp_leaderboard):
        """Test Pareto optimal model detection."""
        temp_leaderboard.register_student(
            "s1", model_size=7.0, quality_score=90.0, latency_ms=100.0, throughput=10.0
        )
        temp_leaderboard.register_student(
            "s2", model_size=13.0, quality_score=80.0, latency_ms=200.0, throughput=5.0
        )
        temp_leaderboard.register_student(
            "s3", model_size=3.0, quality_score=70.0, latency_ms=300.0, throughput=3.0
        )
        pareto = temp_leaderboard.get_pareto_optimal_models()
        assert isinstance(pareto, list)
