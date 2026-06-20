"""Tests for latency/quality Pareto benchmark."""

import tempfile

import pytest

from dvas.benchmarks.latency_quality_pareto import LatencyQualityPoint, LatencyQualityPareto


class TestLatencyQualityPoint:
    """Test LatencyQualityPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        point = LatencyQualityPoint(
            model_id="gpt-4",
            latency_ms=500.0,
            quality_score=85.0,
            cost_per_sample=0.05,
        )
        assert point.model_id == "gpt-4"
        assert point.latency_ms == 500.0
        assert point.quality_score == 85.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        point = LatencyQualityPoint(
            model_id="claude",
            latency_ms=300.0,
            quality_score=80.0,
        )
        data = point.to_dict()
        assert data["model_id"] == "claude"
        assert data["latency_ms"] == 300.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model_id": "together",
            "latency_ms": 200.0,
            "quality_score": 75.0,
            "cost_per_sample": 0.01,
            "metadata": {},
        }
        point = LatencyQualityPoint.from_dict(data)
        assert point.model_id == "together"
        assert point.latency_ms == 200.0


class TestLatencyQualityPareto:
    """Test LatencyQualityPareto."""

    @pytest.fixture
    def temp_pareto(self):
        """Create temporary Pareto directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LatencyQualityPareto(tmpdir)

    def test_init(self, temp_pareto):
        """Test initialization."""
        assert temp_pareto.name == "latency_quality_pareto"
        assert temp_pareto.results_dir.exists()

    def test_add_point(self, temp_pareto):
        """Test adding a point."""
        point = LatencyQualityPoint("gpt-4", 500.0, 85.0)
        temp_pareto.add_point(point)
        assert len(temp_pareto._points) == 1

    def test_add_model(self, temp_pareto):
        """Test adding a model."""
        temp_pareto.add_model("gpt-4", 500.0, 85.0, cost_per_sample=0.05)
        assert len(temp_pareto._points) == 1

    def test_compute_pareto_frontier(self, temp_pareto):
        """Test Pareto frontier computation."""
        temp_pareto.add_model("slow_good", 1000.0, 95.0)
        temp_pareto.add_model("fast_good", 100.0, 90.0)
        temp_pareto.add_model("fast_bad", 50.0, 50.0)
        temp_pareto.add_model("slow_bad", 800.0, 60.0)
        frontier = temp_pareto.compute_pareto_frontier()
        assert isinstance(frontier, list)
        assert len(frontier) > 0

    def test_compute_speed_quality_ratio(self, temp_pareto):
        """Test speed-quality ratio computation."""
        temp_pareto.add_model("m1", 500.0, 85.0)
        temp_pareto.add_model("m2", 1000.0, 90.0)
        ratios = temp_pareto.compute_speed_quality_ratio()
        assert isinstance(ratios, dict)
        assert "m1" in ratios
        assert "m2" in ratios

    def test_compute_auc(self, temp_pareto):
        """Test AUC computation."""
        temp_pareto.add_model("m1", 100.0, 50.0)
        temp_pareto.add_model("m2", 500.0, 80.0)
        temp_pareto.add_model("m3", 1000.0, 95.0)
        auc = temp_pareto.compute_auc()
        assert isinstance(auc, float)
        assert auc >= 0.0

    def test_find_best_model_for_latency_budget(self, temp_pareto):
        """Test finding best model within latency budget."""
        temp_pareto.add_model("m1", 100.0, 70.0)
        temp_pareto.add_model("m2", 500.0, 85.0)
        temp_pareto.add_model("m3", 1000.0, 95.0)
        best = temp_pareto.find_best_model_for_latency_budget(600.0)
        assert best is not None
        assert best in ["m1", "m2"]

    def test_find_fastest_model_for_quality(self, temp_pareto):
        """Test finding fastest model meeting quality threshold."""
        temp_pareto.add_model("m1", 100.0, 60.0)
        temp_pareto.add_model("m2", 500.0, 80.0)
        temp_pareto.add_model("m3", 1000.0, 95.0)
        fastest = temp_pareto.find_fastest_model_for_quality(75.0)
        assert fastest is not None
        assert fastest == "m2"

    def test_compute_tradeoff_slope(self, temp_pareto):
        """Test tradeoff slope computation."""
        temp_pareto.add_model("m1", 100.0, 70.0)
        temp_pareto.add_model("m2", 500.0, 85.0)
        slope = temp_pareto.compute_tradeoff_slope("m1", "m2")
        assert isinstance(slope, (float, type(None)))

    def test_compute_speedup_vs_quality_loss(self, temp_pareto):
        """Test speedup vs quality loss computation."""
        temp_pareto.add_model("m1", 100.0, 90.0)
        temp_pareto.add_model("m2", 500.0, 85.0)
        result = temp_pareto.compute_speedup_vs_quality_loss("m1")
        assert isinstance(result, dict)

    def test_generate_chart_data(self, temp_pareto):
        """Test chart data generation."""
        temp_pareto.add_model("m1", 100.0, 70.0)
        data = temp_pareto.generate_chart_data()
        assert "all_points" in data
        assert "pareto_frontier" in data

    def test_run_benchmark(self, temp_pareto):
        """Test full benchmark run."""
        temp_pareto.add_model("m1", 100.0, 70.0)
        result = temp_pareto.run_benchmark("test_run")
        assert result.benchmark_name == "latency_quality_pareto"
        assert result.model_id == "test_run"

    def test_empty_points(self, temp_pareto):
        """Test with no points."""
        frontier = temp_pareto.compute_pareto_frontier()
        assert frontier == []
        auc = temp_pareto.compute_auc()
        assert auc == 0.0
