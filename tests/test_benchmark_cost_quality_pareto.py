"""Tests for cost/quality Pareto benchmark."""

import tempfile

import pytest

from dvas.benchmarks.cost_quality_pareto import CostQualityPoint, CostQualityPareto


class TestCostQualityPoint:
    """Test CostQualityPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        point = CostQualityPoint(
            model_id="gpt-4",
            cost_per_sample=0.05,
            quality_score=85.0,
            latency_ms=500.0,
        )
        assert point.model_id == "gpt-4"
        assert point.cost_per_sample == 0.05
        assert point.quality_score == 85.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        point = CostQualityPoint(
            model_id="claude",
            cost_per_sample=0.03,
            quality_score=80.0,
        )
        data = point.to_dict()
        assert data["model_id"] == "claude"
        assert data["cost_per_sample"] == 0.03

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model_id": "together",
            "cost_per_sample": 0.01,
            "quality_score": 75.0,
            "latency_ms": 300.0,
            "metadata": {},
        }
        point = CostQualityPoint.from_dict(data)
        assert point.model_id == "together"
        assert point.cost_per_sample == 0.01


class TestCostQualityPareto:
    """Test CostQualityPareto."""

    @pytest.fixture
    def temp_pareto(self):
        """Create temporary Pareto directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield CostQualityPareto(tmpdir)

    def test_init(self, temp_pareto):
        """Test initialization."""
        assert temp_pareto.name == "cost_quality_pareto"
        assert temp_pareto.results_dir.exists()

    def test_add_point(self, temp_pareto):
        """Test adding a point."""
        point = CostQualityPoint("gpt-4", 0.05, 85.0)
        temp_pareto.add_point(point)
        assert len(temp_pareto._points) == 1

    def test_add_model(self, temp_pareto):
        """Test adding a model."""
        temp_pareto.add_model("gpt-4", 0.05, 85.0, latency_ms=500.0)
        assert len(temp_pareto._points) == 1
        assert temp_pareto._points[0].model_id == "gpt-4"

    def test_compute_pareto_frontier(self, temp_pareto):
        """Test Pareto frontier computation."""
        temp_pareto.add_model("expensive_good", 0.10, 95.0)
        temp_pareto.add_model("cheap_good", 0.01, 90.0)
        temp_pareto.add_model("cheap_bad", 0.005, 50.0)
        temp_pareto.add_model("expensive_bad", 0.08, 60.0)
        frontier = temp_pareto.compute_pareto_frontier()
        assert isinstance(frontier, list)
        assert len(frontier) > 0

    def test_compute_efficiency_ratio(self, temp_pareto):
        """Test efficiency ratio computation."""
        temp_pareto.add_model("m1", 0.05, 85.0)
        temp_pareto.add_model("m2", 0.10, 90.0)
        ratios = temp_pareto.compute_efficiency_ratio()
        assert isinstance(ratios, dict)
        assert "m1" in ratios
        assert "m2" in ratios

    def test_compute_auc(self, temp_pareto):
        """Test AUC computation."""
        temp_pareto.add_model("m1", 0.01, 50.0)
        temp_pareto.add_model("m2", 0.05, 80.0)
        temp_pareto.add_model("m3", 0.10, 95.0)
        auc = temp_pareto.compute_auc()
        assert isinstance(auc, float)
        assert auc >= 0.0

    def test_find_best_model_for_budget(self, temp_pareto):
        """Test finding best model within budget."""
        temp_pareto.add_model("m1", 0.01, 70.0)
        temp_pareto.add_model("m2", 0.05, 85.0)
        temp_pareto.add_model("m3", 0.10, 95.0)
        best = temp_pareto.find_best_model_for_budget(0.06)
        assert best is not None
        assert best in ["m1", "m2"]

    def test_find_cheapest_model_for_quality(self, temp_pareto):
        """Test finding cheapest model meeting quality threshold."""
        temp_pareto.add_model("m1", 0.01, 60.0)
        temp_pareto.add_model("m2", 0.05, 80.0)
        temp_pareto.add_model("m3", 0.10, 95.0)
        cheapest = temp_pareto.find_cheapest_model_for_quality(75.0)
        assert cheapest is not None
        assert cheapest == "m2"

    def test_compute_tradeoff_slope(self, temp_pareto):
        """Test tradeoff slope computation."""
        temp_pareto.add_model("m1", 0.01, 70.0)
        temp_pareto.add_model("m2", 0.05, 85.0)
        slope = temp_pareto.compute_tradeoff_slope("m1", "m2")
        assert isinstance(slope, float)

    def test_generate_chart_data(self, temp_pareto):
        """Test chart data generation."""
        temp_pareto.add_model("m1", 0.01, 70.0)
        data = temp_pareto.generate_chart_data()
        assert "all_points" in data
        assert "pareto_frontier" in data

    def test_run_benchmark(self, temp_pareto):
        """Test full benchmark run."""
        temp_pareto.add_model("m1", 0.01, 70.0)
        result = temp_pareto.run_benchmark("test_run")
        assert result.benchmark_name == "cost_quality_pareto"
        assert result.model_id == "test_run"

    def test_empty_points(self, temp_pareto):
        """Test with no points."""
        frontier = temp_pareto.compute_pareto_frontier()
        assert frontier == []
        auc = temp_pareto.compute_auc()
        assert auc == 0.0
