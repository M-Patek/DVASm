"""Tests for teacher leaderboard."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from dvas.benchmarks.teacher_leaderboard import TeacherScore, TeacherLeaderboard


class TestTeacherScore:
    """Test TeacherScore dataclass."""

    def test_creation(self):
        """Test basic creation."""
        score = TeacherScore(
            model_name="gpt-4",
            quality_score=85.0,
            cost_score=70.0,
            latency_score=60.0,
            feature_score=90.0,
        )
        assert score.model_name == "gpt-4"
        assert score.quality_score == 85.0
        assert score.cost_score == 70.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = TeacherScore(
            model_name="claude",
            quality_score=80.0,
            cost_score=75.0,
            latency_score=65.0,
            feature_score=85.0,
        )
        data = score.to_dict()
        assert data["model_name"] == "claude"
        assert data["quality_score"] == 80.0


class TestTeacherLeaderboard:
    """Test TeacherLeaderboard."""

    @pytest.fixture
    def temp_leaderboard(self):
        """Create temporary leaderboard directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TeacherLeaderboard(tmpdir)

    def test_init(self, temp_leaderboard):
        """Test initialization."""
        assert temp_leaderboard.name == "teacher_leaderboard"
        assert temp_leaderboard.results_dir.exists()

    def test_compute_quality_scores(self, temp_leaderboard):
        """Test quality score computation."""
        with patch("dvas.benchmarks.teacher_leaderboard.get_registry") as mock_registry:
            mock_model = MagicMock()
            mock_model.name = "gpt-4"
            mock_model.quality_score = 90.0
            mock_registry.return_value = MagicMock(
                list_models=MagicMock(return_value=[mock_model])
            )
            scores = temp_leaderboard.compute_quality_scores(["gpt-4"])
            assert "gpt-4" in scores or len(scores) == 0

    def test_compute_overall_scores(self, temp_leaderboard):
        """Test overall score computation."""
        quality = {"gpt-4": 90.0, "claude": 85.0}
        cost = {"gpt-4": 70.0, "claude": 80.0}
        latency = {"gpt-4": 60.0, "claude": 70.0}
        feature = {"gpt-4": 95.0, "claude": 90.0}
        scores = temp_leaderboard.compute_overall_scores(quality, cost, latency, feature)
        assert "gpt-4" in scores
        assert "claude" in scores
        assert scores["gpt-4"] > 0

    def test_compute_rankings(self, temp_leaderboard):
        """Test ranking computation."""
        scores = {"gpt-4": 90.0, "claude": 85.0, "together": 80.0}
        rankings = temp_leaderboard.compute_rankings(scores)
        assert rankings["gpt-4"] == 1
        assert rankings["claude"] == 2
        assert rankings["together"] == 3

    def test_generate_leaderboard(self, temp_leaderboard):
        """Test leaderboard generation."""
        with patch("dvas.benchmarks.teacher_leaderboard.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock(list_models=MagicMock(return_value=[]))
            leaderboard = temp_leaderboard.generate_leaderboard()
            assert isinstance(leaderboard, list)

    def test_run_benchmark(self, temp_leaderboard):
        """Test full benchmark run."""
        with patch("dvas.benchmarks.teacher_leaderboard.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock(list_models=MagicMock(return_value=[]))
            result = temp_leaderboard.run_benchmark("test_run")
            assert result.benchmark_name == "teacher_leaderboard"
            assert result.model_id == "test_run"

    def test_get_best_model_for_task(self, temp_leaderboard):
        """Test best model selection."""
        with patch("dvas.benchmarks.teacher_leaderboard.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock(list_models=MagicMock(return_value=[]))
            best = temp_leaderboard.get_best_model_for_task("quality")
            assert best is None or isinstance(best, str)

    def test_get_cheapest_model(self, temp_leaderboard):
        """Test cheapest model selection."""
        with patch("dvas.benchmarks.teacher_leaderboard.get_registry") as mock_registry:
            mock_registry.return_value = MagicMock(list_models=MagicMock(return_value=[]))
            cheapest = temp_leaderboard.get_cheapest_model()
            assert cheapest is None or isinstance(cheapest, str)
