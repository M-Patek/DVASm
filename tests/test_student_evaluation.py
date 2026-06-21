"""Tests for teacher vs student evaluation."""

import pytest

from dvas.models.base import GenerationResult, GenerationStatus, ModelType
from dvas.models.evaluator.metrics import _ROUGE_AVAILABLE
from dvas.models.student.evaluation import (
    ComparisonReport,
    CostComparisonResult,
    LatencyComparisonResult,
    QualityComparisonResult,
    TeacherStudentEvaluator,
    evaluate_with_metrics,
)

# Skip tests requiring ROUGE if not available
rouge_required = pytest.mark.skipif(
    not _ROUGE_AVAILABLE, reason="ROUGE not available (install rouge-score)"
)


class TestQualityComparisonResult:
    """Test QualityComparisonResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QualityComparisonResult(
            metric_name="BLEU",
            teacher_score=0.8,
            student_score=0.75,
            relative_improvement=-0.0625,
            absolute_gap=-0.05,
        )

        data = result.to_dict()
        assert data["metric_name"] == "BLEU"
        assert data["teacher_score"] == 0.8
        assert data["student_score"] == 0.75
        assert data["relative_improvement"] == -0.0625


class TestCostComparisonResult:
    """Test CostComparisonResult dataclass."""

    def test_savings_calculation(self):
        """Test cost savings calculations."""
        result = CostComparisonResult(
            teacher_cost_per_sample=0.05,
            student_cost_per_sample=0.001,
            cost_savings_per_sample=0.049,
            cost_savings_percent=98.0,
            estimated_savings_1k_samples=49.0,
            estimated_savings_10k_samples=490.0,
        )

        data = result.to_dict()
        assert data["cost_savings_percent"] == 98.0
        assert data["estimated_savings_10k_samples"] == 490.0


class TestLatencyComparisonResult:
    """Test LatencyComparisonResult dataclass."""

    def test_speedup_calculation(self):
        """Test speedup calculations."""
        result = LatencyComparisonResult(
            teacher_latency_ms=5000,
            student_latency_ms=500,
            teacher_p50_ms=4500,
            teacher_p95_ms=8000,
            teacher_p99_ms=10000,
            student_p50_ms=450,
            student_p95_ms=800,
            student_p99_ms=1000,
            speedup_ratio=10.0,
            throughput_improvement=10.0,
        )

        data = result.to_dict()
        assert data["speedup_ratio"] == 10.0
        assert data["throughput_improvement"] == 10.0


class TestComparisonReport:
    """Test ComparisonReport dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        quality = QualityComparisonResult(
            metric_name="BLEU",
            teacher_score=0.8,
            student_score=0.75,
            relative_improvement=-0.0625,
            absolute_gap=-0.05,
        )
        cost = CostComparisonResult(
            teacher_cost_per_sample=0.05,
            student_cost_per_sample=0.001,
            cost_savings_per_sample=0.049,
            cost_savings_percent=98.0,
            estimated_savings_1k_samples=49.0,
            estimated_savings_10k_samples=490.0,
        )
        latency = LatencyComparisonResult(
            teacher_latency_ms=5000,
            student_latency_ms=500,
            teacher_p50_ms=4500,
            teacher_p95_ms=8000,
            teacher_p99_ms=10000,
            student_p50_ms=450,
            student_p95_ms=800,
            student_p99_ms=1000,
            speedup_ratio=10.0,
            throughput_improvement=10.0,
        )

        report = ComparisonReport(
            n_samples=100,
            quality_metrics=[quality],
            cost_comparison=cost,
            latency_comparison=latency,
            exact_match_rate=0.1,
            semantic_match_rate=0.5,
        )

        data = report.to_dict()
        assert data["n_samples"] == 100
        assert len(data["quality_metrics"]) == 1
        assert data["cost_comparison"] is not None
        assert data["latency_comparison"] is not None


class TestTeacherStudentEvaluator:
    """Test TeacherStudentEvaluator."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        teacher_preds = [
            GenerationResult(
                text="The person is cutting vegetables on a board.",
                model_type=ModelType.TEACHER_CLAUDE,
                status=GenerationStatus.SUCCESS,
                confidence=0.95,
                latency_ms=5000,
                cost_usd=0.05,
            ),
            GenerationResult(
                text="Washing hands at the sink.",
                model_type=ModelType.TEACHER_CLAUDE,
                status=GenerationStatus.SUCCESS,
                confidence=0.9,
                latency_ms=4500,
                cost_usd=0.05,
            ),
        ]

        student_preds = [
            GenerationResult(
                text="The person is cutting vegetables.",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.8,
                latency_ms=500,
                cost_usd=0.0,
            ),
            GenerationResult(
                text="Washing hands.",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
                confidence=0.75,
                latency_ms=450,
                cost_usd=0.0,
            ),
        ]

        return teacher_preds, student_preds

    @rouge_required
    def test_evaluate_quality(self, sample_predictions):
        """Test quality evaluation."""
        evaluator = TeacherStudentEvaluator()
        teacher_preds, student_preds = sample_predictions

        ground_truth = [
            "The person is cutting vegetables on a cutting board.",
            "Washing hands at the kitchen sink.",
        ]

        results = evaluator.evaluate_quality(teacher_preds, student_preds, ground_truth)

        assert len(results) > 0
        # Should have BLEU and ROUGE metrics
        metric_names = [r.metric_name for r in results]
        assert "BLEU" in metric_names
        assert "ROUGE-L" in metric_names

    def test_evaluate_quality_without_ground_truth(self, sample_predictions):
        """Test quality evaluation without ground truth."""
        evaluator = TeacherStudentEvaluator()
        teacher_preds, student_preds = sample_predictions

        results = evaluator.evaluate_quality(teacher_preds, student_preds)

        # Should still compute agreement metrics
        assert len(results) > 0
        metric_names = [r.metric_name for r in results]
        assert "Agreement Rate" in metric_names

    def test_evaluate_cost(self, sample_predictions):
        """Test cost evaluation."""
        evaluator = TeacherStudentEvaluator()
        teacher_preds, student_preds = sample_predictions

        result = evaluator.evaluate_cost(teacher_preds, student_preds)

        assert isinstance(result, CostComparisonResult)
        assert result.teacher_cost_per_sample == 0.05
        assert result.student_cost_per_sample == 0.0
        assert result.cost_savings_per_sample == 0.05

    def test_evaluate_latency(self, sample_predictions):
        """Test latency evaluation."""
        evaluator = TeacherStudentEvaluator()
        teacher_preds, student_preds = sample_predictions

        result = evaluator.evaluate_latency(teacher_preds, student_preds)

        assert isinstance(result, LatencyComparisonResult)
        assert result.teacher_latency_ms == 4750  # Average
        assert result.student_latency_ms == 475
        assert pytest.approx(result.speedup_ratio, rel=0.01) == 10.0

    def test_compare_on_predictions(self, sample_predictions):
        """Test full comparison."""
        evaluator = TeacherStudentEvaluator()
        teacher_preds, student_preds = sample_predictions

        report = evaluator.compare_on_predictions(teacher_preds, student_preds)

        assert isinstance(report, ComparisonReport)
        assert report.n_samples == 2
        assert len(report.quality_metrics) > 0
        assert report.cost_comparison is not None
        assert report.latency_comparison is not None

    def test_compute_similarity(self):
        """Test similarity computation."""
        evaluator = TeacherStudentEvaluator()

        # Exact match
        assert evaluator._compute_similarity("hello world", "hello world") == 1.0

        # Partial overlap
        sim = evaluator._compute_similarity("hello world", "hello universe")
        assert 0 < sim < 1

        # No overlap
        assert evaluator._compute_similarity("abc", "xyz") == 0.0

        # Empty
        assert evaluator._compute_similarity("", "hello") == 0.0


class TestEvaluateWithMetrics:
    """Test evaluate_with_metrics function."""

    @rouge_required
    def test_basic_evaluation(self):
        """Test basic metric evaluation."""
        predictions = [
            GenerationResult(
                text="The person is cutting vegetables.",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
            ),
            GenerationResult(
                text="Washing hands at the sink.",
                model_type=ModelType.STUDENT_LOCAL,
                status=GenerationStatus.SUCCESS,
            ),
        ]

        ground_truth = [
            "The person is cutting vegetables on a board.",
            "Washing hands.",
        ]

        metrics = evaluate_with_metrics(predictions, ground_truth)

        assert "bleu" in metrics
        assert "rouge_l" in metrics
        assert "avg_prediction_length" in metrics
        assert metrics["bleu"] >= 0
        assert metrics["rouge_l"] >= 0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        predictions = []
        ground_truth = []

        metrics = evaluate_with_metrics(predictions, ground_truth)

        assert metrics["bleu"] == 0.0
        assert metrics["rouge_l"] == 0.0
