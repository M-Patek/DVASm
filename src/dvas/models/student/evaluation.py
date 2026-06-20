"""Teacher vs Student evaluation - quality, cost, and latency comparison.

Provides comprehensive evaluation metrics comparing teacher and student models
across multiple dimensions: annotation quality, inference cost, and latency.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dvas.models.base import GenerationResult, ModelType
from dvas.models.evaluator.metrics import MetricsCalculator
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Global metrics calculator instance
_metrics_calc = MetricsCalculator()

def _calculate_bleu(references: List[str], predictions: List[str]) -> float:
    """Calculate average BLEU score."""
    scores = []
    for ref, pred in zip(references, predictions):
        bleu_scores = _metrics_calc.bleu(ref, pred)
        scores.append(bleu_scores.get("bleu_4", 0.0))
    return sum(scores) / len(scores) if scores else 0.0

def _calculate_rouge(reference: str, prediction: str) -> Dict:
    """Calculate ROUGE scores."""
    return _metrics_calc.rouge(reference, prediction)


@dataclass
class QualityComparisonResult:
    """Quality comparison between teacher and student predictions."""

    metric_name: str
    teacher_score: float
    student_score: float
    relative_improvement: float  # Positive means student better
    absolute_gap: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "metric_name": self.metric_name,
            "teacher_score": self.teacher_score,
            "student_score": self.student_score,
            "relative_improvement": self.relative_improvement,
            "absolute_gap": self.absolute_gap,
        }


@dataclass
class CostComparisonResult:
    """Cost comparison between teacher and student inference."""

    teacher_cost_per_sample: float
    student_cost_per_sample: float
    cost_savings_per_sample: float
    cost_savings_percent: float
    estimated_savings_1k_samples: float
    estimated_savings_10k_samples: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "teacher_cost_per_sample": self.teacher_cost_per_sample,
            "student_cost_per_sample": self.student_cost_per_sample,
            "cost_savings_per_sample": self.cost_savings_per_sample,
            "cost_savings_percent": self.cost_savings_percent,
            "estimated_savings_1k_samples": self.estimated_savings_1k_samples,
            "estimated_savings_10k_samples": self.estimated_savings_10k_samples,
        }


@dataclass
class LatencyComparisonResult:
    """Latency comparison between teacher and student inference."""

    teacher_latency_ms: float
    student_latency_ms: float
    teacher_p50_ms: float
    teacher_p95_ms: float
    teacher_p99_ms: float
    student_p50_ms: float
    student_p95_ms: float
    student_p99_ms: float
    speedup_ratio: float  # Student speedup vs teacher
    throughput_improvement: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "teacher_latency_ms": self.teacher_latency_ms,
            "student_latency_ms": self.student_latency_ms,
            "teacher_p50_ms": self.teacher_p50_ms,
            "teacher_p95_ms": self.teacher_p95_ms,
            "teacher_p99_ms": self.teacher_p99_ms,
            "student_p50_ms": self.student_p50_ms,
            "student_p95_ms": self.student_p95_ms,
            "student_p99_ms": self.student_p99_ms,
            "speedup_ratio": self.speedup_ratio,
            "throughput_improvement": self.throughput_improvement,
        }


@dataclass
class ComparisonReport:
    """Complete teacher vs student comparison report."""

    n_samples: int
    quality_metrics: List[QualityComparisonResult] = field(default_factory=list)
    cost_comparison: Optional[CostComparisonResult] = None
    latency_comparison: Optional[LatencyComparisonResult] = None

    # Agreement statistics
    exact_match_rate: float = 0.0
    semantic_match_rate: float = 0.0

    # Per-sample results
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "quality_metrics": [m.to_dict() for m in self.quality_metrics],
            "cost_comparison": self.cost_comparison.to_dict() if self.cost_comparison else None,
            "latency_comparison": self.latency_comparison.to_dict() if self.latency_comparison else None,
            "exact_match_rate": self.exact_match_rate,
            "semantic_match_rate": self.semantic_match_rate,
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("TEACHER vs STUDENT COMPARISON REPORT")
        print("=" * 60)
        print(f"Samples evaluated: {self.n_samples}")

        print("\n--- QUALITY METRICS ---")
        for metric in self.quality_metrics:
            print(f"{metric.metric_name}:")
            print(f"  Teacher: {metric.teacher_score:.3f}")
            print(f"  Student: {metric.student_score:.3f}")
            print(f"  Gap: {metric.absolute_gap:+.3f} ({metric.relative_improvement:+.1%})")

        if self.cost_comparison:
            print("\n--- COST COMPARISON ---")
            c = self.cost_comparison
            print(f"Teacher cost/sample: ${c.teacher_cost_per_sample:.4f}")
            print(f"Student cost/sample: ${c.student_cost_per_sample:.4f}")
            print(f"Savings: {c.cost_savings_percent:.1%} (${c.cost_savings_per_sample:.4f}/sample)")
            print(f"Est. savings @ 10k samples: ${c.estimated_savings_10k_samples:.2f}")

        if self.latency_comparison:
            print("\n--- LATENCY COMPARISON ---")
            l = self.latency_comparison
            print(f"Teacher latency: {l.teacher_latency_ms:.1f}ms (p95: {l.teacher_p95_ms:.1f}ms)")
            print(f"Student latency: {l.student_latency_ms:.1f}ms (p95: {l.student_p95_ms:.1f}ms)")
            print(f"Speedup: {l.speedup_ratio:.1f}x")

        print("\n--- AGREEMENT ---")
        print(f"Exact match rate: {self.exact_match_rate:.1%}")
        print(f"Semantic match rate: {self.semantic_match_rate:.1%}")
        print("=" * 60)


class TeacherStudentEvaluator:
    """Evaluator for comparing teacher and student models.

    Provides comprehensive evaluation across quality, cost, and latency dimensions.
    """

    def __init__(
        self,
        teacher_model=None,  # Optional: for live evaluation
        student_model=None,  # Optional: for live evaluation
        llm_judge=None,  # Optional: for quality evaluation
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.llm_judge = llm_judge

    def evaluate_quality(
        self,
        teacher_predictions: List[GenerationResult],
        student_predictions: List[GenerationResult],
        ground_truth: Optional[List[str]] = None,
    ) -> List[QualityComparisonResult]:
        """Evaluate and compare annotation quality.

        Args:
            teacher_predictions: Predictions from teacher model
            student_predictions: Predictions from student model
            ground_truth: Optional ground truth annotations

        Returns:
            List of quality comparison metrics
        """
        results = []

        # Extract texts
        teacher_texts = [p.text for p in teacher_predictions]
        student_texts = [p.text for p in student_predictions]

        # Compare against ground truth if available
        if ground_truth:
            # BLEU scores
            teacher_bleu = _calculate_bleu(ground_truth, teacher_texts)
            student_bleu = _calculate_bleu(ground_truth, student_texts)

            results.append(QualityComparisonResult(
                metric_name="BLEU",
                teacher_score=teacher_bleu,
                student_score=student_bleu,
                relative_improvement=(student_bleu - teacher_bleu) / (teacher_bleu + 1e-8),
                absolute_gap=student_bleu - teacher_bleu,
            ))

            # ROUGE scores
            teacher_rouge_scores = [_calculate_rouge(gt, pred) for gt, pred in zip(ground_truth, teacher_texts)]
            student_rouge_scores = [_calculate_rouge(gt, pred) for gt, pred in zip(ground_truth, student_texts)]
            teacher_rouge = np.mean([s.get("rougeL_f", 0.0) for s in teacher_rouge_scores])
            student_rouge = np.mean([s.get("rougeL_f", 0.0) for s in student_rouge_scores])

            results.append(QualityComparisonResult(
                metric_name="ROUGE-L",
                teacher_score=teacher_rouge,
                student_score=student_rouge,
                relative_improvement=(student_rouge - teacher_rouge) / (teacher_rouge + 1e-8),
                absolute_gap=student_rouge - teacher_rouge,
            ))

        # Direct teacher-student comparison metrics
        # Agreement rate
        agreements = sum(
            1 for t, s in zip(teacher_texts, student_texts)
            if self._compute_similarity(t, s) > 0.8
        )
        agreement_rate = agreements / len(teacher_texts) if teacher_texts else 0

        results.append(QualityComparisonResult(
            metric_name="Agreement Rate",
            teacher_score=1.0,  # Teacher is reference
            student_score=agreement_rate,
            relative_improvement=agreement_rate - 1.0,
            absolute_gap=agreement_rate - 1.0,
        ))

        # Confidence calibration comparison
        teacher_confidences = [p.confidence for p in teacher_predictions if p.confidence]
        student_confidences = [p.confidence for p in student_predictions if p.confidence]

        if teacher_confidences and student_confidences:
            results.append(QualityComparisonResult(
                metric_name="Avg Confidence",
                teacher_score=np.mean(teacher_confidences),
                student_score=np.mean(student_confidences),
                relative_improvement=(np.mean(student_confidences) - np.mean(teacher_confidences))
                    / (np.mean(teacher_confidences) + 1e-8),
                absolute_gap=np.mean(student_confidences) - np.mean(teacher_confidences),
            ))

        logger.info(
            "Quality evaluation complete",
            n_samples=len(teacher_predictions),
            metrics=len(results),
        )

        return results

    def evaluate_cost(
        self,
        teacher_predictions: List[GenerationResult],
        student_predictions: List[GenerationResult],
    ) -> CostComparisonResult:
        """Evaluate and compare inference costs.

        Args:
            teacher_predictions: Predictions with cost info
            student_predictions: Predictions with cost info

        Returns:
            Cost comparison results
        """
        teacher_costs = [p.cost_usd for p in teacher_predictions]
        student_costs = [p.cost_usd for p in student_predictions]

        teacher_avg = np.mean(teacher_costs) if teacher_costs else 0
        student_avg = np.mean(student_costs) if student_costs else 0

        savings = teacher_avg - student_avg
        savings_pct = savings / (teacher_avg + 1e-8) * 100

        result = CostComparisonResult(
            teacher_cost_per_sample=teacher_avg,
            student_cost_per_sample=student_avg,
            cost_savings_per_sample=savings,
            cost_savings_percent=savings_pct,
            estimated_savings_1k_samples=savings * 1000,
            estimated_savings_10k_samples=savings * 10000,
        )

        logger.info(
            "Cost evaluation complete",
            teacher_cost=teacher_avg,
            student_cost=student_avg,
            savings_pct=savings_pct,
        )

        return result

    def evaluate_latency(
        self,
        teacher_predictions: List[GenerationResult],
        student_predictions: List[GenerationResult],
    ) -> LatencyComparisonResult:
        """Evaluate and compare inference latency.

        Args:
            teacher_predictions: Predictions with latency info
            student_predictions: Predictions with latency info

        Returns:
            Latency comparison results
        """
        teacher_latencies = [p.latency_ms for p in teacher_predictions if p.latency_ms]
        student_latencies = [p.latency_ms for p in student_predictions if p.latency_ms]

        if not teacher_latencies or not student_latencies:
            return LatencyComparisonResult(
                teacher_latency_ms=0,
                student_latency_ms=0,
                teacher_p50_ms=0,
                teacher_p95_ms=0,
                teacher_p99_ms=0,
                student_p50_ms=0,
                student_p95_ms=0,
                student_p99_ms=0,
                speedup_ratio=1.0,
                throughput_improvement=1.0,
            )

        teacher_avg = np.mean(teacher_latencies)
        student_avg = np.mean(student_latencies)

        result = LatencyComparisonResult(
            teacher_latency_ms=teacher_avg,
            student_latency_ms=student_avg,
            teacher_p50_ms=np.percentile(teacher_latencies, 50),
            teacher_p95_ms=np.percentile(teacher_latencies, 95),
            teacher_p99_ms=np.percentile(teacher_latencies, 99),
            student_p50_ms=np.percentile(student_latencies, 50),
            student_p95_ms=np.percentile(student_latencies, 95),
            student_p99_ms=np.percentile(student_latencies, 99),
            speedup_ratio=teacher_avg / (student_avg + 1e-8),
            throughput_improvement=teacher_avg / (student_avg + 1e-8),
        )

        logger.info(
            "Latency evaluation complete",
            teacher_avg_ms=teacher_avg,
            student_avg_ms=student_avg,
            speedup=result.speedup_ratio,
        )

        return result

    def run_full_comparison(
        self,
        test_videos: List[Path],
        ground_truth: Optional[List[str]] = None,
        prompt: Optional[str] = None,
    ) -> ComparisonReport:
        """Run complete teacher vs student comparison.

        This method runs both models on the test set and compares results.

        Args:
            test_videos: List of test video paths
            ground_truth: Optional ground truth annotations
            prompt: Optional prompt override

        Returns:
            Complete comparison report
        """
        import asyncio

        if not self.teacher_model or not self.student_model:
            raise ValueError("Both teacher_model and student_model required for live evaluation")

        logger.info(
            "Running full comparison",
            n_videos=len(test_videos),
        )

        # Run inference with both models
        async def run_inference():
            teacher_results = []
            student_results = []

            for video_path in test_videos:
                # Teacher
                t_result = await self.teacher_model.annotate(
                    video_path=video_path,
                    prompt=prompt,
                )
                teacher_results.append(t_result)

                # Student
                s_result = await self.student_model.generate(
                    video_path=video_path,
                    prompt=prompt,
                )
                student_results.append(s_result)

            return teacher_results, student_results

        teacher_preds, student_preds = asyncio.run(run_inference())

        # Build report
        report = ComparisonReport(n_samples=len(test_videos))

        # Quality comparison
        report.quality_metrics = self.evaluate_quality(
            teacher_preds, student_preds, ground_truth
        )

        # Cost comparison
        report.cost_comparison = self.evaluate_cost(teacher_preds, student_preds)

        # Latency comparison
        report.latency_comparison = self.evaluate_latency(teacher_preds, student_preds)

        # Agreement statistics
        teacher_texts = [p.text for p in teacher_preds]
        student_texts = [p.text for p in student_preds]

        exact_matches = sum(t == s for t, s in zip(teacher_texts, student_texts))
        report.exact_match_rate = exact_matches / len(teacher_texts) if teacher_texts else 0

        # Semantic matches (using similarity)
        semantic_matches = sum(
            self._compute_similarity(t, s) > 0.7
            for t, s in zip(teacher_texts, student_texts)
        )
        report.semantic_match_rate = semantic_matches / len(teacher_texts) if teacher_texts else 0

        # Per-sample results
        report.per_sample_results = [
            {
                "video": str(test_videos[i]),
                "teacher": teacher_texts[i],
                "student": student_texts[i],
                "similarity": self._compute_similarity(teacher_texts[i], student_texts[i]),
            }
            for i in range(len(test_videos))
        ]

        return report

    def compare_on_predictions(
        self,
        teacher_predictions: List[GenerationResult],
        student_predictions: List[GenerationResult],
        ground_truth: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """Compare models using pre-computed predictions.

        Args:
            teacher_predictions: Pre-computed teacher predictions
            student_predictions: Pre-computed student predictions
            ground_truth: Optional ground truth annotations

        Returns:
            Complete comparison report
        """
        report = ComparisonReport(n_samples=len(teacher_predictions))

        # Quality comparison
        report.quality_metrics = self.evaluate_quality(
            teacher_predictions, student_predictions, ground_truth
        )

        # Cost comparison
        report.cost_comparison = self.evaluate_cost(
            teacher_predictions, student_predictions
        )

        # Latency comparison
        report.latency_comparison = self.evaluate_latency(
            teacher_predictions, student_predictions
        )

        # Agreement statistics
        teacher_texts = [p.text for p in teacher_predictions]
        student_texts = [p.text for p in student_predictions]

        exact_matches = sum(t == s for t, s in zip(teacher_texts, student_texts))
        report.exact_match_rate = exact_matches / len(teacher_texts) if teacher_texts else 0

        semantic_matches = sum(
            self._compute_similarity(t, s) > 0.7
            for t, s in zip(teacher_texts, student_texts)
        )
        report.semantic_match_rate = semantic_matches / len(teacher_texts) if teacher_texts else 0

        return report

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


def evaluate_with_metrics(
    predictions: List[GenerationResult],
    ground_truth: List[str],
) -> Dict[str, float]:
    """Evaluate predictions with standard metrics.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth annotations

    Returns:
        Dictionary of metric scores
    """
    pred_texts = [p.text for p in predictions]

    # Handle empty case
    if not pred_texts or not ground_truth:
        return {
            "bleu": 0.0,
            "rouge_l": 0.0,
            "avg_prediction_length": 0.0,
        }

    # BLEU
    bleu_score = _calculate_bleu(ground_truth, pred_texts)

    # ROUGE-L
    rouge_scores = [_calculate_rouge(gt, pred) for gt, pred in zip(ground_truth, pred_texts)]
    rouge_l = np.mean([s.get("rougeL_f", 0.0) for s in rouge_scores])

    # Average prediction length
    avg_length = np.mean([len(p.split()) for p in pred_texts])

    return {
        "bleu": bleu_score,
        "rouge_l": rouge_l,
        "avg_prediction_length": avg_length,
    }
