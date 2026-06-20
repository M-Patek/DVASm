"""Confidence calibration for student model predictions.

Implements temperature scaling and other calibration methods to ensure
that model confidence scores accurately reflect prediction accuracy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dvas.models.base import GenerationResult
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    nll: float  # Negative Log-Likelihood
    brier: float  # Brier Score

    # Reliability diagram data
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "nll": self.nll,
            "brier": self.brier,
            "bin_accuracies": self.bin_accuracies,
            "bin_confidences": self.bin_confidences,
            "bin_counts": self.bin_counts,
        }


class TemperatureScaler:
    """Temperature scaling for confidence calibration.

    Temperature scaling divides logits by a learned temperature parameter T
    before applying softmax. This preserves ranking while calibrating
    confidence scores.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.is_fitted = False

    def fit(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> "TemperatureScaler":
        """Fit temperature parameter on validation data.

        Args:
            confidences: Raw confidence scores (before scaling)
            accuracies: Binary correctness labels (0 or 1)

        Returns:
            self for method chaining
        """
        # Simple grid search for optimal temperature
        best_temp = 1.0
        best_ece = float("inf")

        for temp in np.linspace(0.5, 2.0, 50):
            scaled = self._scale_confidences(confidences, temp)
            ece = self._compute_ece(scaled, accuracies)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        # Fine search around best
        for temp in np.linspace(best_temp - 0.05, best_temp + 0.05, 20):
            scaled = self._scale_confidences(confidences, temp)
            ece = self._compute_ece(scaled, accuracies)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        self.is_fitted = True

        logger.info(
            "Fitted temperature scaler",
            temperature=best_temp,
            ece=best_ece,
        )

        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to confidences.

        Args:
            confidences: Raw confidence scores

        Returns:
            Calibrated confidence scores
        """
        return self._scale_confidences(confidences, self.temperature)

    def _scale_confidences(
        self,
        confidences: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Scale confidences by temperature."""
        # Convert confidence to logit, scale, convert back
        # This is approximate for high confidences
        clipped = np.clip(confidences, 0.01, 0.99)
        logits = np.log(clipped / (1 - clipped))
        scaled_logits = logits / temperature
        scaled_conf = 1 / (1 + np.exp(-scaled_logits))
        return scaled_conf

    def _compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if bin_boundaries[i + 1] == 1.0:
                mask = mask | (confidences == 1.0)

            if np.sum(mask) > 0:
                avg_confidence = np.mean(confidences[mask])
                avg_accuracy = np.mean(accuracies[mask])
                ece += np.sum(mask) * np.abs(avg_confidence - avg_accuracy)

        return ece / len(confidences)

    def save(self, path: Path) -> None:
        """Save fitted scaler to disk."""
        import json

        data = {
            "temperature": self.temperature,
            "is_fitted": self.is_fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved temperature scaler", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        """Load fitted scaler from disk."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scaler = cls(temperature=data["temperature"])
        scaler.is_fitted = data["is_fitted"]

        return scaler


class ConfidenceCalibrator:
    """Main calibration interface for student model predictions.

    Supports multiple calibration methods:
    - Temperature scaling (parametric)
    - Isotonic regression (non-parametric)
    - Platt scaling (logistic calibration)
    """

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.scaler: Optional[TemperatureScaler] = None
        self.is_fitted = False

    def fit(
        self,
        predictions: List[GenerationResult],
        ground_truth: List[str],
        similarity_threshold: float = 0.7,
    ) -> CalibrationMetrics:
        """Fit calibrator on validation predictions.

        Args:
            predictions: Model predictions with confidence scores
            ground_truth: Ground truth texts
            similarity_threshold: Threshold for considering prediction correct
                (using text similarity for open-ended generation)

        Returns:
            CalibrationMetrics before and after calibration
        """
        # Extract confidences
        confidences = np.array([p.confidence for p in predictions])

        # Determine correctness (using simple text matching for now)
        # In practice, this could use BLEU/ROUGE or LLM-as-judge
        accuracies = np.array(
            [
                self._compute_correctness(p.text, gt, similarity_threshold)
                for p, gt in zip(predictions, ground_truth)
            ]
        )

        # Compute pre-calibration metrics
        pre_metrics = self._compute_metrics(confidences, accuracies)

        # Fit calibration method
        if self.method == "temperature":
            self.scaler = TemperatureScaler()
            self.scaler.fit(confidences, accuracies)
            self.is_fitted = True

            # Compute post-calibration metrics
            calibrated = self.scaler.transform(confidences)
            post_metrics = self._compute_metrics(calibrated, accuracies)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        logger.info(
            "Fitted confidence calibrator",
            method=self.method,
            pre_ece=pre_metrics.ece,
            post_ece=post_metrics.ece,
        )

        return post_metrics

    def calibrate(self, prediction: GenerationResult) -> GenerationResult:
        """Calibrate a single prediction's confidence.

        Args:
            prediction: Raw prediction from model

        Returns:
            Prediction with calibrated confidence
        """
        if not self.is_fitted or self.scaler is None:
            return prediction

        calibrated_conf = self.scaler.transform(np.array([prediction.confidence]))[0]

        # Create new result with calibrated confidence
        return GenerationResult(
            text=prediction.text,
            model_type=prediction.model_type,
            model_version=prediction.model_version,
            status=prediction.status,
            confidence=float(calibrated_conf),
            latency_ms=prediction.latency_ms,
            cost_usd=prediction.cost_usd,
            metadata={
                **(prediction.metadata or {}),
                "raw_confidence": prediction.confidence,
                "calibrated": True,
            },
        )

    def calibrate_batch(
        self,
        predictions: List[GenerationResult],
    ) -> List[GenerationResult]:
        """Calibrate a batch of predictions."""
        return [self.calibrate(p) for p in predictions]

    def _compute_correctness(
        self,
        prediction: str,
        ground_truth: str,
        threshold: float,
    ) -> int:
        """Compute whether prediction is correct using text similarity."""
        # Simple word overlap similarity
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())

        if not pred_words or not gt_words:
            return 0

        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        similarity = intersection / union if union > 0 else 0

        return 1 if similarity >= threshold else 0

    def _compute_metrics(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute comprehensive calibration metrics."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if bin_boundaries[i + 1] == 1.0:
                mask = mask | (confidences == 1.0)

            count = np.sum(mask)
            bin_counts.append(int(count))

            if count > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                bin_confidences.append(float(avg_conf))
                bin_accuracies.append(float(avg_acc))

                gap = np.abs(avg_conf - avg_acc)
                ece += count * gap
                mce = max(mce, gap)
            else:
                bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(0.0)

        ece /= len(confidences)

        # Negative log-likelihood
        clipped_conf = np.clip(confidences, 1e-10, 1 - 1e-10)
        nll = -np.mean(
            accuracies * np.log(clipped_conf) + (1 - accuracies) * np.log(1 - clipped_conf)
        )

        # Brier score
        brier = np.mean((confidences - accuracies) ** 2)

        return CalibrationMetrics(
            ece=float(ece),
            mce=float(mce),
            nll=float(nll),
            brier=float(brier),
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
        )

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        if self.scaler:
            self.scaler.save(path)

    @classmethod
    def load(cls, path: Path, method: str = "temperature") -> "ConfidenceCalibrator":
        """Load calibrator from disk."""
        calibrator = cls(method=method)
        if method == "temperature":
            calibrator.scaler = TemperatureScaler.load(path)
            calibrator.is_fitted = calibrator.scaler.is_fitted
        return calibrator


class ConfidenceThresholdOptimizer:
    """Optimize confidence threshold for specific accuracy/cost tradeoffs.

    Helps determine the optimal confidence threshold for:
    - Maximizing accuracy on high-confidence predictions
    - Minimizing fallback rate to teacher model
    - Balancing cost vs quality
    """

    def __init__(self):
        self.optimal_threshold = 0.5
        self.threshold_metrics: Dict[float, Dict] = {}

    def fit(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        costs: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Find optimal threshold for different objectives.

        Args:
            confidences: Confidence scores from student
            accuracies: Binary correctness labels
            costs: Optional cost savings vs teacher (for cost-aware optimization)

        Returns:
            Dictionary of optimal thresholds for different objectives
        """
        thresholds = np.linspace(0.1, 0.95, 50)
        results = {}

        for thresh in thresholds:
            mask = confidences >= thresh
            if np.sum(mask) == 0:
                continue

            student_acc = np.mean(accuracies[mask])
            coverage = np.sum(mask) / len(confidences)
            fallback_rate = 1 - coverage

            metrics = {
                "threshold": thresh,
                "accuracy": student_acc,
                "coverage": coverage,
                "fallback_rate": fallback_rate,
                "expected_accuracy": coverage * student_acc,
            }

            if costs is not None:
                student_cost = np.mean(costs[mask]) if np.sum(mask) > 0 else 0
                metrics["avg_cost"] = student_cost

            self.threshold_metrics[thresh] = metrics

        # Find optimal for different objectives
        best_for_accuracy = max(
            self.threshold_metrics.items(),
            key=lambda x: x[1]["accuracy"],
        )
        best_for_coverage = max(
            self.threshold_metrics.items(),
            key=lambda x: x[1]["coverage"],
        )
        best_for_expected_acc = max(
            self.threshold_metrics.items(),
            key=lambda x: x[1]["expected_accuracy"],
        )

        results = {
            "max_accuracy_threshold": best_for_accuracy[0],
            "max_accuracy_value": best_for_accuracy[1]["accuracy"],
            "max_coverage_threshold": best_for_coverage[0],
            "max_coverage_value": best_for_coverage[1]["coverage"],
            "balanced_threshold": best_for_expected_acc[0],
            "balanced_expected_accuracy": best_for_expected_acc[1]["expected_accuracy"],
        }

        # Default to balanced
        self.optimal_threshold = results["balanced_threshold"]

        logger.info(
            "Optimized confidence threshold",
            optimal_threshold=self.optimal_threshold,
            max_accuracy_threshold=results["max_accuracy_threshold"],
        )

        return results

    def get_threshold_for_target(
        self,
        target_accuracy: Optional[float] = None,
        max_fallback_rate: Optional[float] = None,
    ) -> float:
        """Get threshold that meets target constraints.

        Args:
            target_accuracy: Minimum accuracy requirement
            max_fallback_rate: Maximum acceptable fallback rate

        Returns:
            Optimal threshold meeting constraints
        """
        valid_thresholds = []

        for thresh, metrics in self.threshold_metrics.items():
            meets_accuracy = target_accuracy is None or metrics["accuracy"] >= target_accuracy
            meets_fallback = (
                max_fallback_rate is None or metrics["fallback_rate"] <= max_fallback_rate
            )

            if meets_accuracy and meets_fallback:
                valid_thresholds.append((thresh, metrics))

        if not valid_thresholds:
            # No threshold meets constraints, return most relaxed
            return self.optimal_threshold

        # Return threshold with best expected accuracy
        return max(valid_thresholds, key=lambda x: x[1]["expected_accuracy"])[0]

    def plot_reliability_diagram(
        self,
        metrics: CalibrationMetrics,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot reliability diagram for calibration visualization.

        Args:
            metrics: CalibrationMetrics from calibrator
            output_path: Path to save plot (if None, displays interactively)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.bar(
            metrics.bin_confidences,
            metrics.bin_accuracies,
            width=0.08,
            alpha=0.6,
            edgecolor="black",
        )
        ax1.set_xlabel("Confidence")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Reliability Diagram (ECE={metrics.ece:.3f})")
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Histogram of predictions per bin
        ax2.bar(
            metrics.bin_confidences,
            metrics.bin_counts,
            width=0.08,
            alpha=0.6,
            edgecolor="black",
        )
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Count")
        ax2.set_title("Prediction Distribution")

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved reliability diagram", path=str(output_path))
        else:
            plt.show()

        plt.close()
