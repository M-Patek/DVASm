"""Drift detection with KS test and PSI for production monitoring.

Provides comprehensive drift detection for input features and prediction
distributions using Kolmogorov-Smirnov test and Population Stability Index.

Usage::

    from dvas.observability.drift_detector import DriftMonitor

    monitor = DriftMonitor(reference_data=ref_annotations)
    report = monitor.check(new_annotations)

    if report.drift_detected:
        print(f"Alert: {report.alerts}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional scipy import
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class DriftReport:
    """Report from drift detection analysis.

    Attributes:
        timestamp: When the check was performed
        drift_detected: Whether any drift was detected
        feature_drift: Dict of feature-level drift scores
        prediction_drift: Dict of prediction-level drift scores
        alerts: List of human-readable alert messages
        ks_results: Kolmogorov-Smirnov test results
        psi_results: Population Stability Index results
        recommendations: Recommended actions
    """

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    drift_detected: bool = False
    feature_drift: Dict[str, Any] = field(default_factory=dict)
    prediction_drift: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    ks_results: Dict[str, Any] = field(default_factory=dict)
    psi_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DriftConfig:
    """Configuration for drift detection.

    Attributes:
        ks_threshold: P-value threshold for KS test (lower = stricter)
        psi_threshold: PSI threshold (higher = stricter)
            - < 0.1: No significant change
            - 0.1 - 0.25: Moderate change
            - > 0.25: Significant change
        window_size: Number of recent samples for rolling reference
        min_samples: Minimum samples required for drift detection
        check_interval_hours: How often to run drift checks
        enable_ks: Whether to enable KS test
        enable_psi: Whether to enable PSI
        alert_on_feature_drift: Whether to alert on input feature drift
        alert_on_prediction_drift: Whether to alert on prediction drift
        feature_columns: List of features to monitor
    """

    ks_threshold: float = 0.05
    psi_threshold: float = 0.25
    window_size: int = 1000
    min_samples: int = 30
    check_interval_hours: int = 1
    enable_ks: bool = True
    enable_psi: bool = True
    alert_on_feature_drift: bool = True
    alert_on_prediction_drift: bool = True
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "resolution",
            "duration",
            "num_segments",
            "caption_length",
            "num_actions",
            "num_objects",
        ]
    )


def ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test.

    Tests whether two samples come from the same distribution.

    Args:
        reference: Reference distribution samples
        current: Current distribution samples

    Returns:
        Tuple of (KS statistic, p-value)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for KS test")

    if len(reference) < 2 or len(current) < 2:
        return 0.0, 1.0

    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index.

    PSI measures how much a distribution has shifted from reference.

    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        bins: Number of bins for discretization

    Returns:
        PSI score
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Create bins based on reference distribution
    min_val = np.min(reference)
    max_val = np.max(reference)

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[-1] += 1e-10  # Ensure max value falls in last bin

    # Compute histograms
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    # Convert to percentages
    ref_pct = ref_hist / len(reference)
    cur_pct = cur_hist / len(current)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    # Compute PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def compute_hellinger_distance(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Hellinger distance between two distributions.

    A symmetric measure of distribution similarity (0 = identical, 1 = completely different).

    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        bins: Number of bins

    Returns:
        Hellinger distance (0-1)
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    min_val = min(np.min(reference), np.min(current))
    max_val = max(np.max(reference), np.max(current))

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[-1] += 1e-10

    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    # Normalize to probabilities
    ref_p = ref_hist / len(reference)
    cur_p = cur_hist / len(current)

    # Hellinger distance
    sqrt_diff = np.sum((np.sqrt(ref_p) - np.sqrt(cur_p)) ** 2)
    distance = np.sqrt(sqrt_diff / 2)

    return float(distance)


class FeatureExtractor:
    """Extract numeric features from annotations for drift detection."""

    @staticmethod
    def extract_features(annotation: Annotation) -> Dict[str, float]:
        """Extract numeric features from an annotation.

        Args:
            annotation: Annotation to extract features from

        Returns:
            Dict of feature name -> value
        """
        features = {
            "resolution": 0.0,
            "duration": 0.0,
            "num_segments": float(len(annotation.segments)),
            "caption_length": 0.0,
            "num_actions": 0.0,
            "num_objects": 0.0,
            "num_qa_pairs": 0.0,
            "quality_score": annotation.quality_score or 0.0,
        }

        # Video metadata features
        if annotation.metadata:
            features["resolution"] = float(
                annotation.metadata.resolution[0] * annotation.metadata.resolution[1]
                if annotation.metadata.resolution
                else 0
            )
            features["duration"] = float(annotation.metadata.duration or 0)
            features["fps"] = float(annotation.metadata.fps or 0)
            features["total_frames"] = float(annotation.metadata.total_frames or 0)

        # Segment-level features
        for segment in annotation.segments:
            features["caption_length"] += len(segment.caption or "")
            features["num_actions"] += len(segment.actions)
            features["num_objects"] += len(segment.objects)
            features["num_qa_pairs"] += len(segment.qa_pairs)

        # Average per segment
        if annotation.segments:
            features["caption_length"] /= len(annotation.segments)
            features["num_actions"] /= len(annotation.segments)
            features["num_objects"] /= len(annotation.segments)
            features["num_qa_pairs"] /= len(annotation.segments)

        return features

    @staticmethod
    def extract_vocabulary_features(annotations: List[Annotation]) -> Dict[str, Any]:
        """Extract vocabulary-level features from annotations.

        Args:
            annotations: List of annotations

        Returns:
            Dict with vocabulary statistics
        """
        verbs = []
        nouns = []
        captions = []
        all_words = []

        for ann in annotations:
            for seg in ann.segments:
                captions.append(seg.caption or "")
                for action in seg.actions:
                    verbs.append(action.verb)
                    nouns.append(action.noun)
                    all_words.extend([action.verb, action.noun])

        # Word frequency
        from collections import Counter

        word_freq = Counter(all_words)
        verb_freq = Counter(verbs)
        noun_freq = Counter(nouns)

        return {
            "unique_words": len(word_freq),
            "total_words": len(all_words),
            "unique_verbs": len(verb_freq),
            "unique_nouns": len(noun_freq),
            "avg_caption_length": np.mean([len(c) for c in captions]) if captions else 0,
            "top_verbs": verb_freq.most_common(10),
            "top_nouns": noun_freq.most_common(10),
            "vocab_diversity": len(word_freq) / len(all_words) if all_words else 0,
        }


class DriftMonitor:
    """Production drift monitor with KS test and PSI.

    Monitors input features and prediction distributions for drift
    and generates alerts when significant changes are detected.

    Usage::

        monitor = DriftMonitor(reference_data=ref_annotations)
        report = monitor.check(new_annotations)

        if report.drift_detected:
            print(f"Drift detected: {report.alerts}")
            print(f"Recommendations: {report.recommendations}")
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        reference_data: Optional[List[Annotation]] = None,
    ):
        self.config = config or DriftConfig()
        self.reference_data = reference_data or []
        self.reference_features: Dict[str, np.ndarray] = {}
        self._history: List[DriftReport] = []

        if reference_data:
            self.set_reference(reference_data)

    def set_reference(self, annotations: List[Annotation]) -> None:
        """Set reference data for drift detection.

        Args:
            annotations: Reference annotations
        """
        if len(annotations) < self.config.min_samples:
            logger.warning(
                "insufficient_reference_samples",
                required=self.config.min_samples,
                actual=len(annotations),
            )

        self.reference_data = annotations
        self.reference_features = self._extract_feature_arrays(annotations)
        logger.info(
            "reference_set",
            samples=len(annotations),
            features=list(self.reference_features.keys()),
        )

    def _extract_feature_arrays(
        self, annotations: List[Annotation]
    ) -> Dict[str, np.ndarray]:
        """Extract feature arrays from annotations.

        Args:
            annotations: List of annotations

        Returns:
            Dict of feature name -> numpy array
        """
        features = {col: [] for col in self.config.feature_columns}

        for annotation in annotations:
            extracted = FeatureExtractor.extract_features(annotation)
            for col in self.config.feature_columns:
                if col in extracted:
                    features[col].append(extracted[col])

        return {k: np.array(v) for k, v in features.items() if v}

    def check(self, current_annotations: List[Annotation]) -> DriftReport:
        """Run drift detection on current annotations.

        Args:
            current_annotations: Current batch of annotations

        Returns:
            DriftReport with detection results
        """
        report = DriftReport()

        if len(current_annotations) < self.config.min_samples:
            report.alerts.append(
                f"Insufficient samples: {len(current_annotations)} < {self.config.min_samples}"
            )
            return report

        if not self.reference_features:
            report.alerts.append("No reference data set")
            return report

        current_features = self._extract_feature_arrays(current_annotations)

        # Feature drift detection
        if self.config.alert_on_feature_drift:
            self._check_feature_drift(report, current_features)

        # Prediction drift detection
        if self.config.alert_on_prediction_drift:
            self._check_prediction_drift(report, current_annotations)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        report.drift_detected = len(report.alerts) > 0

        # Store in history
        self._history.append(report)

        # Trim history
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return report

    def _check_feature_drift(
        self,
        report: DriftReport,
        current_features: Dict[str, np.ndarray],
    ) -> None:
        """Check for drift in input features."""
        for feature_name in self.config.feature_columns:
            if feature_name not in self.reference_features:
                continue

            ref_data = self.reference_features[feature_name]
            cur_data = current_features.get(feature_name, np.array([]))

            if len(cur_data) < self.config.min_samples:
                continue

            # KS test
            if self.config.enable_ks and SCIPY_AVAILABLE:
                ks_stat, p_value = ks_test(ref_data, cur_data)
                report.ks_results[feature_name] = {
                    "statistic": ks_stat,
                    "p_value": p_value,
                    "drift": p_value < self.config.ks_threshold,
                }

                if p_value < self.config.ks_threshold:
                    report.alerts.append(
                        f"Feature drift detected in '{feature_name}': "
                        f"KS p-value={p_value:.4f} < {self.config.ks_threshold}"
                    )

            # PSI
            if self.config.enable_psi:
                psi = compute_psi(ref_data, cur_data)
                report.psi_results[feature_name] = {
                    "psi": psi,
                    "drift": psi > self.config.psi_threshold,
                }

                if psi > self.config.psi_threshold:
                    report.alerts.append(
                        f"Feature PSI alert for '{feature_name}': "
                        f"PSI={psi:.4f} > {self.config.psi_threshold}"
                    )

            # Feature drift summary
            report.feature_drift[feature_name] = {
                "reference_mean": float(np.mean(ref_data)),
                "reference_std": float(np.std(ref_data)),
                "current_mean": float(np.mean(cur_data)),
                "current_std": float(np.std(cur_data)),
            }

    def _check_prediction_drift(
        self,
        report: DriftReport,
        current_annotations: List[Annotation],
    ) -> None:
        """Check for drift in prediction distributions."""
        # Quality score drift
        ref_scores = np.array([a.quality_score or 0 for a in self.reference_data])
        cur_scores = np.array([a.quality_score or 0 for a in current_annotations])

        if len(ref_scores) > 0 and len(cur_scores) > 0:
            if self.config.enable_ks and SCIPY_AVAILABLE:
                ks_stat, p_value = ks_test(ref_scores, cur_scores)
                report.ks_results["quality_score"] = {
                    "statistic": ks_stat,
                    "p_value": p_value,
                }

                if p_value < self.config.ks_threshold:
                    report.alerts.append(
                        f"Prediction drift: Quality score distribution changed "
                        f"(p={p_value:.4f})"
                    )

            psi = compute_psi(ref_scores, cur_scores)
            report.psi_results["quality_score"] = {"psi": psi}

            if psi > self.config.psi_threshold:
                report.alerts.append(
                    f"Prediction PSI alert: Quality score PSI={psi:.4f}"
                )

            report.prediction_drift["quality_score"] = {
                "reference_mean": float(np.mean(ref_scores)),
                "current_mean": float(np.mean(cur_scores)),
            }

        # Vocabulary drift
        ref_vocab = FeatureExtractor.extract_vocabulary_features(self.reference_data)
        cur_vocab = FeatureExtractor.extract_vocabulary_features(current_annotations)

        if ref_vocab["unique_words"] > 0 and cur_vocab["unique_words"] > 0:
            # Check for novel words
            ref_words = set(ref_vocab["top_verbs"])
            cur_words = set(cur_vocab["top_verbs"])
            novel_ratio = len(cur_words - ref_words) / max(len(ref_words), 1)

            report.prediction_drift["vocabulary"] = {
                "reference_unique": ref_vocab["unique_words"],
                "current_unique": cur_vocab["unique_words"],
                "novel_word_ratio": novel_ratio,
            }

            if novel_ratio > 0.3:  # 30% new words
                report.alerts.append(
                    f"Vocabulary drift: {novel_ratio:.1%} novel words detected"
                )

    def _generate_recommendations(self, report: DriftReport) -> List[str]:
        """Generate recommendations based on drift report.

        Args:
            report: Drift report

        Returns:
            List of recommendations
        """
        recommendations = []

        if not report.drift_detected:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations

        # Feature drift recommendations
        feature_drifts = [
            k for k, v in report.ks_results.items()
            if k in self.config.feature_columns and v.get("drift", False)
        ]

        if feature_drifts:
            recommendations.append(
                f"Input feature drift detected in: {', '.join(feature_drifts)}. "
                "Consider data validation and preprocessing checks."
            )

        # Prediction drift recommendations
        if report.psi_results.get("quality_score", {}).get("drift", False):
            recommendations.append(
                "Model output quality has shifted. Consider model retraining or "
                "A/B testing a new model version."
            )

        # Vocabulary drift
        if "vocabulary" in report.prediction_drift:
            vocab = report.prediction_drift["vocabulary"]
            if vocab.get("novel_word_ratio", 0) > 0.3:
                recommendations.append(
                    "Significant vocabulary drift detected. Model may need retraining "
                    "with updated vocabulary."
                )

        # General recommendations
        recommendations.append(
            "Review recent data pipeline changes and model performance metrics."
        )

        return recommendations

    def get_history(self, limit: int = 10) -> List[DriftReport]:
        """Get recent drift detection history.

        Args:
            limit: Maximum number of reports

        Returns:
            List of recent drift reports
        """
        return self._history[-limit:]

    def get_drift_trend(self, feature: str = "quality_score") -> Dict[str, Any]:
        """Get drift trend for a feature over time.

        Args:
            feature: Feature name to analyze

        Returns:
            Dict with trend statistics
        """
        if not self._history:
            return {"error": "No history available"}

        ks_p_values = []
        psi_scores = []

        for report in self._history:
            if feature in report.ks_results:
                ks_p_values.append(report.ks_results[feature].get("p_value", 1.0))
            if feature in report.psi_results:
                psi_scores.append(report.psi_results[feature].get("psi", 0.0))

        return {
            "feature": feature,
            "checks": len(self._history),
            "ks_p_values": ks_p_values,
            "psi_scores": psi_scores,
            "avg_ks_p_value": np.mean(ks_p_values) if ks_p_values else 1.0,
            "avg_psi": np.mean(psi_scores) if psi_scores else 0.0,
            "drift_frequency": sum(1 for p in ks_p_values if p < self.config.ks_threshold)
            / len(ks_p_values)
            if ks_p_values
            else 0.0,
        }

    def save_report(self, report: DriftReport, output_dir: Path) -> Path:
        """Save a drift report to disk.

        Args:
            report: Drift report to save
            output_dir: Output directory

        Returns:
            Path to saved report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"drift_report_{timestamp}.json"

        data = {
            "timestamp": report.timestamp,
            "drift_detected": report.drift_detected,
            "feature_drift": report.feature_drift,
            "prediction_drift": report.prediction_drift,
            "alerts": report.alerts,
            "ks_results": report.ks_results,
            "psi_results": report.psi_results,
            "recommendations": report.recommendations,
        }

        with open(report_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return report_path

    def export_statistics(self) -> Dict[str, Any]:
        """Export comprehensive drift statistics.

        Returns:
            Dict with all drift statistics
        """
        return {
            "config": {
                "ks_threshold": self.config.ks_threshold,
                "psi_threshold": self.config.psi_threshold,
                "min_samples": self.config.min_samples,
                "feature_columns": self.config.feature_columns,
            },
            "reference_samples": len(self.reference_data),
            "history_length": len(self._history),
            "total_alerts": sum(1 for r in self._history if r.drift_detected),
            "features": list(self.reference_features.keys()),
            "latest_report": {
                "timestamp": self._history[-1].timestamp if self._history else None,
                "drift_detected": self._history[-1].drift_detected if self._history else False,
                "alerts": self._history[-1].alerts if self._history else [],
            },
        }
