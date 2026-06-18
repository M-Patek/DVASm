"""A/B testing framework for model comparison."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class TestStatus(str, Enum):
    """A/B test status."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    test_name: str
    variant_a: str  # Model/config name for control
    variant_b: str  # Model/config name for treatment
    traffic_split: float = 0.5  # Traffic to variant B
    min_sample_size: int = 100
    max_sample_size: int = 1000
    primary_metric: str = "quality_score"
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency", "cost"])
    confidence_level: float = 0.95
    mde: float = 0.05  # Minimum detectable effect


@dataclass
class TestResult:
    """Results from an A/B test."""

    test_name: str
    status: TestStatus
    variant_a_stats: Dict
    variant_b_stats: Dict
    winner: Optional[str]
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    recommendations: List[str]
    completed_at: Optional[str] = None


class ABTestManager:
    """Manage A/B tests for model evaluation."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/ab_tests")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.results_cache: Dict[str, TestResult] = {}

    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test."""
        test_id = hashlib.md5(
            f"{config.test_name}_{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        self.active_tests[test_id] = config

        # Save config
        config_path = self.storage_path / f"{test_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        logger.info("ab_test_created", test_id=test_id, name=config.test_name)

        return test_id

    def assign_variant(self, test_id: str, unit_id: str) -> str:
        """Assign a unit (video) to a variant."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        config = self.active_tests[test_id]

        # Deterministic assignment based on hash
        hash_val = int(hashlib.md5(f"{test_id}_{unit_id}".encode()).hexdigest(), 16)
        assignment_prob = (hash_val % 1000) / 1000

        if assignment_prob < config.traffic_split:
            return "B"
        return "A"

    def record_outcome(
        self,
        test_id: str,
        variant: str,
        metrics: Dict[str, float],
    ) -> None:
        """Record an outcome for a variant."""
        outcome_path = self.storage_path / f"{test_id}_outcomes.jsonl"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "metrics": metrics,
        }

        with open(outcome_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def analyze_test(self, test_id: str) -> Optional[TestResult]:
        """Analyze A/B test results."""
        if test_id in self.results_cache:
            return self.results_cache[test_id]

        config = self.active_tests.get(test_id)
        if not config:
            return None

        # Load outcomes
        outcome_path = self.storage_path / f"{test_id}_outcomes.jsonl"
        if not outcome_path.exists():
            return None

        outcomes_a = []
        outcomes_b = []

        with open(outcome_path) as f:
            for line in f:
                record = json.loads(line)
                if record["variant"] == "A":
                    outcomes_a.append(record["metrics"])
                else:
                    outcomes_b.append(record["metrics"])

        # Check sample size
        if len(outcomes_a) < config.min_sample_size or len(outcomes_b) < config.min_sample_size:
            logger.info(
                "insufficient_sample_size",
                test_id=test_id,
                n_a=len(outcomes_a),
                n_b=len(outcomes_b),
                min_required=config.min_sample_size,
            )
            return None

        # Statistical analysis
        primary_metric = config.primary_metric

        values_a = [o.get(primary_metric, 0) for o in outcomes_a]
        values_b = [o.get(primary_metric, 0) for o in outcomes_b]

        # T-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(values_a) ** 2 + np.std(values_b) ** 2) / 2)
        effect_size = (np.mean(values_b) - np.mean(values_a)) / (pooled_std + 1e-10)

        # Confidence interval
        alpha = 1 - config.confidence_level
        mean_diff = np.mean(values_b) - np.mean(values_a)
        se = pooled_std * np.sqrt(1 / len(values_a) + 1 / len(values_b))
        ci = stats.t.interval(
            1 - alpha,
            len(values_a) + len(values_b) - 2,
            loc=mean_diff,
            scale=se,
        )

        # Determine winner
        winner = None
        if p_value < (1 - config.confidence_level):
            if effect_size > 0:
                winner = config.variant_b
            else:
                winner = config.variant_a

        # Generate recommendations
        recommendations = self._generate_recommendations(
            config, p_value, effect_size, len(values_a), len(values_b)
        )

        result = TestResult(
            test_name=config.test_name,
            status=TestStatus.RUNNING,  # Will be updated when stopped
            variant_a_stats={
                "mean": float(np.mean(values_a)),
                "std": float(np.std(values_a)),
                "median": float(np.median(values_a)),
                "count": len(values_a),
            },
            variant_b_stats={
                "mean": float(np.mean(values_b)),
                "std": float(np.std(values_b)),
                "median": float(np.median(values_b)),
                "count": len(values_b),
            },
            winner=winner,
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci[0]), float(ci[1])),
            sample_size_a=len(values_a),
            sample_size_b=len(values_b),
            recommendations=recommendations,
        )

        self.results_cache[test_id] = result
        return result

    def _generate_recommendations(
        self,
        config: ABTestConfig,
        p_value: float,
        effect_size: float,
        n_a: int,
        n_b: int,
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        alpha = 1 - config.confidence_level

        if p_value < alpha:
            if abs(effect_size) >= config.mde:
                recommendations.append(
                    f"Statistically significant effect detected (d={effect_size:.3f}). "
                    f"Recommend adopting {'B' if effect_size > 0 else 'A'}."
                )
            else:
                recommendations.append(
                    f"Effect is significant but below MDE ({config.mde}). "
                    "Consider larger sample or different variant."
                )
        else:
            recommendations.append(
                "No statistically significant difference detected. "
                "Consider running longer or testing different variants."
            )

        total_n = n_a + n_b
        if total_n >= config.max_sample_size:
            recommendations.append("Maximum sample size reached. Test should be concluded.")

        return recommendations

    def stop_test(self, test_id: str) -> TestResult:
        """Stop an A/B test and finalize results."""
        result = self.analyze_test(test_id)

        if not result:
            raise ValueError(f"Cannot analyze test {test_id}")

        result.status = TestStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc).isoformat()

        # Save final results
        result_path = self.storage_path / f"{test_id}_result.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Remove from active tests
        if test_id in self.active_tests:
            del self.active_tests[test_id]

        logger.info("ab_test_completed", test_id=test_id, winner=result.winner)

        return result

    def get_all_tests(self) -> Dict[str, Dict]:
        """Get all tests with their status."""
        tests = {}

        for test_id, config in self.active_tests.items():
            tests[test_id] = {
                "config": asdict(config),
                "status": "running",
                "result": None,
            }

        # Load completed tests
        for result_file in self.storage_path.glob("*_result.json"):
            test_id = result_file.stem.replace("_result", "")
            with open(result_file) as f:
                result = json.load(f)
                tests[test_id] = {
                    "config": {},
                    "status": "completed",
                    "result": result,
                }

        return tests


class DriftDetector:
    """Detect data and model drift over time."""

    def __init__(self, reference_data: Optional[List[Annotation]] = None):
        self.reference_data = reference_data
        self.reference_stats = None

        if reference_data:
            self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, annotations: List[Annotation]) -> Dict:
        """Compute statistical profile of annotations."""
        verbs = []
        nouns = []
        caption_lengths = []
        segment_counts = []

        for ann in annotations:
            segment_counts.append(len(ann.segments))
            for seg in ann.segments:
                caption_lengths.append(len(seg.caption))
                for action in seg.actions:
                    verbs.append(action.verb)
                    nouns.append(action.noun)

        return {
            "verb_distribution": {v: verbs.count(v) for v in set(verbs)},
            "noun_distribution": {n: nouns.count(n) for n in set(nouns)},
            "avg_caption_length": np.mean(caption_lengths) if caption_lengths else 0,
            "avg_segments": np.mean(segment_counts) if segment_counts else 0,
        }

    def detect_drift(self, new_annotations: List[Annotation], threshold: float = 0.1) -> Dict:
        """Detect drift between reference and new data."""
        if not self.reference_stats:
            return {"drift_detected": False, "reason": "No reference data"}

        new_stats = self._compute_stats(new_annotations)

        drifts = []

        # Check caption length drift
        ref_len = self.reference_stats["avg_caption_length"]
        new_len = new_stats["avg_caption_length"]
        if ref_len > 0:
            length_diff = abs(new_len - ref_len) / ref_len
            if length_diff > threshold:
                drifts.append(f"Caption length drift: {length_diff:.1%}")

        # Check segment count drift
        ref_segs = self.reference_stats["avg_segments"]
        new_segs = new_stats["avg_segments"]
        if ref_segs > 0:
            seg_diff = abs(new_segs - ref_segs) / ref_segs
            if seg_diff > threshold:
                drifts.append(f"Segment count drift: {seg_diff:.1%}")

        # Check vocabulary drift (simplified)
        ref_verbs = set(self.reference_stats["verb_distribution"].keys())
        new_verbs = set(new_stats["verb_distribution"].keys())
        novel_verbs = new_verbs - ref_verbs
        if novel_verbs and len(novel_verbs) / len(ref_verbs) > threshold:
            drifts.append(f"Novel verbs: {len(novel_verbs)}")

        return {
            "drift_detected": len(drifts) > 0,
            "drift_reasons": drifts,
            "reference_stats": self.reference_stats,
            "new_stats": new_stats,
        }


class PerformanceMonitor:
    """Monitor system performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, List] = {
            "latency": [],
            "cost": [],
            "quality": [],
            "success_rate": [],
        }
        self.timestamps: List[datetime] = []

    def record(self, metrics: Dict[str, float]) -> None:
        """Record metrics."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key].pop(0)

        self.timestamps.append(datetime.now(timezone.utc))

    def get_statistics(self) -> Dict:
        """Get rolling window statistics."""
        stats = {}

        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "count": len(values),
                }

        return stats

    def check_anomalies(self, threshold_std: float = 3.0) -> List[Dict]:
        """Check for anomalous metrics."""
        anomalies = []

        for metric_name, values in self.metrics.items():
            if len(values) < 10:
                continue

            mean = np.mean(values[:-1])  # Exclude latest
            std = np.std(values[:-1])

            if std > 0:
                latest = values[-1]
                z_score = abs(latest - mean) / std

                if z_score > threshold_std:
                    anomalies.append(
                        {
                            "metric": metric_name,
                            "value": latest,
                            "z_score": float(z_score),
                            "expected_range": (float(mean - std), float(mean + std)),
                        }
                    )

        return anomalies
