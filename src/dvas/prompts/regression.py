"""Prompt regression testing framework.

Provides baseline comparison, golden set validation, and regression
detection for prompt template changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.prompts.registry import PromptTemplate
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class RegressionStatus(str, Enum):
    """Status of a regression test result."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GoldenAnnotation:
    """A golden (reference) annotation for regression testing."""

    id: str
    video_id: str
    expected_output: str
    expected_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "expected_output": self.expected_output,
            "expected_quality": self.expected_quality,
            "metadata": self.metadata.copy(),
        }


@dataclass
class RegressionResult:
    """Result of a single regression test."""

    test_name: str
    prompt_id: str
    status: RegressionStatus
    score: float = 0.0
    baseline_score: float = 0.0
    difference: float = 0.0
    percent_change: float = 0.0
    details: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Compute difference if not explicitly set."""
        if self.difference == 0.0 and (self.score != 0.0 or self.baseline_score != 0.0):
            self.difference = self.score - self.baseline_score
        if self.percent_change == 0.0 and self.baseline_score > 0:
            self.percent_change = (self.difference / self.baseline_score) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "prompt_id": self.prompt_id,
            "status": self.status.value,
            "score": self.score,
            "baseline_score": self.baseline_score,
            "difference": self.difference,
            "percent_change": self.percent_change,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class PromptRegressionTest:
    """Regression testing for prompt templates.

    Compares new prompt versions against a baseline using golden annotations.
    """

    def __init__(self) -> None:
        self._golden_set: Dict[str, List[GoldenAnnotation]] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._results: List[RegressionResult] = []
        self._quality_threshold: float = 0.7
        self._max_regression: float = 0.1  # Max allowed quality drop

    def add_golden_annotation(
        self, annotation: GoldenAnnotation, test_set: str = "default"
    ) -> None:
        """Add a golden annotation to a test set.

        Args:
            annotation: The golden annotation.
            test_set: Name of the test set to add to.
        """
        if test_set not in self._golden_set:
            self._golden_set[test_set] = []
        self._golden_set[test_set].append(annotation)

    def set_baseline(self, prompt_id: str, score: float, test_set: str = "default") -> None:
        """Set the baseline score for a prompt.

        Args:
            prompt_id: The prompt ID.
            score: The baseline quality score.
            test_set: Name of the test set.
        """
        key = f"{test_set}:{prompt_id}"
        if key not in self._baselines:
            self._baselines[key] = {}
        self._baselines[key]["score"] = score

    def run_test(
        self,
        prompt: PromptTemplate,
        test_set: str = "default",
        scorer: Optional[Any] = None,
    ) -> List[RegressionResult]:
        """Run regression test for a prompt against golden set.

        Args:
            prompt: The prompt template to test.
            test_set: Name of the test set to use.
            scorer: Optional scoring function. If None, uses simple comparison.

        Returns:
            List of regression results.
        """
        results: List[RegressionResult] = []

        if test_set not in self._golden_set:
            logger.warning("test_set_not_found", test_set=test_set)
            return results

        key = f"{test_set}:{prompt.id}"
        baseline = self._baselines.get(key, {}).get("score", 0.0)

        for golden in self._golden_set[test_set]:
            # In a real implementation, this would run the prompt
            # For now, use a mock score based on template similarity
            score = self._mock_score(prompt.template, golden.expected_output)

            difference = score - baseline
            percent_change = (difference / baseline * 100) if baseline > 0 else 0.0

            if score >= baseline:
                status = RegressionStatus.PASS
            elif score >= baseline - self._max_regression:
                status = RegressionStatus.WARNING
            else:
                status = RegressionStatus.FAIL

            result = RegressionResult(
                test_name=f"golden_{golden.id}",
                prompt_id=prompt.id,
                status=status,
                score=score,
                baseline_score=baseline,
                difference=difference,
                percent_change=percent_change,
                details=f"Score: {score:.3f}, Baseline: {baseline:.3f}, Diff: {difference:+.3f}",
            )
            results.append(result)
            self._results.append(result)

        return results

    def _mock_score(self, template: str, expected: str) -> float:
        """Mock scoring function for testing.

        In production, replace with actual LLM evaluation.
        """
        # Simple similarity-based score
        template_words = set(template.lower().split())
        expected_words = set(expected.lower().split())

        if not template_words or not expected_words:
            return 0.5

        intersection = len(template_words & expected_words)
        union = len(template_words | expected_words)

        if union == 0:
            return 0.5

        # Jaccard similarity as a proxy
        similarity = intersection / union
        return min(1.0, similarity + 0.3)  # Boost slightly

    def check_regression(
        self,
        prompt_id: str,
        new_score: float,
        baseline_score: float,
    ) -> RegressionResult:
        """Check if a new score represents a regression.

        Args:
            prompt_id: The prompt ID.
            new_score: The new quality score.
            baseline_score: The baseline quality score.

        Returns:
            RegressionResult with status.
        """
        difference = new_score - baseline_score
        percent_change = (difference / baseline_score * 100) if baseline_score > 0 else 0.0

        if new_score >= baseline_score:
            status = RegressionStatus.PASS
        elif new_score >= baseline_score - self._max_regression:
            status = RegressionStatus.WARNING
        else:
            status = RegressionStatus.FAIL

        result = RegressionResult(
            test_name="regression_check",
            prompt_id=prompt_id,
            status=status,
            score=new_score,
            baseline_score=baseline_score,
            difference=difference,
            percent_change=percent_change,
            details=f"Score: {new_score:.3f}, Baseline: {baseline_score:.3f}, Diff: {difference:+.3f}",
        )
        self._results.append(result)
        return result

    def validate_golden_set(self, test_set: str = "default") -> Dict[str, Any]:
        """Validate the golden set for completeness and quality.

        Args:
            test_set: Name of the test set to validate.

        Returns:
            Validation report dictionary.
        """
        if test_set not in self._golden_set:
            return {"valid": False, "error": f"Test set '{test_set}' not found"}

        annotations = self._golden_set[test_set]
        issues: List[str] = []

        for ann in annotations:
            if not ann.expected_output.strip():
                issues.append(f"Empty expected output for {ann.id}")
            if ann.expected_quality < 0 or ann.expected_quality > 1.0:
                issues.append(f"Invalid quality score for {ann.id}")

        return {
            "valid": len(issues) == 0,
            "test_set": test_set,
            "annotation_count": len(annotations),
            "issues": issues,
        }

    def get_summary(self, test_set: str = "default") -> Dict[str, Any]:
        """Get summary of regression test results.

        Args:
            test_set: Name of the test set.

        Returns:
            Summary dictionary.
        """
        results = [r for r in self._results if r.test_name.startswith("golden_")]

        if not results:
            return {"total": 0, "passed": 0, "failed": 0, "warnings": 0}

        passed = sum(1 for r in results if r.status == RegressionStatus.PASS)
        failed = sum(1 for r in results if r.status == RegressionStatus.FAIL)
        warnings = sum(1 for r in results if r.status == RegressionStatus.WARNING)

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": passed / len(results) if results else 0.0,
            "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
        }

    def set_thresholds(self, quality_threshold: float, max_regression: float) -> None:
        """Set regression detection thresholds.

        Args:
            quality_threshold: Minimum acceptable quality score.
            max_regression: Maximum allowed quality drop from baseline.
        """
        self._quality_threshold = quality_threshold
        self._max_regression = max_regression
