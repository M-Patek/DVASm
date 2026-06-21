"""Tests for prompt regression testing."""

import pytest

from dvas.prompts.regression import (
    GoldenAnnotation,
    PromptRegressionTest,
    RegressionResult,
    RegressionStatus,
)
from dvas.prompts.registry import PromptDomain, PromptMetadata, PromptTemplate


class TestGoldenAnnotation:
    """Test suite for GoldenAnnotation."""

    def test_creation(self):
        """Test creating a golden annotation."""
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="The person picks up the cup.",
            expected_quality=0.9,
        )
        assert ann.id == "gold_1"
        assert ann.expected_quality == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="Output",
        )
        d = ann.to_dict()
        assert d["id"] == "gold_1"
        assert d["expected_output"] == "Output"


class TestRegressionResult:
    """Test suite for RegressionResult."""

    def test_creation(self):
        """Test creating a regression result."""
        result = RegressionResult(
            test_name="test_1",
            prompt_id="prompt_1",
            status=RegressionStatus.PASS,
            score=0.85,
            baseline_score=0.80,
        )
        assert result.status == RegressionStatus.PASS
        assert result.difference == pytest.approx(0.05)

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = RegressionResult(
            test_name="test_1",
            prompt_id="prompt_1",
            status=RegressionStatus.PASS,
            score=0.85,
            baseline_score=0.80,
        )
        d = result.to_dict()
        assert d["status"] == "pass"
        assert d["score"] == 0.85


class TestPromptRegressionTest:
    """Test suite for PromptRegressionTest."""

    def test_add_golden_annotation(self):
        """Test adding golden annotations."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="Expected output",
        )
        test.add_golden_annotation(ann, test_set="default")
        assert len(test._golden_set["default"]) == 1

    def test_set_baseline(self):
        """Test setting baseline score."""
        test = PromptRegressionTest()
        test.set_baseline("prompt_1", 0.8, test_set="default")
        assert test._baselines["default:prompt_1"]["score"] == 0.8

    def test_run_test(self):
        """Test running regression test."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="person picks up cup",
        )
        test.add_golden_annotation(ann, test_set="default")
        test.set_baseline("prompt_1", 0.5, test_set="default")

        meta = PromptMetadata(name="test", version="1.0.0", domain=PromptDomain.GENERAL)
        prompt = PromptTemplate(
            id="prompt_1",
            metadata=meta,
            template="person picks up cup and moves it",
        )

        results = test.run_test(prompt, test_set="default")
        assert len(results) > 0
        assert results[0].status in [
            RegressionStatus.PASS,
            RegressionStatus.WARNING,
            RegressionStatus.FAIL,
        ]

    def test_check_regression_pass(self):
        """Test regression check with passing score."""
        test = PromptRegressionTest()
        result = test.check_regression("prompt_1", new_score=0.85, baseline_score=0.80)
        assert result.status == RegressionStatus.PASS
        assert result.difference > 0

    def test_check_regression_warning(self):
        """Test regression check with warning."""
        test = PromptRegressionTest()
        test.set_thresholds(quality_threshold=0.7, max_regression=0.1)
        result = test.check_regression("prompt_1", new_score=0.75, baseline_score=0.80)
        assert result.status == RegressionStatus.WARNING

    def test_check_regression_fail(self):
        """Test regression check with failure."""
        test = PromptRegressionTest()
        test.set_thresholds(quality_threshold=0.7, max_regression=0.1)
        result = test.check_regression("prompt_1", new_score=0.60, baseline_score=0.80)
        assert result.status == RegressionStatus.FAIL

    def test_validate_golden_set_valid(self):
        """Test validating a valid golden set."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="Valid output",
            expected_quality=0.8,
        )
        test.add_golden_annotation(ann, test_set="default")

        report = test.validate_golden_set("default")
        assert report["valid"] is True
        assert report["annotation_count"] == 1

    def test_validate_golden_set_invalid_quality(self):
        """Test validating golden set with invalid quality."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="Output",
            expected_quality=1.5,  # Invalid
        )
        test.add_golden_annotation(ann, test_set="default")

        report = test.validate_golden_set("default")
        assert report["valid"] is False
        assert len(report["issues"]) > 0

    def test_validate_golden_set_empty_output(self):
        """Test validating golden set with empty output."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="",  # Empty
        )
        test.add_golden_annotation(ann, test_set="default")

        report = test.validate_golden_set("default")
        assert report["valid"] is False
        assert len(report["issues"]) > 0

    def test_validate_nonexistent_set(self):
        """Test validating non-existent test set."""
        test = PromptRegressionTest()
        report = test.validate_golden_set("nonexistent")
        assert report["valid"] is False
        assert "not found" in report["error"]

    def test_get_summary(self):
        """Test getting summary of results."""
        test = PromptRegressionTest()
        ann = GoldenAnnotation(
            id="gold_1",
            video_id="vid_1",
            expected_output="output",
        )
        test.add_golden_annotation(ann, test_set="default")
        test.set_baseline("p1", 0.5, test_set="default")

        meta = PromptMetadata(name="test", version="1.0.0", domain=PromptDomain.GENERAL)
        prompt = PromptTemplate(id="p1", metadata=meta, template="output")
        test.run_test(prompt, test_set="default")

        summary = test.get_summary("default")
        assert summary["total"] > 0
        assert "pass_rate" in summary

    def test_get_summary_empty(self):
        """Test getting summary with no results."""
        test = PromptRegressionTest()
        summary = test.get_summary("default")
        assert summary["total"] == 0
        assert summary["passed"] == 0

    def test_set_thresholds(self):
        """Test setting regression thresholds."""
        test = PromptRegressionTest()
        test.set_thresholds(quality_threshold=0.8, max_regression=0.05)
        assert test._quality_threshold == 0.8
        assert test._max_regression == 0.05
