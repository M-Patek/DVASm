"""Tests for validation gate."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies before importing student modules
sys.modules["datasets"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.amp"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.models.student.benchmark import BenchmarkResult, RegressionReport
from dvas.models.student.validation_gate import (
    DEFAULT_THRESHOLDS,
    ValidationGate,
    ValidationGateResult,
    run_validation_cli,
)


class TestValidationGateResult:
    """Test ValidationGateResult dataclass."""

    def test_passed_gate(self):
        result = ValidationGateResult(
            passed=True,
            gate_name="test_gate",
            metrics={"bleu": 0.5},
        )
        assert result.passed is True
        assert result.gate_name == "test_gate"
        assert result.failures == []

    def test_failed_gate(self):
        result = ValidationGateResult(
            passed=False,
            gate_name="test_gate",
            metrics={"bleu": 0.1},
            failures=["bleu: 0.1000 < threshold 0.3000"],
        )
        assert result.passed is False
        assert len(result.failures) == 1

    def test_to_dict(self):
        result = ValidationGateResult(
            passed=True,
            gate_name="benchmark_quality",
            metrics={"bleu": 0.5, "rouge_l": 0.4},
            failures=[],
            warnings=["metric 'cider' not found"],
        )
        data = result.to_dict()
        assert data["passed"] is True
        assert data["gate_name"] == "benchmark_quality"
        assert data["metrics"]["bleu"] == 0.5
        assert data["warnings"][0] == "metric 'cider' not found"

    def test_print_summary_does_not_raise(self, capsys):
        result = ValidationGateResult(
            passed=False,
            gate_name="test",
            metrics={"bleu": 0.1},
            failures=["bleu too low"],
            warnings=["missing metric"],
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "bleu too low" in captured.out


class TestValidationGate:
    """Test ValidationGate class."""

    def test_default_thresholds(self):
        gate = ValidationGate()
        assert gate.thresholds == DEFAULT_THRESHOLDS
        assert gate.thresholds["bleu"] == 0.30
        assert gate.thresholds["rouge_l"] == 0.25

    def test_custom_thresholds(self):
        gate = ValidationGate(thresholds={"bleu": 0.50, "custom": 1.0})
        assert gate.thresholds["bleu"] == 0.50
        assert gate.thresholds["custom"] == 1.0

    def test_check_benchmark_gate_pass(self):
        gate = ValidationGate(thresholds={"bleu": 0.30, "rouge_l": 0.25})

        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_id="model_v1",
            timestamp="2026-06-22T00:00:00",
            metrics={"bleu": 0.50, "rouge_l": 0.40},
            predictions=["pred"],
            references=["ref"],
        )

        result = gate.check_benchmark_gate(benchmark_result)

        assert result.passed is True
        assert result.gate_name == "benchmark_quality"
        assert len(result.failures) == 0

    def test_check_benchmark_gate_fail(self):
        gate = ValidationGate(thresholds={"bleu": 0.30, "rouge_l": 0.25})

        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_id="model_v1",
            timestamp="2026-06-22T00:00:00",
            metrics={"bleu": 0.10, "rouge_l": 0.40},
            predictions=["pred"],
            references=["ref"],
        )

        result = gate.check_benchmark_gate(benchmark_result)

        assert result.passed is False
        assert len(result.failures) == 1
        assert "bleu" in result.failures[0]

    def test_check_benchmark_gate_missing_metric(self):
        gate = ValidationGate(thresholds={"bleu": 0.30, "cider": 1.0})

        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_id="model_v1",
            timestamp="2026-06-22T00:00:00",
            metrics={"bleu": 0.50},
            predictions=["pred"],
            references=["ref"],
        )

        result = gate.check_benchmark_gate(benchmark_result)

        assert result.passed is True  # bleu passes
        assert len(result.warnings) == 1
        assert "cider" in result.warnings[0]

    def test_check_regression_gate_pass(self):
        gate = ValidationGate()

        report = RegressionReport(
            benchmark_name="test",
            current_model="v2",
            baseline_model="v1",
            current_metrics={"bleu": 0.35},
            baseline_metrics={"bleu": 0.34},
            metric_changes={"bleu": 0.029},
            significant_regressions=[],
            significant_improvements=[],
            threshold=0.05,
        )

        result = gate.check_regression_gate(report)

        assert result.passed is True
        assert result.gate_name == "regression"

    def test_check_regression_gate_fail(self):
        gate = ValidationGate()

        report = RegressionReport(
            benchmark_name="test",
            current_model="v2",
            baseline_model="v1",
            current_metrics={"bleu": 0.30},
            baseline_metrics={"bleu": 0.34},
            metric_changes={"bleu": -0.118},
            significant_regressions=["bleu"],
            significant_improvements=[],
            threshold=0.05,
        )

        result = gate.check_regression_gate(report)

        assert result.passed is False
        assert len(result.failures) == 1
        assert "bleu" in result.failures[0]

    def test_save_validation_report(self):
        import tempfile

        gate = ValidationGate()

        results = [
            ValidationGateResult(
                passed=True,
                gate_name="gate1",
                metrics={"bleu": 0.5},
            ),
            ValidationGateResult(
                passed=False,
                gate_name="gate2",
                metrics={"rouge": 0.1},
                failures=["rouge too low"],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"
            gate.save_validation_report(results, output_path)

            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["overall_passed"] is False
            assert len(data["gates"]) == 2
            assert data["gates"][0]["passed"] is True
            assert data["gates"][1]["passed"] is False


class TestRunValidationCli:
    """Test CLI validation runner."""

    @patch("dvas.models.student.validation_gate.ValidationGate")
    def test_cli_pass(self, mock_gate_class):
        mock_gate = MagicMock()
        mock_gate.run_full_validation.return_value = [
            ValidationGateResult(
                passed=True,
                gate_name="benchmark_quality",
                metrics={"bleu": 0.5},
            )
        ]
        mock_gate.thresholds = DEFAULT_THRESHOLDS
        mock_gate_class.return_value = mock_gate

        exit_code = run_validation_cli(
            model_path="/tmp/model",
            benchmark_name="student_regression",
        )

        assert exit_code == 0

    @patch("dvas.models.student.validation_gate.ValidationGate")
    def test_cli_fail(self, mock_gate_class):
        mock_gate = MagicMock()
        mock_gate.run_full_validation.return_value = [
            ValidationGateResult(
                passed=False,
                gate_name="benchmark_quality",
                metrics={"bleu": 0.1},
                failures=["bleu too low"],
            )
        ]
        mock_gate.thresholds = DEFAULT_THRESHOLDS
        mock_gate_class.return_value = mock_gate

        exit_code = run_validation_cli(
            model_path="/tmp/model",
            benchmark_name="student_regression",
        )

        assert exit_code == 1
