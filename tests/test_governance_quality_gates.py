"""Tests for quality gates module.

Tests for QualityGate, QualityGateRunner, QualityGateResult,
QualityThreshold, GateStatus, and QualityDimension.
"""

import pytest

from dvas.governance.quality_gates import (
    GateStatus,
    QualityDimension,
    QualityGate,
    QualityGateResult,
    QualityGateRunner,
    QualityThreshold,
)


class TestGateStatus:
    """Test GateStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert GateStatus.PASS.value == "pass"
        assert GateStatus.FAIL.value == "fail"
        assert GateStatus.PENDING.value == "pending"
        assert GateStatus.SKIP.value == "skip"


class TestQualityDimension:
    """Test QualityDimension enum."""

    def test_dimension_values(self):
        """Test dimension enum values."""
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.ACCURACY.value == "accuracy"
        assert QualityDimension.CONSISTENCY.value == "consistency"
        assert QualityDimension.VALIDITY.value == "validity"
        assert QualityDimension.TIMELINESS.value == "timeliness"


class TestQualityThreshold:
    """Test QualityThreshold dataclass."""

    def test_threshold_creation(self):
        """Test creating a threshold."""
        threshold = QualityThreshold(
            dimension=QualityDimension.COMPLETENESS,
            min_value=0.8,
            max_value=1.0,
            weight=1.0,
        )
        assert threshold.dimension == QualityDimension.COMPLETENESS
        assert threshold.min_value == 0.8
        assert threshold.max_value == 1.0

    def test_threshold_check_pass(self):
        """Test threshold check passing."""
        threshold = QualityThreshold(
            dimension=QualityDimension.COMPLETENESS,
            min_value=0.8,
            max_value=1.0,
        )
        assert threshold.check(0.9) is True
        assert threshold.check(0.8) is True
        assert threshold.check(1.0) is True

    def test_threshold_check_fail(self):
        """Test threshold check failing."""
        threshold = QualityThreshold(
            dimension=QualityDimension.COMPLETENESS,
            min_value=0.8,
            max_value=1.0,
        )
        assert threshold.check(0.7) is False
        assert threshold.check(1.1) is False

    def test_threshold_to_dict(self):
        """Test converting threshold to dict."""
        threshold = QualityThreshold(
            dimension=QualityDimension.ACCURACY,
            min_value=0.9,
        )
        d = threshold.to_dict()
        assert d["dimension"] == "accuracy"
        assert d["min_value"] == 0.9


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        result = QualityGateResult(
            gate_id="test",
            status=GateStatus.PASS,
            score=0.95,
        )
        assert result.gate_id == "test"
        assert result.status == GateStatus.PASS
        assert result.score == 0.95

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = QualityGateResult(
            gate_id="test",
            status=GateStatus.PASS,
            score=0.95,
            failures=[],
        )
        d = result.to_dict()
        assert d["gate_id"] == "test"
        assert d["status"] == "pass"
        assert d["score"] == 0.95


class TestQualityGate:
    """Test QualityGate class."""

    def test_init(self):
        """Test initialization."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        assert gate.gate_id == "quality"
        assert len(gate.thresholds) == 1

    def test_run_pass(self):
        """Test gate run passing."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        result = gate.run("item_001", {QualityDimension.COMPLETENESS: 0.9})
        assert result.status == GateStatus.PASS
        assert result.score == pytest.approx(0.9)

    def test_run_fail(self):
        """Test gate run failing."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        result = gate.run("item_001", {QualityDimension.COMPLETENESS: 0.5})
        assert result.status == GateStatus.FAIL
        assert len(result.failures) == 1

    def test_run_multiple_thresholds(self):
        """Test gate with multiple thresholds."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                QualityThreshold(QualityDimension.ACCURACY, min_value=0.9),
            ],
        )
        result = gate.run(
            "item_001",
            {
                QualityDimension.COMPLETENESS: 0.85,
                QualityDimension.ACCURACY: 0.92,
            },
        )
        assert result.status == GateStatus.PASS

    def test_run_multiple_one_fails(self):
        """Test gate where one threshold fails."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                QualityThreshold(QualityDimension.ACCURACY, min_value=0.9),
            ],
        )
        result = gate.run(
            "item_001",
            {
                QualityDimension.COMPLETENESS: 0.85,
                QualityDimension.ACCURACY: 0.5,
            },
        )
        assert result.status == GateStatus.FAIL

    def test_run_missing_dimension(self):
        """Test gate with missing dimension score."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        result = gate.run("item_001", {})
        assert result.status == GateStatus.FAIL
        assert result.score == 0.0

    def test_run_weighted(self):
        """Test gate with weighted thresholds."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.0, weight=2.0),
                QualityThreshold(QualityDimension.ACCURACY, min_value=0.0, weight=1.0),
            ],
        )
        result = gate.run(
            "item_001",
            {
                QualityDimension.COMPLETENESS: 1.0,
                QualityDimension.ACCURACY: 0.0,
            },
        )
        # Score = (1.0*2.0 + 0.0*1.0) / 3.0 = 0.667
        assert result.score == pytest.approx(0.667, rel=0.01)

    def test_run_require_all_false(self):
        """Test gate with require_all=False."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
            require_all=False,
        )
        result = gate.run("item_001", {QualityDimension.COMPLETENESS: 0.9})
        assert result.status == GateStatus.PASS

    def test_run_max_value(self):
        """Test gate with max value threshold."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.0, max_value=0.5),
            ],
        )
        result = gate.run("item_001", {QualityDimension.COMPLETENESS: 0.3})
        assert result.status == GateStatus.PASS

        result = gate.run("item_001", {QualityDimension.COMPLETENESS: 0.6})
        assert result.status == GateStatus.FAIL

    def test_to_dict(self):
        """Test converting gate to dict."""
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
            description="Quality gate",
        )
        d = gate.to_dict()
        assert d["gate_id"] == "quality"
        assert d["description"] == "Quality gate"
        assert d["require_all"] is True


class TestQualityGateRunner:
    """Test QualityGateRunner class."""

    def test_init(self):
        """Test initialization."""
        runner = QualityGateRunner()
        assert runner is not None
        assert len(runner.list_gates()) == 0

    def test_register_gate(self):
        """Test registering a gate."""
        runner = QualityGateRunner()
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        runner.register_gate(gate)
        assert len(runner.list_gates()) == 1

    def test_unregister_gate(self):
        """Test unregistering a gate."""
        runner = QualityGateRunner()
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        runner.register_gate(gate)
        assert runner.unregister_gate("quality") is True
        assert len(runner.list_gates()) == 0

    def test_unregister_gate_not_found(self):
        """Test unregistering non-existent gate."""
        runner = QualityGateRunner()
        assert runner.unregister_gate("nonexistent") is False

    def test_run(self):
        """Test running a gate."""
        runner = QualityGateRunner()
        gate = QualityGate(
            gate_id="quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
            ],
        )
        runner.register_gate(gate)
        result = runner.run("quality", "item_001", {QualityDimension.COMPLETENESS: 0.9})
        assert result.status == GateStatus.PASS

    def test_run_not_found(self):
        """Test running non-existent gate."""
        runner = QualityGateRunner()
        with pytest.raises(ValueError):
            runner.run("nonexistent", "item_001", {})

    def test_run_all(self):
        """Test running all gates."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="gate1",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                ],
            )
        )
        runner.register_gate(
            QualityGate(
                gate_id="gate2",
                thresholds=[
                    QualityThreshold(QualityDimension.ACCURACY, min_value=0.9),
                ],
            )
        )
        results = runner.run_all(
            "item_001",
            {
                QualityDimension.COMPLETENESS: 0.85,
                QualityDimension.ACCURACY: 0.92,
            },
        )
        assert len(results) == 2
        assert results["gate1"].status == GateStatus.PASS
        assert results["gate2"].status == GateStatus.PASS

    def test_get_history(self):
        """Test getting history."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="quality",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                ],
            )
        )
        runner.run("quality", "item_001", {QualityDimension.COMPLETENESS: 0.9})
        history = runner.get_history("quality")
        assert len(history) == 1

    def test_get_history_filtered(self):
        """Test getting filtered history."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="gate1",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                ],
            )
        )
        runner.run("gate1", "item_001", {QualityDimension.COMPLETENESS: 0.9})
        history = runner.get_history("nonexistent")
        assert len(history) == 0

    def test_get_trend(self):
        """Test getting trend."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="quality",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.0),
                ],
            )
        )
        runner.run("quality", "item_001", {QualityDimension.COMPLETENESS: 0.9})
        runner.run("quality", "item_002", {QualityDimension.COMPLETENESS: 0.8})
        trend = runner.get_trend("quality", window=10)
        assert trend is not None
        assert trend["gate_id"] == "quality"
        assert trend["window"] == 2
        assert "average_score" in trend
        assert "pass_rate" in trend

    def test_get_trend_empty(self):
        """Test getting trend with no history."""
        runner = QualityGateRunner()
        trend = runner.get_trend("quality")
        assert trend is None

    def test_clear_history(self):
        """Test clearing history."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="quality",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                ],
            )
        )
        runner.run("quality", "item_001", {QualityDimension.COMPLETENESS: 0.9})
        runner.clear_history()
        history = runner.get_history("quality")
        assert len(history) == 0

    def test_list_gates(self):
        """Test listing gates."""
        runner = QualityGateRunner()
        runner.register_gate(
            QualityGate(
                gate_id="gate_b",
                thresholds=[],
            )
        )
        runner.register_gate(
            QualityGate(
                gate_id="gate_a",
                thresholds=[],
            )
        )
        gates = runner.list_gates()
        assert gates == ["gate_a", "gate_b"]

    def test_get_gate(self):
        """Test getting a registered gate."""
        runner = QualityGateRunner()
        gate = QualityGate(
            gate_id="quality",
            thresholds=[],
        )
        runner.register_gate(gate)
        retrieved = runner.get_gate("quality")
        assert retrieved is not None
        assert retrieved.gate_id == "quality"

    def test_get_gate_not_found(self):
        """Test getting non-existent gate."""
        runner = QualityGateRunner()
        assert runner.get_gate("nonexistent") is None

    def test_run_all_empty(self):
        """Test running all gates when none registered."""
        runner = QualityGateRunner()
        results = runner.run_all("item_001", {})
        assert len(results) == 0

    def test_history_limit(self):
        """Test history size limit."""
        runner = QualityGateRunner()
        runner._history_limit = 5
        runner.register_gate(
            QualityGate(
                gate_id="quality",
                thresholds=[
                    QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.0),
                ],
            )
        )
        for i in range(10):
            runner.run("quality", f"item_{i}", {QualityDimension.COMPLETENESS: 0.9})

        history = runner.get_history("quality")
        assert len(history) == 5
