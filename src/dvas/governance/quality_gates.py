"""Quality gate definitions for DVAS governance.

Provides QualityGate and QualityGateRunner for defining and running
quality thresholds across multiple dimensions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class GateStatus(str, Enum):
    """Status of a quality gate."""

    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    SKIP = "skip"


class QualityDimension(str, Enum):
    """Dimensions of quality measurement."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"


@dataclass
class QualityThreshold:
    """Threshold for a single quality dimension."""

    dimension: QualityDimension
    min_value: float = 0.0
    max_value: float = 1.0
    weight: float = 1.0
    description: str = ""

    def check(self, value: float) -> bool:
        """Check if a value meets the threshold.

        Args:
            value: The value to check.

        Returns:
            True if the value is within threshold.
        """
        return self.min_value <= value <= self.max_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class QualityGateResult:
    """Result of running a quality gate."""

    gate_id: str
    status: GateStatus
    score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "score": self.score,
            "dimension_scores": self.dimension_scores,
            "failures": self.failures,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class QualityGate:
    """A quality gate with configurable thresholds.

    Usage::

        gate = QualityGate(
            gate_id="annotation_quality",
            thresholds=[
                QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.8),
                QualityThreshold(QualityDimension.ACCURACY, min_value=0.9),
            ],
        )

        result = gate.run("ann_001", {
            QualityDimension.COMPLETENESS: 0.85,
            QualityDimension.ACCURACY: 0.92,
        })
    """

    def __init__(
        self,
        gate_id: str,
        thresholds: List[QualityThreshold],
        description: str = "",
        require_all: bool = True,
    ) -> None:
        """Initialize a quality gate.

        Args:
            gate_id: Unique gate identifier.
            thresholds: List of quality thresholds.
            description: Gate description.
            require_all: Whether all thresholds must pass.
        """
        self.gate_id = gate_id
        self.thresholds = thresholds
        self.description = description
        self.require_all = require_all

    def run(self, item_id: str, scores: Dict[QualityDimension, float]) -> QualityGateResult:
        """Run the quality gate against scores.

        Args:
            item_id: The item being checked.
            scores: Dictionary of dimension scores.

        Returns:
            QualityGateResult with pass/fail status.
        """
        dimension_scores: Dict[str, float] = {}
        failures: List[str] = []
        total_score = 0.0
        total_weight = 0.0
        all_pass = True

        for threshold in self.thresholds:
            dimension = threshold.dimension
            score = scores.get(dimension, 0.0)
            dimension_scores[dimension.value] = score

            if not threshold.check(score):
                failures.append(
                    f"{dimension.value}: {score:.4f} not in [{threshold.min_value:.4f}, {threshold.max_value:.4f}]"
                )
                all_pass = False

            total_score += score * threshold.weight
            total_weight += threshold.weight

        weighted_score = total_score / total_weight if total_weight > 0 else 0.0

        if self.require_all:
            status = GateStatus.PASS if all_pass else GateStatus.FAIL
        else:
            status = GateStatus.PASS if weighted_score >= 0.5 else GateStatus.FAIL

        return QualityGateResult(
            gate_id=self.gate_id,
            status=status,
            score=weighted_score,
            dimension_scores=dimension_scores,
            failures=failures,
            metadata={"item_id": item_id},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "description": self.description,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "require_all": self.require_all,
        }


class QualityGateRunner:
    """Runner for executing multiple quality gates.

    Usage::

        runner = QualityGateRunner()
        runner.register_gate(quality_gate)
        results = runner.run_all("ann_001", scores)
    """

    def __init__(self) -> None:
        """Initialize the quality gate runner."""
        self._gates: Dict[str, QualityGate] = {}
        self._history: List[QualityGateResult] = []
        self._history_limit = 10000

    def register_gate(self, gate: QualityGate) -> None:
        """Register a quality gate.

        Args:
            gate: The gate to register.
        """
        self._gates[gate.gate_id] = gate

    def unregister_gate(self, gate_id: str) -> bool:
        """Unregister a quality gate.

        Args:
            gate_id: The gate ID to remove.

        Returns:
            True if the gate was removed.
        """
        if gate_id in self._gates:
            del self._gates[gate_id]
            return True
        return False

    def run(
        self,
        gate_id: str,
        item_id: str,
        scores: Dict[QualityDimension, float],
    ) -> QualityGateResult:
        """Run a specific quality gate.

        Args:
            gate_id: The gate ID.
            item_id: The item to check.
            scores: Dimension scores.

        Returns:
            QualityGateResult.

        Raises:
            ValueError: If the gate is not found.
        """
        gate = self._gates.get(gate_id)
        if not gate:
            raise ValueError(f"Quality gate {gate_id} not found")

        result = gate.run(item_id, scores)
        self._add_to_history(result)
        return result

    def run_all(
        self,
        item_id: str,
        scores: Dict[QualityDimension, float],
    ) -> Dict[str, QualityGateResult]:
        """Run all registered quality gates.

        Args:
            item_id: The item to check.
            scores: Dimension scores.

        Returns:
            Dictionary mapping gate IDs to results.
        """
        results: Dict[str, QualityGateResult] = {}
        for gate_id, gate in self._gates.items():
            results[gate_id] = self.run(gate_id, item_id, scores)
        return results

    def get_history(self, gate_id: Optional[str] = None, limit: int = 100) -> List[QualityGateResult]:
        """Get evaluation history.

        Args:
            gate_id: Optional gate ID filter.
            limit: Maximum results.

        Returns:
            List of results.
        """
        if gate_id:
            results = [h for h in self._history if h.gate_id == gate_id]
        else:
            results = self._history

        return results[-limit:]

    def get_trend(self, gate_id: str, window: int = 10) -> Optional[Dict[str, Any]]:
        """Get trend for a gate.

        Args:
            gate_id: The gate ID.
            window: Number of recent results.

        Returns:
            Trend data or None.
        """
        results = self.get_history(gate_id, limit=window)
        if not results:
            return None

        scores = [r.score for r in results]
        passes = sum(1 for r in results if r.status == GateStatus.PASS)

        return {
            "gate_id": gate_id,
            "window": len(results),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": passes / len(results),
        }

    def _add_to_history(self, result: QualityGateResult) -> None:
        """Add a result to history."""
        self._history.append(result)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._history = []

    def list_gates(self) -> List[str]:
        """List all registered gate IDs."""
        return sorted(self._gates.keys())

    def get_gate(self, gate_id: str) -> Optional[QualityGate]:
        """Get a registered gate."""
        return self._gates.get(gate_id)


__all__ = [
    "QualityGate",
    "QualityGateRunner",
    "QualityGateResult",
    "QualityThreshold",
    "GateStatus",
    "QualityDimension",
]
