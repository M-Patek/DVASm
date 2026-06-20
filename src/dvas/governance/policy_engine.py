"""Policy and rule engine for DVAS governance.

Provides Policy and PolicyEngine for evaluating governance policies
with rule-based validation, conditional policies, versioning, and
compliance reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class PolicyStatus(str, Enum):
    """Status of a policy."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    DRAFT = "draft"


class PolicyScope(str, Enum):
    """Scope of a policy."""

    GLOBAL = "global"
    PROJECT = "project"
    USER = "user"
    ANNOTATION = "annotation"
    SYSTEM = "system"


class RuleOperator(str, Enum):
    """Operators for rule conditions."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    EXISTS = "exists"


@dataclass
class Rule:
    """A single rule within a policy."""

    name: str
    field: str
    operator: RuleOperator
    value: Any = None
    description: str = ""
    weight: float = 1.0

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this rule against data.

        Args:
            data: The data to evaluate against.

        Returns:
            True if the rule passes.
        """
        actual = data.get(self.field)

        if self.operator == RuleOperator.EXISTS:
            return self.field in data and data[self.field] is not None

        if actual is None:
            return False

        if self.operator == RuleOperator.EQ:
            return actual == self.value
        elif self.operator == RuleOperator.NE:
            return actual != self.value
        elif self.operator == RuleOperator.GT:
            return actual > self.value  # type: ignore[operator]
        elif self.operator == RuleOperator.GTE:
            return actual >= self.value  # type: ignore[operator]
        elif self.operator == RuleOperator.LT:
            return actual < self.value  # type: ignore[operator]
        elif self.operator == RuleOperator.LTE:
            return actual <= self.value  # type: ignore[operator]
        elif self.operator == RuleOperator.IN:
            return actual in self.value
        elif self.operator == RuleOperator.NOT_IN:
            return actual not in self.value
        elif self.operator == RuleOperator.CONTAINS:
            return self.value in actual

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class Policy:
    """A governance policy definition.

    Usage::

        policy = Policy(
            id="quality_threshold",
            name="Quality Threshold Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
            ],
        )
    """

    id: str
    name: str
    scope: PolicyScope
    rules: List[Rule] = field(default_factory=list)
    description: str = ""
    status: PolicyStatus = PolicyStatus.ACTIVE
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    updated_at: Optional[float] = None
    effective_from: Optional[float] = None
    effective_until: Optional[float] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, data: Dict[str, Any]) -> "PolicyResult":
        """Evaluate all rules in this policy.

        Args:
            data: The data to evaluate.

        Returns:
            PolicyResult with pass/fail and details.
        """
        if self.status != PolicyStatus.ACTIVE:
            return PolicyResult(
                policy_id=self.id,
                passed=False,
                message=f"Policy is {self.status.value}",
            )

        # Check time-based conditions
        now = time.time()
        if self.effective_from and now < self.effective_from:
            return PolicyResult(
                policy_id=self.id,
                passed=False,
                message="Policy not yet effective",
            )
        if self.effective_until and now > self.effective_until:
            return PolicyResult(
                policy_id=self.id,
                passed=False,
                message="Policy expired",
            )

        rule_results: List[Dict[str, Any]] = []
        total_weight = 0.0
        passed_weight = 0.0

        for rule in self.rules:
            passed = rule.evaluate(data)
            total_weight += rule.weight
            if passed:
                passed_weight += rule.weight

            rule_results.append({
                "name": rule.name,
                "passed": passed,
                "field": rule.field,
                "operator": rule.operator.value,
                "expected": rule.value,
                "actual": data.get(rule.field),
            })

        all_passed = all(r["passed"] for r in rule_results)
        weighted_score = passed_weight / total_weight if total_weight > 0 else 1.0

        return PolicyResult(
            policy_id=self.id,
            passed=all_passed,
            score=weighted_score,
            rule_results=rule_results,
            message="All rules passed" if all_passed else "Some rules failed",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "scope": self.scope.value,
            "description": self.description,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "rules": [r.to_dict() for r in self.rules],
            "conditions": self.conditions,
            "metadata": self.metadata,
        }

    def bump_version(self, bump_type: str = "patch") -> None:
        """Bump the policy version.

        Args:
            bump_type: One of "major", "minor", "patch".
        """
        parts = [int(p) for p in self.version.split(".")]
        if bump_type == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif bump_type == "minor":
            parts[1] += 1
            parts[2] = 0
        else:
            parts[2] += 1
        self.version = f"{parts[0]}.{parts[1]}.{parts[2]}"
        self.updated_at = time.time()


@dataclass
class PolicyResult:
    """Result of evaluating a policy."""

    policy_id: str
    passed: bool
    score: float = 1.0
    rule_results: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "passed": self.passed,
            "score": self.score,
            "rule_results": self.rule_results,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class PolicyEngine:
    """Engine for evaluating governance policies.

    Usage::

        engine = PolicyEngine()
        engine.register_policy(quality_policy)

        result = engine.evaluate("quality_threshold", annotation_data)
        results = engine.evaluate_all(annotation_data)
    """

    def __init__(self) -> None:
        """Initialize the policy engine."""
        self._policies: Dict[str, Policy] = {}
        self._history: List[PolicyResult] = []
        self._history_limit = 10000
        self._custom_evaluators: Dict[str, Callable[[Dict[str, Any]], PolicyResult]] = {}

    def register_policy(self, policy: Policy) -> None:
        """Register a policy with the engine.

        Args:
            policy: The policy to register.
        """
        self._policies[policy.id] = policy
        logger.info("policy_registered", policy_id=policy.id, name=policy.name)

    def unregister_policy(self, policy_id: str) -> bool:
        """Unregister a policy.

        Args:
            policy_id: The policy ID to remove.

        Returns:
            True if the policy was removed.
        """
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a registered policy."""
        return self._policies.get(policy_id)

    def list_policies(self, scope: Optional[PolicyScope] = None) -> List[Policy]:
        """List all registered policies.

        Args:
            scope: Optional scope filter.

        Returns:
            List of policies.
        """
        policies = list(self._policies.values())
        if scope:
            policies = [p for p in policies if p.scope == scope]
        return policies

    def evaluate(self, policy_id: str, data: Dict[str, Any]) -> PolicyResult:
        """Evaluate a single policy against data.

        Args:
            policy_id: The policy ID.
            data: The data to evaluate.

        Returns:
            PolicyResult.

        Raises:
            ValueError: If the policy is not found.
        """
        if policy_id in self._custom_evaluators:
            result = self._custom_evaluators[policy_id](data)
            self._add_to_history(result)
            return result

        policy = self._policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")

        result = policy.evaluate(data)
        self._add_to_history(result)
        return result

    def evaluate_all(self, data: Dict[str, Any]) -> Dict[str, PolicyResult]:
        """Evaluate all active policies against data.

        Args:
            data: The data to evaluate.

        Returns:
            Dictionary mapping policy IDs to results.
        """
        results: Dict[str, PolicyResult] = {}
        for policy_id, policy in self._policies.items():
            if policy.status == PolicyStatus.ACTIVE:
                results[policy_id] = self.evaluate(policy_id, data)
        return results

    def evaluate_by_scope(self, scope: PolicyScope, data: Dict[str, Any]) -> Dict[str, PolicyResult]:
        """Evaluate all policies in a scope.

        Args:
            scope: The policy scope.
            data: The data to evaluate.

        Returns:
            Dictionary mapping policy IDs to results.
        """
        results: Dict[str, PolicyResult] = {}
        for policy_id, policy in self._policies.items():
            if policy.scope == scope and policy.status == PolicyStatus.ACTIVE:
                results[policy_id] = self.evaluate(policy_id, data)
        return results

    def add_custom_evaluator(
        self,
        policy_id: str,
        evaluator: Callable[[Dict[str, Any]], PolicyResult],
    ) -> None:
        """Add a custom policy evaluator.

        Args:
            policy_id: The policy ID.
            evaluator: Function that takes data and returns PolicyResult.
        """
        self._custom_evaluators[policy_id] = evaluator

    def get_compliance_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get a compliance report for all evaluated policies.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            Compliance report dictionary.
        """
        history = self._history
        if start_time:
            history = [h for h in history if h.timestamp >= start_time]
        if end_time:
            history = [h for h in history if h.timestamp <= end_time]

        total = len(history)
        passed = sum(1 for h in history if h.passed)
        failed = total - passed

        by_policy: Dict[str, Dict[str, int]] = {}
        for h in history:
            if h.policy_id not in by_policy:
                by_policy[h.policy_id] = {"passed": 0, "failed": 0}
            if h.passed:
                by_policy[h.policy_id]["passed"] += 1
            else:
                by_policy[h.policy_id]["failed"] += 1

        return {
            "total_evaluations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 1.0,
            "by_policy": by_policy,
            "period_start": start_time,
            "period_end": end_time,
        }

    def get_policy_history(self, policy_id: str, limit: int = 100) -> List[PolicyResult]:
        """Get evaluation history for a specific policy.

        Args:
            policy_id: The policy ID.
            limit: Maximum number of results.

        Returns:
            List of policy results.
        """
        results = [h for h in self._history if h.policy_id == policy_id]
        return results[-limit:]

    def _add_to_history(self, result: PolicyResult) -> None:
        """Add a result to history, maintaining size limit."""
        self._history.append(result)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._history = []


__all__ = [
    "Policy",
    "PolicyEngine",
    "PolicyResult",
    "PolicyScope",
    "PolicyStatus",
    "Rule",
    "RuleOperator",
]
