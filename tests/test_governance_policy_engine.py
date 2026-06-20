"""Tests for policy engine module.

Tests for Policy, PolicyEngine, PolicyResult, PolicyScope, PolicyStatus,
Rule, and RuleOperator.
"""

import time

import pytest

from dvas.governance.policy_engine import (
    Policy,
    PolicyEngine,
    PolicyResult,
    PolicyScope,
    PolicyStatus,
    Rule,
    RuleOperator,
)


class TestRuleOperator:
    """Test RuleOperator enum."""

    def test_operator_values(self):
        """Test operator enum values."""
        assert RuleOperator.EQ.value == "eq"
        assert RuleOperator.NE.value == "ne"
        assert RuleOperator.GT.value == "gt"
        assert RuleOperator.GTE.value == "gte"
        assert RuleOperator.LT.value == "lt"
        assert RuleOperator.LTE.value == "lte"
        assert RuleOperator.IN.value == "in"
        assert RuleOperator.NOT_IN.value == "not_in"
        assert RuleOperator.CONTAINS.value == "contains"
        assert RuleOperator.EXISTS.value == "exists"


class TestRule:
    """Test Rule dataclass."""

    def test_rule_creation(self):
        """Test creating a rule."""
        rule = Rule(
            name="min_confidence",
            field="confidence",
            operator=RuleOperator.GTE,
            value=0.8,
            description="Minimum confidence threshold",
        )
        assert rule.name == "min_confidence"
        assert rule.field == "confidence"
        assert rule.operator == RuleOperator.GTE
        assert rule.value == 0.8

    def test_rule_evaluate_eq(self):
        """Test equality rule."""
        rule = Rule("test", "field", RuleOperator.EQ, "value")
        assert rule.evaluate({"field": "value"}) is True
        assert rule.evaluate({"field": "other"}) is False

    def test_rule_evaluate_ne(self):
        """Test not-equal rule."""
        rule = Rule("test", "field", RuleOperator.NE, "value")
        assert rule.evaluate({"field": "other"}) is True
        assert rule.evaluate({"field": "value"}) is False

    def test_rule_evaluate_gt(self):
        """Test greater-than rule."""
        rule = Rule("test", "field", RuleOperator.GT, 5)
        assert rule.evaluate({"field": 10}) is True
        assert rule.evaluate({"field": 3}) is False
        assert rule.evaluate({"field": 5}) is False

    def test_rule_evaluate_gte(self):
        """Test greater-than-or-equal rule."""
        rule = Rule("test", "field", RuleOperator.GTE, 5)
        assert rule.evaluate({"field": 10}) is True
        assert rule.evaluate({"field": 5}) is True
        assert rule.evaluate({"field": 3}) is False

    def test_rule_evaluate_lt(self):
        """Test less-than rule."""
        rule = Rule("test", "field", RuleOperator.LT, 5)
        assert rule.evaluate({"field": 3}) is True
        assert rule.evaluate({"field": 10}) is False

    def test_rule_evaluate_lte(self):
        """Test less-than-or-equal rule."""
        rule = Rule("test", "field", RuleOperator.LTE, 5)
        assert rule.evaluate({"field": 3}) is True
        assert rule.evaluate({"field": 5}) is True
        assert rule.evaluate({"field": 10}) is False

    def test_rule_evaluate_in(self):
        """Test in rule."""
        rule = Rule("test", "field", RuleOperator.IN, ["a", "b", "c"])
        assert rule.evaluate({"field": "a"}) is True
        assert rule.evaluate({"field": "d"}) is False

    def test_rule_evaluate_not_in(self):
        """Test not-in rule."""
        rule = Rule("test", "field", RuleOperator.NOT_IN, ["a", "b"])
        assert rule.evaluate({"field": "c"}) is True
        assert rule.evaluate({"field": "a"}) is False

    def test_rule_evaluate_contains(self):
        """Test contains rule."""
        rule = Rule("test", "field", RuleOperator.CONTAINS, "sub")
        assert rule.evaluate({"field": "substring"}) is True
        assert rule.evaluate({"field": "other"}) is False

    def test_rule_evaluate_exists(self):
        """Test exists rule."""
        rule = Rule("test", "field", RuleOperator.EXISTS)
        assert rule.evaluate({"field": "value"}) is True
        assert rule.evaluate({"field": None}) is False
        assert rule.evaluate({"other": "value"}) is False

    def test_rule_evaluate_missing_field(self):
        """Test rule with missing field."""
        rule = Rule("test", "field", RuleOperator.EQ, "value")
        assert rule.evaluate({}) is False

    def test_rule_to_dict(self):
        """Test converting rule to dict."""
        rule = Rule("test", "field", RuleOperator.EQ, "value")
        d = rule.to_dict()
        assert d["name"] == "test"
        assert d["field"] == "field"
        assert d["operator"] == "eq"


class TestPolicy:
    """Test Policy class."""

    def test_policy_creation(self):
        """Test creating a policy."""
        policy = Policy(
            id="quality_threshold",
            name="Quality Threshold",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
            ],
        )
        assert policy.id == "quality_threshold"
        assert policy.scope == PolicyScope.ANNOTATION
        assert policy.status == PolicyStatus.ACTIVE

    def test_policy_evaluate_pass(self):
        """Test policy evaluation that passes."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
            ],
        )
        result = policy.evaluate({"confidence": 0.9})
        assert result.passed is True
        assert result.score == 1.0

    def test_policy_evaluate_fail(self):
        """Test policy evaluation that fails."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
            ],
        )
        result = policy.evaluate({"confidence": 0.5})
        assert result.passed is False
        assert len(result.rule_results) == 1
        assert result.rule_results[0]["passed"] is False

    def test_policy_inactive(self):
        """Test inactive policy."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            status=PolicyStatus.INACTIVE,
        )
        result = policy.evaluate({})
        assert result.passed is False

    def test_policy_effective_from_future(self):
        """Test policy with future effective date."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            effective_from=time.time() + 86400,
        )
        result = policy.evaluate({})
        assert result.passed is False
        assert "not yet effective" in result.message

    def test_policy_effective_until_past(self):
        """Test policy with past expiration."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            effective_until=time.time() - 86400,
        )
        result = policy.evaluate({})
        assert result.passed is False
        assert "expired" in result.message

    def test_policy_multiple_rules(self):
        """Test policy with multiple rules."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
                Rule("max_error", "error_rate", RuleOperator.LTE, 0.1),
            ],
        )
        result = policy.evaluate({"confidence": 0.9, "error_rate": 0.05})
        assert result.passed is True

    def test_policy_multiple_rules_one_fails(self):
        """Test policy where one rule fails."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("min_confidence", "confidence", RuleOperator.GTE, 0.8),
                Rule("max_error", "error_rate", RuleOperator.LTE, 0.1),
            ],
        )
        result = policy.evaluate({"confidence": 0.9, "error_rate": 0.2})
        assert result.passed is False

    def test_policy_weighted_rules(self):
        """Test policy with weighted rules."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            rules=[
                Rule("rule1", "field1", RuleOperator.EQ, 1, weight=2.0),
                Rule("rule2", "field2", RuleOperator.EQ, 2, weight=1.0),
            ],
        )
        result = policy.evaluate({"field1": 1, "field2": 2})
        assert result.passed is True
        assert result.score == 1.0

    def test_policy_bump_version(self):
        """Test bumping policy version."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
            version="1.0.0",
        )
        policy.bump_version("patch")
        assert policy.version == "1.0.1"
        policy.bump_version("minor")
        assert policy.version == "1.1.0"
        policy.bump_version("major")
        assert policy.version == "2.0.0"

    def test_policy_to_dict(self):
        """Test converting policy to dict."""
        policy = Policy(
            id="test",
            name="Test Policy",
            scope=PolicyScope.ANNOTATION,
        )
        d = policy.to_dict()
        assert d["id"] == "test"
        assert d["scope"] == "annotation"
        assert d["status"] == "active"


class TestPolicyResult:
    """Test PolicyResult dataclass."""

    def test_result_creation(self):
        """Test creating a policy result."""
        result = PolicyResult(
            policy_id="test",
            passed=True,
            score=0.95,
            message="All good",
        )
        assert result.policy_id == "test"
        assert result.passed is True
        assert result.score == 0.95

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = PolicyResult(
            policy_id="test",
            passed=True,
            score=0.95,
        )
        d = result.to_dict()
        assert d["policy_id"] == "test"
        assert d["passed"] is True
        assert d["score"] == 0.95


class TestPolicyEngine:
    """Test PolicyEngine class."""

    def test_init(self):
        """Test initialization."""
        engine = PolicyEngine()
        assert engine is not None
        assert len(engine.list_policies()) == 0

    def test_register_policy(self):
        """Test registering a policy."""
        engine = PolicyEngine()
        policy = Policy(
            id="quality",
            name="Quality",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("min", "score", RuleOperator.GTE, 0.8)],
        )
        engine.register_policy(policy)
        assert len(engine.list_policies()) == 1

    def test_unregister_policy(self):
        """Test unregistering a policy."""
        engine = PolicyEngine()
        policy = Policy(
            id="quality",
            name="Quality",
            scope=PolicyScope.ANNOTATION,
        )
        engine.register_policy(policy)
        assert engine.unregister_policy("quality") is True
        assert len(engine.list_policies()) == 0

    def test_unregister_not_found(self):
        """Test unregistering non-existent policy."""
        engine = PolicyEngine()
        assert engine.unregister_policy("nonexistent") is False

    def test_get_policy(self):
        """Test getting a policy."""
        engine = PolicyEngine()
        policy = Policy(
            id="quality",
            name="Quality",
            scope=PolicyScope.ANNOTATION,
        )
        engine.register_policy(policy)
        retrieved = engine.get_policy("quality")
        assert retrieved is not None
        assert retrieved.id == "quality"

    def test_evaluate(self):
        """Test evaluating a policy."""
        engine = PolicyEngine()
        policy = Policy(
            id="quality",
            name="Quality",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("min", "score", RuleOperator.GTE, 0.8)],
        )
        engine.register_policy(policy)
        result = engine.evaluate("quality", {"score": 0.9})
        assert result.passed is True

    def test_evaluate_not_found(self):
        """Test evaluating non-existent policy."""
        engine = PolicyEngine()
        with pytest.raises(ValueError):
            engine.evaluate("nonexistent", {})

    def test_evaluate_all(self):
        """Test evaluating all policies."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1",
            name="P1",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r1", "score", RuleOperator.GTE, 0.8)],
        ))
        engine.register_policy(Policy(
            id="p2",
            name="P2",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r2", "count", RuleOperator.GTE, 5)],
        ))
        results = engine.evaluate_all({"score": 0.9, "count": 10})
        assert len(results) == 2
        assert results["p1"].passed is True
        assert results["p2"].passed is True

    def test_evaluate_by_scope(self):
        """Test evaluating policies by scope."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1",
            name="P1",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r1", "score", RuleOperator.GTE, 0.8)],
        ))
        engine.register_policy(Policy(
            id="p2",
            name="P2",
            scope=PolicyScope.GLOBAL,
            rules=[Rule("r2", "count", RuleOperator.GTE, 5)],
        ))
        results = engine.evaluate_by_scope(PolicyScope.ANNOTATION, {"score": 0.9})
        assert len(results) == 1
        assert "p1" in results

    def test_custom_evaluator(self):
        """Test custom evaluator."""
        engine = PolicyEngine()

        def custom_eval(data):
            return PolicyResult(
                policy_id="custom",
                passed=data.get("ok", False),
            )

        engine.add_custom_evaluator("custom", custom_eval)
        result = engine.evaluate("custom", {"ok": True})
        assert result.passed is True

    def test_compliance_report(self):
        """Test compliance report."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1",
            name="P1",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r1", "score", RuleOperator.GTE, 0.8)],
        ))
        engine.evaluate("p1", {"score": 0.9})
        report = engine.get_compliance_report()
        assert report["total_evaluations"] == 1
        assert report["passed"] == 1

    def test_policy_history(self):
        """Test policy evaluation history."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1",
            name="P1",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r1", "score", RuleOperator.GTE, 0.8)],
        ))
        engine.evaluate("p1", {"score": 0.9})
        engine.evaluate("p1", {"score": 0.5})
        history = engine.get_policy_history("p1")
        assert len(history) == 2

    def test_clear_history(self):
        """Test clearing history."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1",
            name="P1",
            scope=PolicyScope.ANNOTATION,
            rules=[Rule("r1", "score", RuleOperator.GTE, 0.8)],
        ))
        engine.evaluate("p1", {"score": 0.9})
        engine.clear_history()
        history = engine.get_policy_history("p1")
        assert len(history) == 0

    def test_list_policies_filter_scope(self):
        """Test listing policies with scope filter."""
        engine = PolicyEngine()
        engine.register_policy(Policy(
            id="p1", name="P1", scope=PolicyScope.ANNOTATION,
        ))
        engine.register_policy(Policy(
            id="p2", name="P2", scope=PolicyScope.GLOBAL,
        ))
        annotation_policies = engine.list_policies(PolicyScope.ANNOTATION)
        assert len(annotation_policies) == 1
        assert annotation_policies[0].id == "p1"


class TestPolicyScope:
    """Test PolicyScope enum."""

    def test_scope_values(self):
        """Test scope values."""
        assert PolicyScope.GLOBAL.value == "global"
        assert PolicyScope.PROJECT.value == "project"
        assert PolicyScope.USER.value == "user"
        assert PolicyScope.ANNOTATION.value == "annotation"
        assert PolicyScope.SYSTEM.value == "system"


class TestPolicyStatus:
    """Test PolicyStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert PolicyStatus.ACTIVE.value == "active"
        assert PolicyStatus.INACTIVE.value == "inactive"
        assert PolicyStatus.DEPRECATED.value == "deprecated"
        assert PolicyStatus.DRAFT.value == "draft"
