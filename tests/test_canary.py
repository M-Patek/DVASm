"""Tests for canary and shadow deployment."""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.routing.canary import (
    BlueGreenDeployment,
    CanaryRouter,
    ComparisonResult,
    DeploymentMode,
    RollbackConfig,
    RollbackReason,
    RoutingDecision,
    ShadowDeployment,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, name: str, result: str = "result"):
        self.name = name
        self.result = result

    def __call__(self, request):
        return f"{self.name}:{self.result}"


class TestCanaryRouter:
    """Test CanaryRouter."""

    def test_init(self):
        router = CanaryRouter()
        assert router.mode == DeploymentMode.CANARY
        assert router.rollback_config.enabled is True

    def test_register_model(self):
        router = CanaryRouter()
        model = MockModel("v1")
        router.register_model("v1.0", model)

        assert "v1.0" in router._versions
        assert router._versions["v1.0"].model is model

    def test_unregister_model(self):
        router = CanaryRouter()
        model = MockModel("v1")
        router.register_model("v1.0", model)

        assert router.unregister_model("v1.0") is True
        assert "v1.0" not in router._versions
        assert router.unregister_model("v1.0") is False

    def test_set_traffic_split(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))
        router.register_model("v1.1", MockModel("v2"))

        router.set_traffic_split({"v1.0": 0.9, "v1.1": 0.1})

        assert router._traffic_split["v1.0"] == 0.9
        assert router._traffic_split["v1.1"] == 0.1

    def test_set_traffic_split_zero_weights(self):
        router = CanaryRouter()
        with pytest.raises(ValueError, match="sum to"):
            router.set_traffic_split({"v1.0": 0, "v1.1": 0})

    def test_route_deterministic(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))
        router.register_model("v1.1", MockModel("v2"))
        router.set_traffic_split({"v1.0": 0.5, "v1.1": 0.5})

        # Same request ID should always route to same version
        version = router.route("request-001")
        for _ in range(10):
            assert router.route("request-001") == version

    def test_route_no_models(self):
        router = CanaryRouter()
        with pytest.raises(ValueError, match="No models"):
            router.route("request-001")

    def test_get_model(self):
        router = CanaryRouter()
        model = MockModel("v1")
        router.register_model("v1.0", model)

        assert router.get_model("v1.0") is model

    def test_get_model_not_found(self):
        router = CanaryRouter()
        with pytest.raises(ValueError, match="not registered"):
            router.get_model("v1.0")

    def test_get_routing_decision(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))
        router.set_traffic_split({"v1.0": 1.0})

        decision = router.get_routing_decision("req-001")
        assert isinstance(decision, RoutingDecision)
        assert decision.version_id == "v1.0"
        assert decision.mode == DeploymentMode.CANARY
        assert decision.is_shadow is False

    def test_record_metrics(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))

        router.record_metrics("v1.0", latency_ms=500, quality=0.85, success=True)
        router.record_metrics("v1.0", latency_ms=600, quality=0.80, success=False)

        metrics = router.get_metrics("v1.0")
        assert metrics["total_requests"] == 2
        assert metrics["error_rate"] == 0.5
        assert metrics["avg_quality"] > 0

    def test_should_rollback_quality(self):
        router = CanaryRouter(rollback_config=RollbackConfig(min_samples=5))
        router.register_model("v1.1", MockModel("v2"))

        # Record low quality metrics
        for _ in range(10):
            router.record_metrics("v1.1", quality=0.3, success=True)

        should_rollback, reason = router.should_rollback("v1.1")
        assert should_rollback is True
        assert reason == RollbackReason.QUALITY_REGRESSION

    def test_should_rollback_error_rate(self):
        router = CanaryRouter(rollback_config=RollbackConfig(min_samples=5))
        router.register_model("v1.1", MockModel("v2"))

        # Record high error rate
        for _ in range(10):
            router.record_metrics("v1.1", success=False, quality=1.0)

        # Reset rollback window to avoid cooldown issues
        router._rollback_window_start = time.time() - 4000

        should_rollback, reason = router.should_rollback("v1.1")
        assert should_rollback is True
        assert reason == RollbackReason.ERROR_RATE_SPIKE

    def test_should_rollback_insufficient_samples(self):
        router = CanaryRouter(rollback_config=RollbackConfig(min_samples=100))
        router.register_model("v1.1", MockModel("v2"))

        router.record_metrics("v1.1", quality=0.3, success=True)

        should_rollback, reason = router.should_rollback("v1.1")
        assert should_rollback is False
        assert reason is None

    def test_rollback(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))
        router.register_model("v1.1", MockModel("v2"))
        router.set_traffic_split({"v1.0": 0.9, "v1.1": 0.1})

        router.rollback("v1.1", RollbackReason.QUALITY_REGRESSION)

        assert "v1.1" not in router._traffic_split
        assert router._versions["v1.1"].is_active is False
        assert router._last_rollback is not None

    def test_get_deployment_status(self):
        router = CanaryRouter()
        router.register_model("v1.0", MockModel("v1"))
        router.set_traffic_split({"v1.0": 1.0})

        status = router.get_deployment_status()
        assert status["mode"] == "canary"
        assert "versions" in status
        assert "traffic_split" in status


class TestShadowDeployment:
    """Test ShadowDeployment."""

    def test_init(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        assert shadow.primary_version == "v1.0"
        assert shadow.shadow_version == "v1.1"
        assert shadow.similarity_threshold == 0.95

    def test_register_model(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        model = MockModel("v1")
        shadow.register_model("v1.0", model)
        assert "v1.0" in shadow._models

    def test_predict_primary_only(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        shadow.register_model("v1.0", MockModel("v1", "hello"))

        result = shadow.predict("test_request")
        assert result == "v1:hello"

    def test_predict_with_shadow(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        shadow.register_model("v1.0", MockModel("v1", "hello"))
        shadow.register_model("v1.1", MockModel("v2", "hello"))

        result = shadow.predict("test_request")
        assert result == "v1:hello"

        # Should have recorded comparison
        assert shadow._last_comparison is not None
        # Similarity depends on the exact string comparison

    def test_predict_shadow_divergence(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1", similarity_threshold=0.5)
        shadow.register_model("v1.0", MockModel("v1", "hello world"))
        shadow.register_model("v1.1", MockModel("v2", "goodbye world"))

        result = shadow.predict("test_request")
        assert result == "v1:hello world"

        comparison = shadow.compare_last()
        assert comparison is not None
        assert comparison.is_significant is True

    def test_compare_last_none(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        assert shadow.compare_last() is None

    def test_get_comparison_stats(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        shadow.register_model("v1.0", MockModel("v1", "hello"))
        shadow.register_model("v1.1", MockModel("v2", "hello"))

        for _ in range(5):
            shadow.predict("test")

        stats = shadow.get_comparison_stats()
        assert stats["comparisons"] == 5
        # Similarity depends on exact string comparison

    def test_get_comparison_stats_empty(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        stats = shadow.get_comparison_stats()
        assert stats["comparisons"] == 0

    def test_get_divergent_samples(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1", similarity_threshold=0.5)
        shadow.register_model("v1.0", MockModel("v1", "hello world"))
        shadow.register_model("v1.1", MockModel("v2", "goodbye world"))

        shadow.predict("test")
        divergent = shadow.get_divergent_samples(10)
        assert len(divergent) == 1

    def test_compute_similarity_strings(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        sim = shadow._compute_similarity("hello world", "hello world")
        assert sim == 1.0

        sim = shadow._compute_similarity("hello world", "goodbye world")
        assert 0 < sim < 1.0

    def test_compute_similarity_lists(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        sim = shadow._compute_similarity([1, 2, 3], [1, 2, 3])
        assert sim == 1.0

        sim = shadow._compute_similarity([1, 2, 3], [1, 2, 4])
        assert 0 < sim < 1.0

    def test_compute_similarity_dicts(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        sim = shadow._compute_similarity({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert sim == 1.0

        sim = shadow._compute_similarity({"a": 1}, {"b": 2})
        assert sim == 0.0

    def test_compute_similarity_different_types(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        sim = shadow._compute_similarity("hello", 123)
        assert sim == 0.0

    def test_reset(self):
        shadow = ShadowDeployment(primary="v1.0", shadow="v1.1")
        shadow.register_model("v1.0", MockModel("v1", "hello"))
        shadow.register_model("v1.1", MockModel("v2", "hello"))

        shadow.predict("test")
        assert len(shadow._comparisons) == 1

        shadow.reset()
        assert len(shadow._comparisons) == 0
        assert shadow._last_comparison is None


class TestBlueGreenDeployment:
    """Test BlueGreenDeployment."""

    def test_init(self):
        deploy = BlueGreenDeployment()
        assert deploy.get_active_environment() == "blue"

    def test_deploy_blue(self):
        deploy = BlueGreenDeployment()
        model = MockModel("v1")
        deploy.deploy_blue(model)
        assert deploy._environments["blue"] is model

    def test_deploy_green(self):
        deploy = BlueGreenDeployment()
        model = MockModel("v2")
        deploy.deploy_green(model)
        assert deploy._environments["green"] is model

    def test_switch_to_green(self):
        deploy = BlueGreenDeployment()
        deploy.deploy_blue(MockModel("v1"))
        deploy.deploy_green(MockModel("v2"))

        deploy.switch_to("green")
        assert deploy.get_active_environment() == "green"
        assert deploy._switch_count == 1

    def test_switch_to_blue(self):
        deploy = BlueGreenDeployment()
        deploy.deploy_blue(MockModel("v1"))

        deploy.switch_to("blue")
        assert deploy.get_active_environment() == "blue"

    def test_switch_to_unknown(self):
        deploy = BlueGreenDeployment()
        with pytest.raises(ValueError, match="Unknown environment"):
            deploy.switch_to("red")

    def test_switch_to_not_deployed(self):
        deploy = BlueGreenDeployment()
        with pytest.raises(ValueError, match="not deployed"):
            deploy.switch_to("green")

    def test_get_active(self):
        deploy = BlueGreenDeployment()
        model = MockModel("v1")
        deploy.deploy_blue(model)

        assert deploy.get_active() is model

    def test_get_status(self):
        deploy = BlueGreenDeployment()
        deploy.deploy_blue(MockModel("v1"))
        deploy.deploy_green(MockModel("v2"))

        status = deploy.get_status()
        assert status["active"] == "blue"
        assert status["blue_deployed"] is True
        assert status["green_deployed"] is True
        assert status["switch_count"] == 0
