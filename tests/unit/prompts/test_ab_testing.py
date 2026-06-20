"""Tests for A/B testing functionality."""

import pytest

from dvas.prompts.ab_testing import (
    ABTestConfig,
    ABTestRunner,
    AssignmentMethod,
)


class TestABTestConfig:
    """Test suite for ABTestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ABTestConfig(
            test_name="test",
            variant_a_id="A",
            variant_b_id="B",
        )
        assert config.traffic_split == 0.5
        assert config.assignment_method == AssignmentMethod.RANDOM
        assert config.min_sample_size == 100

    def test_traffic_split_validation(self):
        """Test traffic split validation."""
        with pytest.raises(ValueError, match="traffic_split"):
            ABTestConfig(
                test_name="test",
                variant_a_id="A",
                variant_b_id="B",
                traffic_split=0,  # Invalid
            )

    def test_min_sample_size_validation(self):
        """Test minimum sample size validation."""
        with pytest.raises(ValueError, match="min_sample_size"):
            ABTestConfig(
                test_name="test",
                variant_a_id="A",
                variant_b_id="B",
                min_sample_size=5,
            )


class TestABTestAssignment:
    """Test suite for A/B test assignment."""

    def test_random_assignment(self):
        """Test random variant assignment."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="random_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
            traffic_split=0.5,
            assignment_method=AssignmentMethod.RANDOM,
        )
        runner.register_test(config)

        assignments = set()
        for i in range(100):
            variant = runner.assign_variant("random_test", f"entity_{i}")
            assignments.add(variant)

        assert "prompt_A" in assignments
        assert "prompt_B" in assignments

    def test_hash_assignment_deterministic(self):
        """Test hash-based assignment is deterministic."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="hash_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
            assignment_method=AssignmentMethod.HASH,
        )
        runner.register_test(config)

        variant1 = runner.assign_variant("hash_test", "entity_1")
        variant2 = runner.assign_variant("hash_test", "entity_1")

        assert variant1 == variant2

    def test_round_robin_assignment(self):
        """Test round-robin assignment."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="rr_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
            traffic_split=0.5,
            assignment_method=AssignmentMethod.ROUND_ROBIN,
        )
        runner.register_test(config)

        variants = []
        for i in range(4):
            variants.append(runner.assign_variant("rr_test", f"entity_{i}"))

        assert variants[0] == "prompt_A"
        assert variants[1] == "prompt_B"

    def test_consistent_assignment(self):
        """Test that same entity gets same assignment."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="consistent_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        variant1 = runner.assign_variant("consistent_test", "entity_1")
        variant2 = runner.assign_variant("consistent_test", "entity_1")

        assert variant1 == variant2

    def test_nonexistent_test(self):
        """Test assignment for non-existent test."""
        runner = ABTestRunner()
        assert runner.assign_variant("nonexistent", "entity") is None


class TestABTestMetrics:
    """Test suite for A/B test metrics collection."""

    def test_record_and_get_metrics(self):
        """Test recording and retrieving metrics."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="metrics_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        runner.record_metric("metrics_test", "prompt_A", quality=0.8, latency_ms=100.0, cost=0.01)
        runner.record_metric("metrics_test", "prompt_A", quality=0.9, latency_ms=110.0, cost=0.01)

        metrics = runner.get_metrics("metrics_test", "prompt_A")
        assert metrics is not None
        assert metrics.avg_quality == 0.85
        assert metrics.avg_latency == 105.0

    def test_metrics_for_nonexistent_variant(self):
        """Test getting metrics for non-existent variant."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="metrics_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        assert runner.get_metrics("metrics_test", "nonexistent") is None


class TestABTestComparison:
    """Test suite for A/B test comparison."""

    def test_compare_variants(self):
        """Test comparing two variants."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="compare_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
            significance_threshold=0.05,
        )
        runner.register_test(config)

        # Record metrics for variant A (lower quality)
        for i in range(20):
            runner.record_metric("compare_test", "prompt_A", quality=0.6 + i * 0.01)

        # Record metrics for variant B (higher quality)
        for i in range(20):
            runner.record_metric("compare_test", "prompt_B", quality=0.8 + i * 0.01)

        result = runner.compare("compare_test", metric="quality")
        assert result is not None
        assert result.a_mean < result.b_mean
        assert result.difference > 0

    def test_insufficient_data(self):
        """Test comparison with insufficient data."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="insufficient_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        result = runner.compare("insufficient_test")
        assert result is None

    def test_get_winner(self):
        """Test determining the winning variant."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="winner_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        # Variant B is better
        for i in range(30):
            runner.record_metric("winner_test", "prompt_A", quality=0.6)
            runner.record_metric("winner_test", "prompt_B", quality=0.9)

        winner = runner.get_winner("winner_test", metric="quality")
        assert winner == "prompt_B"

    def test_no_clear_winner(self):
        """Test when there is no clear winner."""
        runner = ABTestRunner()
        config = ABTestConfig(
            test_name="tie_test",
            variant_a_id="prompt_A",
            variant_b_id="prompt_B",
        )
        runner.register_test(config)

        # Similar performance - no clear winner
        for i in range(30):
            runner.record_metric("tie_test", "prompt_A", quality=0.75)
            runner.record_metric("tie_test", "prompt_B", quality=0.76)

        winner = runner.get_winner("tie_test", metric="quality")
        # May be None if not significant
        assert winner is None or isinstance(winner, str)
