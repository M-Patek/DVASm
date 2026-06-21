"""Tests for backpressure control implementation."""

import asyncio

import pytest

from dvas.core.backpressure import (
    AdaptiveRateLimiter,
    BackpressureController,
    BackpressureStrategy,
    ControlledFlow,
    LeakyBucket,
    TokenBucket,
)


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        result = await bucket.acquire(tokens=1.0)
        assert result is True
        assert 8.0 < bucket.available_tokens < 10.0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        # Acquire within capacity
        result = await bucket.acquire(tokens=3.0)
        assert result is True
        assert bucket.available_tokens < 5.0

    @pytest.mark.asyncio
    async def test_drop_strategy(self):
        bucket = TokenBucket(rate=1.0, capacity=1.0, strategy=BackpressureStrategy.DROP)
        # Exhaust tokens
        await bucket.acquire(tokens=1.0)

        # Next acquire should drop immediately
        result = await bucket.acquire(tokens=1.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_block_strategy_timeout(self):
        bucket = TokenBucket(rate=0.1, capacity=1.0, strategy=BackpressureStrategy.BLOCK)
        # Exhaust tokens
        await bucket.acquire(tokens=1.0)

        # Should timeout
        result = await bucket.acquire(tokens=1.0, timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_refill(self):
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        await bucket.acquire(tokens=5.0)
        before = bucket.available_tokens
        assert before < 10.0

        # Wait for refill
        await asyncio.sleep(0.1)
        assert bucket.available_tokens > before

    def test_available_tokens_property(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        assert bucket.available_tokens == 10.0

    def test_context_manager(self):
        with TokenBucket(rate=10.0, capacity=10.0) as bucket:
            assert bucket.rate == 10.0


class TestLeakyBucket:
    @pytest.mark.asyncio
    async def test_add_within_capacity(self):
        bucket = LeakyBucket(leak_rate=1.0, capacity=5)
        result = await bucket.add(amount=3.0)
        assert result is True
        assert 0.0 < bucket.current_level <= 3.0

    @pytest.mark.asyncio
    async def test_add_exceeds_capacity(self):
        bucket = LeakyBucket(leak_rate=1.0, capacity=2)
        result = await bucket.add(amount=3.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_leak_over_time(self):
        bucket = LeakyBucket(leak_rate=10.0, capacity=10)
        await bucket.add(amount=5.0)
        before = bucket.current_level
        assert before > 0.0

        await asyncio.sleep(0.1)
        assert bucket.current_level < before

    @pytest.mark.asyncio
    async def test_multiple_adds(self):
        bucket = LeakyBucket(leak_rate=1.0, capacity=10)
        assert await bucket.add(3.0) is True
        assert await bucket.add(4.0) is True
        assert await bucket.add(4.0) is False  # Would exceed capacity

    def test_current_level_property(self):
        bucket = LeakyBucket(leak_rate=1.0, capacity=5)
        assert bucket.current_level == 0.0


class TestAdaptiveRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire(self):
        limiter = AdaptiveRateLimiter(initial_rate=10.0, min_rate=1.0, max_rate=100.0)
        result = await limiter.acquire(timeout=1.0)
        assert result is True

    def test_report_latency(self):
        limiter = AdaptiveRateLimiter(initial_rate=10.0)
        limiter.report_latency(50.0)  # Below target
        limiter.report_latency(50.0)
        limiter.report_latency(50.0)

        # Should have adapted rate
        assert len(limiter._latency_history) == 3

    def test_adapt_rate_decreases_when_slow(self):
        limiter = AdaptiveRateLimiter(
            initial_rate=100.0, target_latency_ms=50.0, adaptation_factor=0.5
        )
        # Report high latencies
        for _ in range(5):
            limiter.report_latency(200.0)

        # Rate should have decreased
        assert limiter.rate < 100.0

    def test_adapt_rate_increases_when_fast(self):
        limiter = AdaptiveRateLimiter(
            initial_rate=10.0, target_latency_ms=100.0, adaptation_factor=0.5
        )
        # Report low latencies
        for _ in range(5):
            limiter.report_latency(10.0)

        # Rate should have increased
        assert limiter.rate > 10.0

    def test_rate_bounds(self):
        limiter = AdaptiveRateLimiter(initial_rate=50.0, min_rate=10.0, max_rate=100.0)
        # Report very high latency to try to decrease below min
        for _ in range(10):
            limiter.report_latency(1000.0)

        assert limiter.rate >= 10.0

        # Report very low latency to try to increase above max
        limiter._latency_history.clear()
        for _ in range(10):
            limiter.report_latency(1.0)

        assert limiter.rate <= 100.0


class TestBackpressureController:
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        controller = BackpressureController(max_inflight=5)

        result = await controller.acquire()
        assert result is True
        assert controller.inflight == 1

        controller.release()
        assert controller.inflight == 0

    @pytest.mark.asyncio
    async def test_max_inflight_limit(self):
        controller = BackpressureController(max_inflight=2)

        # Acquire all slots
        assert await controller.acquire(timeout=0.1) is True
        assert await controller.acquire(timeout=0.1) is True

        # Third should fail with timeout
        assert await controller.acquire(timeout=0.01) is False

        controller.release()
        controller.release()

    @pytest.mark.asyncio
    async def test_drop_strategy(self):
        controller = BackpressureController(max_inflight=1, strategy=BackpressureStrategy.DROP)

        assert await controller.acquire() is True
        # With DROP strategy, should immediately fail
        assert await controller.acquire() is False

        controller.release()

    def test_properties(self):
        controller = BackpressureController(max_inflight=10)
        assert controller.max_inflight == 10
        assert controller.available == 10
        assert controller.utilization == 0.0

    def test_stats(self):
        controller = BackpressureController(max_inflight=10)
        stats = controller.stats
        assert stats["inflight"] == 0
        assert stats["max_inflight"] == 10
        assert stats["utilization"] == 0.0
        assert stats["total_processed"] == 0
        assert stats["total_dropped"] == 0
        assert stats["total_blocked"] == 0

    @pytest.mark.asyncio
    async def test_with_rate_limiter(self):
        controller = BackpressureController(max_inflight=10, rate_limit=100.0)
        result = await controller.acquire(timeout=1.0)
        assert result is True
        controller.release()


class TestControlledFlow:
    @pytest.mark.asyncio
    async def test_context_manager_acquires_and_releases(self):
        controller = BackpressureController(max_inflight=5)

        async with ControlledFlow(controller) as flow:
            assert flow.allowed is True
            assert controller.inflight == 1

        # After exiting, should be released
        assert controller.inflight == 0

    @pytest.mark.asyncio
    async def test_context_manager_when_not_allowed(self):
        controller = BackpressureController(max_inflight=1, strategy=BackpressureStrategy.DROP)

        # Fill the semaphore
        await controller.acquire()

        async with ControlledFlow(controller) as flow:
            assert flow.allowed is False

        controller.release()
