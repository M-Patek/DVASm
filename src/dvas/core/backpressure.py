"""Backpressure control for flow management.

Implements token bucket, leaky bucket, and adaptive rate limiting
to prevent overwhelming downstream components.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class BackpressureStrategy(Enum):
    """Strategies for handling backpressure."""

    BLOCK = "block"  # Block until capacity available
    DROP = "drop"  # Drop new items
    SHED = "shed"  # Drop oldest items
    ADAPTIVE = "adaptive"  # Adjust rate dynamically


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """Token bucket rate limiter with backpressure support.

    Usage::

        bucket = TokenBucket(rate=10.0, capacity=20.0)

        if await bucket.acquire():
            process(item)
        else:
            # Handle rate limit
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: float,
        strategy: BackpressureStrategy = BackpressureStrategy.BLOCK,
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self.strategy = strategy
        self._tokens = capacity
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: asyncio.Queue = asyncio.Queue()

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket.

        Returns True if tokens were acquired, False if timed out.
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if self.strategy == BackpressureStrategy.DROP:
                return False

            if self.strategy == BackpressureStrategy.BLOCK:
                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.rate

                if timeout is not None and wait_time > timeout:
                    return False

        # Wait outside lock to allow other operations
        try:
            await asyncio.sleep(wait_time)
            return await self.acquire(tokens, timeout)
        except asyncio.TimeoutError:
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """Get currently available tokens."""
        self._refill()
        return self._tokens

    def __enter__(self) -> TokenBucket:
        return self

    def __exit__(self, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Leaky Bucket
# ---------------------------------------------------------------------------


class LeakyBucket:
    """Leaky bucket rate limiter.

    Items are added to a bucket that leaks at a fixed rate.
    If the bucket overflows, new items are rejected.
    """

    def __init__(
        self,
        leak_rate: float,  # items per second
        capacity: int,
    ) -> None:
        self.leak_rate = leak_rate
        self.capacity = capacity
        self._water = 0.0
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def add(self, amount: float = 1.0) -> bool:
        """Add water to the bucket.

        Returns True if accepted, False if bucket would overflow.
        """
        async with self._lock:
            self._leak()

            if self._water + amount > self.capacity:
                return False

            self._water += amount
            return True

    def _leak(self) -> None:
        """Leak water based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._water = max(0.0, self._water - elapsed * self.leak_rate)
        self._last_update = now

    @property
    def current_level(self) -> float:
        """Get current water level."""
        self._leak()
        return self._water


# ---------------------------------------------------------------------------
# Adaptive Rate Limiter
# ---------------------------------------------------------------------------


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on downstream feedback.

    Monitors downstream latency and adjusts rate accordingly.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 100.0,
        target_latency_ms: float = 100.0,
        adaptation_factor: float = 0.1,
    ) -> None:
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_latency_ms = target_latency_ms
        self.adaptation_factor = adaptation_factor
        self._bucket = TokenBucket(rate=initial_rate, capacity=initial_rate * 2)
        self._latency_history: list[float] = []
        self._history_size = 10

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to proceed."""
        return await self._bucket.acquire(timeout=timeout)

    def report_latency(self, latency_ms: float) -> None:
        """Report observed latency for rate adaptation."""
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._history_size:
            self._latency_history.pop(0)

        if len(self._latency_history) >= 3:
            self._adapt_rate()

    def _adapt_rate(self) -> None:
        """Adjust rate based on observed latency."""
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        error = (avg_latency - self.target_latency_ms) / self.target_latency_ms

        # Adjust rate inversely proportional to latency error
        adjustment = -error * self.adaptation_factor
        new_rate = self.rate * (1 + adjustment)
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))

        if abs(new_rate - self.rate) / self.rate > 0.05:  # 5% threshold
            old_rate = self.rate
            self.rate = new_rate
            self._bucket.rate = new_rate
            self._bucket.capacity = new_rate * 2
            logger.info(
                "rate_adapted",
                old_rate=old_rate,
                new_rate=new_rate,
                avg_latency_ms=avg_latency,
                target_latency_ms=self.target_latency_ms,
            )


# ---------------------------------------------------------------------------
# Backpressure Controller
# ---------------------------------------------------------------------------


class BackpressureController:
    """High-level backpressure controller for pipeline stages.

    Combines multiple strategies to provide robust flow control.
    """

    def __init__(
        self,
        max_inflight: int = 100,
        rate_limit: Optional[float] = None,
        strategy: BackpressureStrategy = BackpressureStrategy.BLOCK,
    ) -> None:
        self.max_inflight = max_inflight
        self.strategy = strategy
        self._inflight = 0
        self._semaphore = asyncio.Semaphore(max_inflight)
        self._rate_limiter: Optional[TokenBucket] = None

        if rate_limit:
            self._rate_limiter = TokenBucket(
                rate=rate_limit,
                capacity=rate_limit * 2,
                strategy=strategy,
            )

        self._total_processed = 0
        self._total_dropped = 0
        self._total_blocked = 0

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to process an item.

        Returns True if the item should be processed, False if it should be dropped.
        """
        # Check rate limit first
        if self._rate_limiter:
            if not await self._rate_limiter.acquire(timeout=timeout):
                self._total_dropped += 1
                return False

        # Check concurrency limit
        if self.strategy == BackpressureStrategy.BLOCK:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                self._inflight += 1
                return True
            except asyncio.TimeoutError:
                self._total_blocked += 1
                return False
        else:
            if self._semaphore.locked():
                self._total_dropped += 1
                return False
            await self._semaphore.acquire()
            self._inflight += 1
            return True

    def release(self) -> None:
        """Release a slot after processing is complete."""
        self._semaphore.release()
        self._inflight -= 1
        self._total_processed += 1

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def available(self) -> int:
        return self.max_inflight - self._inflight

    @property
    def utilization(self) -> float:
        return self._inflight / self.max_inflight if self.max_inflight > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "inflight": self._inflight,
            "max_inflight": self.max_inflight,
            "utilization": self.utilization,
            "total_processed": self._total_processed,
            "total_dropped": self._total_dropped,
            "total_blocked": self._total_blocked,
        }


# ---------------------------------------------------------------------------
# Context manager for automatic backpressure
# ---------------------------------------------------------------------------


class ControlledFlow:
    """Context manager for automatic backpressure control.

    Usage::

        controller = BackpressureController(max_inflight=50)

        async with ControlledFlow(controller) as flow:
            if flow.allowed:
                await process(item)
    """

    def __init__(self, controller: BackpressureController) -> None:
        self.controller = controller
        self.allowed = False

    async def __aenter__(self) -> ControlledFlow:
        self.allowed = await self.controller.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        if self.allowed:
            self.controller.release()
