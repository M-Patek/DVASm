"""Circuit breaker pattern for fault tolerance.

Prevents cascading failures by temporarily rejecting requests
to a failing service, allowing it time to recover.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5  # failures before opening
    recovery_timeout: float = 30.0  # seconds before half-open
    half_open_max_calls: int = 3  # test calls in half-open state
    success_threshold: int = 2  # successes needed to close
    timeout: float = 10.0  # request timeout in seconds


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Usage::

        breaker = CircuitBreaker("api_service", CircuitBreakerConfig())

        try:
            result = await breaker.call(api_function, arg1, arg2)
        except CircuitBreakerOpen:
            # Circuit is open, service unavailable
            pass
    """

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._stats: Dict[str, int] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpen: If the circuit is open
            CircuitBreakerTimeout: If the call times out
        """
        async with self._lock:
            await self._update_state()

            if self._state == CircuitState.OPEN:
                self._stats["rejected_calls"] += 1
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Last failure: {self._last_failure_time}"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats["rejected_calls"] += 1
                    raise CircuitBreakerOpen(
                        f"Circuit '{self.name}' is HALF_OPEN with max calls reached"
                    )
                self._half_open_calls += 1

        # Execute the call outside the lock
        self._stats["total_calls"] += 1

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout,
            )
            await self._record_success()
            return result

        except asyncio.TimeoutError:
            await self._record_failure()
            raise CircuitBreakerTimeout(
                f"Circuit '{self.name}' timed out after {self.config.timeout}s"
            )

        except Exception:
            await self._record_failure()
            raise

    async def _update_state(self) -> None:
        """Update circuit state based on time and failures."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time > self.config.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                self._stats["state_changes"] += 1
                logger.info(
                    "circuit_half_open",
                    name=self.name,
                    after_seconds=self.config.recovery_timeout,
                )

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._success_count += 1
            self._stats["successful_calls"] += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._stats["state_changes"] += 1
                    logger.info("circuit_closed", name=self.name)

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._stats["failed_calls"] += 1

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._stats["state_changes"] += 1
                logger.warning(
                    "circuit_opened",
                    name=self.name,
                    reason="half_open_failure",
                )
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._stats["state_changes"] += 1
                logger.warning(
                    "circuit_opened",
                    name=self.name,
                    reason="failure_threshold_reached",
                    failures=self._failure_count,
                )

    @property
    def stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            **self._stats,
        }

    def __repr__(self) -> str:
        return f"CircuitBreaker(name={self.name}, state={self._state.name})"


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _breakers: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_or_create(
        cls,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(name, config or CircuitBreakerConfig())
        return cls._breakers[name]

    @classmethod
    def get(cls, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return cls._breakers.get(name)

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in cls._breakers:
            del cls._breakers[name]
            return True
        return False

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.stats for name, breaker in cls._breakers.items()}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open."""

    pass


class CircuitBreakerTimeout(Exception):
    """Raised when a call through the circuit breaker times out."""

    pass


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable:
    """Decorator that wraps a function with a circuit breaker.

    Usage::

        @circuit_breaker("api_service")
        async def call_api() -> dict:
            return await fetch_data()
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable:
        breaker = CircuitBreakerRegistry.get_or_create(name, config)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call(func, *args, **kwargs)

        wrapper.__circuit_breaker__ = breaker  # type: ignore
        return wrapper

    return decorator
