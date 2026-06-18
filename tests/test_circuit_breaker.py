"""Tests for circuit breaker pattern."""

import asyncio

import pytest

from dvas.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitBreakerTimeout,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreakerConfig:
    def test_default_config(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.timeout == 10.0

    def test_custom_config(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=5,
            success_threshold=1,
            timeout=5.0,
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 1
        assert config.timeout == 5.0


class TestCircuitBreaker:
    @pytest.fixture
    def config(self):
        return CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
            success_threshold=1,
            timeout=1.0,
        )

    @pytest.fixture
    def breaker(self, config):
        return CircuitBreaker("test_service", config)

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.stats["total_calls"] == 1
        assert breaker.stats["successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_failure_opens_circuit(self, breaker):
        async def fail_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        assert breaker.state == CircuitState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects(self, breaker):
        async def fail_func():
            raise ValueError("test error")

        # Trigger failure threshold
        for _ in range(2):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(CircuitBreakerOpen):
            await breaker.call(fail_func)

    @pytest.mark.asyncio
    async def test_timeout(self, breaker):
        async def slow_func():
            await asyncio.sleep(2.0)

        with pytest.raises(CircuitBreakerTimeout):
            await breaker.call(slow_func)

    @pytest.mark.asyncio
    async def test_half_open_recovery(self, breaker):
        async def fail_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(2):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should transition to half-open
        async def success_func():
            return "ok"

        result = await breaker.call(success_func)
        assert result == "ok"
        # After success, circuit should be closed
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stats(self, breaker):
        async def success_func():
            return "ok"

        async def fail_func():
            raise ValueError("error")

        await breaker.call(success_func)

        try:
            await breaker.call(fail_func)
        except ValueError:
            pass

        stats = breaker.stats
        assert stats["name"] == "test_service"
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1

    def test_repr(self, breaker):
        assert repr(breaker) == "CircuitBreaker(name=test_service, state=CLOSED)"


class TestCircuitBreakerRegistry:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        # Clean registry before each test
        CircuitBreakerRegistry._breakers.clear()
        yield
        CircuitBreakerRegistry._breakers.clear()

    def test_get_or_create(self):
        breaker = CircuitBreakerRegistry.get_or_create("service1")
        assert breaker.name == "service1"
        assert breaker.state == CircuitState.CLOSED

        # Second call should return same instance
        breaker2 = CircuitBreakerRegistry.get_or_create("service1")
        assert breaker2 is breaker

    def test_get_existing(self):
        breaker = CircuitBreakerRegistry.get_or_create("service2")
        retrieved = CircuitBreakerRegistry.get("service2")
        assert retrieved is breaker

    def test_get_nonexistent(self):
        assert CircuitBreakerRegistry.get("nonexistent") is None

    def test_remove(self):
        CircuitBreakerRegistry.get_or_create("service3")
        assert CircuitBreakerRegistry.remove("service3") is True
        assert CircuitBreakerRegistry.get("service3") is None
        assert CircuitBreakerRegistry.remove("service3") is False

    def test_get_all_stats(self):
        CircuitBreakerRegistry.get_or_create("svc1")
        CircuitBreakerRegistry.get_or_create("svc2")
        stats = CircuitBreakerRegistry.get_all_stats()
        assert len(stats) == 2
        assert "svc1" in stats
        assert "svc2" in stats


class TestCircuitBreakerDecorator:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        CircuitBreakerRegistry._breakers.clear()
        yield
        CircuitBreakerRegistry._breakers.clear()

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)

        @circuit_breaker("decorated_service", config)
        async def my_function():
            return "decorated_result"

        result = await my_function()
        assert result == "decorated_result"

    @pytest.mark.asyncio
    async def test_decorator_circuit_open(self):
        config = CircuitBreakerConfig(failure_threshold=1, timeout=1.0)

        @circuit_breaker("failing_service", config)
        async def fail_function():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await fail_function()

        with pytest.raises(CircuitBreakerOpen):
            await fail_function()
