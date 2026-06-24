"""Circuit breaker pattern for resilient API calls.

Prevents cascading failures by temporarily blocking requests
to failing services. Implements the standard circuit breaker
state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

Usage::

    from dvas.models.teacher.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0
    )
    breaker = CircuitBreaker("openai", config)

    # Wrap API calls with circuit breaker
    result = await breaker.call(openai_client.chat.completions.create, **params)
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import Callable, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # 正常状态，请求正常通过
    OPEN = auto()  # 熔断状态，请求立即失败
    HALF_OPEN = auto()  # 半开状态，测试服务是否恢复


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, circuit_name: str = ""):
        super().__init__(message)
        self.circuit_name = circuit_name


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        half_open_timeout: float = 5.0,
    ):
        """初始化熔断器配置.

        Args:
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断后等待恢复的时间(秒)
            half_open_max_calls: 半开状态最大测试请求数
            success_threshold: 半开状态恢复所需成功次数
            half_open_timeout: 半开状态超时时间
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.half_open_timeout = half_open_timeout


class CircuitBreaker:
    """熔断器防止级联故障.

    当服务连续失败次数超过阈值时，熔断器打开，
    后续请求立即失败，避免等待超时。经过恢复时间后，
    进入半开状态，允许少量请求测试服务是否恢复。

    Attributes:
        name: 熔断器名称(用于日志)
        config: 熔断器配置
        state: 当前状态(CLOSED/OPEN/HALF_OPEN)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """初始化熔断器.

        Args:
            name: 熔断器标识名称
            config: 熔断器配置，使用默认配置如果未提供
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_half_open_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """获取当前熔断器状态."""
        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """在熔断器保护下执行函数.

        Args:
            func: 要执行的异步函数
            *args: 函数位置参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            CircuitBreakerOpen: 熔断器打开时
            Exception: 函数执行失败时
        """
        # 检查状态
        async with self._lock:
            current_state = await self._check_and_update_state()

            if current_state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit {self.name} is OPEN - too many failures",
                    circuit_name=self.name,
                )

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        f"Circuit {self.name} is HALF_OPEN - max test calls reached",
                        circuit_name=self.name,
                    )
                self._half_open_calls += 1
                self._last_half_open_time = time.time()

        # 执行实际调用
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _check_and_update_state(self) -> CircuitState:
        """检查并更新熔断器状态.

        Returns:
            当前状态
        """
        if self._state == CircuitState.OPEN:
            # 检查是否可以进入半开状态
            if self._last_failure_time and (
                time.time() - self._last_failure_time > self.config.recovery_timeout
            ):
                logger.info(
                    "Circuit entering HALF_OPEN state",
                    circuit=self.name,
                    failure_count=self._failure_count,
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self._half_open_calls = 0
                self._last_half_open_time = time.time()

        elif self._state == CircuitState.HALF_OPEN:
            # 检查半开状态是否超时
            if self._last_half_open_time and (
                time.time() - self._last_half_open_time > self.config.half_open_timeout
            ):
                logger.warning(
                    "Circuit HALF_OPEN timeout, returning to OPEN",
                    circuit=self.name,
                )
                self._state = CircuitState.OPEN
                self._failure_count = self.config.failure_threshold
                self._half_open_calls = 0

        return self._state

    async def _on_success(self):
        """处理成功调用."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        "Circuit CLOSED - service recovered",
                        circuit=self.name,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                # 正常状态下，减少失败计数(但不低于0)
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self):
        """处理失败调用."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态失败，重新熔断
                logger.warning(
                    "Circuit OPEN - recovery test failed",
                    circuit=self.name,
                    failure_count=self._failure_count,
                )
                self._state = CircuitState.OPEN

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                # 达到失败阈值，打开熔断器
                logger.error(
                    "Circuit OPEN - failure threshold reached",
                    circuit=self.name,
                    threshold=self.config.failure_threshold,
                )
                self._state = CircuitState.OPEN

    def get_stats(self) -> dict:
        """获取熔断器统计信息.

        Returns:
            包含状态、失败次数等信息的字典
        """
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_calls": self._half_open_calls,
            "last_failure_time": self._last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
        }

    def reset(self):
        """手动重置熔断器状态为CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        logger.info("Circuit manually reset", circuit=self.name)


class CircuitBreakerRegistry:
    """熔断器注册表，管理多个服务的熔断器."""

    _breakers: dict[str, CircuitBreaker] = {}
    _configs: dict[str, CircuitBreakerConfig] = {}

    @classmethod
    def get(
        cls,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """获取或创建熔断器.

        Args:
            name: 服务名称
            config: 可选配置

        Returns:
            CircuitBreaker实例
        """
        if name not in cls._breakers:
            cfg = config or cls._configs.get(name)
            cls._breakers[name] = CircuitBreaker(name, cfg)
        return cls._breakers[name]

    @classmethod
    def register_config(cls, name: str, config: CircuitBreakerConfig) -> None:
        """注册服务配置."""
        cls._configs[name] = config

    @classmethod
    def get_all_stats(cls) -> dict:
        """获取所有熔断器统计."""
        return {name: breaker.get_stats() for name, breaker in cls._breakers.items()}

    @classmethod
    def reset_all(cls) -> None:
        """重置所有熔断器."""
        for breaker in cls._breakers.values():
            breaker.reset()
