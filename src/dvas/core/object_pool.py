"""Object pool for efficient memory reuse.

Provides object pools for numpy arrays and other expensive objects
to reduce GC pressure and allocation overhead.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Generic, Optional, TypeVar

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class PoolStats:
    """Statistics for an object pool."""

    hits: int = 0
    misses: int = 0
    created: int = 0
    released: int = 0
    evicted: int = 0
    current_size: int = 0
    peak_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses


class ObjectPool(Generic[T]):
    """Thread-safe object pool for reusable objects.

    Usage::

        pool = ObjectPool(
            factory=lambda: np.zeros((448, 448, 3), dtype=np.uint8),
            reset=lambda obj: obj.fill(0),
            max_size=32,
        )

        obj = pool.acquire()
        # Use obj...
        pool.release(obj)
    """

    def __init__(
        self,
        factory: Callable[[], T],
        reset: Optional[Callable[[T], None]] = None,
        max_size: int = 32,
        name: str = "pool",
    ) -> None:
        self.factory = factory
        self.reset = reset
        self.max_size = max_size
        self.name = name
        self._pool: Deque[T] = deque()
        self._lock = threading.Lock()
        self._stats = PoolStats()
        self._in_use: int = 0

    def acquire(self) -> T:
        """Acquire an object from the pool.

        Returns a pooled object if available, otherwise creates a new one.
        """
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._stats.hits += 1
                self._stats.current_size = len(self._pool)
                self._in_use += 1

                if self.reset:
                    self.reset(obj)
                return obj

            self._stats.misses += 1
            self._stats.created += 1
            self._in_use += 1

        # Create outside lock to avoid blocking
        return self.factory()

    def release(self, obj: T) -> None:
        """Return an object to the pool.

        If the pool is full, the object is discarded.
        """
        with self._lock:
            self._in_use -= 1
            self._stats.released += 1

            if len(self._pool) < self.max_size:
                self._pool.append(obj)
                self._stats.current_size = len(self._pool)
                self._stats.peak_size = max(self._stats.peak_size, len(self._pool))
            else:
                self._stats.evicted += 1

    def clear(self) -> int:
        """Clear all objects from the pool.

        Returns the number of objects cleared.
        """
        with self._lock:
            count = len(self._pool)
            self._pool.clear()
            self._stats.current_size = 0
            return count

    @property
    def stats(self) -> PoolStats:
        """Get pool statistics."""
        with self._lock:
            return PoolStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                created=self._stats.created,
                released=self._stats.released,
                evicted=self._stats.evicted,
                current_size=len(self._pool),
                peak_size=self._stats.peak_size,
            )

    def __enter__(self) -> ObjectPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.clear()


class NumpyArrayPool(ObjectPool[np.ndarray]):
    """Specialized pool for numpy arrays with shape validation.

    Ensures all pooled arrays have the same shape and dtype.
    """

    def __init__(
        self,
        shape: tuple,
        dtype: np.dtype = np.uint8,
        max_size: int = 32,
        name: str = "numpy_pool",
    ) -> None:
        self.shape = shape
        self.dtype = dtype

        def factory() -> np.ndarray:
            return np.zeros(shape, dtype=dtype)

        def reset(arr: np.ndarray) -> None:
            arr.fill(0)

        super().__init__(factory=factory, reset=reset, max_size=max_size, name=name)

    def acquire(self) -> np.ndarray:
        """Acquire an array from the pool."""
        arr = super().acquire()
        if arr.shape != self.shape or arr.dtype != self.dtype:
            # Shape mismatch - create new array
            logger.warning(
                "pool_shape_mismatch",
                expected_shape=self.shape,
                actual_shape=arr.shape,
                expected_dtype=str(self.dtype),
                actual_dtype=str(arr.dtype),
            )
            return np.zeros(self.shape, dtype=self.dtype)
        return arr


class PoolRegistry:
    """Registry for managing multiple object pools.

    Provides centralized pool management with lifecycle hooks.
    """

    _pools: Dict[str, ObjectPool] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, pool: ObjectPool) -> None:
        """Register a pool."""
        with cls._lock:
            cls._pools[name] = pool

    @classmethod
    def get(cls, name: str) -> Optional[ObjectPool]:
        """Get a registered pool."""
        with cls._lock:
            return cls._pools.get(name)

    @classmethod
    def create_numpy_pool(
        cls,
        name: str,
        shape: tuple,
        dtype: np.dtype = np.uint8,
        max_size: int = 32,
    ) -> NumpyArrayPool:
        """Create and register a numpy array pool."""
        pool = NumpyArrayPool(shape=shape, dtype=dtype, max_size=max_size, name=name)
        cls.register(name, pool)
        return pool

    @classmethod
    def get_stats(cls) -> Dict[str, PoolStats]:
        """Get statistics for all registered pools."""
        with cls._lock:
            return {name: pool.stats for name, pool in cls._pools.items()}

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registered pools."""
        with cls._lock:
            for pool in cls._pools.values():
                pool.clear()
            cls._pools.clear()


# ---------------------------------------------------------------------------
# Context manager for automatic pool management
# ---------------------------------------------------------------------------

class PooledObject:
    """Context manager for automatic pool acquire/release.

    Usage::

        pool = NumpyArrayPool(shape=(448, 448, 3))

        with PooledObject(pool) as arr:
            # arr is acquired from pool
            process(arr)
        # arr is automatically released back to pool
    """

    def __init__(self, pool: ObjectPool[T]) -> None:
        self.pool = pool
        self.obj: Optional[T] = None

    def __enter__(self) -> T:
        self.obj = self.pool.acquire()
        return self.obj

    def __exit__(self, *args: Any) -> None:
        if self.obj is not None:
            self.pool.release(self.obj)
            self.obj = None
