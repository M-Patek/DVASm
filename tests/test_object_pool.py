"""Tests for object pool implementation."""

import numpy as np
import pytest

from dvas.core.object_pool import (
    NumpyArrayPool,
    ObjectPool,
    PoolRegistry,
    PooledObject,
)


class TestObjectPool:
    @pytest.fixture
    def pool(self):
        return ObjectPool(
            factory=lambda: {"data": []},
            reset=lambda obj: obj["data"].clear(),
            max_size=3,
            name="test_pool",
        )

    def test_acquire_creates_new(self, pool):
        obj = pool.acquire()
        assert obj == {"data": []}
        assert pool.stats.misses == 1
        assert pool.stats.created == 1

    def test_acquire_reuses_pooled(self, pool):
        obj1 = pool.acquire()
        pool.release(obj1)

        obj2 = pool.acquire()
        assert obj2 is obj1  # Same object reused
        assert pool.stats.hits == 1

    def test_release_returns_to_pool(self, pool):
        obj = pool.acquire()
        pool.release(obj)
        assert pool.stats.released == 1
        assert pool.stats.current_size == 1

    def test_release_evicts_when_full(self, pool):
        # Fill pool beyond capacity
        objs = [pool.acquire() for _ in range(5)]
        for obj in objs:
            pool.release(obj)

        assert pool.stats.evicted > 0

    def test_clear(self, pool):
        obj = pool.acquire()
        pool.release(obj)
        assert pool.stats.current_size == 1

        cleared = pool.clear()
        assert cleared == 1
        assert pool.stats.current_size == 0

    def test_stats(self, pool):
        obj1 = pool.acquire()
        pool.release(obj1)

        obj2 = pool.acquire()
        pool.release(obj2)

        stats = pool.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.created == 1
        assert stats.released == 2
        assert stats.current_size == 1
        assert stats.total_requests == 2
        assert stats.hit_rate == 0.5

    def test_context_manager(self):
        with ObjectPool(factory=lambda: [], max_size=2) as pool:
            obj = pool.acquire()
            assert obj == []
            pool.release(obj)

    def test_in_use_tracking(self):
        pool = ObjectPool(factory=lambda: "x", max_size=2)
        assert pool._in_use == 0

        obj = pool.acquire()
        assert pool._in_use == 1

        pool.release(obj)
        assert pool._in_use == 0


class TestNumpyArrayPool:
    def test_create_and_acquire(self):
        pool = NumpyArrayPool(shape=(64, 64, 3), dtype=np.uint8, max_size=2)
        arr = pool.acquire()

        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8
        assert np.all(arr == 0)

    def test_reset_zeros_array(self):
        pool = NumpyArrayPool(shape=(10, 10), dtype=np.float32, max_size=2)
        arr = pool.acquire()
        arr.fill(1.0)
        pool.release(arr)

        arr2 = pool.acquire()
        assert np.all(arr2 == 0)  # Should be reset to zeros

    def test_shape_mismatch_creates_new(self):
        pool = NumpyArrayPool(shape=(10, 10), dtype=np.float32, max_size=2)
        # Manually put wrong-shaped array into pool
        pool._pool.append(np.zeros((5, 5), dtype=np.float32))

        arr = pool.acquire()
        assert arr.shape == (10, 10)  # Should create correct shape

    def test_dtype_mismatch_creates_new(self):
        pool = NumpyArrayPool(shape=(10, 10), dtype=np.float32, max_size=2)
        # Manually put wrong dtype array into pool
        pool._pool.append(np.zeros((10, 10), dtype=np.int32))

        arr = pool.acquire()
        assert arr.dtype == np.float32


class TestPoolRegistry:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        PoolRegistry.clear_all()
        yield
        PoolRegistry.clear_all()

    def test_register_and_get(self):
        pool = ObjectPool(factory=lambda: [], name="test")
        PoolRegistry.register("my_pool", pool)

        retrieved = PoolRegistry.get("my_pool")
        assert retrieved is pool

    def test_get_nonexistent(self):
        assert PoolRegistry.get("nonexistent") is None

    def test_create_numpy_pool(self):
        pool = PoolRegistry.create_numpy_pool("np_pool", (100, 100, 3), np.uint8, max_size=5)
        assert isinstance(pool, NumpyArrayPool)
        assert pool.shape == (100, 100, 3)

        retrieved = PoolRegistry.get("np_pool")
        assert retrieved is pool

    def test_get_stats(self):
        pool1 = ObjectPool(factory=lambda: [], name="p1")
        pool2 = ObjectPool(factory=lambda: [], name="p2")
        PoolRegistry.register("p1", pool1)
        PoolRegistry.register("p2", pool2)

        stats = PoolRegistry.get_stats()
        assert "p1" in stats
        assert "p2" in stats

    def test_clear_all(self):
        PoolRegistry.register("p1", ObjectPool(factory=lambda: []))
        PoolRegistry.clear_all()
        assert PoolRegistry.get("p1") is None


class TestPooledObject:
    def test_context_manager_acquire_release(self):
        pool = ObjectPool(factory=lambda: [1, 2, 3], max_size=2)

        with PooledObject(pool) as obj:
            assert obj == [1, 2, 3]

        # After context, object should be back in pool
        assert pool.stats.released == 1
