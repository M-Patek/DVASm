"""Tests for semantic cache with perceptual hashing."""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvas.utils.semantic_cache import (
    CacheEntry,
    CostAwareCache,
    SemanticCache,
    SemanticCacheConfig,
    compute_phash,
    cosine_similarity,
    phash_similarity,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_default_creation(self):
        entry = CacheEntry(key="test", value="hello")
        assert entry.key == "test"
        assert entry.value == "hello"
        assert entry.perceptual_hash == ""
        assert entry.embedding is None
        assert entry.cost == 0.0
        assert entry.access_count == 0
        assert entry.ttl == 3600

    def test_is_expired(self):
        import time

        # Entry with 0 TTL should be expired
        entry = CacheEntry(key="test", value="x", ttl=0)
        assert entry.is_expired is True

        # Entry with long TTL should not be expired
        entry = CacheEntry(key="test", value="x", ttl=3600)
        assert entry.is_expired is False

    def test_touch(self):
        entry = CacheEntry(key="test", value="x")
        entry.touch()
        assert entry.access_count == 1
        assert entry.last_accessed > entry.created_at


class TestSemanticCacheConfig:
    """Test SemanticCacheConfig dataclass."""

    def test_default_config(self):
        config = SemanticCacheConfig()
        assert config.default_ttl == 3600
        assert config.cost_aware_ttl is True
        assert config.similarity_threshold == 0.95
        assert config.max_entries == 10000
        assert config.enable_phash is True
        assert config.phash_size == 8

    def test_custom_config(self):
        config = SemanticCacheConfig(
            default_ttl=7200,
            similarity_threshold=0.90,
            max_entries=500,
        )
        assert config.default_ttl == 7200
        assert config.similarity_threshold == 0.90
        assert config.max_entries == 500


class TestPHashFunctions:
    """Test perceptual hash functions."""

    def test_compute_phash_grayscale(self):
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        hash_val = compute_phash(image, hash_size=8)
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    def test_compute_phash_rgb(self):
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        hash_val = compute_phash(image, hash_size=8)
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    def test_phash_similarity_identical(self):
        hash_val = "abcdef1234567890"
        similarity = phash_similarity(hash_val, hash_val)
        assert similarity == 1.0

    def test_phash_similarity_different(self):
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"
        similarity = phash_similarity(hash1, hash2)
        assert 0.0 <= similarity < 1.0

    def test_phash_similarity_empty(self):
        assert phash_similarity("", "abc") == 0.0
        assert phash_similarity("abc", "") == 0.0


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == 1.0

    def test_orthogonal_vectors(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2)) < 0.01

    def test_opposite_vectors(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2) - (-1.0)) < 0.01

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


class TestSemanticCache:
    """Test SemanticCache."""

    def test_init(self):
        cache = SemanticCache()
        assert cache._redis is None
        assert cache._memory == {}

    def test_get_ttl_cost_aware(self):
        config = SemanticCacheConfig(cost_aware_ttl=True, min_cost_premium_ttl=0.01)
        cache = SemanticCache(config)

        # Low cost should get default TTL
        assert cache._get_ttl(cost=0.005) == 3600

        # High cost should get premium TTL
        assert cache._get_ttl(cost=0.02) == 10800

    def test_get_ttl_disabled(self):
        config = SemanticCacheConfig(cost_aware_ttl=False)
        cache = SemanticCache(config)
        assert cache._get_ttl(cost=0.1) == 3600

    def test_put_and_get(self):
        cache = SemanticCache()
        cache.put("key1", "value1", cost=0.01)

        result = cache.get("key1")
        assert result == "value1"

    def test_get_not_found(self):
        cache = SemanticCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_delete(self):
        cache = SemanticCache()
        cache.put("key1", "value1")

        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_delete_not_found(self):
        cache = SemanticCache()
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        cache = SemanticCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_expired_entry(self):
        cache = SemanticCache(SemanticCacheConfig(default_ttl=0))
        cache.put("key1", "value1", ttl=0)

        # Entry should be expired
        result = cache.get("key1")
        assert result is None

    def test_similarity_search_exact_match(self):
        cache = SemanticCache(SemanticCacheConfig(similarity_threshold=0.95))
        cache.put("key1", "value1", perceptual_hash="abcdef1234567890")

        result = cache.get_similar(perceptual_hash="abcdef1234567890")
        assert result == "value1"

    def test_similarity_search_no_match(self):
        cache = SemanticCache(SemanticCacheConfig(similarity_threshold=0.95))
        cache.put("key1", "value1", perceptual_hash="0000000000000000")

        result = cache.get_similar(perceptual_hash="ffffffffffffffff")
        assert result is None

    def test_get_stats(self):
        cache = SemanticCache()
        cache.put("key1", "value1", cost=0.01)
        cache.put("key2", "value2", cost=0.02)

        stats = cache.get_stats()
        assert stats["memory_entries"] == 2
        assert stats["redis_connected"] is False
        assert "config" in stats

    def test_eviction(self):
        config = SemanticCacheConfig(max_entries=2)
        cache = SemanticCache(config)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Should have evicted oldest entry
        assert len(cache._memory) <= 2


class TestCostAwareCache:
    """Test CostAwareCache wrapper."""

    def test_init(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)
        assert cache.total_requests == 0
        assert cache.cache_hits == 0
        assert cache.total_savings == 0.0

    def test_put_and_track_cost(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)

        cache.put("key1", "value1", cost=0.02)
        assert cache.total_cost == 0.02

    def test_get_hit(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)

        cache.put("key1", "value1", cost=0.02)
        result, was_cached = cache.get("key1")

        assert result == "value1"
        assert was_cached is True
        assert cache.cache_hits == 1
        assert cache.total_requests == 1

    def test_get_miss(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)

        result, was_cached = cache.get("nonexistent")
        assert result is None
        assert was_cached is False
        assert cache.cache_hits == 0
        assert cache.total_requests == 1

    def test_hit_rate(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)

        assert cache.hit_rate == 0.0

        cache.put("key1", "value1", cost=0.02)
        cache.get("key1")
        cache.get("key1")

        assert cache.hit_rate == 1.0

    def test_get_report(self):
        inner = SemanticCache()
        cache = CostAwareCache(inner)

        cache.put("key1", "value1", cost=0.02)
        cache.get("key1")

        report = cache.get_report()
        assert report["total_requests"] == 1
        assert report["cache_hits"] == 1
        assert report["hit_rate"] == 1.0
        assert report["total_cost_incurred"] == 0.02

    def test_get_similar_hit(self):
        inner = SemanticCache(SemanticCacheConfig(similarity_threshold=0.95))
        cache = CostAwareCache(inner)

        cache.put("key1", "value1", perceptual_hash="abcdef1234567890", cost=0.02)
        result, was_cached = cache.get_similar(perceptual_hash="abcdef1234567890")

        assert result == "value1"
        assert was_cached is True
        assert cache.cache_hits == 1

    def test_get_similar_miss(self):
        inner = SemanticCache(SemanticCacheConfig(similarity_threshold=0.95))
        cache = CostAwareCache(inner)

        cache.put("key1", "value1", perceptual_hash="0000000000000000", cost=0.02)
        result, was_cached = cache.get_similar(perceptual_hash="ffffffffffffffff")

        assert result is None
        assert was_cached is False
        assert cache.cache_hits == 0
