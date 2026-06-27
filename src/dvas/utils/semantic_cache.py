"""Semantic cache with Redis backend and perceptual hashing.

Provides SemanticCache for intelligent caching of API responses
based on perceptual hashing and embedding similarity. Reduces
redundant API calls and costs.

Usage::

    from dvas.utils.semantic_cache import SemanticCache

    cache = SemanticCache(redis_url="redis://localhost:6379")

    # Store with perceptual hash
    cache.put("video_hash", "annotation_result", cost=0.01)

    # Retrieve with similarity matching
    result = cache.get_similar("video_hash", threshold=0.95)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional Redis import
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        key: Cache key
        value: Cached value
        perceptual_hash: Perceptual hash for similarity matching
        embedding: Optional embedding vector for semantic matching
        cost: API cost of generating this result
        created_at: Timestamp when cached
        access_count: Number of times accessed
        last_accessed: Last access timestamp
        ttl: Time to live in seconds
    """

    key: str
    value: Any
    perceptual_hash: str = ""
    embedding: Optional[List[float]] = None
    cost: float = 0.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: int = 3600

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        # Use >= for Windows compatibility (time.time() precision issues)
        if self.ttl <= 0:
            return True
        return time.time() - self.created_at >= self.ttl

    @property
    def age_hours(self) -> float:
        """Age of the entry in hours."""
        return (time.time() - self.created_at) / 3600

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        # Ensure last_accessed is always > created_at on Windows
        now = time.time()
        if now <= self.created_at:
            now = self.created_at + 0.001  # Add 1ms to ensure ordering
        self.last_accessed = now


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache.

    Attributes:
        redis_url: Redis connection URL
        default_ttl: Default TTL in seconds
        cost_aware_ttl: Whether to use cost-aware TTL (higher cost = longer TTL)
        min_cost_premium_ttl: Minimum cost for premium TTL
        premium_ttl_multiplier: TTL multiplier for high-cost entries
        similarity_threshold: Threshold for perceptual hash similarity (0-1)
        semantic_threshold: Threshold for embedding similarity (0-1)
        max_entries: Maximum number of entries to keep in memory
        enable_phash: Whether to enable perceptual hashing
        phash_size: Size of perceptual hash (8x8 = 64 bits)
        enable_semantic: Whether to enable semantic similarity
        embedding_dim: Dimension of embedding vectors
    """

    redis_url: Optional[str] = None
    default_ttl: int = 3600
    cost_aware_ttl: bool = True
    min_cost_premium_ttl: float = 0.01  # $0.01 threshold
    premium_ttl_multiplier: float = 3.0
    similarity_threshold: float = 0.95
    semantic_threshold: float = 0.90
    max_entries: int = 10000
    enable_phash: bool = True
    phash_size: int = 8
    enable_semantic: bool = False
    embedding_dim: int = 768


def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute perceptual hash (average hash) for an image.

    Args:
        image: Image as numpy array (H, W, C) or (H, W)
        hash_size: Size of the hash (hash_size x hash_size bits)

    Returns:
        Hex string representing the perceptual hash
    """
    if not PIL_AVAILABLE:
        # Fallback: simple hash of resized array
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Resize to hash_size x hash_size
        from scipy.ndimage import zoom

        if gray.shape[0] > hash_size and gray.shape[1] > hash_size:
            zoom_factors = (hash_size / gray.shape[0], hash_size / gray.shape[1])
            small = zoom(gray, zoom_factors, order=1)
        else:
            small = gray

        # Compute average and binary hash
        avg = np.mean(small)
        bits = (small > avg).flatten()
        hash_int = sum(1 << i for i, b in enumerate(bits) if b)
        return f"{hash_int:0{hash_size * hash_size // 4}x}"

    # PIL-based pHash
    if image.ndim == 3:
        # Convert to grayscale
        img = Image.fromarray(image.astype(np.uint8)).convert("L")
    else:
        img = Image.fromarray(image.astype(np.uint8))

    # Resize to hash_size x hash_size
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

    # Compute average hash
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)

    # Convert to hex
    return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)


def phash_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity between two perceptual hashes.

    Args:
        hash1: First perceptual hash (hex string)
        hash2: Second perceptual hash (hex string)

    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if not hash1 or not hash2:
        return 0.0

    try:
        # Convert hex to integers
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)

        # XOR to find differing bits
        xor = int1 ^ int2

        # Count differing bits (Hamming distance)
        diff_bits = bin(xor).count("1")
        total_bits = len(hash1) * 4

        # Similarity = 1 - (distance / total_bits)
        return max(0.0, 1.0 - diff_bits / total_bits)
    except ValueError:
        return 0.0


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


class SemanticCache:
    """Semantic cache with perceptual hashing and Redis backend.

    Provides intelligent caching that can match similar inputs
    using perceptual hashing or embedding similarity.

    Attributes:
        config: SemanticCacheConfig
        _redis: Redis client (if available)
        _memory: In-memory cache fallback
    """

    def __init__(self, config: Optional[SemanticCacheConfig] = None) -> None:
        self.config = config or SemanticCacheConfig()
        self._redis: Optional[Any] = None
        self._memory: Dict[str, CacheEntry] = {}

        # Try to connect to Redis
        if self.config.redis_url and REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(self.config.redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("redis_connected", url=self.config.redis_url)
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))
                self._redis = None

    def _get_ttl(self, cost: float = 0.0) -> int:
        """Get TTL based on cost (cost-aware caching).

        Higher cost results get longer TTL to maximize savings.

        Args:
            cost: API cost of the result

        Returns:
            TTL in seconds
        """
        if not self.config.cost_aware_ttl:
            return self.config.default_ttl

        if cost >= self.config.min_cost_premium_ttl:
            return int(self.config.default_ttl * self.config.premium_ttl_multiplier)

        return self.config.default_ttl

    def _make_key(self, identifier: str, prefix: str = "semantic") -> str:
        """Create a cache key.

        Args:
            identifier: Unique identifier (e.g., video hash)
            prefix: Key prefix

        Returns:
            Formatted cache key
        """
        return f"{prefix}:{identifier}"

    def put(
        self,
        key: str,
        value: Any,
        perceptual_hash: str = "",
        embedding: Optional[List[float]] = None,
        cost: float = 0.0,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            perceptual_hash: Optional perceptual hash for similarity matching
            embedding: Optional embedding vector for semantic matching
            cost: API cost of generating this result
            ttl: Optional custom TTL (uses cost-aware TTL if None)
        """
        effective_ttl = ttl or self._get_ttl(cost)
        cache_key = self._make_key(key)

        entry = CacheEntry(
            key=cache_key,
            value=value,
            perceptual_hash=perceptual_hash,
            embedding=embedding,
            cost=cost,
            ttl=effective_ttl,
        )

        # Store in Redis if available
        if self._redis is not None:
            try:
                data = {
                    "value": json.dumps(value),
                    "perceptual_hash": perceptual_hash,
                    "embedding": json.dumps(embedding) if embedding else "",
                    "cost": str(cost),
                    "created_at": str(time.time()),
                    "access_count": "0",
                    "ttl": str(effective_ttl),
                }
                self._redis.hset(cache_key, mapping=data)
                self._redis.expire(cache_key, effective_ttl)
                logger.debug("cache_put_redis", key=cache_key, ttl=effective_ttl)
            except Exception as e:
                logger.warning("redis_put_failed", key=cache_key, error=str(e))

        # Store in memory
        self._memory[cache_key] = entry

        # Evict old entries if over limit
        self._evict_if_needed()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache by exact key match.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = self._make_key(key)

        # Try Redis first
        if self._redis is not None:
            try:
                data = self._redis.hgetall(cache_key)
                if data:
                    value = json.loads(data.get("value", "null"))
                    # Update access count
                    self._redis.hincrby(cache_key, "access_count", 1)
                    self._redis.hset(cache_key, "last_accessed", str(time.time()))
                    logger.debug("cache_hit_redis", key=cache_key)
                    return value
            except Exception as e:
                logger.warning("redis_get_failed", key=cache_key, error=str(e))

        # Fallback to memory
        entry = self._memory.get(cache_key)
        if entry is not None:
            if entry.is_expired:
                del self._memory[cache_key]
                return None

            entry.touch()
            logger.debug("cache_hit_memory", key=cache_key)
            return entry.value

        return None

    def get_similar(
        self,
        perceptual_hash: str = "",
        embedding: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Any]:
        """Get a value from cache using similarity matching.

        Tries perceptual hash similarity first, then embedding similarity.

        Args:
            perceptual_hash: Perceptual hash to match
            embedding: Embedding vector to match
            threshold: Similarity threshold (uses config default if None)

        Returns:
            Best matching cached value or None
        """
        sim_threshold = threshold or self.config.similarity_threshold

        # Try memory cache first
        best_match: Optional[CacheEntry] = None
        best_score = 0.0

        for entry in self._memory.values():
            if entry.is_expired:
                continue

            score = 0.0

            # Check perceptual hash similarity
            if self.config.enable_phash and perceptual_hash and entry.perceptual_hash:
                score = phash_similarity(perceptual_hash, entry.perceptual_hash)

            # Check embedding similarity
            if self.config.enable_semantic and embedding and entry.embedding:
                emb_score = cosine_similarity(embedding, entry.embedding)
                score = max(score, emb_score)

            if score > best_score and score >= sim_threshold:
                best_score = score
                best_match = entry

        if best_match is not None:
            best_match.touch()
            logger.info(
                "cache_similarity_hit",
                score=best_score,
                key=best_match.key,
            )
            return best_match.value

        # Try Redis scan (slower but persistent)
        if self._redis is not None:
            try:
                # Scan all keys with prefix
                for redis_key in self._redis.scan_iter(match="semantic:*"):
                    data = self._redis.hgetall(redis_key)
                    if not data:
                        continue

                    stored_hash = data.get("perceptual_hash", "")
                    stored_embedding = data.get("embedding", "")

                    score = 0.0

                    if self.config.enable_phash and perceptual_hash and stored_hash:
                        score = phash_similarity(perceptual_hash, stored_hash)

                    if self.config.enable_semantic and embedding and stored_embedding:
                        emb = json.loads(stored_embedding)
                        if emb:
                            emb_score = cosine_similarity(embedding, emb)
                            score = max(score, emb_score)

                    if score > best_score and score >= sim_threshold:
                        best_score = score
                        best_match = CacheEntry(
                            key=redis_key,
                            value=json.loads(data.get("value", "null")),
                            perceptual_hash=stored_hash,
                        )

                if best_match is not None:
                    logger.info("cache_similarity_hit_redis", score=best_score)
                    return best_match.value
            except Exception as e:
                logger.warning("redis_similarity_search_failed", error=str(e))

        return None

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_key = self._make_key(key)
        deleted = False

        # Delete from Redis
        if self._redis is not None:
            try:
                result = self._redis.delete(cache_key)
                deleted = result > 0
            except Exception as e:
                logger.warning("redis_delete_failed", key=cache_key, error=str(e))

        # Delete from memory
        if cache_key in self._memory:
            del self._memory[cache_key]
            deleted = True

        return deleted

    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear Redis
        if self._redis is not None:
            try:
                for key in self._redis.scan_iter(match="semantic:*"):
                    self._redis.delete(key)
                logger.info("redis_cache_cleared")
            except Exception as e:
                logger.warning("redis_clear_failed", error=str(e))

        # Clear memory
        self._memory.clear()
        logger.info("memory_cache_cleared")

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over max size."""
        if len(self._memory) <= self.config.max_entries:
            return

        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self._memory.items(),
            key=lambda x: x[1].last_accessed,
        )

        to_remove = len(sorted_entries) - self.config.max_entries
        for i in range(to_remove):
            del self._memory[sorted_entries[i][0]]

        logger.debug("cache_evicted", count=to_remove, remaining=len(self._memory))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        stats = {
            "memory_entries": len(self._memory),
            "redis_connected": self._redis is not None,
            "config": {
                "default_ttl": self.config.default_ttl,
                "cost_aware_ttl": self.config.cost_aware_ttl,
                "similarity_threshold": self.config.similarity_threshold,
                "max_entries": self.config.max_entries,
                "enable_phash": self.config.enable_phash,
                "enable_semantic": self.config.enable_semantic,
            },
        }

        # Memory cache stats
        if self._memory:
            total_accesses = sum(e.access_count for e in self._memory.values())
            total_cost = sum(e.cost for e in self._memory.values())
            expired = sum(1 for e in self._memory.values() if e.is_expired)

            stats["memory"] = {
                "total_entries": len(self._memory),
                "total_accesses": total_accesses,
                "total_cost_saved": total_cost,
                "expired_entries": expired,
                "avg_access_count": total_accesses / len(self._memory),
            }

        # Redis stats
        if self._redis is not None:
            try:
                info = self._redis.info()
                stats["redis"] = {
                    "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_keys": self._redis.dbsize(),
                }
            except Exception:
                pass

        return stats

    def get_cache_hit_rate(self, total_requests: int, cache_hits: int) -> float:
        """Calculate cache hit rate.

        Args:
            total_requests: Total number of requests
            cache_hits: Number of cache hits

        Returns:
            Hit rate (0-1)
        """
        if total_requests == 0:
            return 0.0
        return cache_hits / total_requests


class CostAwareCache:
    """Cost-aware cache wrapper that tracks savings.

    Wraps SemanticCache and provides cost tracking for API calls.

    Usage::

        cache = CostAwareCache(SemanticCache())
        cache.put("key", "value", cost=0.02)

        result = cache.get("key")
        print(f"Saved: ${cache.total_savings:.4f}")
    """

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache
        self.total_requests = 0
        self.cache_hits = 0
        self.total_savings = 0.0
        self.total_cost = 0.0

    def put(
        self,
        key: str,
        value: Any,
        perceptual_hash: str = "",
        embedding: Optional[List[float]] = None,
        cost: float = 0.0,
        ttl: Optional[int] = None,
    ) -> None:
        """Store value in cache and track cost."""
        self.total_cost += cost
        self.cache.put(key, value, perceptual_hash, embedding, cost, ttl)

    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache and track hit/miss.

        Returns:
            Tuple of (value, was_cached)
        """
        self.total_requests += 1
        result = self.cache.get(key)

        if result is not None:
            self.cache_hits += 1
            # Find the entry to track savings
            cache_key = self.cache._make_key(key)
            entry = self.cache._memory.get(cache_key)
            if entry:
                self.total_savings += entry.cost
            return result, True

        return None, False

    def get_similar(
        self,
        perceptual_hash: str = "",
        embedding: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[Any], bool]:
        """Get similar value and track hit/miss."""
        self.total_requests += 1
        result = self.cache.get_similar(perceptual_hash, embedding, threshold)

        if result is not None:
            self.cache_hits += 1
            return result, True

        return None, False

    @property
    def hit_rate(self) -> float:
        """Current cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def estimated_savings(self) -> float:
        """Estimated cost savings from caching."""
        return self.total_savings

    def get_report(self) -> Dict[str, Any]:
        """Get cost savings report.

        Returns:
            Dict with savings statistics
        """
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.total_requests - self.cache_hits,
            "hit_rate": self.hit_rate,
            "total_cost_incurred": self.total_cost,
            "total_savings": self.total_savings,
            "net_cost": self.total_cost - self.total_savings,
            "efficiency": self.hit_rate * 100,
        }
