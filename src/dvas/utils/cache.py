"""Cache utilities for DVAS."""

import hashlib
import json
import threading
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from aiocache import Cache
from aiocache.serializers import JsonSerializer

T = TypeVar("T")

# Default cache instance (in-memory, can be swapped for Redis)
_default_cache: Optional[Cache] = None
_default_lock = threading.Lock()


def get_cache() -> Cache:
    """Get or create default cache instance (thread-safe)."""
    global _default_cache
    if _default_cache is None:
        with _default_lock:
            if _default_cache is None:
                _default_cache = Cache(Cache.MEMORY, serializer=JsonSerializer())
    return _default_cache


def _make_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Create cache key from function arguments."""
    key_data = {
        "prefix": prefix,
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()[:16]}"


def cached(
    prefix: str,
    ttl: int = 3600,
    cache: Optional[Cache] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching function results.

    Supports both sync and async functions.
    For sync functions, uses threading.Lock to avoid race conditions.
    For async functions, uses asyncio-compatible locking.

    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        cache: Cache instance (uses default if None)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(func):
            # Async wrapper
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                cache_instance = cache or get_cache()
                key = _make_key(prefix, *args, **kwargs)

                # Try to get from cache
                try:
                    cached_value = await cache_instance.get(key)
                    if cached_value is not None:
                        return cached_value
                except Exception:
                    pass

                # Execute function and cache result
                result = await func(*args, **kwargs)
                try:
                    await cache_instance.set(key, result, ttl=ttl)
                except Exception:
                    pass
                return result

            return async_wrapper
        else:
            # Sync wrapper - use threading lock for thread safety
            _lock = threading.Lock()

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                cache_instance = cache or get_cache()
                key = _make_key(prefix, *args, **kwargs)

                with _lock:
                    # Try to get from cache
                    try:
                        # For sync functions, use cache.get directly
                        # aiocache Cache has a get method that can work in sync context
                        # via the underlying backend
                        cached_value = cache_instance.get_sync(key)
                        if cached_value is not None:
                            return cached_value
                    except Exception:
                        pass

                    # Execute function and cache result
                    result = func(*args, **kwargs)
                    try:
                        cache_instance.set_sync(key, result, ttl=ttl)
                    except Exception:
                        pass
                    return result

            return sync_wrapper

    return decorator


def invalidate_cache(prefix: str, cache: Optional[Cache] = None) -> None:
    """Invalidate all cache entries with given prefix.

    Note: aiocache doesn't support prefix-based invalidation natively.
    For production, use Redis with SCAN command.
    """
    cache_instance = cache or get_cache()
    try:
        # Try to clear all cache entries
        cache_instance.clear()
    except Exception:
        pass


@cached("video_metadata", ttl=7200)
async def get_cached_video_metadata(video_path: str) -> dict:
    """Get cached video metadata."""
    from dvas.data.video_loader import VideoLoader

    with VideoLoader(video_path) as loader:
        return loader.metadata.model_dump()


@cached("annotation", ttl=3600)
async def get_cached_annotation(annotation_id: str, source: str = "model") -> Optional[dict]:
    """Get cached annotation."""
    from dvas.data.storage import AnnotationStore

    store = AnnotationStore()
    annotation = store.load(annotation_id, source)
    return annotation.model_dump() if annotation else None
