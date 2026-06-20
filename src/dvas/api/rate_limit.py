"""API rate limiting with per-tenant support.

Provides per-tenant rate limits using token bucket implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from dvas.core.backpressure import TokenBucket
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TenantRateLimitConfig:
    """Per-tenant rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: float = 20.0
    max_concurrent: int = 10
    daily_quota: Optional[int] = None
    monthly_quota: Optional[int] = None


class TenantRateLimiter:
    """Token bucket rate limiter with per-tenant support."""

    def __init__(self, default_config: Optional[TenantRateLimitConfig] = None) -> None:
        self._default_config = default_config or TenantRateLimitConfig()
        self._tenant_configs: Dict[str, TenantRateLimitConfig] = {}
        self._tenant_buckets: Dict[str, Dict[str, Any]] = {}
        self._tenant_usage: Dict[str, Dict[str, Any]] = {}

    def configure_tenant(self, tenant_id: str, config: TenantRateLimitConfig) -> None:
        self._tenant_configs[tenant_id] = config
        self._tenant_buckets[tenant_id] = {}
        logger.info(
            "tenant_rate_limit_configured",
            tenant_id=tenant_id,
            rps=config.requests_per_second,
            burst=config.burst_size,
        )

    def get_tenant_config(self, tenant_id: str) -> TenantRateLimitConfig:
        return self._tenant_configs.get(tenant_id, self._default_config)

    def _get_bucket(self, tenant_id: str, path: str) -> TokenBucket:
        if tenant_id not in self._tenant_buckets:
            self._tenant_buckets[tenant_id] = {}
        bucket_key = self._categorize_path(path)
        if bucket_key not in self._tenant_buckets[tenant_id]:
            config = self.get_tenant_config(tenant_id)
            self._tenant_buckets[tenant_id][bucket_key] = TokenBucket(
                rate=config.requests_per_second,
                capacity=config.burst_size,
            )
        return self._tenant_buckets[tenant_id][bucket_key]

    def _categorize_path(self, path: str) -> str:
        if path.startswith("/api/v1/videos"):
            return "videos"
        elif path.startswith("/api/v1/annotations"):
            return "annotations"
        elif path.startswith("/api/v1/export"):
            return "export"
        elif path.startswith("/api/v1/batch"):
            return "batch"
        elif path.startswith("/api/v1/datasets"):
            return "datasets"
        elif path.startswith("/api/v1/review"):
            return "review"
        elif path.startswith("/api/v1/models"):
            return "models"
        elif path.startswith("/api/v1/prompts"):
            return "prompts"
        else:
            return "default"

    def try_acquire(self, tenant_id: str, path: str) -> bool:
        bucket = self._get_bucket(tenant_id, path)
        if bucket.available_tokens >= 1.0:
            bucket._tokens -= 1
            now = time.time()
            day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
            if tenant_id not in self._tenant_usage:
                self._tenant_usage[tenant_id] = {}
            if day_key not in self._tenant_usage[tenant_id]:
                self._tenant_usage[tenant_id][day_key] = 0
            self._tenant_usage[tenant_id][day_key] += 1
            config = self.get_tenant_config(tenant_id)
            if config.daily_quota and self._tenant_usage[tenant_id][day_key] > config.daily_quota:
                return False
            return True
        logger.warning("tenant_rate_limit_exceeded", tenant_id=tenant_id, path=path)
        return False

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        config = self.get_tenant_config(tenant_id)
        buckets = self._tenant_buckets.get(tenant_id, {})
        bucket_stats = {}
        for key, bucket in buckets.items():
            bucket_stats[key] = {
                "available_tokens": bucket.available_tokens,
                "rate": bucket.rate,
                "capacity": bucket.capacity,
            }
        usage = self._tenant_usage.get(tenant_id, {})
        today = time.strftime("%Y-%m-%d", time.gmtime())
        daily_usage = usage.get(today, 0)
        return {
            "tenant_id": tenant_id,
            "config": {
                "requests_per_second": config.requests_per_second,
                "burst_size": config.burst_size,
                "max_concurrent": config.max_concurrent,
                "daily_quota": config.daily_quota,
            },
            "daily_usage": daily_usage,
            "quota_remaining": (config.daily_quota - daily_usage if config.daily_quota else None),
            "buckets": bucket_stats,
        }

    def get_all_stats(self) -> Dict[str, Any]:
        return {
            tenant_id: self.get_tenant_stats(tenant_id)
            for tenant_id in self._tenant_configs
        }


class TenantRateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for per-tenant rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[TenantRateLimiter] = None,
        excluded_paths: Optional[list] = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter or TenantRateLimiter()
        self._excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/api/v1/health",
            "/api/v1/ready",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        if any(request.url.path.startswith(path) for path in self._excluded_paths):
            return await call_next(request)
        tenant_id = getattr(request.state, "tenant_id", None) or "default"
        if not self._limiter.try_acquire(tenant_id, request.url.path):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for tenant",
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


def get_tenant_rate_limiter() -> TenantRateLimiter:
    return TenantRateLimiter()
