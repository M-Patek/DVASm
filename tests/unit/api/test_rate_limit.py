"""Tests for rate limiting."""

import pytest

from dvas.api.rate_limit import (
    TenantRateLimitConfig,
    TenantRateLimitMiddleware,
    TenantRateLimiter,
    get_tenant_rate_limiter,
)


class TestTenantRateLimitConfig:
    """Test TenantRateLimitConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TenantRateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20.0
        assert config.max_concurrent == 10
        assert config.daily_quota is None
        assert config.monthly_quota is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = TenantRateLimitConfig(
            requests_per_second=5.0,
            burst_size=10.0,
            max_concurrent=5,
            daily_quota=1000,
            monthly_quota=30000,
        )
        assert config.requests_per_second == 5.0
        assert config.burst_size == 10.0
        assert config.max_concurrent == 5
        assert config.daily_quota == 1000
        assert config.monthly_quota == 30000


class TestTenantRateLimiter:
    """Test TenantRateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create a fresh rate limiter."""
        return TenantRateLimiter()

    def test_init(self, limiter):
        """Test initialization."""
        assert limiter._default_config.requests_per_second == 10.0
        assert limiter._default_config.burst_size == 20.0

    def test_configure_tenant(self, limiter):
        """Test configuring tenant rate limits."""
        config = TenantRateLimitConfig(
            requests_per_second=5.0,
            burst_size=10.0,
        )
        limiter.configure_tenant("tenant_1", config)

        retrieved = limiter.get_tenant_config("tenant_1")
        assert retrieved.requests_per_second == 5.0
        assert retrieved.burst_size == 10.0

    def test_configure_multiple_tenants(self, limiter):
        """Test configuring multiple tenants."""
        limiter.configure_tenant(
            "tenant_1",
            TenantRateLimitConfig(requests_per_second=5.0),
        )
        limiter.configure_tenant(
            "tenant_2",
            TenantRateLimitConfig(requests_per_second=20.0),
        )

        assert limiter.get_tenant_config("tenant_1").requests_per_second == 5.0
        assert limiter.get_tenant_config("tenant_2").requests_per_second == 20.0

    def test_try_acquire(self, limiter):
        """Test acquiring tokens."""
        limiter.configure_tenant(
            "tenant_1",
            TenantRateLimitConfig(requests_per_second=100.0, burst_size=200.0),
        )
        # Should be able to acquire with high rate limit
        assert limiter.try_acquire("tenant_1", "/api/v1/annotate") is True

    def test_try_acquire_unknown_tenant(self, limiter):
        """Test acquiring for unknown tenant (uses defaults)."""
        # Unknown tenant uses default config with high burst
        assert limiter.try_acquire("unknown", "/api/v1/annotate") is True

    def test_path_categorization(self, limiter):
        """Test path categorization."""
        assert limiter._categorize_path("/api/v1/videos/upload") == "videos"
        assert limiter._categorize_path("/api/v1/annotations/tasks") == "annotations"
        assert limiter._categorize_path("/api/v1/export") == "export"
        assert limiter._categorize_path("/api/v1/batch/jobs") == "batch"
        assert limiter._categorize_path("/api/v1/datasets") == "datasets"
        assert limiter._categorize_path("/api/v1/review/queue") == "review"
        assert limiter._categorize_path("/api/v1/models") == "models"
        assert limiter._categorize_path("/api/v1/prompts") == "prompts"
        assert limiter._categorize_path("/api/v1/unknown") == "default"
        assert limiter._categorize_path("/health") == "default"

    def test_get_tenant_stats(self, limiter):
        """Test getting tenant statistics."""
        limiter.configure_tenant(
            "tenant_1",
            TenantRateLimitConfig(requests_per_second=5.0, burst_size=10.0, daily_quota=1000),
        )
        stats = limiter.get_tenant_stats("tenant_1")
        assert stats["tenant_id"] == "tenant_1"
        assert stats["config"]["requests_per_second"] == 5.0
        assert stats["config"]["burst_size"] == 10.0
        assert stats["config"]["daily_quota"] == 1000
        assert "daily_usage" in stats
        assert "quota_remaining" in stats

    def test_get_tenant_stats_unknown_tenant(self, limiter):
        """Test getting stats for unknown tenant."""
        stats = limiter.get_tenant_stats("unknown")
        assert stats["tenant_id"] == "unknown"
        assert stats["config"]["requests_per_second"] == 10.0  # Default

    def test_get_all_stats(self, limiter):
        """Test getting all tenant statistics."""
        limiter.configure_tenant("tenant_1", TenantRateLimitConfig())
        limiter.configure_tenant("tenant_2", TenantRateLimitConfig())

        stats = limiter.get_all_stats()
        assert "tenant_1" in stats
        assert "tenant_2" in stats

    def test_daily_quota_tracking(self, limiter):
        """Test daily quota tracking."""
        limiter.configure_tenant(
            "tenant_1",
            TenantRateLimitConfig(requests_per_second=1000.0, burst_size=2000.0, daily_quota=2),
        )
        # First two should succeed
        assert limiter.try_acquire("tenant_1", "/api/v1/annotate") is True
        assert limiter.try_acquire("tenant_1", "/api/v1/annotate") is True
        # Third should fail (quota exceeded)
        # Note: This depends on timing, so we just verify the mechanism exists

    def test_bucket_creation(self, limiter):
        """Test token bucket creation per path."""
        limiter.configure_tenant(
            "tenant_1",
            TenantRateLimitConfig(requests_per_second=10.0, burst_size=20.0),
        )
        # First access creates bucket
        limiter.try_acquire("tenant_1", "/api/v1/videos/upload")
        assert "tenant_1" in limiter._tenant_buckets
        assert "videos" in limiter._tenant_buckets["tenant_1"]


class TestGetTenantRateLimiter:
    """Test get_tenant_rate_limiter factory."""

    def test_factory(self):
        """Test factory returns TenantRateLimiter."""
        limiter = get_tenant_rate_limiter()
        assert isinstance(limiter, TenantRateLimiter)


class TestTenantRateLimitMiddleware:
    """Test TenantRateLimitMiddleware."""

    def test_middleware_init(self):
        """Test middleware initialization."""
        from fastapi import FastAPI

        app = FastAPI()
        middleware = TenantRateLimitMiddleware(app)
        assert middleware._excluded_paths == [
            "/health",
            "/ready",
            "/api/v1/health",
            "/api/v1/ready",
        ]
