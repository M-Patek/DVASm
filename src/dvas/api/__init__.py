"""API module for DVAS."""

from dvas.api.auth import require_auth, require_auth_strict, get_auth_status
from dvas.api.middleware import (
    APIVersion,
    CompressionMiddleware,
    HealthChecker,
    HealthStatus,
    InputValidator,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimiter,
    RequestTracker,
    api_error,
    api_response,
)

__all__ = [
    # Middleware
    "APIVersion",
    "CompressionMiddleware",
    "HealthChecker",
    "HealthStatus",
    "InputValidator",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimiter",
    "RequestTracker",
    "api_error",
    "api_response",
    # Auth
    "require_auth",
    "require_auth_strict",
    "get_auth_status",
]
