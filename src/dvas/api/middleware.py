"""API middleware and utilities for DVAS.

Provides rate limiting, request/response compression, health checks,
and API versioning.
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dvas.core.backpressure import TokenBucket
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_size: float = 20.0
    key_func: Optional[Callable[..., str]] = None
    excluded_paths: List[str] = field(default_factory=list)


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Usage::

        limiter = RateLimiter(RateLimitConfig(requests_per_second=10.0))
        if not limiter.allow_request(client_ip, "/api/v1/annotate"):
            raise RateLimitExceeded()
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._buckets: Dict[str, TokenBucket] = {}
        self._default_bucket = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.burst_size,
        )

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for a key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                rate=self.config.requests_per_second,
                capacity=self.config.burst_size,
            )
        return self._buckets[key]

    def allow_request(self, key: str, path: str) -> bool:
        """Check if a request is allowed.

        Returns True if the request should proceed, False if rate limited.
        """
        # Check excluded paths
        for excluded in self.config.excluded_paths:
            if path.startswith(excluded):
                return True

        bucket = self._get_bucket(key)
        # Use a very short timeout for non-blocking check
        import asyncio

        try:
            _loop = asyncio.get_running_loop()
            # In async context, we can't do blocking acquire
            # Return based on available tokens
            return bucket.available_tokens >= 1.0
        except RuntimeError:
            # No running loop, use sync check
            return bucket.available_tokens >= 1.0

    def consume(self, key: str) -> bool:
        """Consume a token for a request."""
        bucket = self._get_bucket(key)
        import asyncio

        try:
            _loop = asyncio.get_running_loop()
            # Can't do blocking acquire in async context
            if bucket.available_tokens >= 1.0:
                # Manually consume a token
                bucket._tokens -= 1
                return True
            return False
        except RuntimeError:
            if bucket.available_tokens >= 1.0:
                bucket._tokens -= 1
                return True
            return False

    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        if key:
            bucket = self._get_bucket(key)
            return {
                "key": key,
                "available_tokens": bucket.available_tokens,
                "rate": bucket.rate,
                "capacity": bucket.capacity,
            }

        return {
            "total_buckets": len(self._buckets),
            "default_rate": self.config.requests_per_second,
            "default_capacity": self.config.burst_size,
        }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")


# ---------------------------------------------------------------------------
# Request/Response Compression
# ---------------------------------------------------------------------------

class CompressionMiddleware:
    """Middleware for request/response compression.

    Compresses responses larger than a threshold using gzip.
    """

    def __init__(self, min_size: int = 1024, level: int = 6) -> None:
        self.min_size = min_size
        self.level = level

    def should_compress(self, content_type: str, content_length: int) -> bool:
        """Check if response should be compressed."""
        if content_length < self.min_size:
            return False

        # Don't compress already compressed content
        compressed_types = {
            "image/",
            "video/",
            "audio/",
            "application/gzip",
            "application/zip",
        }
        for prefix in compressed_types:
            if content_type.startswith(prefix):
                return False

        return True

    def compress(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)


# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Health check manager with liveness and readiness probes.

    Usage::

        checker = HealthChecker()
        checker.register("database", check_database)
        checker.register("storage", check_storage)

        # Liveness probe
        status = checker.liveness()

        # Readiness probe
        status = checker.readiness()
    """

    def __init__(self) -> None:
        self._checks: Dict[str, Callable[..., HealthCheck]] = {}
        self._startup_time = time.time()

    def register(
        self,
        name: str,
        check_fn: Callable[..., HealthCheck],
        critical: bool = False,
    ) -> None:
        """Register a health check."""
        self._checks[name] = {"fn": check_fn, "critical": critical}

    def _run_check(self, name: str) -> HealthCheck:
        """Run a single health check."""
        check_info = self._checks[name]
        start = time.time()

        try:
            result = check_info["fn"]()
            result.latency_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
            )

    def liveness(self) -> Dict[str, Any]:
        """Liveness probe - is the process running?"""
        uptime = time.time() - self._startup_time

        return {
            "status": HealthStatus.HEALTHY.value,
            "uptime_seconds": uptime,
            "checks": [],
        }

    def readiness(self) -> Dict[str, Any]:
        """Readiness probe - is the service ready to accept traffic?"""
        checks = []
        any_critical_failed = False

        for name in self._checks:
            check = self._run_check(name)
            checks.append(
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "latency_ms": round(check.latency_ms, 2),
                }
            )

            if check.status == HealthStatus.UNHEALTHY and self._checks[name]["critical"]:
                any_critical_failed = True

        status = HealthStatus.UNHEALTHY if any_critical_failed else HealthStatus.HEALTHY

        # If any non-critical check is degraded, overall is degraded
        if not any_critical_failed:
            for check in checks:
                if check["status"] == HealthStatus.DEGRADED.value:
                    status = HealthStatus.DEGRADED
                    break

        return {
            "status": status.value,
            "checks": checks,
        }


# ---------------------------------------------------------------------------
# API Versioning
# ---------------------------------------------------------------------------

class APIVersion:
    """API version information."""

    CURRENT = "v1"
    DEPRECATED: List[str] = []
    SUPPORTED = ["v1"]

    @classmethod
    def is_supported(cls, version: str) -> bool:
        """Check if a version is supported."""
        return version in cls.SUPPORTED

    @classmethod
    def is_deprecated(cls, version: str) -> bool:
        """Check if a version is deprecated."""
        return version in cls.DEPRECATED

    @classmethod
    def get_path_prefix(cls, version: Optional[str] = None) -> str:
        """Get the path prefix for a version."""
        return f"/api/{version or cls.CURRENT}"


# ---------------------------------------------------------------------------
# Request ID Tracking
# ---------------------------------------------------------------------------

class RequestTracker:
    """Track API requests with unique IDs for distributed tracing.

    Usage::

        tracker = RequestTracker()
        request_id = tracker.start_request("POST", "/api/v1/annotate")
        # ... process request ...
        tracker.end_request(request_id, status_code=200)
    """

    def __init__(self) -> None:
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_requests = 0
        self._total_errors = 0
        self._response_times: List[float] = []

    def start_request(self, method: str, path: str) -> str:
        """Start tracking a request."""
        import uuid

        request_id = str(uuid.uuid4())[:12]
        self._requests[request_id] = {
            "id": request_id,
            "method": method,
            "path": path,
            "start_time": time.time(),
            "status_code": None,
        }
        self._total_requests += 1
        return request_id

    def end_request(self, request_id: str, status_code: int) -> None:
        """End tracking a request."""
        if request_id not in self._requests:
            return

        req = self._requests[request_id]
        req["status_code"] = status_code
        req["duration_ms"] = (time.time() - req["start_time"]) * 1000

        self._response_times.append(req["duration_ms"])

        # Keep only last 1000 response times
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]

        if status_code >= 400:
            self._total_errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get request statistics."""
        if not self._response_times:
            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": 0.0,
            }

        sorted_times = sorted(self._response_times)
        n = len(sorted_times)

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_requests, 1),
            "response_time_ms": {
                "p50": sorted_times[n // 2],
                "p95": sorted_times[int(n * 0.95)],
                "p99": sorted_times[int(n * 0.99)],
                "max": sorted_times[-1],
            },
        }


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class InputValidator:
    """Validate and sanitize API inputs."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            raise ValueError("Expected string input")

        # Trim whitespace
        value = value.strip()

        # Limit length
        if len(value) > max_length:
            value = value[:max_length]

        return value

    @staticmethod
    def validate_video_id(video_id: str) -> str:
        """Validate a video ID."""
        if not video_id or len(video_id) < 3:
            raise ValueError("Video ID must be at least 3 characters")

        # Allow alphanumeric, underscore, hyphen
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", video_id):
            raise ValueError("Video ID contains invalid characters")

        return video_id

    @staticmethod
    def validate_file_size(size_bytes: int, max_mb: float = 500.0) -> bool:
        """Validate file size."""
        max_bytes = int(max_mb * 1024 * 1024)
        if size_bytes > max_bytes:
            raise ValueError(f"File size exceeds maximum of {max_mb}MB")
        return True


# ---------------------------------------------------------------------------
# API Response Utilities
# ---------------------------------------------------------------------------

def api_response(
    data: Any = None,
    message: str = "",
    status: str = "success",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a standardized API response.

    Usage::

        return api_response(
            data=annotation.model_dump(),
            message="Annotation completed",
        )
    """
    response = {
        "status": status,
        "message": message,
        "data": data,
    }

    if metadata:
        response["metadata"] = metadata

    return response


def api_error(
    message: str,
    error_code: str = "INTERNAL_ERROR",
    status_code: int = 500,
    details: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a standardized API error response."""
    response = {
        "status": "error",
        "error": {
            "code": error_code,
            "message": message,
            "status_code": status_code,
        },
    }

    if details:
        response["error"]["details"] = details

    return response
