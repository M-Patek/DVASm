"""API module for DVAS."""

from dvas.api.auth import require_auth, require_auth_strict, get_auth_status
from dvas.api.dependencies import AppState
from dvas.api.middleware import (
    APIVersion,
    CompressionMiddleware,
    HealthChecker,
    HealthStatus,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimiter,
    RequestTracker,
    api_error,
    api_response,
)
from dvas.api.task_store import (
    InMemoryTaskStore,
    PostgresTaskStore,
    RedisTaskStore,
    Task,
    TaskStatus,
    TaskStore,
    TaskType,
)
from dvas.api.task_queue import (
    CeleryTaskQueue,
    InMemoryTaskQueue,
    QueueBackend,
    QueueConfig,
    TaskQueue,
    create_task_queue,
)
from dvas.api.task_lifecycle import (
    Checkpoint,
    RetryConfig,
    RetryPolicy,
    TaskLifecycleManager,
)
from dvas.api.tenant import (
    Tenant,
    TenantContext,
    TenantMiddleware,
    TenantScopedAccess,
    TenantStore,
)
from dvas.api.rate_limit import (
    TenantRateLimitConfig,
    TenantRateLimitMiddleware,
    TenantRateLimiter,
    get_tenant_rate_limiter,
)
from dvas.api.audit_log import (
    AuditLogEntry,
    AuditLogStore,
    AuditLogger,
    get_audit_logger,
)
from dvas.security.validation import InputValidator

__all__ = [
    # Dependencies
    "AppState",
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
    # Task Store
    "Task",
    "TaskStatus",
    "TaskStore",
    "TaskType",
    "InMemoryTaskStore",
    "RedisTaskStore",
    "PostgresTaskStore",
    # Task Queue
    "TaskQueue",
    "InMemoryTaskQueue",
    "CeleryTaskQueue",
    "QueueBackend",
    "QueueConfig",
    "create_task_queue",
    # Task Lifecycle
    "TaskLifecycleManager",
    "RetryConfig",
    "RetryPolicy",
    "Checkpoint",
    # Tenant
    "Tenant",
    "TenantContext",
    "TenantMiddleware",
    "TenantRateLimitConfig",
    "TenantRateLimiter",
    "TenantScopedAccess",
    "TenantStore",
    # Rate Limit
    "TenantRateLimitMiddleware",
    "get_tenant_rate_limiter",
    # Audit Log
    "AuditLogEntry",
    "AuditLogStore",
    "AuditLogger",
    "get_audit_logger",
]
