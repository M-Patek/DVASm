"""Tenant isolation middleware for DVAS API.

Provides tenant-scoped data access with middleware for extracting tenant
from requests and enforcing tenant boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Tenant:
    """Tenant identity."""

    id: str
    name: str
    plan: str = "basic"
    features: Optional[List[str]] = None
    rate_limit_multiplier: float = 1.0
    max_concurrent_tasks: int = 10

    def __post_init__(self):
        if self.features is None:
            self.features = []


class TenantStore:
    """In-memory tenant store for development."""

    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._api_keys: Dict[str, str] = {}
        self.register(
            Tenant(
                id="default",
                name="Default Tenant",
                plan="basic",
                features=["annotation", "export"],
            )
        )

    def register(self, tenant: Tenant) -> None:
        self._tenants[tenant.id] = tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def get_by_api_key(self, api_key: str) -> Optional[Tenant]:
        tenant_id = self._api_keys.get(api_key)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    def assign_api_key(self, tenant_id: str, api_key: str) -> None:
        self._api_keys[api_key] = tenant_id

    def revoke_api_key(self, api_key: str) -> bool:
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False

    def list_tenants(self) -> List[Tenant]:
        return list(self._tenants.values())

    def delete(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            self._api_keys = {k: v for k, v in self._api_keys.items() if v != tenant_id}
            return True
        return False


class TenantContext:
    """Tenant context for request-scoped tenant access."""

    _tenant_header = "X-Tenant-ID"
    _api_key_header = "X-API-Key"

    @classmethod
    def get_tenant_id(cls, request: Request) -> Optional[str]:
        return request.headers.get(cls._tenant_header)

    @classmethod
    def get_current_tenant(
        cls, request: Request, store: Optional[TenantStore] = None
    ) -> Optional[Tenant]:
        tenant_id = cls.get_tenant_id(request)
        if tenant_id and store:
            return store.get(tenant_id)
        api_key = request.headers.get(cls._api_key_header)
        if api_key and store:
            return store.get_by_api_key(api_key)
        return None


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and validate tenant from requests."""

    def __init__(
        self,
        app: ASGIApp,
        tenant_store: Optional[TenantStore] = None,
        require_tenant: bool = False,
        excluded_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__(app)
        self._store = tenant_store or TenantStore()
        self._require_tenant = require_tenant
        self._excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/api/v1/health",
            "/api/v1/ready",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        if any(request.url.path.startswith(path) for path in self._excluded_paths):
            return await call_next(request)
        tenant = TenantContext.get_current_tenant(request, self._store)
        if self._require_tenant and tenant is None:
            logger.warning(
                "tenant_required",
                path=request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant identification required",
                headers={"WWW-Authenticate": "X-Tenant-ID"},
            )
        request.state.tenant = tenant
        request.state.tenant_id = tenant.id if tenant else None
        if tenant:
            logger.debug(
                "tenant_identified",
                tenant_id=tenant.id,
                path=request.url.path,
            )
        return await call_next(request)


def get_tenant_id(request: Request) -> Optional[str]:
    return getattr(request.state, "tenant_id", None)


def require_tenant(request: Request) -> Tenant:
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant identification required",
        )
    return tenant


def tenant_scope(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    return {"tenant_id": tenant_id}


class TenantScopedAccess:
    """Helper for tenant-scoped data access."""

    def __init__(self, store: Any, tenant_id: Optional[str] = None) -> None:
        self._store = store
        self._tenant_id = tenant_id

    async def create(self, item: Any) -> Any:
        if hasattr(item, "tenant_id"):
            item.tenant_id = self._tenant_id
        return await self._store.create(item)

    async def get(self, item_id: str) -> Optional[Any]:
        return await self._store.get(item_id, tenant_id=self._tenant_id)

    async def list(self, **kwargs: Any) -> List[Any]:
        kwargs["tenant_id"] = self._tenant_id
        return await self._store.list(**kwargs)

    async def update(self, item: Any) -> Any:
        return await self._store.update(item)

    async def delete(self, item_id: str) -> bool:
        return await self._store.delete(item_id, tenant_id=self._tenant_id)
