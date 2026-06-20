"""Tenant-level data isolation for DVAS.

Provides multi-tenant data separation with tenant-scoped storage,
access control, and data isolation guarantees.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Tenant:
    """A tenant in the multi-tenant system."""

    id: str
    name: str
    created_at: str
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    allowed_domains: List[str] = field(default_factory=list)
    max_users: int = 100
    max_storage_gb: int = 100
    features: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "metadata": self.metadata,
            "allowed_domains": self.allowed_domains,
            "max_users": self.max_users,
            "max_storage_gb": self.max_storage_gb,
            "features": list(self.features),
        }


class TenantIsolationError(Exception):
    """Raised when a tenant isolation rule is violated."""
    pass


class TenantManager:
    """Manage tenants and enforce data isolation.

    Usage::

        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.activate_tenant(tenant.id)

        # Store data scoped to tenant
        manager.store_data(tenant.id, "annotation_001", {...})

        # Retrieve tenant-scoped data
        data = manager.get_data(tenant.id, "annotation_001")
    """

    def __init__(self) -> None:
        """Initialize the tenant manager."""
        self._tenants: Dict[str, Tenant] = {}
        self._data: Dict[str, Dict[str, Any]] = {}
        self._user_tenants: Dict[str, str] = {}
        self._tenant_users: Dict[str, Set[str]] = {}

    def create_tenant(
        self,
        name: str,
        tenant_id: Optional[str] = None,
        **metadata,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            name: The tenant name.
            tenant_id: Optional tenant ID. Generated if not provided.
            **metadata: Additional tenant metadata.

        Returns:
            The created Tenant.
        """
        tenant_id = tenant_id or str(uuid.uuid4())
        tenant = Tenant(
            id=tenant_id,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

        self._tenants[tenant_id] = tenant
        self._data[tenant_id] = {}
        self._tenant_users[tenant_id] = set()

        logger.info("tenant_created", tenant_id=tenant_id, name=name)
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID.

        Args:
            tenant_id: The tenant ID.

        Returns:
            The Tenant, or None if not found.
        """
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[Tenant]:
        """List all tenants.

        Returns:
            List of all tenants.
        """
        return list(self._tenants.values())

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if deactivated, False if not found.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False

        tenant.status = "inactive"
        logger.info("tenant_deactivated", tenant_id=tenant_id)
        return True

    def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if activated, False if not found.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False

        tenant.status = "active"
        logger.info("tenant_activated", tenant_id=tenant_id)
        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if deleted, False if not found.
        """
        if tenant_id not in self._tenants:
            return False

        del self._tenants[tenant_id]
        del self._data[tenant_id]

        # Remove user associations
        for user_id in list(self._user_tenants.keys()):
            if self._user_tenants[user_id] == tenant_id:
                del self._user_tenants[user_id]

        if tenant_id in self._tenant_users:
            del self._tenant_users[tenant_id]

        logger.info("tenant_deleted", tenant_id=tenant_id)
        return True

    def assign_user(self, user_id: str, tenant_id: str) -> None:
        """Assign a user to a tenant.

        Args:
            user_id: The user ID.
            tenant_id: The tenant ID.

        Raises:
            TenantIsolationError: If the tenant does not exist.
        """
        if tenant_id not in self._tenants:
            raise TenantIsolationError(f"Tenant not found: {tenant_id}")

        self._user_tenants[user_id] = tenant_id
        self._tenant_users[tenant_id].add(user_id)
        logger.info("user_assigned_to_tenant", user_id=user_id, tenant_id=tenant_id)

    def get_user_tenant(self, user_id: str) -> Optional[str]:
        """Get the tenant ID for a user.

        Args:
            user_id: The user ID.

        Returns:
            The tenant ID, or None if not assigned.
        """
        return self._user_tenants.get(user_id)

    def store_data(self, tenant_id: str, key: str, data: Any) -> None:
        """Store data scoped to a tenant.

        Args:
            tenant_id: The tenant ID.
            key: The data key.
            data: The data to store.

        Raises:
            TenantIsolationError: If the tenant does not exist.
        """
        self._ensure_tenant_exists(tenant_id)
        self._data[tenant_id][key] = data

    def get_data(self, tenant_id: str, key: str) -> Any:
        """Retrieve data scoped to a tenant.

        Args:
            tenant_id: The tenant ID.
            key: The data key.

        Returns:
            The stored data, or None if not found.
        """
        return self._data.get(tenant_id, {}).get(key)

    def delete_data(self, tenant_id: str, key: str) -> bool:
        """Delete data scoped to a tenant.

        Args:
            tenant_id: The tenant ID.
            key: The data key.

        Returns:
            True if deleted, False if not found.
        """
        tenant_data = self._data.get(tenant_id, {})
        if key not in tenant_data:
            return False

        del tenant_data[key]
        return True

    def list_data_keys(self, tenant_id: str) -> List[str]:
        """List all data keys for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            List of data keys.
        """
        return list(self._data.get(tenant_id, {}).keys())

    def get_tenant_data(self, tenant_id: str) -> Dict[str, Any]:
        """Get all data for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dictionary of all tenant data.
        """
        return self._data.get(tenant_id, {}).copy()

    def is_tenant_active(self, tenant_id: str) -> bool:
        """Check if a tenant is active.

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if the tenant exists and is active.
        """
        tenant = self._tenants.get(tenant_id)
        return tenant is not None and tenant.status == "active"

    def enforce_isolation(self, user_id: str, target_tenant_id: str) -> None:
        """Enforce tenant isolation for a user.

        Args:
            user_id: The user ID.
            target_tenant_id: The tenant ID being accessed.

        Raises:
            TenantIsolationError: If the user is not in the target tenant.
        """
        user_tenant = self._user_tenants.get(user_id)
        if user_tenant is None:
            raise TenantIsolationError(
                f"User {user_id} is not assigned to any tenant"
            )

        if user_tenant != target_tenant_id:
            raise TenantIsolationError(
                f"User {user_id} (tenant: {user_tenant}) cannot access "
                f"data from tenant {target_tenant_id}"
            )

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dictionary with tenant statistics.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return {}

        data_count = len(self._data.get(tenant_id, {}))
        user_count = len(self._tenant_users.get(tenant_id, set()))

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "status": tenant.status,
            "data_count": data_count,
            "user_count": user_count,
            "created_at": tenant.created_at,
        }

    def _ensure_tenant_exists(self, tenant_id: str) -> None:
        """Ensure a tenant exists."""
        if tenant_id not in self._tenants:
            raise TenantIsolationError(f"Tenant not found: {tenant_id}")


class TenantScopedStorage:
    """Storage wrapper that automatically scopes data to a tenant."""

    def __init__(self, tenant_manager: TenantManager, tenant_id: str) -> None:
        """Initialize tenant-scoped storage.

        Args:
            tenant_manager: The tenant manager.
            tenant_id: The tenant ID.
        """
        self.tenant_manager = tenant_manager
        self.tenant_id = tenant_id

    def store(self, key: str, data: Any) -> None:
        """Store data in the tenant scope.

        Args:
            key: The data key.
            data: The data to store.
        """
        self.tenant_manager.store_data(self.tenant_id, key, data)

    def get(self, key: str) -> Any:
        """Retrieve data from the tenant scope.

        Args:
            key: The data key.

        Returns:
            The stored data, or None if not found.
        """
        return self.tenant_manager.get_data(self.tenant_id, key)

    def delete(self, key: str) -> bool:
        """Delete data from the tenant scope.

        Args:
            key: The data key.

        Returns:
            True if deleted, False if not found.
        """
        return self.tenant_manager.delete_data(self.tenant_id, key)

    def list_keys(self) -> List[str]:
        """List all keys in the tenant scope.

        Returns:
            List of data keys.
        """
        return self.tenant_manager.list_data_keys(self.tenant_id)


__all__ = [
    "TenantManager",
    "Tenant",
    "TenantScopedStorage",
    "TenantIsolationError",
]
