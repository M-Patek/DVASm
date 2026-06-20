"""Tests for tenant-level data isolation module.

Tests for TenantManager, Tenant, TenantScopedStorage, and TenantIsolationError.
"""

import pytest

from dvas.security.tenant import (
    Tenant,
    TenantIsolationError,
    TenantManager,
    TenantScopedStorage,
)


class TestTenant:
    """Test Tenant dataclass."""

    def test_tenant_creation(self):
        """Test creating a tenant."""
        tenant = Tenant(
            id="tenant_001",
            name="Acme Corp",
            created_at="2024-01-01T00:00:00+00:00",
        )
        assert tenant.id == "tenant_001"
        assert tenant.name == "Acme Corp"
        assert tenant.status == "active"

    def test_tenant_to_dict(self):
        """Test converting tenant to dict."""
        tenant = Tenant(
            id="tenant_001",
            name="Acme Corp",
            created_at="2024-01-01T00:00:00+00:00",
            metadata={"industry": "tech"},
            features={"export", "api"},
        )
        d = tenant.to_dict()
        assert d["id"] == "tenant_001"
        assert d["name"] == "Acme Corp"
        assert d["metadata"]["industry"] == "tech"
        assert "export" in d["features"]


class TestTenantManager:
    """Test TenantManager class."""

    def test_init(self):
        """Test initialization."""
        manager = TenantManager()
        assert manager is not None

    def test_create_tenant(self):
        """Test creating a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert tenant.name == "Acme Corp"
        assert tenant.id is not None
        assert tenant.status == "active"

    def test_create_tenant_with_id(self):
        """Test creating tenant with specific ID."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp", tenant_id="custom_id")
        assert tenant.id == "custom_id"

    def test_create_tenant_with_metadata(self):
        """Test creating tenant with metadata."""
        manager = TenantManager()
        tenant = manager.create_tenant(
            "Acme Corp",
            tenant_id="tenant_001",
            industry="tech",
            region="us-east",
        )
        assert tenant.metadata["industry"] == "tech"
        assert tenant.metadata["region"] == "us-east"

    def test_get_tenant(self):
        """Test getting a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        retrieved = manager.get_tenant(tenant.id)
        assert retrieved is not None
        assert retrieved.name == "Acme Corp"

    def test_get_tenant_not_found(self):
        """Test getting non-existent tenant."""
        manager = TenantManager()
        assert manager.get_tenant("nonexistent") is None

    def test_list_tenants(self):
        """Test listing tenants."""
        manager = TenantManager()
        manager.create_tenant("Acme Corp")
        manager.create_tenant("Beta Inc")
        tenants = manager.list_tenants()
        assert len(tenants) == 2

    def test_deactivate_tenant(self):
        """Test deactivating a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert manager.deactivate_tenant(tenant.id) is True
        assert tenant.status == "inactive"

    def test_deactivate_tenant_not_found(self):
        """Test deactivating non-existent tenant."""
        manager = TenantManager()
        assert manager.deactivate_tenant("nonexistent") is False

    def test_activate_tenant(self):
        """Test activating a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.deactivate_tenant(tenant.id)
        assert manager.activate_tenant(tenant.id) is True
        assert tenant.status == "active"

    def test_activate_tenant_not_found(self):
        """Test activating non-existent tenant."""
        manager = TenantManager()
        assert manager.activate_tenant("nonexistent") is False

    def test_delete_tenant(self):
        """Test deleting a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert manager.delete_tenant(tenant.id) is True
        assert manager.get_tenant(tenant.id) is None

    def test_delete_tenant_not_found(self):
        """Test deleting non-existent tenant."""
        manager = TenantManager()
        assert manager.delete_tenant("nonexistent") is False

    def test_delete_tenant_removes_data(self):
        """Test that deleting tenant removes data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.store_data(tenant.id, "key_001", "value_001")
        manager.delete_tenant(tenant.id)
        assert manager.get_data(tenant.id, "key_001") is None

    def test_delete_tenant_removes_users(self):
        """Test that deleting tenant removes user associations."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        manager.delete_tenant(tenant.id)
        assert manager.get_user_tenant("user_001") is None

    def test_assign_user(self):
        """Test assigning user to tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        assert manager.get_user_tenant("user_001") == tenant.id

    def test_assign_user_to_nonexistent_tenant(self):
        """Test assigning user to non-existent tenant."""
        manager = TenantManager()
        with pytest.raises(TenantIsolationError):
            manager.assign_user("user_001", "nonexistent")

    def test_get_user_tenant(self):
        """Test getting user tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        assert manager.get_user_tenant("user_001") == tenant.id

    def test_get_user_tenant_not_found(self):
        """Test getting tenant for unassigned user."""
        manager = TenantManager()
        assert manager.get_user_tenant("user_001") is None

    def test_store_data(self):
        """Test storing data scoped to tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.store_data(tenant.id, "key_001", "value_001")
        assert manager.get_data(tenant.id, "key_001") == "value_001"

    def test_store_data_nonexistent_tenant(self):
        """Test storing data for non-existent tenant."""
        manager = TenantManager()
        with pytest.raises(TenantIsolationError):
            manager.store_data("nonexistent", "key_001", "value_001")

    def test_get_data_not_found(self):
        """Test getting non-existent data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert manager.get_data(tenant.id, "nonexistent") is None

    def test_delete_data(self):
        """Test deleting data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.store_data(tenant.id, "key_001", "value_001")
        assert manager.delete_data(tenant.id, "key_001") is True
        assert manager.get_data(tenant.id, "key_001") is None

    def test_delete_data_not_found(self):
        """Test deleting non-existent data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert manager.delete_data(tenant.id, "nonexistent") is False

    def test_list_data_keys(self):
        """Test listing data keys."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.store_data(tenant.id, "key_001", "value_001")
        manager.store_data(tenant.id, "key_002", "value_002")
        keys = manager.list_data_keys(tenant.id)
        assert len(keys) == 2
        assert "key_001" in keys
        assert "key_002" in keys

    def test_get_tenant_data(self):
        """Test getting all tenant data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.store_data(tenant.id, "key_001", "value_001")
        manager.store_data(tenant.id, "key_002", "value_002")
        data = manager.get_tenant_data(tenant.id)
        assert data["key_001"] == "value_001"
        assert data["key_002"] == "value_002"

    def test_is_tenant_active(self):
        """Test checking if tenant is active."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        assert manager.is_tenant_active(tenant.id) is True
        manager.deactivate_tenant(tenant.id)
        assert manager.is_tenant_active(tenant.id) is False

    def test_is_tenant_active_nonexistent(self):
        """Test checking non-existent tenant."""
        manager = TenantManager()
        assert manager.is_tenant_active("nonexistent") is False

    def test_enforce_isolation_same_tenant(self):
        """Test isolation enforcement for same tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        # Should not raise
        manager.enforce_isolation("user_001", tenant.id)

    def test_enforce_isolation_different_tenant(self):
        """Test isolation enforcement for different tenant."""
        manager = TenantManager()
        tenant1 = manager.create_tenant("Acme Corp")
        tenant2 = manager.create_tenant("Beta Inc")
        manager.assign_user("user_001", tenant1.id)
        with pytest.raises(TenantIsolationError):
            manager.enforce_isolation("user_001", tenant2.id)

    def test_enforce_isolation_unassigned_user(self):
        """Test isolation enforcement for unassigned user."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        with pytest.raises(TenantIsolationError):
            manager.enforce_isolation("user_001", tenant.id)

    def test_get_tenant_stats(self):
        """Test getting tenant statistics."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        manager.store_data(tenant.id, "key_001", "value_001")
        stats = manager.get_tenant_stats(tenant.id)
        assert stats["name"] == "Acme Corp"
        assert stats["data_count"] == 1
        assert stats["user_count"] == 1

    def test_get_tenant_stats_nonexistent(self):
        """Test getting stats for non-existent tenant."""
        manager = TenantManager()
        stats = manager.get_tenant_stats("nonexistent")
        assert stats == {}

    def test_multiple_tenants_isolation(self):
        """Test data isolation between multiple tenants."""
        manager = TenantManager()
        tenant1 = manager.create_tenant("Acme Corp")
        tenant2 = manager.create_tenant("Beta Inc")

        manager.store_data(tenant1.id, "key_001", "tenant1_value")
        manager.store_data(tenant2.id, "key_001", "tenant2_value")

        assert manager.get_data(tenant1.id, "key_001") == "tenant1_value"
        assert manager.get_data(tenant2.id, "key_001") == "tenant2_value"

    def test_multiple_users_same_tenant(self):
        """Test multiple users in same tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        manager.assign_user("user_001", tenant.id)
        manager.assign_user("user_002", tenant.id)

        assert manager.get_user_tenant("user_001") == tenant.id
        assert manager.get_user_tenant("user_002") == tenant.id


class TestTenantScopedStorage:
    """Test TenantScopedStorage class."""

    def test_init(self):
        """Test initialization."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        storage = TenantScopedStorage(manager, tenant.id)
        assert storage.tenant_id == tenant.id

    def test_store_and_get(self):
        """Test storing and getting data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        storage = TenantScopedStorage(manager, tenant.id)
        storage.store("key_001", "value_001")
        assert storage.get("key_001") == "value_001"

    def test_delete(self):
        """Test deleting data."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        storage = TenantScopedStorage(manager, tenant.id)
        storage.store("key_001", "value_001")
        assert storage.delete("key_001") is True
        assert storage.get("key_001") is None

    def test_list_keys(self):
        """Test listing keys."""
        manager = TenantManager()
        tenant = manager.create_tenant("Acme Corp")
        storage = TenantScopedStorage(manager, tenant.id)
        storage.store("key_001", "value_001")
        storage.store("key_002", "value_002")
        keys = storage.list_keys()
        assert len(keys) == 2


class TestTenantIsolationError:
    """Test TenantIsolationError exception."""

    def test_exception_message(self):
        """Test exception message."""
        error = TenantIsolationError("Access denied")
        assert str(error) == "Access denied"

    def test_exception_is_exception(self):
        """Test that it's an Exception."""
        error = TenantIsolationError("Access denied")
        assert isinstance(error, Exception)
