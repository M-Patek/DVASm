"""Tests for RBAC (Role-Based Access Control) module.

Tests for RBAC, Permission, Role, AccessPolicy, and PermissionChecker classes.
"""

import pytest

from dvas.security.rbac import (
    ROLE_PERMISSIONS,
    AccessPolicy,
    Permission,
    PermissionChecker,
    RBAC,
    Role,
)


class TestPermission:
    """Test Permission enum."""

    def test_permission_values(self):
        """Test permission values."""
        assert Permission.ANNOTATION_READ.value == "annotation:read"
        assert Permission.ANNOTATION_WRITE.value == "annotation:write"
        assert Permission.ANNOTATION_DELETE.value == "annotation:delete"
        assert Permission.ANNOTATION_EXPORT.value == "annotation:export"
        assert Permission.ANNOTATION_APPROVE.value == "annotation:approve"
        assert Permission.VIDEO_READ.value == "video:read"
        assert Permission.SYSTEM_ADMIN.value == "system:admin"
        assert Permission.USER_MANAGE.value == "user:manage"


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        """Test role values."""
        assert Role.ADMIN.value == "admin"
        assert Role.REVIEWER.value == "reviewer"
        assert Role.ANNOTATOR.value == "annotator"
        assert Role.VIEWER.value == "viewer"
        assert Role.API_CLIENT.value == "api_client"


class TestRolePermissions:
    """Test ROLE_PERMISSIONS mapping."""

    def test_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.ANNOTATION_READ in admin_perms
        assert Permission.ANNOTATION_WRITE in admin_perms
        assert Permission.ANNOTATION_DELETE in admin_perms
        assert Permission.SYSTEM_ADMIN in admin_perms
        assert Permission.USER_MANAGE in admin_perms

    def test_viewer_has_only_read(self):
        """Test viewer has only read permissions."""
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.ANNOTATION_READ in viewer_perms
        assert Permission.ANNOTATION_WRITE not in viewer_perms
        assert Permission.ANNOTATION_DELETE not in viewer_perms

    def test_annotator_has_read_write(self):
        """Test annotator has read and write."""
        annotator_perms = ROLE_PERMISSIONS[Role.ANNOTATOR]
        assert Permission.ANNOTATION_READ in annotator_perms
        assert Permission.ANNOTATION_WRITE in annotator_perms
        assert Permission.ANNOTATION_DELETE not in annotator_perms

    def test_reviewer_has_approve(self):
        """Test reviewer has approve permission."""
        reviewer_perms = ROLE_PERMISSIONS[Role.REVIEWER]
        assert Permission.ANNOTATION_APPROVE in reviewer_perms
        assert Permission.REVIEW_APPROVE in reviewer_perms


class TestAccessPolicy:
    """Test AccessPolicy dataclass."""

    def test_policy_creation(self):
        """Test creating an access policy."""
        policy = AccessPolicy(
            resource_type="annotation",
            action="read",
            allowed=True,
        )
        assert policy.resource_type == "annotation"
        assert policy.action == "read"
        assert policy.allowed is True


class TestRBAC:
    """Test RBAC class."""

    def test_init(self):
        """Test initialization."""
        rbac = RBAC()
        assert rbac is not None

    def test_assign_role(self):
        """Test assigning a role."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        assert rbac.get_role("user_001") == Role.ADMIN

    def test_assign_role_by_string(self):
        """Test assigning a role by string."""
        rbac = RBAC()
        rbac.assign_role("user_001", "admin")
        assert rbac.get_role("user_001") == Role.ADMIN

    def test_assign_invalid_role(self):
        """Test assigning invalid role."""
        rbac = RBAC()
        with pytest.raises(ValueError):
            rbac.assign_role("user_001", "invalid_role")

    def test_revoke_role(self):
        """Test revoking a role."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        rbac.revoke_role("user_001")
        assert rbac.get_role("user_001") is None

    def test_revoke_nonexistent_role(self):
        """Test revoking role for unassigned user."""
        rbac = RBAC()
        # Should not raise
        rbac.revoke_role("user_001")

    def test_set_owner(self):
        """Test setting resource owner."""
        rbac = RBAC()
        rbac.set_owner("resource_001", "user_001")
        assert rbac.get_owner("resource_001") == "user_001"

    def test_owner_has_full_access(self):
        """Test that owner has full access."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        rbac.set_owner("resource_001", "user_001")
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)

    def test_non_owner_lacks_permission(self):
        """Test that non-owner lacks permissions based on role."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        rbac.set_owner("resource_001", "user_002")
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)
        assert rbac.has_permission("user_001", "resource_001", Permission.SYSTEM_ADMIN)

    def test_annotator_has_read_write(self):
        """Test annotator has read and write."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)

    def test_viewer_has_only_read(self):
        """Test viewer has only read."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)

    def test_reviewer_has_approve(self):
        """Test reviewer has approve."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.REVIEWER)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_APPROVE)
        assert rbac.has_permission("user_001", "resource_001", Permission.REVIEW_APPROVE)

    def test_api_client_has_export(self):
        """Test API client has export."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.API_CLIENT)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_EXPORT)

    def test_unassigned_user_is_viewer(self):
        """Test unassigned user defaults to viewer."""
        rbac = RBAC()
        # No role assigned
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_check_permission_granted(self):
        """Test check_permission when granted."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        # Should not raise
        rbac.check_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_check_permission_denied(self):
        """Test check_permission when denied."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        with pytest.raises(PermissionError):
            rbac.check_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_add_custom_permission(self):
        """Test adding custom permission."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        rbac.add_custom_permission("user_001", Permission.ANNOTATION_WRITE)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_remove_custom_permission(self):
        """Test removing custom permission."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        rbac.add_custom_permission("user_001", Permission.ANNOTATION_WRITE)
        rbac.remove_custom_permission("user_001", Permission.ANNOTATION_WRITE)
        assert not rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)

    def test_add_policy(self):
        """Test adding access policy."""
        rbac = RBAC()
        policy = AccessPolicy(
            resource_type="annotation",
            action="read",
            allowed=True,
        )
        rbac.add_policy(policy)
        assert len(rbac._policies) == 1

    def test_get_user_permissions(self):
        """Test getting user permissions."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        perms = rbac.get_user_permissions("user_001")
        assert Permission.ANNOTATION_READ in perms
        assert Permission.ANNOTATION_WRITE in perms
        assert Permission.ANNOTATION_DELETE not in perms

    def test_list_roles(self):
        """Test listing roles."""
        rbac = RBAC()
        roles = rbac.list_roles()
        assert "admin" in roles
        assert "reviewer" in roles
        assert "annotator" in roles
        assert "viewer" in roles
        assert "api_client" in roles

    def test_list_role_permissions(self):
        """Test listing role permissions."""
        rbac = RBAC()
        perms = rbac.list_role_permissions("viewer")
        assert "annotation:read" in perms
        assert "annotation:write" not in perms

    def test_is_admin(self):
        """Test checking if user is admin."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        assert rbac.is_admin("user_001") is True

    def test_is_not_admin(self):
        """Test checking non-admin user."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        assert rbac.is_admin("user_001") is False

    def test_role_inheritance_annotator_inherits_viewer(self):
        """Test that annotator inherits viewer permissions."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        # Annotator should have viewer permissions
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)
        assert rbac.has_permission("user_001", "resource_001", Permission.VIDEO_READ)

    def test_role_inheritance_reviewer_inherits_annotator(self):
        """Test that reviewer inherits annotator permissions."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.REVIEWER)
        # Reviewer should have annotator permissions
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_READ)

    def test_role_inheritance_admin_inherits_all(self):
        """Test that admin inherits all permissions."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        # Admin should have all permissions
        assert rbac.has_permission("user_001", "resource_001", Permission.SYSTEM_ADMIN)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_APPROVE)
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_WRITE)


class TestPermissionChecker:
    """Test PermissionChecker class."""

    def test_can_read(self):
        """Test checking read permission."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        checker = PermissionChecker(rbac, "user_001")
        assert checker.can_read("resource_001") is True

    def test_can_write_denied(self):
        """Test checking write permission denied."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)
        checker = PermissionChecker(rbac, "user_001")
        assert checker.can_write("resource_001") is False

    def test_can_delete_denied(self):
        """Test checking delete permission denied."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        checker = PermissionChecker(rbac, "user_001")
        assert checker.can_delete("resource_001") is False

    def test_can_export(self):
        """Test checking export permission."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.API_CLIENT)
        checker = PermissionChecker(rbac, "user_001")
        assert checker.can_export("resource_001") is True


class TestRBACDecorator:
    """Test RBAC decorator functionality."""

    def test_require_permission_decorator(self):
        """Test require_permission decorator."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)

        @rbac.require_permission("resource_001", Permission.ANNOTATION_WRITE)
        def protected_function(user_id, data):
            return f"success: {data}"

        result = protected_function("user_001", "test_data")
        assert "success" in result

    def test_require_permission_denied(self):
        """Test require_permission decorator denied."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.VIEWER)

        @rbac.require_permission("resource_001", Permission.ANNOTATION_WRITE)
        def protected_function(user_id, data):
            return f"success: {data}"

        with pytest.raises(PermissionError):
            protected_function("user_001", "test_data")


class TestRBACEEdgeCases:
    """Test edge cases for RBAC."""

    def test_multiple_users_same_resource(self):
        """Test multiple users with same resource."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ADMIN)
        rbac.assign_role("user_002", Role.VIEWER)
        rbac.set_owner("resource_001", "user_001")

        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)
        assert not rbac.has_permission("user_002", "resource_001", Permission.ANNOTATION_DELETE)

    def test_user_multiple_resources(self):
        """Test user with multiple resources."""
        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        rbac.set_owner("resource_001", "user_001")
        rbac.set_owner("resource_002", "user_002")

        # Owner of resource_001
        assert rbac.has_permission("user_001", "resource_001", Permission.ANNOTATION_DELETE)
        # Not owner of resource_002, but has annotator role
        assert rbac.has_permission("user_001", "resource_002", Permission.ANNOTATION_WRITE)
        assert not rbac.has_permission("user_001", "resource_002", Permission.ANNOTATION_DELETE)

    def test_empty_user_id(self):
        """Test with empty user ID."""
        rbac = RBAC()
        rbac.assign_role("", Role.VIEWER)
        assert rbac.has_permission("", "resource_001", Permission.ANNOTATION_READ)
