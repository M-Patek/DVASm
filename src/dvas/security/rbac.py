"""Role-based access control (RBAC) for DVAS.

Provides role-based access control with support for roles, permissions,
resource ownership, and hierarchical role inheritance.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class Permission(str, Enum):
    """Available permissions in the system."""

    # Annotation permissions
    ANNOTATION_READ = "annotation:read"
    ANNOTATION_WRITE = "annotation:write"
    ANNOTATION_DELETE = "annotation:delete"
    ANNOTATION_EXPORT = "annotation:export"
    ANNOTATION_APPROVE = "annotation:approve"

    # Video permissions
    VIDEO_READ = "video:read"
    VIDEO_UPLOAD = "video:upload"
    VIDEO_DELETE = "video:delete"

    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_MANAGE = "user:manage"

    # System
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"

    # Export
    EXPORT_READ = "export:read"
    EXPORT_CREATE = "export:create"
    EXPORT_APPROVE = "export:approve"

    # Review
    REVIEW_READ = "review:read"
    REVIEW_WRITE = "review:write"
    REVIEW_APPROVE = "review:approve"


class Role(str, Enum):
    """Built-in roles."""

    ADMIN = "admin"
    REVIEWER = "reviewer"
    ANNOTATOR = "annotator"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


# Role-to-permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.ANNOTATION_READ,
        Permission.ANNOTATION_WRITE,
        Permission.ANNOTATION_DELETE,
        Permission.ANNOTATION_EXPORT,
        Permission.ANNOTATION_APPROVE,
        Permission.VIDEO_READ,
        Permission.VIDEO_UPLOAD,
        Permission.VIDEO_DELETE,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_DELETE,
        Permission.USER_MANAGE,
        Permission.SYSTEM_ADMIN,
        Permission.SYSTEM_CONFIG,
        Permission.SYSTEM_AUDIT,
        Permission.EXPORT_READ,
        Permission.EXPORT_CREATE,
        Permission.EXPORT_APPROVE,
        Permission.REVIEW_READ,
        Permission.REVIEW_WRITE,
        Permission.REVIEW_APPROVE,
    },
    Role.REVIEWER: {
        Permission.ANNOTATION_READ,
        Permission.ANNOTATION_WRITE,
        Permission.ANNOTATION_EXPORT,
        Permission.ANNOTATION_APPROVE,
        Permission.VIDEO_READ,
        Permission.EXPORT_READ,
        Permission.EXPORT_CREATE,
        Permission.REVIEW_READ,
        Permission.REVIEW_WRITE,
        Permission.REVIEW_APPROVE,
    },
    Role.ANNOTATOR: {
        Permission.ANNOTATION_READ,
        Permission.ANNOTATION_WRITE,
        Permission.VIDEO_READ,
        Permission.EXPORT_READ,
        Permission.REVIEW_READ,
    },
    Role.VIEWER: {
        Permission.ANNOTATION_READ,
        Permission.VIDEO_READ,
        Permission.REVIEW_READ,
    },
    Role.API_CLIENT: {
        Permission.ANNOTATION_READ,
        Permission.ANNOTATION_EXPORT,
        Permission.EXPORT_READ,
        Permission.EXPORT_CREATE,
    },
}


@dataclass
class AccessPolicy:
    """A policy rule for access control."""

    resource_type: str
    action: str
    allowed: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)


class RBAC:
    """Role-based access control for DVAS.

    Usage::

        rbac = RBAC()
        rbac.assign_role("user_001", Role.ANNOTATOR)
        rbac.set_owner("annotation_001", "user_001")

        # Check permission
        if rbac.has_permission("user_001", "annotation_001", Permission.ANNOTATION_WRITE):
            # Allow write

        # Decorator
        @rbac.require_permission("annotation_001", Permission.ANNOTATION_WRITE)
        def update_annotation(user_id, ...):
            ...
    """

    def __init__(self) -> None:
        """Initialize the RBAC system."""
        self._user_roles: Dict[str, Role] = {}
        self._resource_owners: Dict[str, str] = {}
        self._custom_permissions: Dict[str, Set[Permission]] = {}
        self._policies: List[AccessPolicy] = []
        self._role_hierarchy: Dict[Role, Set[Role]] = {
            Role.ADMIN: {Role.REVIEWER},
            Role.REVIEWER: {Role.ANNOTATOR},
            Role.ANNOTATOR: {Role.VIEWER},
        }

    def assign_role(self, user_id: str, role: Union[Role, str]) -> None:
        """Assign a role to a user.

        Args:
            user_id: The user ID.
            role: The role to assign.

        Raises:
            ValueError: If the role is invalid.
        """
        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                raise ValueError(f"Invalid role: {role}")

        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Invalid role: {role}")

        self._user_roles[user_id] = role
        logger.info("role_assigned", user_id=user_id, role=role.value)

    def revoke_role(self, user_id: str) -> None:
        """Revoke a user's role.

        Args:
            user_id: The user ID.
        """
        if user_id in self._user_roles:
            del self._user_roles[user_id]
            logger.info("role_revoked", user_id=user_id)

    def get_role(self, user_id: str) -> Optional[Role]:
        """Get a user's role.

        Args:
            user_id: The user ID.

        Returns:
            The user's role, or None if not assigned.
        """
        return self._user_roles.get(user_id)

    def set_owner(self, resource_id: str, user_id: str) -> None:
        """Set the owner of a resource.

        Args:
            resource_id: The resource ID.
            user_id: The owner user ID.
        """
        self._resource_owners[resource_id] = user_id

    def get_owner(self, resource_id: str) -> Optional[str]:
        """Get the owner of a resource.

        Args:
            resource_id: The resource ID.

        Returns:
            The owner user ID, or None if not set.
        """
        return self._resource_owners.get(resource_id)

    def has_permission(
        self,
        user_id: str,
        resource_id: str,
        permission: Permission,
    ) -> bool:
        """Check if a user has a specific permission for a resource.

        Args:
            user_id: The user ID.
            resource_id: The resource ID.
            permission: The permission to check.

        Returns:
            True if the user has the permission.
        """
        # Owner always has full access
        if self._resource_owners.get(resource_id) == user_id:
            return True

        # Get user's permissions
        user_permissions = self._get_effective_permissions(user_id)
        return permission in user_permissions

    def check_permission(
        self,
        user_id: str,
        resource_id: str,
        permission: Permission,
    ) -> None:
        """Check permission and raise if not granted.

        Args:
            user_id: The user ID.
            resource_id: The resource ID.
            permission: The permission to check.

        Raises:
            PermissionError: If the user does not have the permission.
        """
        if not self.has_permission(user_id, resource_id, permission):
            role = self._user_roles.get(user_id, "unknown")
            raise PermissionError(
                f"User {user_id} (role: {role}) lacks permission {permission.value} "
                f"for resource {resource_id}"
            )

    def add_custom_permission(self, user_id: str, permission: Permission) -> None:
        """Add a custom permission for a user.

        Args:
            user_id: The user ID.
            permission: The permission to add.
        """
        if user_id not in self._custom_permissions:
            self._custom_permissions[user_id] = set()
        self._custom_permissions[user_id].add(permission)

    def remove_custom_permission(self, user_id: str, permission: Permission) -> None:
        """Remove a custom permission from a user.

        Args:
            user_id: The user ID.
            permission: The permission to remove.
        """
        if user_id in self._custom_permissions:
            self._custom_permissions[user_id].discard(permission)

    def add_policy(self, policy: AccessPolicy) -> None:
        """Add an access policy.

        Args:
            policy: The policy to add.
        """
        self._policies.append(policy)

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user.

        Args:
            user_id: The user ID.

        Returns:
            Set of permissions.
        """
        return self._get_effective_permissions(user_id)

    def require_permission(self, resource_id: str, permission: Permission):
        """Decorator to require a permission.

        Args:
            resource_id: The resource ID.
            permission: The required permission.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(user_id: str, *args, **kwargs):
                self.check_permission(user_id, resource_id, permission)
                return func(user_id, *args, **kwargs)
            return wrapper
        return decorator

    def _get_effective_permissions(self, user_id: str) -> Set[Permission]:
        """Get effective permissions for a user including role hierarchy."""
        role = self._user_roles.get(user_id)
        if not role:
            # Default to VIEWER role for unassigned users
            role = Role.VIEWER

        permissions: Set[Permission] = set()

        # Add permissions from role and inherited roles
        permissions.update(self._get_role_permissions(role))

        # Add custom permissions
        if user_id in self._custom_permissions:
            permissions.update(self._custom_permissions[user_id])

        return permissions

    def _get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a role, including inherited ones."""
        permissions = set(ROLE_PERMISSIONS.get(role, set()))

        # Add permissions from child roles (roles that this role encompasses)
        for child_role, _ in self._role_hierarchy.items():
            # Check if this role is the parent of child_role in the hierarchy
            # by checking if role is in the hierarchy chain of child_role
            pass

        # Recursively collect permissions from child roles
        visited = {role}
        queue = list(self._role_hierarchy.get(role, set()))
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                permissions.update(ROLE_PERMISSIONS.get(current, set()))
                queue.extend(self._role_hierarchy.get(current, set()))

        return permissions

    def list_roles(self) -> List[str]:
        """List all available role names.

        Returns:
            List of role names.
        """
        return [r.value for r in Role]

    def list_role_permissions(self, role: Union[Role, str]) -> List[str]:
        """List permissions for a role.

        Args:
            role: The role to query.

        Returns:
            List of permission names.
        """
        if isinstance(role, str):
            role = Role(role)

        perms = self._get_role_permissions(role)
        return sorted([p.value for p in perms])

    def is_admin(self, user_id: str) -> bool:
        """Check if a user is an admin.

        Args:
            user_id: The user ID.

        Returns:
            True if the user has the admin role.
        """
        return self._user_roles.get(user_id) == Role.ADMIN


class PermissionChecker:
    """Convenience class for checking multiple permissions."""

    def __init__(self, rbac: RBAC, user_id: str) -> None:
        """Initialize permission checker.

        Args:
            rbac: The RBAC instance.
            user_id: The user ID to check permissions for.
        """
        self.rbac = rbac
        self.user_id = user_id

    def can_read(self, resource_id: str) -> bool:
        """Check if user can read a resource."""
        return self.rbac.has_permission(self.user_id, resource_id, Permission.ANNOTATION_READ)

    def can_write(self, resource_id: str) -> bool:
        """Check if user can write a resource."""
        return self.rbac.has_permission(self.user_id, resource_id, Permission.ANNOTATION_WRITE)

    def can_delete(self, resource_id: str) -> bool:
        """Check if user can delete a resource."""
        return self.rbac.has_permission(self.user_id, resource_id, Permission.ANNOTATION_DELETE)

    def can_export(self, resource_id: str) -> bool:
        """Check if user can export a resource."""
        return self.rbac.has_permission(self.user_id, resource_id, Permission.ANNOTATION_EXPORT)


__all__ = [
    "RBAC",
    "Permission",
    "Role",
    "AccessPolicy",
    "PermissionChecker",
    "ROLE_PERMISSIONS",
]
