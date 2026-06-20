"""Secret management for DVAS.

Provides secure secret handling, including storage, retrieval,
rotation, and access control for sensitive configuration values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Secret:
    """A stored secret with metadata."""

    name: str
    value: str
    created_at: str
    updated_at: str
    version: int
    tags: List[str]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (value is NOT included)."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "tags": self.tags,
            "description": self.description,
        }


class SecretManager:
    """Secure secret management for DVAS.

    Usage::

        manager = SecretManager()
        manager.set_secret("api_key", "sk-...")
        value = manager.get_secret("api_key")

        # With encryption
        from dvas.security.encryption import EncryptionAtRest
        encryptor = EncryptionAtRest()
        manager = SecretManager(encryptor=encryptor)
        manager.set_secret("password", "my_secret")
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encryptor: Optional[Any] = None,
    ) -> None:
        """Initialize the secret manager.

        Args:
            storage_path: Path for persistent storage. If None, only in-memory.
            encryptor: Optional encryption instance for at-rest encryption.
        """
        self.storage_path = storage_path
        self.encryptor = encryptor
        self._secrets: Dict[str, Secret] = {}
        self._access_log: List[Dict[str, Any]] = []

        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_secrets()

    def set_secret(
        self,
        name: str,
        value: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a secret.

        Args:
            name: The secret name.
            value: The secret value.
            description: Optional description.
            tags: Optional tags for categorization.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Check if secret exists
        if name in self._secrets:
            existing = self._secrets[name]
            secret = Secret(
                name=name,
                value=value,
                created_at=existing.created_at,
                updated_at=now,
                version=existing.version + 1,
                tags=tags or existing.tags,
                description=description or existing.description,
            )
        else:
            secret = Secret(
                name=name,
                value=value,
                created_at=now,
                updated_at=now,
                version=1,
                tags=tags or [],
                description=description,
            )

        self._secrets[name] = secret
        self._persist_secrets()
        logger.info("secret_set", name=name, version=secret.version)

    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value.

        Args:
            name: The secret name.

        Returns:
            The secret value, or None if not found.
        """
        secret = self._secrets.get(name)
        if secret is None:
            return None

        # Log access
        self._access_log.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "read",
            }
        )

        return secret.value

    def delete_secret(self, name: str) -> bool:
        """Delete a secret.

        Args:
            name: The secret name.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self._secrets:
            return False

        del self._secrets[name]
        self._persist_secrets()
        logger.info("secret_deleted", name=name)
        return True

    def has_secret(self, name: str) -> bool:
        """Check if a secret exists.

        Args:
            name: The secret name.

        Returns:
            True if the secret exists.
        """
        return name in self._secrets

    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets (without values).

        Returns:
            List of secret metadata dictionaries.
        """
        return [s.to_dict() for s in self._secrets.values()]

    def rotate_secret(self, name: str, new_value: str) -> None:
        """Rotate a secret to a new value.

        Args:
            name: The secret name.
            new_value: The new secret value.

        Raises:
            KeyError: If the secret does not exist.
        """
        if name not in self._secrets:
            raise KeyError(f"Secret not found: {name}")

        self.set_secret(name, new_value)
        logger.info("secret_rotated", name=name)

    def get_secret_version(self, name: str) -> int:
        """Get the version of a secret.

        Args:
            name: The secret name.

        Returns:
            The version number.

        Raises:
            KeyError: If the secret does not exist.
        """
        if name not in self._secrets:
            raise KeyError(f"Secret not found: {name}")

        return self._secrets[name].version

    def _persist_secrets(self) -> None:
        """Persist secrets to storage."""
        if not self.storage_path:
            return

        data = {}
        for name, secret in self._secrets.items():
            value = secret.value
            if self.encryptor:
                value = self.encryptor.encrypt_field(value)

            data[name] = {
                "value": value,
                "created_at": secret.created_at,
                "updated_at": secret.updated_at,
                "version": secret.version,
                "tags": secret.tags,
                "description": secret.description,
            }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_secrets(self) -> None:
        """Load secrets from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for name, secret_data in data.items():
                value = secret_data["value"]
                if self.encryptor:
                    try:
                        value = self.encryptor.decrypt_field(value)
                    except Exception:
                        pass  # Not encrypted

                self._secrets[name] = Secret(
                    name=name,
                    value=value,
                    created_at=secret_data["created_at"],
                    updated_at=secret_data["updated_at"],
                    version=secret_data["version"],
                    tags=secret_data.get("tags", []),
                    description=secret_data.get("description", ""),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("failed_to_load_secrets", error=str(e))


class EnvironmentSecretProvider:
    """Provider for secrets from environment variables."""

    def __init__(self, prefix: str = "DVAS_") -> None:
        """Initialize the provider.

        Args:
            prefix: Prefix for environment variable names.
        """
        self.prefix = prefix

    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret from environment variables.

        Args:
            name: The secret name.

        Returns:
            The secret value, or None if not found.
        """
        env_name = f"{self.prefix}{name.upper()}"
        return os.environ.get(env_name)

    def set_secret(self, name: str, value: str) -> None:
        """Set a secret in environment variables.

        Args:
            name: The secret name.
            value: The secret value.
        """
        env_name = f"{self.prefix}{name.upper()}"
        os.environ[env_name] = value

    def list_secrets(self) -> List[str]:
        """List all secrets with the configured prefix.

        Returns:
            List of secret names.
        """
        return [
            key[len(self.prefix) :].lower() for key in os.environ if key.startswith(self.prefix)
        ]


class CompositeSecretProvider:
    """Composite provider that tries multiple sources."""

    def __init__(self, providers: List[Any]) -> None:
        """Initialize with multiple providers.

        Args:
            providers: List of secret providers to try in order.
        """
        self.providers = providers

    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret from any provider.

        Args:
            name: The secret name.

        Returns:
            The secret value, or None if not found in any provider.
        """
        for provider in self.providers:
            value = provider.get_secret(name)
            if value is not None:
                return value
        return None

    def set_secret(self, name: str, value: str) -> None:
        """Set a secret in the first provider.

        Args:
            name: The secret name.
            value: The secret value.
        """
        if self.providers:
            self.providers[0].set_secret(name, value)


__all__ = [
    "SecretManager",
    "Secret",
    "EnvironmentSecretProvider",
    "CompositeSecretProvider",
]
