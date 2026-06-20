"""Tests for secret management module.

Tests for SecretManager, Secret, EnvironmentSecretProvider, and CompositeSecretProvider.
"""

import os
from pathlib import Path

import pytest

from dvas.security.secrets import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    Secret,
    SecretManager,
)


class TestSecret:
    """Test Secret dataclass."""

    def test_secret_creation(self):
        """Test creating a secret."""
        secret = Secret(
            name="api_key",
            value="sk-test-12345",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            version=1,
            tags=["api", "production"],
            description="Test API key",
        )
        assert secret.name == "api_key"
        assert secret.value == "sk-test-12345"
        assert secret.version == 1

    def test_secret_to_dict_excludes_value(self):
        """Test that to_dict excludes the value."""
        secret = Secret(
            name="api_key",
            value="sk-test-12345",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            version=1,
            tags=[],
        )
        d = secret.to_dict()
        assert "value" not in d
        assert d["name"] == "api_key"
        assert d["version"] == 1


class TestSecretManager:
    """Test SecretManager class."""

    def test_init(self):
        """Test initialization."""
        manager = SecretManager()
        assert manager is not None

    def test_init_with_storage(self, tmp_path):
        """Test initialization with storage path."""
        storage = tmp_path / "secrets.json"
        manager = SecretManager(storage_path=storage)
        assert manager.storage_path == storage

    def test_set_secret(self):
        """Test setting a secret."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        assert manager.has_secret("api_key")

    def test_set_secret_with_description(self):
        """Test setting a secret with description."""
        manager = SecretManager()
        manager.set_secret(
            "api_key",
            "sk-test-12345",
            description="Production API key",
            tags=["production"],
        )
        secrets = manager.list_secrets()
        assert secrets[0]["description"] == "Production API key"
        assert secrets[0]["tags"] == ["production"]

    def test_get_secret(self):
        """Test getting a secret."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        value = manager.get_secret("api_key")
        assert value == "sk-test-12345"

    def test_get_secret_not_found(self):
        """Test getting non-existent secret."""
        manager = SecretManager()
        value = manager.get_secret("nonexistent")
        assert value is None

    def test_has_secret(self):
        """Test checking if secret exists."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        assert manager.has_secret("api_key") is True
        assert manager.has_secret("nonexistent") is False

    def test_delete_secret(self):
        """Test deleting a secret."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        assert manager.delete_secret("api_key") is True
        assert manager.has_secret("api_key") is False

    def test_delete_secret_not_found(self):
        """Test deleting non-existent secret."""
        manager = SecretManager()
        assert manager.delete_secret("nonexistent") is False

    def test_list_secrets(self):
        """Test listing secrets."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        manager.set_secret("password", "secret123")
        secrets = manager.list_secrets()
        assert len(secrets) == 2
        names = [s["name"] for s in secrets]
        assert "api_key" in names
        assert "password" in names

    def test_list_secrets_no_values(self):
        """Test that listed secrets don't include values."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        secrets = manager.list_secrets()
        assert "value" not in secrets[0]

    def test_rotate_secret(self):
        """Test rotating a secret."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        manager.rotate_secret("api_key", "sk-test-new-67890")
        value = manager.get_secret("api_key")
        assert value == "sk-test-new-67890"

    def test_rotate_secret_increments_version(self):
        """Test that rotation increments version."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        old_version = manager.get_secret_version("api_key")
        manager.rotate_secret("api_key", "sk-test-new-67890")
        new_version = manager.get_secret_version("api_key")
        assert new_version == old_version + 1

    def test_rotate_secret_not_found(self):
        """Test rotating non-existent secret."""
        manager = SecretManager()
        with pytest.raises(KeyError):
            manager.rotate_secret("nonexistent", "new_value")

    def test_get_secret_version(self):
        """Test getting secret version."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        version = manager.get_secret_version("api_key")
        assert version == 1

    def test_get_secret_version_not_found(self):
        """Test getting version of non-existent secret."""
        manager = SecretManager()
        with pytest.raises(KeyError):
            manager.get_secret_version("nonexistent")

    def test_persistence(self, tmp_path):
        """Test secret persistence."""
        storage = tmp_path / "secrets.json"
        manager = SecretManager(storage_path=storage)
        manager.set_secret("api_key", "sk-test-12345")

        # Create new manager pointing to same file
        manager2 = SecretManager(storage_path=storage)
        value = manager2.get_secret("api_key")
        assert value == "sk-test-12345"

    def test_persistence_with_encryption(self, tmp_path):
        """Test persistence with encryption."""
        try:
            from dvas.security.encryption import EncryptionAtRest

            encryptor = EncryptionAtRest()
            storage = tmp_path / "secrets.json"
            manager = SecretManager(storage_path=storage, encryptor=encryptor)
            manager.set_secret("api_key", "sk-test-12345")

            # Verify file is encrypted
            content = storage.read_text()
            assert "sk-test-12345" not in content

            # Create new manager with same encryptor
            manager2 = SecretManager(storage_path=storage, encryptor=encryptor)
            value = manager2.get_secret("api_key")
            assert value == "sk-test-12345"
        except ImportError:
            pytest.skip("cryptography not available")

    def test_update_existing_secret(self):
        """Test updating an existing secret."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345")
        manager.set_secret("api_key", "sk-test-67890")
        value = manager.get_secret("api_key")
        assert value == "sk-test-67890"

    def test_update_preserves_tags(self):
        """Test that update preserves existing tags."""
        manager = SecretManager()
        manager.set_secret("api_key", "sk-test-12345", tags=["production"])
        manager.set_secret("api_key", "sk-test-67890")
        secrets = manager.list_secrets()
        assert secrets[0]["tags"] == ["production"]


class TestEnvironmentSecretProvider:
    """Test EnvironmentSecretProvider class."""

    def test_init(self):
        """Test initialization."""
        provider = EnvironmentSecretProvider()
        assert provider.prefix == "DVAS_"

    def test_init_custom_prefix(self):
        """Test initialization with custom prefix."""
        provider = EnvironmentSecretProvider(prefix="APP_")
        assert provider.prefix == "APP_"

    def test_get_secret(self):
        """Test getting secret from environment."""
        provider = EnvironmentSecretProvider()
        os.environ["DVAS_API_KEY"] = "sk-test-12345"
        value = provider.get_secret("api_key")
        assert value == "sk-test-12345"
        del os.environ["DVAS_API_KEY"]

    def test_get_secret_not_found(self):
        """Test getting non-existent secret."""
        provider = EnvironmentSecretProvider()
        value = provider.get_secret("nonexistent")
        assert value is None

    def test_set_secret(self):
        """Test setting secret in environment."""
        provider = EnvironmentSecretProvider()
        provider.set_secret("api_key", "sk-test-12345")
        assert os.environ.get("DVAS_API_KEY") == "sk-test-12345"
        del os.environ["DVAS_API_KEY"]

    def test_list_secrets(self):
        """Test listing secrets."""
        provider = EnvironmentSecretProvider()
        os.environ["DVAS_KEY1"] = "value1"
        os.environ["DVAS_KEY2"] = "value2"
        os.environ["OTHER_KEY"] = "value3"

        secrets = provider.list_secrets()
        assert "key1" in secrets
        assert "key2" in secrets
        assert "other_key" not in secrets

        del os.environ["DVAS_KEY1"]
        del os.environ["DVAS_KEY2"]


class TestCompositeSecretProvider:
    """Test CompositeSecretProvider class."""

    def test_init(self):
        """Test initialization."""
        provider1 = EnvironmentSecretProvider()
        provider2 = SecretManager()
        composite = CompositeSecretProvider([provider1, provider2])
        assert len(composite.providers) == 2

    def test_get_secret_first_provider(self):
        """Test getting secret from first provider."""
        os.environ["DVAS_API_KEY"] = "from_env"
        provider1 = EnvironmentSecretProvider()
        provider2 = SecretManager()
        composite = CompositeSecretProvider([provider1, provider2])

        value = composite.get_secret("api_key")
        assert value == "from_env"
        del os.environ["DVAS_API_KEY"]

    def test_get_secret_second_provider(self):
        """Test getting secret from second provider."""
        provider1 = EnvironmentSecretProvider()
        provider2 = SecretManager()
        provider2.set_secret("api_key", "from_manager")
        composite = CompositeSecretProvider([provider1, provider2])

        value = composite.get_secret("api_key")
        assert value == "from_manager"

    def test_get_secret_not_found(self):
        """Test getting secret not in any provider."""
        provider1 = EnvironmentSecretProvider()
        provider2 = SecretManager()
        composite = CompositeSecretProvider([provider1, provider2])

        value = composite.get_secret("nonexistent")
        assert value is None

    def test_set_secret(self):
        """Test setting secret in first provider."""
        provider1 = SecretManager()
        provider2 = SecretManager()
        composite = CompositeSecretProvider([provider1, provider2])

        composite.set_secret("api_key", "sk-test-12345")
        assert provider1.get_secret("api_key") == "sk-test-12345"
        assert provider2.get_secret("api_key") is None


class TestSecretManagerEdgeCases:
    """Test edge cases for SecretManager."""

    def test_empty_secret_value(self):
        """Test setting empty secret value."""
        manager = SecretManager()
        manager.set_secret("empty", "")
        assert manager.get_secret("empty") == ""

    def test_secret_with_special_chars(self):
        """Test secret with special characters."""
        manager = SecretManager()
        value = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        manager.set_secret("special", value)
        assert manager.get_secret("special") == value

    def test_many_secrets(self):
        """Test managing many secrets."""
        manager = SecretManager()
        for i in range(100):
            manager.set_secret(f"key_{i}", f"value_{i}")

        assert len(manager.list_secrets()) == 100
        assert manager.get_secret("key_50") == "value_50"
