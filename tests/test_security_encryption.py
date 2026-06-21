"""Tests for encryption at rest module.

Tests for EncryptionAtRest, FieldEncryption, KeyManager, and EncryptionConfig.
"""

import pytest

from dvas.security.encryption import (
    EncryptionAtRest,
    EncryptionConfig,
    FieldEncryption,
    KeyManager,
)


class TestEncryptionConfig:
    """Test EncryptionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = EncryptionConfig()
        assert config.algorithm == "fernet"
        assert config.key_rotation_days == 90
        assert config.key_derivation_iterations == 100000


class TestEncryptionAtRest:
    """Test EncryptionAtRest class."""

    def test_init_default_key(self):
        """Test initialization with default key."""
        encryptor = EncryptionAtRest()
        assert encryptor is not None

    def test_init_with_key(self):
        """Test initialization with provided key."""
        key = EncryptionAtRest.generate_key()
        encryptor = EncryptionAtRest(key=key)
        assert encryptor is not None

    def test_generate_key(self):
        """Test generating a new key."""
        key = EncryptionAtRest.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_encrypt_decrypt_field(self):
        """Test encrypting and decrypting a field."""
        encryptor = EncryptionAtRest()
        plaintext = "sensitive data"
        encrypted = encryptor.encrypt_field(plaintext)
        assert encrypted != plaintext
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == plaintext

    def test_encrypt_field_different_ciphertext(self):
        """Test that same plaintext produces different ciphertext."""
        encryptor = EncryptionAtRest()
        plaintext = "test data"
        encrypted1 = encryptor.encrypt_field(plaintext)
        encrypted2 = encryptor.encrypt_field(plaintext)
        # Fernet uses random IV, so ciphertexts should differ
        assert encrypted1 != encrypted2

    def test_encrypt_decrypt_bytes(self):
        """Test encrypting and decrypting bytes."""
        encryptor = EncryptionAtRest()
        data = b"binary sensitive data"
        encrypted = encryptor.encrypt_bytes(data)
        assert encrypted != data
        decrypted = encryptor.decrypt_bytes(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_dict(self):
        """Test encrypting and decrypting a dictionary."""
        encryptor = EncryptionAtRest()
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }
        encrypted = encryptor.encrypt_dict(data)
        assert encrypted["name"] != "John Doe"
        assert encrypted["email"] != "john@example.com"
        assert encrypted["age"] == 30  # Non-string not encrypted

        decrypted = encryptor.decrypt_dict(encrypted)
        assert decrypted["name"] == "John Doe"
        assert decrypted["email"] == "john@example.com"
        assert decrypted["age"] == 30

    def test_encrypt_decrypt_dict_with_fields(self):
        """Test encrypting specific fields."""
        encryptor = EncryptionAtRest()
        data = {
            "public": "public data",
            "private": "private data",
        }
        encrypted = encryptor.encrypt_dict(data, sensitive_fields=["private"])
        assert encrypted["public"] == "public data"
        assert encrypted["private"] != "private data"

        decrypted = encryptor.decrypt_dict(encrypted, encrypted_fields=["private"])
        assert decrypted["public"] == "public data"
        assert decrypted["private"] == "private data"

    def test_decrypt_dict_not_encrypted(self):
        """Test decrypting dict with non-encrypted values."""
        encryptor = EncryptionAtRest()
        data = {
            "field1": "plain text",
            "field2": "not encrypted",
        }
        decrypted = encryptor.decrypt_dict(data)
        assert decrypted["field1"] == "plain text"
        assert decrypted["field2"] == "not encrypted"

    def test_encrypt_file(self, tmp_path):
        """Test encrypting a file."""
        encryptor = EncryptionAtRest()
        input_file = tmp_path / "test.txt"
        input_file.write_text("sensitive content")
        output_file = tmp_path / "test.txt.enc"

        encryptor.encrypt_file(input_file, output_file)
        assert output_file.exists()
        assert output_file.read_text() != "sensitive content"

    def test_decrypt_file(self, tmp_path):
        """Test decrypting a file."""
        encryptor = EncryptionAtRest()
        input_file = tmp_path / "test.txt"
        input_file.write_text("sensitive content")
        encrypted_file = tmp_path / "test.txt.enc"
        decrypted_file = tmp_path / "test_decrypted.txt"

        encryptor.encrypt_file(input_file, encrypted_file)
        encryptor.decrypt_file(encrypted_file, decrypted_file)
        assert decrypted_file.read_text() == "sensitive content"

    def test_rotate_key(self):
        """Test key rotation."""
        encryptor = EncryptionAtRest()
        old_key_hash = encryptor.get_key_hash()
        new_key = encryptor.rotate_key()
        new_key_hash = encryptor.get_key_hash()

        assert new_key is not None
        assert new_key_hash != old_key_hash

    def test_from_password(self):
        """Test creating encryptor from password."""
        encryptor = EncryptionAtRest.from_password("my_password")
        plaintext = "test data"
        encrypted = encryptor.encrypt_field(plaintext)
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == plaintext

    def test_from_password_consistency(self):
        """Test password-based encryption consistency."""
        encryptor1 = EncryptionAtRest.from_password("my_password", salt=b"fixed_salt")
        encryptor2 = EncryptionAtRest.from_password("my_password", salt=b"fixed_salt")

        plaintext = "test data"
        encrypted = encryptor1.encrypt_field(plaintext)
        decrypted = encryptor2.decrypt_field(encrypted)
        assert decrypted == plaintext

    def test_get_key_hash(self):
        """Test getting key hash."""
        encryptor = EncryptionAtRest()
        hash_value = encryptor.get_key_hash()
        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_encrypt_empty_string(self):
        """Test encrypting empty string."""
        encryptor = EncryptionAtRest()
        encrypted = encryptor.encrypt_field("")
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == ""

    def test_encrypt_unicode(self):
        """Test encrypting unicode text."""
        encryptor = EncryptionAtRest()
        plaintext = "Hello, 世界! éà"
        encrypted = encryptor.encrypt_field(plaintext)
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == plaintext


class TestFieldEncryption:
    """Test FieldEncryption class."""

    def test_init(self):
        """Test initialization."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        assert field_enc.encryptor is encryptor

    def test_encrypt_field_with_marker(self):
        """Test encrypting field with marker."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        encrypted = field_enc.encrypt_field("sensitive data")
        assert encrypted.startswith("enc:")

    def test_decrypt_field_with_marker(self):
        """Test decrypting field with marker."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        encrypted = field_enc.encrypt_field("sensitive data")
        decrypted = field_enc.decrypt_field(encrypted)
        assert decrypted == "sensitive data"

    def test_decrypt_field_without_marker(self):
        """Test decrypting field without marker."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        with pytest.raises(ValueError):
            field_enc.decrypt_field("not_encrypted")

    def test_is_encrypted_true(self):
        """Test checking encrypted field."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        encrypted = field_enc.encrypt_field("data")
        assert field_enc.is_encrypted(encrypted) is True

    def test_is_encrypted_false(self):
        """Test checking non-encrypted field."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        assert field_enc.is_encrypted("plain text") is False

    def test_is_encrypted_non_string(self):
        """Test checking non-string."""
        encryptor = EncryptionAtRest()
        field_enc = FieldEncryption(encryptor)
        assert field_enc.is_encrypted(123) is False


class TestKeyManager:
    """Test KeyManager class."""

    def test_init_no_dir(self):
        """Test initialization without directory."""
        manager = KeyManager()
        assert manager is not None

    def test_generate_key(self):
        """Test generating a key."""
        manager = KeyManager()
        key = manager.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_generate_key_with_version(self):
        """Test generating key with version."""
        manager = KeyManager()
        manager.generate_key(version="v2")
        assert manager.get_current_version() == "v2"

    def test_get_key(self):
        """Test getting a key."""
        manager = KeyManager()
        key = manager.generate_key()
        retrieved = manager.get_key()
        assert retrieved == key

    def test_get_key_specific_version(self):
        """Test getting key by version."""
        manager = KeyManager()
        key_v1 = manager.generate_key(version="v1")
        key_v2 = manager.generate_key(version="v2")

        assert manager.get_key("v1") == key_v1
        assert manager.get_key("v2") == key_v2

    def test_get_key_not_found(self):
        """Test getting non-existent key."""
        manager = KeyManager()
        with pytest.raises(KeyError):
            manager.get_key("nonexistent")

    def test_list_versions(self):
        """Test listing key versions."""
        manager = KeyManager()
        manager.generate_key(version="v1")
        manager.generate_key(version="v2")
        versions = manager.list_versions()
        assert "v1" in versions
        assert "v2" in versions

    def test_key_persistence(self, tmp_path):
        """Test key persistence."""
        key_dir = tmp_path / "keys"
        manager = KeyManager(key_dir=key_dir)
        key = manager.generate_key(version="v1")

        # Create new manager pointing to same dir
        manager2 = KeyManager(key_dir=key_dir)
        retrieved = manager2.get_key("v1")
        assert retrieved == key

    def test_multiple_keys(self):
        """Test managing multiple keys."""
        manager = KeyManager()
        manager.generate_key(version="v1")
        manager.generate_key(version="v2")
        manager.generate_key(version="v3")

        assert len(manager.list_versions()) == 3


class TestEncryptionEdgeCases:
    """Test edge cases for encryption."""

    def test_very_long_plaintext(self):
        """Test encrypting very long plaintext."""
        encryptor = EncryptionAtRest()
        plaintext = "a" * 100000
        encrypted = encryptor.encrypt_field(plaintext)
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == plaintext

    def test_special_characters(self):
        """Test encrypting special characters."""
        encryptor = EncryptionAtRest()
        plaintext = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        encrypted = encryptor.encrypt_field(plaintext)
        decrypted = encryptor.decrypt_field(encrypted)
        assert decrypted == plaintext

    def test_binary_data_in_dict(self):
        """Test dictionary with binary data."""
        encryptor = EncryptionAtRest()
        data = {
            "text": "plain text",
            "number": 42,
            "list": ["item1", "item2"],
        }
        encrypted = encryptor.encrypt_dict(data)
        decrypted = encryptor.decrypt_dict(encrypted)
        assert decrypted["text"] == "plain text"
        assert decrypted["number"] == 42
        assert decrypted["list"] == ["item1", "item2"]
