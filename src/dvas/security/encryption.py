"""Encryption at rest for DVAS data.

Provides data encryption utilities for encrypting data at rest,
including file encryption, field-level encryption, and key management.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""

    algorithm: str = "fernet"
    key_rotation_days: int = 90
    key_derivation_iterations: int = 100000


class EncryptionAtRest:
    """Encrypt data at rest using Fernet encryption.

    Usage::

        encryptor = EncryptionAtRest()
        encrypted = encryptor.encrypt_field("sensitive data")
        decrypted = encryptor.decrypt_field(encrypted)

        # Encrypt a file
        encryptor.encrypt_file(
            Path("data.json"),
            Path("data.json.enc"),
        )
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        config: Optional[EncryptionConfig] = None,
    ) -> None:
        """Initialize the encryptor.

        Args:
            key: Encryption key (32 bytes). If None, generates a random key.
            config: Encryption configuration.
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            raise ImportError(
                "cryptography package required. Install with: pip install cryptography"
            )

        self.config = config or EncryptionConfig()

        if key is None:
            key = Fernet.generate_key()

        self._fernet = Fernet(key)
        self._key = key

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new encryption key.

        Returns:
            A new Fernet key.
        """
        from cryptography.fernet import Fernet

        return Fernet.generate_key()

    @classmethod
    def from_password(
        cls,
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
    ) -> "EncryptionAtRest":
        """Create an encryptor from a password.

        Args:
            password: The password to derive the key from.
            salt: Optional salt. Generated randomly if not provided.
            iterations: Number of PBKDF2 iterations.

        Returns:
            EncryptionAtRest instance.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key=key)

    def encrypt_field(self, data: str) -> str:
        """Encrypt a string field.

        Args:
            data: The string to encrypt.

        Returns:
            Base64-encoded encrypted string.
        """
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt_field(self, data: str) -> str:
        """Decrypt an encrypted string field.

        Args:
            data: Base64-encoded encrypted string.

        Returns:
            Decrypted string.
        """
        return self._fernet.decrypt(data.encode()).decode()

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt bytes.

        Args:
            data: The bytes to encrypt.

        Returns:
            Encrypted bytes.
        """
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt bytes.

        Args:
            data: Encrypted bytes.

        Returns:
            Decrypted bytes.
        """
        return self._fernet.decrypt(data)

    def encrypt_dict(self, data: Dict[str, Any], sensitive_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Encrypt sensitive fields in a dictionary.

        Args:
            data: The dictionary to encrypt.
            sensitive_fields: List of field names to encrypt. If None, encrypts all string values.

        Returns:
            Dictionary with encrypted fields.
        """
        result = {}
        for key, value in data.items():
            if sensitive_fields and key not in sensitive_fields:
                result[key] = value
            elif isinstance(value, str):
                result[key] = self.encrypt_field(value)
            elif isinstance(value, dict):
                result[key] = self.encrypt_dict(value, sensitive_fields)
            elif isinstance(value, list):
                result[key] = [
                    self.encrypt_field(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def decrypt_dict(self, data: Dict[str, Any], encrypted_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Decrypt encrypted fields in a dictionary.

        Args:
            data: The dictionary with encrypted fields.
            encrypted_fields: List of field names to decrypt. If None, attempts to decrypt all string values.

        Returns:
            Dictionary with decrypted fields.
        """
        result = {}
        for key, value in data.items():
            if encrypted_fields and key not in encrypted_fields:
                result[key] = value
            elif isinstance(value, str):
                try:
                    result[key] = self.decrypt_field(value)
                except Exception:
                    # Not encrypted, keep as is
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.decrypt_dict(value, encrypted_fields)
            elif isinstance(value, list):
                decrypted_list = []
                for item in value:
                    if isinstance(item, str):
                        try:
                            decrypted_list.append(self.decrypt_field(item))
                        except Exception:
                            decrypted_list.append(item)
                    else:
                        decrypted_list.append(item)
                result[key] = decrypted_list
            else:
                result[key] = value
        return result

    def encrypt_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Encrypt a file.

        Args:
            input_path: Path to the input file.
            output_path: Path for the encrypted output file.
        """
        with open(input_path, "rb") as f:
            data = f.read()

        encrypted = self.encrypt_bytes(data)

        with open(output_path, "wb") as f:
            f.write(encrypted)

        logger.info("file_encrypted", input=str(input_path), output=str(output_path))

    def decrypt_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Decrypt a file.

        Args:
            input_path: Path to the encrypted file.
            output_path: Path for the decrypted output file.
        """
        with open(input_path, "rb") as f:
            data = f.read()

        decrypted = self.decrypt_bytes(data)

        with open(output_path, "wb") as f:
            f.write(decrypted)

        logger.info("file_decrypted", input=str(input_path), output=str(output_path))

    def rotate_key(self) -> bytes:
        """Rotate the encryption key.

        Returns:
            The new encryption key.
        """
        from cryptography.fernet import Fernet

        new_key = Fernet.generate_key()
        self._key = new_key
        self._fernet = Fernet(new_key)
        logger.info("key_rotated")
        return new_key

    def get_key_hash(self) -> str:
        """Get a hash of the current key for identification.

        Returns:
            Hex-encoded key hash.
        """
        return hashlib.sha256(self._key).hexdigest()[:16]


class FieldEncryption:
    """Field-level encryption for database records."""

    def __init__(self, encryptor: EncryptionAtRest) -> None:
        """Initialize field encryption.

        Args:
            encryptor: The encryption instance to use.
        """
        self.encryptor = encryptor
        self._encrypted_marker = "enc:"

    def encrypt_field(self, value: str) -> str:
        """Encrypt a field value with a marker prefix.

        Args:
            value: The value to encrypt.

        Returns:
            Encrypted value with marker prefix.
        """
        encrypted = self.encryptor.encrypt_field(value)
        return self._encrypted_marker + encrypted

    def decrypt_field(self, value: str) -> str:
        """Decrypt a field value.

        Args:
            value: The encrypted value with marker prefix.

        Returns:
            Decrypted value.

        Raises:
            ValueError: If the value is not marked as encrypted.
        """
        if not value.startswith(self._encrypted_marker):
            raise ValueError("Value is not encrypted")

        encrypted = value[len(self._encrypted_marker):]
        return self.encryptor.decrypt_field(encrypted)

    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted.

        Args:
            value: The value to check.

        Returns:
            True if the value has the encrypted marker.
        """
        return isinstance(value, str) and value.startswith(self._encrypted_marker)


class KeyManager:
    """Manage encryption keys with versioning."""

    def __init__(self, key_dir: Optional[Path] = None) -> None:
        """Initialize the key manager.

        Args:
            key_dir: Directory to store keys. If None, keys are not persisted.
        """
        self.key_dir = key_dir
        self._keys: Dict[str, bytes] = {}
        self._current_version: str = "v1"

        if self.key_dir:
            self.key_dir.mkdir(parents=True, exist_ok=True)
            self._load_keys()

    def generate_key(self, version: Optional[str] = None) -> bytes:
        """Generate a new key.

        Args:
            version: Optional version identifier.

        Returns:
            The new key.
        """
        version = version or f"v{int(time.time())}"
        key = EncryptionAtRest.generate_key()
        self._keys[version] = key
        self._current_version = version

        if self.key_dir:
            self._save_key(version, key)

        return key

    def get_key(self, version: Optional[str] = None) -> bytes:
        """Get a key by version.

        Args:
            version: The key version. Uses current version if not specified.

        Returns:
            The encryption key.

        Raises:
            KeyError: If the version is not found.
        """
        version = version or self._current_version
        if version not in self._keys:
            raise KeyError(f"Key version not found: {version}")
        return self._keys[version]

    def get_current_version(self) -> str:
        """Get the current key version.

        Returns:
            The current version string.
        """
        return self._current_version

    def list_versions(self) -> List[str]:
        """List all key versions.

        Returns:
            List of version strings.
        """
        return sorted(self._keys.keys())

    def _save_key(self, version: str, key: bytes) -> None:
        """Save a key to disk."""
        if not self.key_dir:
            return

        key_file = self.key_dir / f"key_{version}.key"
        with open(key_file, "wb") as f:
            f.write(key)

    def _load_keys(self) -> None:
        """Load keys from disk."""
        if not self.key_dir:
            return

        for key_file in self.key_dir.glob("key_*.key"):
            version = key_file.stem.replace("key_", "")
            with open(key_file, "rb") as f:
                self._keys[version] = f.read()


import time


__all__ = [
    "EncryptionAtRest",
    "FieldEncryption",
    "KeyManager",
    "EncryptionConfig",
]
