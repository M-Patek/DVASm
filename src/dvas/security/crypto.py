"""Data encryption and password hashing for DVAS.

Provides Fernet-based encryption, bcrypt password hashing,
and secure random token generation.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional


class DataEncryptor:
    """Simple data encryption using Fernet (from cryptography).

    Usage::

        encryptor = DataEncryptor(key=b"...")
        encrypted = encryptor.encrypt("sensitive data")
        decrypted = encryptor.decrypt(encrypted)
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        """Initialize encryptor with a key.

        Args:
            key: Encryption key (32 bytes). If None, generates a random key.
        """
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError(
                "cryptography package required. Install with: pip install cryptography"
            )

        if key is None:
            key = Fernet.generate_key()

        self._fernet = Fernet(key)
        self._key = key

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new encryption key."""
        from cryptography.fernet import Fernet

        return Fernet.generate_key()

    def encrypt(self, data: str) -> str:
        """Encrypt string data.

        Returns:
            Base64-encoded encrypted string
        """
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        """Decrypt string data.

        Args:
            data: Base64-encoded encrypted string

        Returns:
            Decrypted string
        """
        return self._fernet.decrypt(data.encode()).decode()

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt bytes."""
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt bytes."""
        return self._fernet.decrypt(data)


class PasswordHasher:
    """Secure password hashing using bcrypt.

    Usage::

        hasher = PasswordHasher()
        hashed = hasher.hash_password("my_password")
        assert hasher.verify_password("my_password", hashed)
    """

    def __init__(self, rounds: int = 12) -> None:
        self.rounds = rounds

    def hash_password(self, password: str) -> str:
        """Hash a password.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        try:
            import bcrypt

            password_bytes = password.encode("utf-8")
            salt = bcrypt.gensalt(rounds=self.rounds)
            return bcrypt.hashpw(password_bytes, salt).decode("utf-8")
        except ImportError:
            # Fallback to simple hash (NOT for production)
            salt = secrets.token_hex(16)
            hash_value = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000,
            )
            return f"fallback:{salt}:{hash_value.hex()}"

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against a hash.

        Args:
            password: Plain text password
            hashed: Previously hashed password

        Returns:
            True if password matches
        """
        if hashed.startswith("fallback:"):
            parts = hashed.split(":")
            if len(parts) != 3:
                return False
            salt = parts[1]
            expected_hash = parts[2]
            hash_value = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000,
            )
            return hash_value.hex() == expected_hash

        try:
            import bcrypt

            password_bytes = password.encode("utf-8")
            hashed_bytes = hashed.encode("utf-8")
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except ImportError:
            return False


def secure_token(length: int = 32) -> str:
    """Generate a secure random token.

    Args:
        length: Length of the token in bytes

    Returns:
        URL-safe base64-encoded token
    """
    return secrets.token_urlsafe(length)


def secure_hex(length: int = 32) -> str:
    """Generate a secure random hex string.

    Args:
        length: Number of random bytes

    Returns:
        Hex-encoded string
    """
    return secrets.token_hex(length)


__all__ = ["DataEncryptor", "PasswordHasher", "secure_token", "secure_hex"]
