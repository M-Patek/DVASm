"""Security utilities for DVAS.

Provides input sanitization, audit logging, data encryption,
and security hardening utilities.

.. deprecated::
   This module is kept for backward compatibility. Import from the
   specialized submodules directly:
   - `dvas.security.validation` — InputValidator
   - `dvas.security.audit_log` — AuditEventType, AuditEvent, AuditLogger
   - `dvas.security.crypto` — DataEncryptor, PasswordHasher, secure_token, secure_hex
   - `dvas.security.headers` — CSRFProtection, ContentSecurityPolicy, SecurityHeaders, APIKeyManager
"""

from __future__ import annotations

# Re-export from specialized modules for backward compatibility
from dvas.security.audit_log import AuditEvent, AuditEventType, AuditLogger
from dvas.security.crypto import DataEncryptor, PasswordHasher, secure_hex, secure_token
from dvas.security.headers import (
    APIKeyManager,
    ContentSecurityPolicy,
    CSRFProtection,
    SecurityHeaders,
)
from dvas.security.validation import InputValidator

__all__ = [
    # Input validation
    "InputValidator",
    # Audit logging
    "AuditEventType",
    "AuditEvent",
    "AuditLogger",
    # Encryption
    "DataEncryptor",
    # Password hashing
    "PasswordHasher",
    # CSRF
    "CSRFProtection",
    # CSP
    "ContentSecurityPolicy",
    # Security headers
    "SecurityHeaders",
    # API keys
    "APIKeyManager",
    # Secure random
    "secure_token",
    "secure_hex",
]
