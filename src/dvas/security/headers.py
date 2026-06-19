"""CSRF protection, CSP builder, security headers, and API key management.

Provides CSRF token generation/validation, Content Security Policy building,
HTTP security headers, and API key management.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, List, Optional


class CSRFProtection:
    """CSRF token generation and validation.

    Usage::

        csrf = CSRFProtection()
        token = csrf.generate_token("user_001")
        assert csrf.validate_token(token, "user_001")
    """

    def __init__(self, secret_key: Optional[str] = None) -> None:
        self.secret_key = secret_key or secrets.token_hex(32)

    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token for a session."""
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{data}:{signature}"

    def validate_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate a CSRF token.

        Args:
            token: The token to validate
            session_id: Expected session ID
            max_age: Maximum age in seconds

        Returns:
            True if valid
        """
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False

            token_session, timestamp_str, signature = parts
            timestamp = int(timestamp_str)

            # Check session
            if token_session != session_id:
                return False

            # Check age
            if int(time.time()) - timestamp > max_age:
                return False

            # Verify signature
            data = f"{token_session}:{timestamp_str}"
            expected = hmac.new(
                self.secret_key.encode(),
                data.encode(),
                hashlib.sha256,
            ).hexdigest()

            return hmac.compare_digest(signature, expected)

        except (ValueError, IndexError):
            return False


class ContentSecurityPolicy:
    """Content Security Policy builder.

    Usage::

        csp = ContentSecurityPolicy()
        csp.add_directive("default-src", ["'self'"])
        csp.add_directive("script-src", ["'self'", "'unsafe-inline'"])
        headers = csp.to_headers()
    """

    def __init__(self) -> None:
        self._directives: Dict[str, List[str]] = {}

    def add_directive(self, name: str, values: List[str]) -> None:
        """Add a directive."""
        self._directives[name] = values

    def to_header(self) -> str:
        """Generate CSP header value."""
        parts = []
        for name, values in self._directives.items():
            parts.append(f"{name} {' '.join(values)}")
        return "; ".join(parts)

    def to_headers(self) -> Dict[str, str]:
        """Generate response headers dict."""
        return {
            "Content-Security-Policy": self.to_header(),
        }

    @classmethod
    def default(cls) -> "ContentSecurityPolicy":
        """Create a default CSP."""
        csp = cls()
        csp.add_directive("default-src", ["'self'"])
        csp.add_directive("script-src", ["'self'"])
        csp.add_directive("style-src", ["'self'", "'unsafe-inline'"])
        csp.add_directive("img-src", ["'self'", "data:", "blob:"])
        csp.add_directive("font-src", ["'self'"])
        csp.add_directive("connect-src", ["'self'"])
        csp.add_directive("frame-ancestors", ["'none'"])
        csp.add_directive("base-uri", ["'self'"])
        csp.add_directive("form-action", ["'self'"])
        return csp


class SecurityHeaders:
    """Security headers for HTTP responses.

    Usage::

        headers = SecurityHeaders.default()
        response_headers = headers.to_dict()
    """

    def __init__(self) -> None:
        self._headers: Dict[str, str] = {}

    def add_header(self, name: str, value: str) -> None:
        """Add a header."""
        self._headers[name] = value

    def to_dict(self) -> Dict[str, str]:
        """Get all headers as dict."""
        return self._headers.copy()

    @classmethod
    def default(cls) -> "SecurityHeaders":
        """Create default security headers."""
        headers = cls()
        headers.add_header("X-Content-Type-Options", "nosniff")
        headers.add_header("X-Frame-Options", "DENY")
        headers.add_header("X-XSS-Protection", "1; mode=block")
        headers.add_header("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.add_header("Permissions-Policy", "geolocation=(), microphone=(), camera()")
        headers.add_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        headers.add_header("X-Content-Security-Policy", "default-src 'self'")
        return headers


class APIKeyManager:
    """Manage API keys for authentication.

    Usage::

        manager = APIKeyManager()
        key = manager.create_key("user_001")
        assert manager.validate_key(key)
    """

    def __init__(self) -> None:
        self._keys: Dict[str, Dict[str, Any]] = {}

    def create_key(self, user_id: str, name: str = "") -> str:
        """Create a new API key.

        Returns:
            The API key string
        """
        key = secrets.token_urlsafe(32)
        self._keys[key] = {
            "user_id": user_id,
            "name": name,
            "created_at": time.time(),
            "last_used": None,
            "usage_count": 0,
        }
        return key

    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        if key not in self._keys:
            return False

        self._keys[key]["last_used"] = time.time()
        self._keys[key]["usage_count"] += 1
        return True

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key.

        Returns:
            True if key was revoked
        """
        if key in self._keys:
            del self._keys[key]
            return True
        return False

    def get_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key."""
        return self._keys.get(key)


__all__ = [
    "CSRFProtection",
    "ContentSecurityPolicy",
    "SecurityHeaders",
    "APIKeyManager",
]
