"""Security utilities for DVAS.

Provides input sanitization, audit logging, data encryption,
and security hardening utilities.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input Validation and Sanitization
# ---------------------------------------------------------------------------

class InputValidator:
    """Validate and sanitize user inputs.

    Usage::

        validator = InputValidator()
        clean = validator.sanitize_string(user_input)
        validator.validate_video_id(video_id)
    """

    # Patterns for common attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
        r"(\b(OR|AND)\s+\d+=\d+)",
        r"(--|#|\/\*)",
        r"(\bEXEC\b|\bEXECUTE\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.",
        r"^/",
        r"^\\\\",
    ]

    def __init__(self) -> None:
        self._sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self._xss_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS]
        self._path_patterns = [re.compile(p) for p in self.PATH_TRAVERSAL_PATTERNS]

    def sanitize_string(self, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize a string input.

        Removes control characters, trims whitespace, and optionally escapes HTML.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")

        # Trim whitespace
        value = value.strip()

        # Remove control characters except newlines and tabs
        value = "".join(c for c in value if c == "\n" or c == "\t" or ord(c) >= 32)

        # Escape HTML if not allowed
        if not allow_html:
            value = html.escape(value)

        # Limit length
        if len(value) > max_length:
            value = value[:max_length]

        return value

    def validate_video_id(self, video_id: str) -> str:
        """Validate a video ID.

        Args:
            video_id: The video ID to validate

        Returns:
            The validated video ID

        Raises:
            ValueError: If the video ID is invalid
        """
        if not video_id or len(video_id) < 3:
            raise ValueError("Video ID must be at least 3 characters")

        if len(video_id) > 256:
            raise ValueError("Video ID must be at most 256 characters")

        # Allow alphanumeric, underscore, hyphen, and dots
        if not re.match(r"^[a-zA-Z0-9_.-]+$", video_id):
            raise ValueError("Video ID contains invalid characters")

        # Check for path traversal
        if ".." in video_id or "/" in video_id or "\\" in video_id:
            raise ValueError("Video ID cannot contain path separators")

        return video_id

    def validate_file_path(self, path: str, allowed_prefixes: Optional[List[str]] = None) -> str:
        """Validate a file path to prevent path traversal.

        Args:
            path: The path to validate
            allowed_prefixes: List of allowed path prefixes

        Returns:
            The validated path

        Raises:
            ValueError: If the path is invalid or outside allowed prefixes
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Normalize the path
        normalized = Path(path).resolve()

        # Check for path traversal
        for pattern in self._path_patterns:
            if pattern.search(str(normalized)):
                raise ValueError(f"Path contains traversal characters: {path}")

        # Check allowed prefixes
        if allowed_prefixes:
            allowed = False
            for prefix in allowed_prefixes:
                prefix_path = Path(prefix).resolve()
                try:
                    normalized.relative_to(prefix_path)
                    allowed = True
                    break
                except ValueError:
                    continue

            if not allowed:
                raise ValueError(f"Path outside allowed directories: {path}")

        return str(normalized)

    def check_sql_injection(self, value: str) -> List[str]:
        """Check for SQL injection patterns.

        Returns:
            List of matched patterns (empty if clean)
        """
        matches = []
        for pattern in self._sql_patterns:
            if pattern.search(value):
                matches.append(pattern.pattern)
        return matches

    def check_xss(self, value: str) -> List[str]:
        """Check for XSS patterns.

        Returns:
            List of matched patterns (empty if clean)
        """
        matches = []
        for pattern in self._xss_patterns:
            if pattern.search(value):
                matches.append(pattern.pattern)
        return matches

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal and injection.

        Args:
            filename: The filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators
        filename = filename.replace("/", "_").replace("\\", "_")

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Remove control characters
        filename = "".join(c for c in filename if ord(c) >= 32)

        # Limit length
        if len(filename) > 255:
            filename = filename[:255]

        # Ensure not empty
        if not filename:
            filename = "unnamed"

        return filename


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """Types of audit events."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    IMPORT_ = "import"
    API_CALL = "api_call"
    SYSTEM = "system"


@dataclass
class AuditEvent:
    """An audit event record."""

    event_type: AuditEventType
    user_id: str
    resource_type: str
    resource_id: str
    action: str
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
        }


class AuditLogger:
    """Audit logger for security events.

    Usage::

        audit = AuditLogger()
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        ))
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = log_dir
        self._events: List[AuditEvent] = []
        self._max_memory_events = 10000

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self._events.append(event)

        # Keep only last N events in memory
        if len(self._events) > self._max_memory_events:
            self._events = self._events[-self._max_memory_events:]

        # Write to file if configured
        if self.log_dir:
            self._write_event(event)

        # Also log to structured logger
        logger.info(
            "audit_event",
            event_type=event.event_type.value,
            user_id=event.user_id,
            resource=event.resource_type,
            action=event.action,
            success=event.success,
        )

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        if not self.log_dir:
            return

        # Use daily log files
        from datetime import datetime
        date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events matching filters."""
        results = []

        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)

            if len(results) >= limit:
                break

        return results

    def get_user_activity(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get activity for a specific user."""
        return self.get_events(user_id=user_id, limit=limit)


# ---------------------------------------------------------------------------
# Data Encryption
# ---------------------------------------------------------------------------

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
            raise ImportError("cryptography package required. Install with: pip install cryptography")

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


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

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
            import hashlib
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
            import hashlib
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


# ---------------------------------------------------------------------------
# CSRF Protection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rate Limiting (Token Bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Rate limiter for API endpoints.

    Usage::

        limiter = RateLimiter(requests_per_second=10, burst_size=20)
        if limiter.allow_request("client_001"):
            process_request()
        else:
            raise RateLimitExceeded()
    """

    def __init__(self, requests_per_second: float = 10.0, burst_size: float = 20.0) -> None:
        self.rate = requests_per_second
        self.capacity = burst_size
        self._clients: Dict[str, Dict[str, float]] = {}

    def allow_request(self, client_id: str) -> bool:
        """Check if a request is allowed.

        Returns:
            True if the request should proceed
        """
        now = time.time()

        if client_id not in self._clients:
            self._clients[client_id] = {
                "tokens": self.capacity,
                "last_update": now,
            }

        client = self._clients[client_id]

        # Add tokens based on time elapsed
        elapsed = now - client["last_update"]
        client["tokens"] = min(
            self.capacity,
            client["tokens"] + elapsed * self.rate,
        )
        client["last_update"] = now

        # Check if we have enough tokens
        if client["tokens"] >= 1.0:
            client["tokens"] -= 1.0
            return True

        return False

    def get_remaining(self, client_id: str) -> float:
        """Get remaining tokens for a client."""
        if client_id not in self._clients:
            return self.capacity

        client = self._clients[client_id]
        now = time.time()
        elapsed = now - client["last_update"]
        return min(self.capacity, client["tokens"] + elapsed * self.rate)


# ---------------------------------------------------------------------------
# Content Security Policy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Security Headers
# ---------------------------------------------------------------------------

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
        headers.add_header("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        headers.add_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        headers.add_header("X-Content-Security-Policy", "default-src 'self'")
        return headers


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Secure Random
# ---------------------------------------------------------------------------

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


# Convenience exports
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
    # Rate limiting
    "RateLimiter",
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
