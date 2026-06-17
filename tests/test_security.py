"""Security tests for DVAS.

Tests for input validation, audit logging, encryption, and security utilities.
"""

import pytest
from unittest.mock import MagicMock, patch

from dvas.security.audit import (
    APIKeyManager,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    CSRFProtection,
    ContentSecurityPolicy,
    DataEncryptor,
    InputValidator,
    PasswordHasher,
    RateLimiter,
    SecurityHeaders,
    secure_hex,
    secure_token,
)


class TestInputValidator:
    """Test input validation and sanitization."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        validator = InputValidator()
        assert validator.sanitize_string("  hello  ") == "hello"

    def test_sanitize_string_html_escaping(self):
        """Test HTML escaping in string sanitization."""
        validator = InputValidator()
        result = validator.sanitize_string("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_sanitize_string_max_length(self):
        """Test max length truncation."""
        validator = InputValidator()
        result = validator.sanitize_string("a" * 1000, max_length=10)
        assert len(result) == 10

    def test_sanitize_string_control_chars(self):
        """Test removal of control characters."""
        validator = InputValidator()
        result = validator.sanitize_string("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_string_non_string_input(self):
        """Test that non-string input raises error."""
        validator = InputValidator()
        with pytest.raises(ValueError):
            validator.sanitize_string(123)

    def test_validate_video_id_valid(self):
        """Test valid video ID validation."""
        validator = InputValidator()
        assert validator.validate_video_id("vid_001") == "vid_001"
        assert validator.validate_video_id("video-123") == "video-123"
        assert validator.validate_video_id("test.video") == "test.video"

    def test_validate_video_id_too_short(self):
        """Test video ID too short."""
        validator = InputValidator()
        with pytest.raises(ValueError):
            validator.validate_video_id("ab")

    def test_validate_video_id_path_traversal(self):
        """Test video ID with path traversal."""
        validator = InputValidator()
        with pytest.raises(ValueError):
            validator.validate_video_id("../etc/passwd")

    def test_validate_video_id_invalid_chars(self):
        """Test video ID with invalid characters."""
        validator = InputValidator()
        with pytest.raises(ValueError):
            validator.validate_video_id("vid<001>")

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        validator = InputValidator()
        assert validator.sanitize_filename("../test.txt") == ".._test.txt"
        assert validator.sanitize_filename("file\x00name") == "filename"
        assert validator.sanitize_filename("") == "unnamed"

    def test_check_sql_injection(self):
        """Test SQL injection detection."""
        validator = InputValidator()
        matches = validator.check_sql_injection("SELECT * FROM users")
        assert len(matches) > 0

    def test_check_sql_injection_clean(self):
        """Test clean input passes SQL injection check."""
        validator = InputValidator()
        matches = validator.check_sql_injection("Hello world")
        assert len(matches) == 0

    def test_check_xss(self):
        """Test XSS detection."""
        validator = InputValidator()
        matches = validator.check_xss("<script>alert('xss')</script>")
        assert len(matches) > 0

    def test_check_xss_clean(self):
        """Test clean input passes XSS check."""
        validator = InputValidator()
        matches = validator.check_xss("Hello world")
        assert len(matches) == 0


class TestAuditLogger:
    """Test audit logging."""

    def test_log_event(self):
        """Test logging an audit event."""
        audit = AuditLogger()
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created annotation",
        )
        audit.log_event(event)
        assert len(audit._events) == 1

    def test_log_event_with_details(self):
        """Test logging event with details."""
        audit = AuditLogger()
        event = AuditEvent(
            event_type=AuditEventType.UPDATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="updated segments",
            details={"segments_changed": 2},
        )
        audit.log_event(event)
        assert audit._events[0].details["segments_changed"] == 2

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        audit = AuditLogger()
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created",
        ))
        audit.log_event(AuditEvent(
            event_type=AuditEventType.READ,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="read",
        ))

        create_events = audit.get_events(event_type=AuditEventType.CREATE)
        assert len(create_events) == 1
        assert create_events[0].event_type == AuditEventType.CREATE

    def test_get_events_by_user(self):
        """Test filtering events by user."""
        audit = AuditLogger()
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created",
        ))
        audit.log_event(AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_002",
            resource_type="annotation",
            resource_id="ann_002",
            action="created",
        ))

        user_events = audit.get_events(user_id="user_001")
        assert len(user_events) == 1
        assert user_events[0].user_id == "user_001"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            user_id="user_001",
            resource_type="annotation",
            resource_id="ann_001",
            action="created",
            success=True,
        )
        d = event.to_dict()
        assert d["event_type"] == "create"
        assert d["user_id"] == "user_001"
        assert d["success"] is True

    def test_memory_limit(self):
        """Test that memory events are limited."""
        audit = AuditLogger()
        audit._max_memory_events = 5

        for i in range(10):
            audit.log_event(AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id=f"user_{i}",
                resource_type="annotation",
                resource_id=f"ann_{i}",
                action="created",
            ))

        assert len(audit._events) == 5


class TestCSRFProtection:
    """Test CSRF protection."""

    def test_generate_token(self):
        """Test CSRF token generation."""
        csrf = CSRFProtection(secret_key="test_secret")
        token = csrf.generate_token("session_001")
        assert isinstance(token, str)
        assert ":" in token

    def test_validate_token(self):
        """Test CSRF token validation."""
        csrf = CSRFProtection(secret_key="test_secret")
        token = csrf.generate_token("session_001")
        assert csrf.validate_token(token, "session_001")

    def test_validate_wrong_session(self):
        """Test CSRF token with wrong session."""
        csrf = CSRFProtection(secret_key="test_secret")
        token = csrf.generate_token("session_001")
        assert not csrf.validate_token(token, "session_002")

    def test_validate_expired_token(self):
        """Test expired CSRF token."""
        csrf = CSRFProtection(secret_key="test_secret")
        # Generate a token with an old timestamp by monkeypatching time.time
        import time as time_module
        old_time = time_module.time() - 10  # 10 seconds ago
        with patch.object(time_module, "time", return_value=old_time):
            token = csrf.generate_token("session_001")
        # Now validate with max_age=5 (should be expired)
        assert not csrf.validate_token(token, "session_001", max_age=5)

    def test_validate_invalid_token(self):
        """Test invalid CSRF token format."""
        csrf = CSRFProtection(secret_key="test_secret")
        assert not csrf.validate_token("invalid", "session_001")

    def test_validate_tampered_token(self):
        """Test tampered CSRF token."""
        csrf = CSRFProtection(secret_key="test_secret")
        token = csrf.generate_token("session_001")
        tampered = token[:-10] + "tampered123"
        assert not csrf.validate_token(tampered, "session_001")


class TestRateLimiter:
    """Test rate limiting."""

    def test_allow_request_within_limit(self):
        """Test request within rate limit."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        assert limiter.allow_request("client_001")

    def test_allow_request_exceeds_limit(self):
        """Test request exceeds rate limit."""
        limiter = RateLimiter(requests_per_second=1, burst_size=1)
        assert limiter.allow_request("client_001")
        assert not limiter.allow_request("client_001")

    def test_different_clients_independent(self):
        """Test rate limits are per-client."""
        limiter = RateLimiter(requests_per_second=1, burst_size=1)
        assert limiter.allow_request("client_001")
        assert limiter.allow_request("client_002")

    def test_get_remaining(self):
        """Test getting remaining tokens."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        assert limiter.get_remaining("client_001") == 5.0
        limiter.allow_request("client_001")
        assert limiter.get_remaining("client_001") >= 3.9


class TestSecurityHeaders:
    """Test security headers."""

    def test_default_headers(self):
        """Test default security headers."""
        headers = SecurityHeaders.default()
        h = headers.to_dict()
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert h["X-XSS-Protection"] == "1; mode=block"
        assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Strict-Transport-Security" in h

    def test_add_header(self):
        """Test adding custom header."""
        headers = SecurityHeaders()
        headers.add_header("X-Custom", "value")
        assert headers.to_dict()["X-Custom"] == "value"


class TestContentSecurityPolicy:
    """Test Content Security Policy."""

    def test_default_csp(self):
        """Test default CSP."""
        csp = ContentSecurityPolicy.default()
        header = csp.to_header()
        assert "default-src 'self'" in header
        assert "frame-ancestors 'none'" in header

    def test_add_directive(self):
        """Test adding custom directive."""
        csp = ContentSecurityPolicy()
        csp.add_directive("script-src", ["'self'", "https://example.com"])
        assert "script-src 'self' https://example.com" in csp.to_header()

    def test_to_headers(self):
        """Test generating response headers."""
        csp = ContentSecurityPolicy.default()
        headers = csp.to_headers()
        assert "Content-Security-Policy" in headers


class TestAPIKeyManager:
    """Test API key management."""

    def test_create_key(self):
        """Test API key creation."""
        manager = APIKeyManager()
        key = manager.create_key("user_001", name="test_key")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_validate_key(self):
        """Test API key validation."""
        manager = APIKeyManager()
        key = manager.create_key("user_001")
        assert manager.validate_key(key)

    def test_validate_invalid_key(self):
        """Test invalid API key."""
        manager = APIKeyManager()
        assert not manager.validate_key("invalid_key")

    def test_revoke_key(self):
        """Test API key revocation."""
        manager = APIKeyManager()
        key = manager.create_key("user_001")
        assert manager.validate_key(key)
        assert manager.revoke_key(key)
        assert not manager.validate_key(key)

    def test_revoke_nonexistent_key(self):
        """Test revoking non-existent key."""
        manager = APIKeyManager()
        assert not manager.revoke_key("nonexistent")

    def test_get_key_info(self):
        """Test getting key information."""
        manager = APIKeyManager()
        key = manager.create_key("user_001", name="test")
        info = manager.get_key_info(key)
        assert info is not None
        assert info["user_id"] == "user_001"
        assert info["name"] == "test"

    def test_key_usage_count(self):
        """Test key usage counting."""
        manager = APIKeyManager()
        key = manager.create_key("user_001")
        manager.validate_key(key)
        manager.validate_key(key)
        info = manager.get_key_info(key)
        assert info["usage_count"] == 2


class TestSecureRandom:
    """Test secure random generation."""

    def test_secure_token(self):
        """Test secure token generation."""
        token = secure_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_secure_token_unique(self):
        """Test secure tokens are unique."""
        tokens = [secure_token() for _ in range(100)]
        assert len(set(tokens)) == 100

    def test_secure_hex(self):
        """Test secure hex generation."""
        hex_str = secure_hex()
        assert isinstance(hex_str, str)
        assert all(c in "0123456789abcdef" for c in hex_str)

    def test_secure_hex_length(self):
        """Test secure hex length."""
        hex_str = secure_hex(length=16)
        assert len(hex_str) == 32  # 16 bytes = 32 hex chars


class TestPasswordHasher:
    """Test password hashing."""

    def test_hash_password(self):
        """Test password hashing."""
        hasher = PasswordHasher()
        hashed = hasher.hash_password("my_password")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_verify_password(self):
        """Test password verification."""
        hasher = PasswordHasher()
        hashed = hasher.hash_password("my_password")
        assert hasher.verify_password("my_password", hashed)

    def test_verify_wrong_password(self):
        """Test wrong password verification."""
        hasher = PasswordHasher()
        hashed = hasher.hash_password("my_password")
        assert not hasher.verify_password("wrong_password", hashed)

    def test_different_passwords_different_hashes(self):
        """Test different passwords produce different hashes."""
        hasher = PasswordHasher()
        hash1 = hasher.hash_password("password1")
        hash2 = hasher.hash_password("password2")
        assert hash1 != hash2

    def test_same_password_different_hashes(self):
        """Test same password produces different hashes (due to salt)."""
        hasher = PasswordHasher()
        hash1 = hasher.hash_password("my_password")
        hash2 = hasher.hash_password("my_password")
        assert hash1 != hash2


class TestDataEncryptor:
    """Test data encryption."""

    def test_encrypt_decrypt(self):
        """Test encryption and decryption."""
        encryptor = DataEncryptor()
        original = "sensitive data"
        encrypted = encryptor.encrypt(original)
        assert encrypted != original
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_bytes(self):
        """Test byte encryption."""
        encryptor = DataEncryptor()
        original = b"sensitive bytes"
        encrypted = encryptor.encrypt_bytes(original)
        assert encrypted != original
        decrypted = encryptor.decrypt_bytes(encrypted)
        assert decrypted == original

    def test_different_data_different_ciphertext(self):
        """Test different data produces different ciphertext."""
        encryptor = DataEncryptor()
        cipher1 = encryptor.encrypt("data1")
        cipher2 = encryptor.encrypt("data2")
        assert cipher1 != cipher2

    def test_generate_key(self):
        """Test key generation."""
        key = DataEncryptor.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_encrypt_with_custom_key(self):
        """Test encryption with custom key."""
        key = DataEncryptor.generate_key()
        encryptor = DataEncryptor(key=key)
        original = "test data"
        encrypted = encryptor.encrypt(original)

        # Decrypt with same key
        decryptor = DataEncryptor(key=key)
        decrypted = decryptor.decrypt(encrypted)
        assert decrypted == original
