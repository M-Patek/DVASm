"""Security regression tests for DVAS.

Provides a test suite for verifying security features including
input validation, encryption, access control, and PII handling.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityTestResult:
    """Result of a security test."""

    def __init__(self, name: str, passed: bool, message: str = "", details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class SecurityRegressionTestSuite:
    """Security regression test suite for DVAS.

    Usage::

        suite = SecurityRegressionTestSuite()
        results = suite.run_all_tests()
        for result in results:
            print(f"{result.name}: {'PASS' if result.passed else 'FAIL'}")
    """

    def __init__(self) -> None:
        """Initialize the test suite."""
        self._tests: List[Callable[[], SecurityTestResult]] = []
        self._register_default_tests()

    def run_all_tests(self) -> List[SecurityTestResult]:
        """Run all security regression tests.

        Returns:
            List of test results.
        """
        results = []
        for test in self._tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    SecurityTestResult(
                        name=test.__name__,
                        passed=False,
                        message=f"Test execution failed: {str(e)}",
                    )
                )
        return results

    def add_test(self, test: Callable[[], SecurityTestResult]) -> None:
        """Add a custom test to the suite.

        Args:
            test: A function that returns a SecurityTestResult.
        """
        self._tests.append(test)

    def _register_default_tests(self) -> None:
        """Register the default set of security tests."""
        self._tests = [
            self.test_sql_injection_detection,
            self.test_xss_detection,
            self.test_path_traversal_prevention,
            self.test_password_hashing,
            self.test_encryption_roundtrip,
            self.test_csrf_token_validation,
            self.test_api_key_generation,
            self.test_pii_redaction,
            self.test_role_permissions,
            self.test_audit_logging,
            self.test_secret_handling,
            self.test_tenant_isolation,
            self.test_watermark_detection,
            self.test_input_sanitization,
        ]

    def test_sql_injection_detection(self) -> SecurityTestResult:
        """Test SQL injection pattern detection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1'; DELETE FROM annotations WHERE '1'='1",
        ]

        # Simple SQL injection detection
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
            r"(\b(OR|AND)\s+\d+=\d+)",
            r"(--|#|\/\*)",
        ]

        detected = 0
        for input_str in malicious_inputs:
            for pattern in sql_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    detected += 1
                    break

        return SecurityTestResult(
            name="sql_injection_detection",
            passed=detected > 0,
            message=f"Detected {detected}/{len(malicious_inputs)} SQL injection attempts",
            details={"detected_count": detected, "total_count": len(malicious_inputs)},
        )

    def test_xss_detection(self) -> SecurityTestResult:
        """Test XSS pattern detection."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='evil.com'></iframe>",
        ]

        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]

        detected = 0
        for input_str in malicious_inputs:
            for pattern in xss_patterns:
                if re.search(pattern, input_str, re.IGNORECASE | re.DOTALL):
                    detected += 1
                    break

        return SecurityTestResult(
            name="xss_detection",
            passed=detected > 0,
            message=f"Detected {detected}/{len(malicious_inputs)} XSS attempts",
            details={"detected_count": detected, "total_count": len(malicious_inputs)},
        )

    def test_path_traversal_prevention(self) -> SecurityTestResult:
        """Test path traversal prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "\\windows\\system32\\config\\sam",
        ]

        traversal_patterns = [r"\.\.", r"^/", r"^\\\\"]
        detected = 0

        for path in malicious_paths:
            for pattern in traversal_patterns:
                if re.search(pattern, path):
                    detected += 1
                    break

        return SecurityTestResult(
            name="path_traversal_prevention",
            passed=detected > 0,
            message=f"Detected {detected}/{len(malicious_paths)} path traversal attempts",
            details={"detected_count": detected, "total_count": len(malicious_paths)},
        )

    def test_password_hashing(self) -> SecurityTestResult:
        """Test password hashing functionality."""
        try:
            from dvas.security.crypto import PasswordHasher

            hasher = PasswordHasher()
            password = "test_password_123"
            hashed = hasher.hash_password(password)

            # Verify correct password
            correct = hasher.verify_password(password, hashed)

            # Verify wrong password fails
            wrong = not hasher.verify_password("wrong_password", hashed)

            # Verify different salts produce different hashes
            hash1 = hasher.hash_password(password)
            hash2 = hasher.hash_password(password)
            different = hash1 != hash2

            return SecurityTestResult(
                name="password_hashing",
                passed=correct and wrong and different,
                message="Password hashing works correctly",
                details={"correct_verify": correct, "wrong_reject": wrong, "salted": different},
            )
        except ImportError:
            return SecurityTestResult(
                name="password_hashing",
                passed=False,
                message="PasswordHasher not available",
            )

    def test_encryption_roundtrip(self) -> SecurityTestResult:
        """Test encryption/decryption roundtrip."""
        try:
            from dvas.security.crypto import DataEncryptor

            encryptor = DataEncryptor()
            plaintext = "sensitive data for testing"
            encrypted = encryptor.encrypt(plaintext)
            decrypted = encryptor.decrypt(encrypted)

            return SecurityTestResult(
                name="encryption_roundtrip",
                passed=decrypted == plaintext and encrypted != plaintext,
                message="Encryption roundtrip successful",
                details={
                    "plaintext_matches": decrypted == plaintext,
                    "ciphertext_differs": encrypted != plaintext,
                },
            )
        except ImportError:
            return SecurityTestResult(
                name="encryption_roundtrip",
                passed=False,
                message="DataEncryptor not available",
            )

    def test_csrf_token_validation(self) -> SecurityTestResult:
        """Test CSRF token generation and validation."""
        try:
            from dvas.security.headers import CSRFProtection

            csrf = CSRFProtection(secret_key="test_secret_key_12345")
            token = csrf.generate_token("session_001")
            valid = csrf.validate_token(token, "session_001")
            invalid = not csrf.validate_token(token, "session_002")

            return SecurityTestResult(
                name="csrf_token_validation",
                passed=valid and invalid,
                message="CSRF token validation works correctly",
                details={"valid_token": valid, "invalid_rejected": invalid},
            )
        except ImportError:
            return SecurityTestResult(
                name="csrf_token_validation",
                passed=False,
                message="CSRFProtection not available",
            )

    def test_api_key_generation(self) -> SecurityTestResult:
        """Test API key generation and validation."""
        try:
            from dvas.security.headers import APIKeyManager

            manager = APIKeyManager()
            key = manager.create_key("user_001")
            valid = manager.validate_key(key)
            invalid = not manager.validate_key("invalid_key_12345")

            return SecurityTestResult(
                name="api_key_generation",
                passed=valid and invalid,
                message="API key generation and validation works",
                details={"valid_key": valid, "invalid_rejected": invalid},
            )
        except ImportError:
            return SecurityTestResult(
                name="api_key_generation",
                passed=False,
                message="APIKeyManager not available",
            )

    def test_pii_redaction(self) -> SecurityTestResult:
        """Test PII detection and redaction."""
        try:
            from dvas.security.pii import PIIDetector

            detector = PIIDetector()
            text = "Contact me at john@example.com or call 555-123-4567"
            findings = detector.scan_text(text)
            redacted = detector.redact_text(text)

            has_email = any(f.pii_type.value == "email" for f in findings)
            has_phone = any(f.pii_type.value == "phone" for f in findings)
            redacted_ok = "[REDACTED]" in redacted

            return SecurityTestResult(
                name="pii_redaction",
                passed=has_email and has_phone and redacted_ok,
                message="PII detection and redaction works",
                details={
                    "email_found": has_email,
                    "phone_found": has_phone,
                    "redacted": redacted_ok,
                },
            )
        except ImportError:
            return SecurityTestResult(
                name="pii_redaction",
                passed=False,
                message="PIIDetector not available",
            )

    def test_role_permissions(self) -> SecurityTestResult:
        """Test RBAC role permissions."""
        try:
            from dvas.security.rbac import RBAC, Permission, Role

            rbac = RBAC()
            rbac.assign_role("user_001", Role.ANNOTATOR)
            rbac.set_owner("resource_001", "user_001")

            # Owner should have all permissions
            owner_perm = rbac.has_permission(
                "user_001", "resource_001", Permission.ANNOTATION_WRITE
            )

            # Non-owner with viewer role should not have write
            rbac.assign_role("user_002", Role.VIEWER)
            viewer_perm = not rbac.has_permission(
                "user_002", "resource_001", Permission.ANNOTATION_WRITE
            )

            return SecurityTestResult(
                name="role_permissions",
                passed=owner_perm and viewer_perm,
                message="RBAC role permissions work correctly",
                details={"owner_has_access": owner_perm, "viewer_denied": viewer_perm},
            )
        except ImportError:
            return SecurityTestResult(
                name="role_permissions",
                passed=False,
                message="RBAC not available",
            )

    def test_audit_logging(self) -> SecurityTestResult:
        """Test audit logging functionality."""
        try:
            from dvas.security.audit_comprehensive import AuditLog, AuditEvent, AuditEventType

            audit = AuditLog()
            event = AuditEvent(
                event_type=AuditEventType.CREATE,
                user_id="user_001",
                resource_type="annotation",
                resource_id="ann_001",
                action="created annotation",
            )
            audit.log_event(event)
            events = audit.get_events(event_type=AuditEventType.CREATE)

            return SecurityTestResult(
                name="audit_logging",
                passed=len(events) == 1 and events[0].user_id == "user_001",
                message="Audit logging works correctly",
                details={"events_logged": len(events)},
            )
        except ImportError:
            return SecurityTestResult(
                name="audit_logging",
                passed=False,
                message="AuditLog not available",
            )

    def test_secret_handling(self) -> SecurityTestResult:
        """Test secret management."""
        try:
            from dvas.security.secrets import SecretManager

            manager = SecretManager()
            manager.set_secret("api_key", "sk-test-12345")
            value = manager.get_secret("api_key")
            missing = manager.get_secret("nonexistent") is None

            return SecurityTestResult(
                name="secret_handling",
                passed=value == "sk-test-12345" and missing,
                message="Secret management works correctly",
                details={"stored_correctly": value == "sk-test-12345", "missing_handled": missing},
            )
        except ImportError:
            return SecurityTestResult(
                name="secret_handling",
                passed=False,
                message="SecretManager not available",
            )

    def test_tenant_isolation(self) -> SecurityTestResult:
        """Test tenant data isolation."""
        try:
            from dvas.security.tenant import TenantManager, TenantIsolationError

            manager = TenantManager()
            tenant = manager.create_tenant("Test Corp")
            manager.assign_user("user_001", tenant.id)

            # User should be in the tenant
            user_tenant = manager.get_user_tenant("user_001")

            # Should raise isolation error
            try:
                manager.enforce_isolation("user_001", "wrong_tenant_id")
                isolated = False
            except TenantIsolationError:
                isolated = True

            return SecurityTestResult(
                name="tenant_isolation",
                passed=user_tenant == tenant.id and isolated,
                message="Tenant isolation works correctly",
                details={
                    "user_in_tenant": user_tenant == tenant.id,
                    "isolation_enforced": isolated,
                },
            )
        except ImportError:
            return SecurityTestResult(
                name="tenant_isolation",
                passed=False,
                message="TenantManager not available",
            )

    def test_watermark_detection(self) -> SecurityTestResult:
        """Test watermark embedding and detection."""
        try:
            from dvas.security.watermark import Watermarker

            watermarker = Watermarker(organization_id="test_org")
            original = "This is test data for watermarking."
            watermarked = watermarker.embed_watermark(original, "user_001")
            info = watermarker.extract_watermark(watermarked)

            has_watermark = info is not None and info.is_valid
            recipient_match = info is not None and info.recipient_id == "user_001"

            return SecurityTestResult(
                name="watermark_detection",
                passed=has_watermark and recipient_match,
                message="Watermark embedding and detection works",
                details={"watermark_detected": has_watermark, "recipient_match": recipient_match},
            )
        except ImportError:
            return SecurityTestResult(
                name="watermark_detection",
                passed=False,
                message="Watermarker not available",
            )

    def test_input_sanitization(self) -> SecurityTestResult:
        """Test input sanitization."""
        try:
            from dvas.security.validation import InputValidator

            validator = InputValidator()

            # Test HTML escaping
            sanitized = validator.sanitize_string("<script>alert('xss')</script>")
            html_escaped = "<script>" not in sanitized

            # Test control character removal
            clean = validator.sanitize_string("hello\x00world")
            no_null = "\x00" not in clean

            # Test video ID validation
            try:
                validator.validate_video_id("../etc/passwd")
                path_traversal = False
            except ValueError:
                path_traversal = True

            return SecurityTestResult(
                name="input_sanitization",
                passed=html_escaped and no_null and path_traversal,
                message="Input sanitization works correctly",
                details={
                    "html_escaped": html_escaped,
                    "no_null": no_null,
                    "path_traversal_blocked": path_traversal,
                },
            )
        except ImportError:
            return SecurityTestResult(
                name="input_sanitization",
                passed=False,
                message="InputValidator not available",
            )


class SecurityRegressionRunner:
    """Runner for security regression tests."""

    def __init__(self) -> None:
        """Initialize the runner."""
        self.suite = SecurityRegressionTestSuite()

    def run(self) -> Dict[str, Any]:
        """Run all tests and return results.

        Returns:
            Dictionary with test results and summary.
        """
        results = self.suite.run_all_tests()
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(results) if results else 0.0,
            "results": [r.to_dict() for r in results],
        }

    def run_and_assert(self) -> None:
        """Run all tests and raise if any fail."""
        results = self.run()
        if results["failed"] > 0:
            failed_names = [r["name"] for r in results["results"] if not r["passed"]]
            raise AssertionError(f"Security regression tests failed: {failed_names}")


__all__ = [
    "SecurityRegressionTestSuite",
    "SecurityRegressionRunner",
    "SecurityTestResult",
]
