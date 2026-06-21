"""Tests for security regression tests module.

Tests for SecurityRegressionTestSuite, SecurityRegressionRunner, and SecurityTestResult.
"""

from dvas.security.regression import (
    SecurityRegressionRunner,
    SecurityRegressionTestSuite,
    SecurityTestResult,
)


class TestSecurityTestResult:
    """Test SecurityTestResult class."""

    def test_creation_passed(self):
        """Test creating a passed result."""
        result = SecurityTestResult(
            name="test_pass",
            passed=True,
            message="Test passed",
            details={"key": "value"},
        )
        assert result.name == "test_pass"
        assert result.passed is True
        assert result.message == "Test passed"
        assert result.details["key"] == "value"

    def test_creation_failed(self):
        """Test creating a failed result."""
        result = SecurityTestResult(
            name="test_fail",
            passed=False,
            message="Test failed",
        )
        assert result.name == "test_fail"
        assert result.passed is False

    def test_to_dict(self):
        """Test converting to dict."""
        result = SecurityTestResult(
            name="test_pass",
            passed=True,
            message="Test passed",
            details={"key": "value"},
        )
        d = result.to_dict()
        assert d["name"] == "test_pass"
        assert d["passed"] is True
        assert d["message"] == "Test passed"
        assert d["details"]["key"] == "value"


class TestSecurityRegressionTestSuite:
    """Test SecurityRegressionTestSuite class."""

    def test_init(self):
        """Test initialization."""
        suite = SecurityRegressionTestSuite()
        assert suite is not None
        assert len(suite._tests) > 0

    def test_add_test(self):
        """Test adding a custom test."""
        suite = SecurityRegressionTestSuite()
        initial_count = len(suite._tests)

        def custom_test():
            return SecurityTestResult("custom", True, "Custom test")

        suite.add_test(custom_test)
        assert len(suite._tests) == initial_count + 1

    def test_run_all_tests(self):
        """Test running all tests."""
        suite = SecurityRegressionTestSuite()
        results = suite.run_all_tests()
        assert len(results) > 0
        # All tests should return results (passed or failed)
        assert all(isinstance(r, SecurityTestResult) for r in results)

    def test_sql_injection_detection(self):
        """Test SQL injection detection test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_sql_injection_detection()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "sql_injection_detection"
        # Should detect at least some SQL injection attempts
        assert result.details["detected_count"] > 0

    def test_xss_detection(self):
        """Test XSS detection test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_xss_detection()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "xss_detection"
        assert result.details["detected_count"] > 0

    def test_path_traversal_prevention(self):
        """Test path traversal prevention test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_path_traversal_prevention()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "path_traversal_prevention"

    def test_password_hashing(self):
        """Test password hashing test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_password_hashing()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "password_hashing"

    def test_encryption_roundtrip(self):
        """Test encryption roundtrip test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_encryption_roundtrip()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "encryption_roundtrip"

    def test_csrf_token_validation(self):
        """Test CSRF token validation test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_csrf_token_validation()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "csrf_token_validation"

    def test_api_key_generation(self):
        """Test API key generation test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_api_key_generation()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "api_key_generation"

    def test_pii_redaction(self):
        """Test PII redaction test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_pii_redaction()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "pii_redaction"

    def test_role_permissions(self):
        """Test role permissions test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_role_permissions()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "role_permissions"

    def test_audit_logging(self):
        """Test audit logging test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_audit_logging()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "audit_logging"

    def test_secret_handling(self):
        """Test secret handling test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_secret_handling()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "secret_handling"

    def test_tenant_isolation(self):
        """Test tenant isolation test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_tenant_isolation()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "tenant_isolation"

    def test_watermark_detection(self):
        """Test watermark detection test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_watermark_detection()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "watermark_detection"

    def test_input_sanitization(self):
        """Test input sanitization test."""
        suite = SecurityRegressionTestSuite()
        result = suite.test_input_sanitization()
        assert isinstance(result, SecurityTestResult)
        assert result.name == "input_sanitization"

    def test_custom_test_execution(self):
        """Test that custom tests are executed."""
        suite = SecurityRegressionTestSuite()

        custom_called = []

        def custom_test():
            custom_called.append(True)
            return SecurityTestResult("custom", True, "Custom test passed")

        suite.add_test(custom_test)
        results = suite.run_all_tests()
        # Find our custom test result
        custom_results = [r for r in results if r.name == "custom"]
        assert len(custom_results) == 1
        assert custom_results[0].passed is True

    def test_failing_test_handling(self):
        """Test that failing tests are handled gracefully."""
        suite = SecurityRegressionTestSuite()

        def failing_test():
            raise RuntimeError("Test failure")

        suite.add_test(failing_test)
        results = suite.run_all_tests()
        # Find our failing test result
        failing_results = [r for r in results if r.name == "failing_test"]
        assert len(failing_results) == 1
        assert failing_results[0].passed is False


class TestSecurityRegressionRunner:
    """Test SecurityRegressionRunner class."""

    def test_init(self):
        """Test initialization."""
        runner = SecurityRegressionRunner()
        assert runner.suite is not None

    def test_run(self):
        """Test running all tests."""
        runner = SecurityRegressionRunner()
        results = runner.run()
        assert "total" in results
        assert "passed" in results
        assert "failed" in results
        assert "success_rate" in results
        assert "results" in results
        assert results["total"] > 0
        assert results["passed"] + results["failed"] == results["total"]

    def test_run_and_assert(self):
        """Test run_and_assert method."""
        runner = SecurityRegressionRunner()
        # Should not raise since default tests should pass
        runner.run_and_assert()

    def test_run_results_structure(self):
        """Test structure of run results."""
        runner = SecurityRegressionRunner()
        results = runner.run()
        assert isinstance(results["results"], list)
        assert all(isinstance(r, dict) for r in results["results"])
        assert all("name" in r for r in results["results"])
        assert all("passed" in r for r in results["results"])

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        runner = SecurityRegressionRunner()
        results = runner.run()
        if results["total"] > 0:
            expected_rate = results["passed"] / results["total"]
            assert results["success_rate"] == expected_rate


class TestSecurityRegressionEdgeCases:
    """Test edge cases for security regression tests."""

    def test_empty_suite(self):
        """Test with empty test suite."""
        suite = SecurityRegressionTestSuite()
        suite._tests = []
        results = suite.run_all_tests()
        assert results == []

    def test_all_tests_return_results(self):
        """Test that all default tests return results."""
        suite = SecurityRegressionTestSuite()
        results = suite.run_all_tests()
        assert len(results) == len(suite._tests)
        # Every test should have a name
        assert all(r.name for r in results)

    def test_no_duplicate_test_names(self):
        """Test that there are no duplicate test names."""
        suite = SecurityRegressionTestSuite()
        results = suite.run_all_tests()
        names = [r.name for r in results]
        assert len(names) == len(set(names))

    def test_suite_with_only_custom_tests(self):
        """Test suite with only custom tests."""
        suite = SecurityRegressionTestSuite()
        suite._tests = []

        def test1():
            return SecurityTestResult("test1", True, "Pass")

        def test2():
            return SecurityTestResult("test2", False, "Fail")

        suite.add_test(test1)
        suite.add_test(test2)
        results = suite.run_all_tests()
        assert len(results) == 2
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) == 1
        assert len(failed) == 1
