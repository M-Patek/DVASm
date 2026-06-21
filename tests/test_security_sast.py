"""Tests for Bandit/SAST gate module.

Tests for BanditScanner, SASTGate, SASTGateConfig, SASTFinding,
SecurityScanner, Severity, and Confidence.
"""

from dvas.security.sast import (
    BanditScanner,
    SASTGate,
    SASTGateConfig,
    SASTFinding,
    SecurityScanner,
    Severity,
    Confidence,
)


class TestSeverity:
    """Test Severity enum."""

    def test_severity_values(self):
        """Test severity values."""
        assert Severity.LOW.value == "LOW"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.HIGH.value == "HIGH"
        assert Severity.CRITICAL.value == "CRITICAL"


class TestConfidence:
    """Test Confidence enum."""

    def test_confidence_values(self):
        """Test confidence values."""
        assert Confidence.LOW.value == "LOW"
        assert Confidence.MEDIUM.value == "MEDIUM"
        assert Confidence.HIGH.value == "HIGH"


class TestSASTFinding:
    """Test SASTFinding dataclass."""

    def test_creation(self):
        """Test creating a SAST finding."""
        finding = SASTFinding(
            tool="bandit",
            rule_id="B101",
            message="Hardcoded password",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file_path="src/app.py",
            line_number=42,
            code_snippet="password = 'secret123'",
            remediation="Use environment variables",
        )
        assert finding.tool == "bandit"
        assert finding.rule_id == "B101"
        assert finding.severity == Severity.HIGH
        assert finding.confidence == Confidence.MEDIUM

    def test_to_dict(self):
        """Test converting to dict."""
        finding = SASTFinding(
            tool="bandit",
            rule_id="B101",
            message="Hardcoded password",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            file_path="src/app.py",
            line_number=42,
        )
        d = finding.to_dict()
        assert d["tool"] == "bandit"
        assert d["rule_id"] == "B101"
        assert d["severity"] == "HIGH"
        assert d["confidence"] == "MEDIUM"
        assert d["file_path"] == "src/app.py"


class TestSASTGateConfig:
    """Test SASTGateConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SASTGateConfig()
        assert config.max_high_severity == 0
        assert config.max_medium_severity == 5
        assert config.max_low_severity == 10
        assert config.min_confidence == Confidence.MEDIUM

    def test_custom_config(self):
        """Test custom configuration."""
        config = SASTGateConfig(
            max_high_severity=1,
            max_medium_severity=10,
            max_low_severity=20,
            min_confidence=Confidence.LOW,
        )
        assert config.max_high_severity == 1
        assert config.max_medium_severity == 10
        assert config.max_low_severity == 20
        assert config.min_confidence == Confidence.LOW

    def test_default_exclude_patterns(self):
        """Test default exclude patterns."""
        config = SASTGateConfig()
        assert "*/tests/*" in config.exclude_patterns
        assert "*/test_*.py" in config.exclude_patterns


class TestBanditScanner:
    """Test BanditScanner class."""

    def test_init(self):
        """Test initialization."""
        scanner = BanditScanner()
        assert scanner is not None

    def test_init_with_config(self, tmp_path):
        """Test initialization with config path."""
        config = tmp_path / "bandit.yaml"
        config.write_text("skips: [B101]")
        scanner = BanditScanner(config_path=config)
        assert scanner.config_path == config

    def test_scan_file_not_found(self):
        """Test scanning non-existent file."""
        scanner = BanditScanner()
        findings = scanner.scan_file("/nonexistent/path/file.py")
        assert findings == []

    def test_scan_directory_not_found(self):
        """Test scanning non-existent directory."""
        scanner = BanditScanner()
        findings = scanner.scan_directory("/nonexistent/path")
        assert findings == []

    def test_parse_bandit_output_empty(self):
        """Test parsing empty output."""
        scanner = BanditScanner()
        findings = scanner._parse_bandit_output("")
        assert findings == []

    def test_parse_bandit_output_invalid_json(self):
        """Test parsing invalid JSON."""
        scanner = BanditScanner()
        findings = scanner._parse_bandit_output("not json")
        assert findings == []

    def test_parse_bandit_output_valid(self):
        """Test parsing valid Bandit output."""
        scanner = BanditScanner()
        output = """{"results": [{"test_id": "B101", "issue_text": "Hardcoded password", "issue_severity": "HIGH", "issue_confidence": "MEDIUM", "filename": "test.py", "line_number": 10, "code": "password = 'secret'"}]}"""
        findings = scanner._parse_bandit_output(output)
        assert len(findings) == 1
        assert findings[0].rule_id == "B101"
        assert findings[0].severity == Severity.HIGH
        assert findings[0].confidence == Confidence.MEDIUM

    def test_parse_bandit_output_no_results(self):
        """Test parsing output with no results."""
        scanner = BanditScanner()
        output = '{"results": []}'
        findings = scanner._parse_bandit_output(output)
        assert findings == []


class TestSASTGate:
    """Test SASTGate class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        gate = SASTGate()
        assert gate.config is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SASTGateConfig(max_high_severity=5)
        gate = SASTGate(config)
        assert gate.config.max_high_severity == 5

    def test_evaluate_no_findings(self):
        """Test evaluation with no findings."""
        gate = SASTGate()
        result = gate.evaluate([])
        assert result["passed"] is True
        assert result["total_findings"] == 0
        assert result["high_count"] == 0
        assert result["medium_count"] == 0
        assert result["low_count"] == 0

    def test_evaluate_passing_findings(self):
        """Test evaluation with passing findings."""
        gate = SASTGate(SASTGateConfig(max_high_severity=1, max_medium_severity=2))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["passed"] is True
        assert result["total_findings"] == 1

    def test_evaluate_failing_high(self):
        """Test evaluation failing due to high severity."""
        gate = SASTGate(SASTGateConfig(max_high_severity=0))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["passed"] is False
        assert result["high_count"] == 1

    def test_evaluate_failing_medium(self):
        """Test evaluation failing due to medium severity."""
        gate = SASTGate(SASTGateConfig(max_medium_severity=0))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["passed"] is False
        assert result["medium_count"] == 1

    def test_evaluate_failing_low(self):
        """Test evaluation failing due to low severity."""
        gate = SASTGate(SASTGateConfig(max_low_severity=0))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["passed"] is False
        assert result["low_count"] == 1

    def test_evaluate_critical_as_high(self):
        """Test that CRITICAL severity counts as HIGH."""
        gate = SASTGate(SASTGateConfig(max_high_severity=0))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.CRITICAL,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["passed"] is False
        assert result["high_count"] == 1

    def test_evaluate_confidence_filter(self):
        """Test confidence filtering."""
        gate = SASTGate(SASTGateConfig(min_confidence=Confidence.HIGH))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        # MEDIUM confidence should be filtered out when min is HIGH
        assert result["total_findings"] == 0

    def test_evaluate_report_contains_findings(self):
        """Test that report contains finding details."""
        gate = SASTGate()
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Hardcoded password",
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=42,
            ),
        ]
        result = gate.evaluate(findings)
        assert "SAST Gate Report" in result["report"]
        assert "test.py:42" in result["report"]
        assert "Hardcoded password" in result["report"]

    def test_evaluate_findings_list(self):
        """Test that findings are included in result."""
        gate = SASTGate()
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="Test",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
        ]
        result = gate.evaluate(findings)
        assert len(result["findings"]) == 1
        assert result["findings"][0]["rule_id"] == "B101"

    def test_confidence_value_mapping(self):
        """Test confidence value mapping."""
        gate = SASTGate()
        assert gate._confidence_value(Confidence.LOW) == 1
        assert gate._confidence_value(Confidence.MEDIUM) == 2
        assert gate._confidence_value(Confidence.HIGH) == 3


class TestSecurityScanner:
    """Test SecurityScanner class."""

    def test_init(self):
        """Test initialization."""
        scanner = SecurityScanner()
        assert scanner is not None

    def test_add_scanner(self):
        """Test adding a custom scanner."""
        scanner = SecurityScanner()
        initial_count = len(scanner.scanners)
        custom = BanditScanner()
        scanner.add_scanner(custom)
        assert len(scanner.scanners) == initial_count + 1

    def test_scan_with_no_scanners(self, tmp_path):
        """Test scanning with no scanners."""
        scanner = SecurityScanner()
        scanner.scanners = []
        findings = scanner.scan(tmp_path)
        assert findings == []


class TestSASTEdgeCases:
    """Test edge cases for SAST module."""

    def test_sast_gate_with_mixed_findings(self):
        """Test gate with mixed severity findings."""
        gate = SASTGate(SASTGateConfig(max_high_severity=1, max_medium_severity=2))
        findings = [
            SASTFinding(
                tool="bandit",
                rule_id="B101",
                message="High",
                severity=Severity.HIGH,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=1,
            ),
            SASTFinding(
                tool="bandit",
                rule_id="B102",
                message="Medium",
                severity=Severity.MEDIUM,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=2,
            ),
            SASTFinding(
                tool="bandit",
                rule_id="B103",
                message="Low",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                file_path="test.py",
                line_number=3,
            ),
        ]
        result = gate.evaluate(findings)
        assert result["total_findings"] == 3
        assert result["high_count"] == 1
        assert result["medium_count"] == 1
        assert result["low_count"] == 1

    def test_sast_finding_with_empty_fields(self):
        """Test SASTFinding with empty optional fields."""
        finding = SASTFinding(
            tool="bandit",
            rule_id="B101",
            message="Test",
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            file_path="",
            line_number=0,
        )
        assert finding.code_snippet == ""
        assert finding.remediation == ""
        assert finding.metadata == {}

    def test_sast_gate_config_with_custom_patterns(self):
        """Test SASTGateConfig with custom patterns."""
        config = SASTGateConfig(
            exclude_patterns=["*/vendor/*"],
            include_patterns=["*.py", "*.js"],
        )
        assert "*/vendor/*" in config.exclude_patterns
        assert "*.py" in config.include_patterns
        assert "*.js" in config.include_patterns

    def test_bandit_scanner_availability(self):
        """Test checking Bandit availability."""
        scanner = BanditScanner()
        # _available is set based on whether bandit is installed
        assert isinstance(scanner._available, bool)
