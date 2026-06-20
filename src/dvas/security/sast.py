"""Bandit/SAST gate integration for DVAS.

Provides integration with static analysis security testing (SAST) tools
like Bandit for automated security scanning of Python code.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """Severity levels for SAST findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Confidence(str, Enum):
    """Confidence levels for SAST findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SASTFinding:
    """A SAST finding from static analysis."""

    tool: str
    rule_id: str
    message: str
    severity: Severity
    confidence: Confidence
    file_path: str
    line_number: int
    code_snippet: str = ""
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool,
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "remediation": self.remediation,
            "metadata": self.metadata,
        }


class SASTGateConfig:
    """Configuration for the SAST gate."""

    def __init__(
        self,
        max_high_severity: int = 0,
        max_medium_severity: int = 5,
        max_low_severity: int = 10,
        min_confidence: Confidence = Confidence.MEDIUM,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize SAST gate configuration.

        Args:
            max_high_severity: Maximum allowed high severity findings.
            max_medium_severity: Maximum allowed medium severity findings.
            max_low_severity: Maximum allowed low severity findings.
            min_confidence: Minimum confidence level to report.
            exclude_patterns: Patterns to exclude from scanning.
            include_patterns: Patterns to include in scanning.
        """
        self.max_high_severity = max_high_severity
        self.max_medium_severity = max_medium_severity
        self.max_low_severity = max_low_severity
        self.min_confidence = min_confidence
        self.exclude_patterns = exclude_patterns or ["*/tests/*", "*/test_*.py"]
        self.include_patterns = include_patterns or ["*.py"]


class BanditScanner:
    """Bandit SAST scanner integration.

    Usage::

        scanner = BanditScanner()
        findings = scanner.scan_directory(Path("src/dvas"))

        gate = SASTGate()
        result = gate.evaluate(findings)
        if not result["passed"]:
            print("SAST gate failed!")
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the Bandit scanner.

        Args:
            config_path: Optional path to Bandit configuration file.
        """
        self.config_path = config_path
        self._available = self._check_bandit()

    def _check_bandit(self) -> bool:
        """Check if Bandit is installed."""
        try:
            subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("bandit_not_found")
            return False

    def scan_file(self, file_path: Path) -> List[SASTFinding]:
        """Scan a single file with Bandit.

        Args:
            file_path: Path to the Python file.

        Returns:
            List of SAST findings.
        """
        if not self._available:
            logger.warning("bandit_not_available", file=str(file_path))
            return []

        try:
            cmd = ["bandit", "-f", "json", "-ll", "-ii"]
            if self.config_path:
                cmd.extend(["-c", str(self.config_path)])
            cmd.append(str(file_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            return self._parse_bandit_output(result.stdout)

        except Exception as e:
            logger.error("bandit_scan_failed", file=str(file_path), error=str(e))
            return []

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[SASTFinding]:
        """Scan a directory with Bandit.

        Args:
            directory: Path to the directory to scan.
            recursive: Whether to scan recursively.

        Returns:
            List of SAST findings.
        """
        if not self._available:
            logger.warning("bandit_not_available", directory=str(directory))
            return []

        try:
            cmd = ["bandit", "-f", "json", "-ll", "-ii"]
            if self.config_path:
                cmd.extend(["-c", str(self.config_path)])
            if recursive:
                cmd.append("-r")
            cmd.append(str(directory))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            return self._parse_bandit_output(result.stdout)

        except Exception as e:
            logger.error("bandit_scan_failed", directory=str(directory), error=str(e))
            return []

    def _parse_bandit_output(self, output: str) -> List[SASTFinding]:
        """Parse Bandit JSON output into SAST findings."""
        findings = []

        try:
            data = json.loads(output)
            for result in data.get("results", []):
                severity = Severity(result.get("issue_severity", "LOW").upper())
                confidence = Confidence(result.get("issue_confidence", "LOW").upper())

                finding = SASTFinding(
                    tool="bandit",
                    rule_id=result.get("test_id", ""),
                    message=result.get("issue_text", ""),
                    severity=severity,
                    confidence=confidence,
                    file_path=result.get("filename", ""),
                    line_number=result.get("line_number", 0),
                    code_snippet=result.get("code", ""),
                    remediation=result.get("more_info", ""),
                )
                findings.append(finding)

        except json.JSONDecodeError:
            logger.warning("bandit_output_parse_failed")

        return findings


class SASTGate:
    """SAST gate for enforcing security quality standards.

    Usage::

        config = SASTGateConfig(
            max_high_severity=0,
            max_medium_severity=3,
        )
        gate = SASTGate(config)

        scanner = BanditScanner()
        findings = scanner.scan_directory(Path("src/dvas"))

        result = gate.evaluate(findings)
        if not result["passed"]:
            print(result["report"])
    """

    def __init__(self, config: Optional[SASTGateConfig] = None) -> None:
        """Initialize the SAST gate.

        Args:
            config: SAST gate configuration.
        """
        self.config = config or SASTGateConfig()

    def evaluate(self, findings: List[SASTFinding]) -> Dict[str, Any]:
        """Evaluate findings against the gate configuration.

        Args:
            findings: List of SAST findings.

        Returns:
            Dictionary with evaluation results.
        """
        # Filter by confidence
        filtered = [
            f for f in findings
            if self._confidence_value(f.confidence) >= self._confidence_value(self.config.min_confidence)
        ]

        # Count by severity
        high_count = sum(1 for f in filtered if f.severity == Severity.HIGH or f.severity == Severity.CRITICAL)
        medium_count = sum(1 for f in filtered if f.severity == Severity.MEDIUM)
        low_count = sum(1 for f in filtered if f.severity == Severity.LOW)

        # Evaluate gate
        passed = (
            high_count <= self.config.max_high_severity
            and medium_count <= self.config.max_medium_severity
            and low_count <= self.config.max_low_severity
        )

        # Build report
        report_lines = ["SAST Gate Report", "=" * 40]
        report_lines.append(f"Total findings: {len(filtered)}")
        report_lines.append(f"  High/Critical: {high_count} (max: {self.config.max_high_severity})")
        report_lines.append(f"  Medium: {medium_count} (max: {self.config.max_medium_severity})")
        report_lines.append(f"  Low: {low_count} (max: {self.config.max_low_severity})")
        report_lines.append(f"  Gate: {'PASSED' if passed else 'FAILED'}")

        if filtered:
            report_lines.append("\nFindings:")
            for finding in filtered:
                report_lines.append(
                    f"  [{finding.severity.value}] {finding.file_path}:{finding.line_number} "
                    f"- {finding.message}"
                )

        return {
            "passed": passed,
            "total_findings": len(filtered),
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "report": "\n".join(report_lines),
            "findings": [f.to_dict() for f in filtered],
        }

    def _confidence_value(self, confidence: Confidence) -> int:
        """Get numeric value for confidence comparison."""
        mapping = {
            Confidence.LOW: 1,
            Confidence.MEDIUM: 2,
            Confidence.HIGH: 3,
        }
        return mapping.get(confidence, 0)


class SecurityScanner:
    """Unified security scanner that runs multiple SAST tools."""

    def __init__(self) -> None:
        """Initialize the security scanner."""
        self.scanners: List[Any] = []
        self._register_default_scanners()

    def _register_default_scanners(self) -> None:
        """Register default security scanners."""
        bandit = BanditScanner()
        if bandit._available:
            self.scanners.append(bandit)

    def scan(self, target: Path) -> List[SASTFinding]:
        """Scan a target with all registered scanners.

        Args:
            target: Path to scan.

        Returns:
            Combined list of findings from all scanners.
        """
        all_findings: List[SASTFinding] = []

        for scanner in self.scanners:
            if hasattr(scanner, "scan_directory"):
                findings = scanner.scan_directory(target)
            elif hasattr(scanner, "scan_file"):
                findings = scanner.scan_file(target)
            else:
                continue

            all_findings.extend(findings)

        return all_findings

    def add_scanner(self, scanner: Any) -> None:
        """Add a custom scanner.

        Args:
            scanner: Scanner instance with scan_directory or scan_file method.
        """
        self.scanners.append(scanner)


__all__ = [
    "BanditScanner",
    "SASTGate",
    "SASTGateConfig",
    "SASTFinding",
    "SecurityScanner",
    "Severity",
    "Confidence",
]
