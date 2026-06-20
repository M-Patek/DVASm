"""PII detection and redaction for DVAS annotations and video metadata.

Provides pattern-based and heuristic detection of personally identifiable
information in text fields, with configurable redaction strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Set, Tuple

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class PIIType(str, Enum):
    """Types of personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    URL = "url"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    USERNAME = "username"
    FULL_NAME = "full_name"


@dataclass
class PIIFinding:
    """A detected PII occurrence."""

    pii_type: PIIType
    value: str
    position: Tuple[int, int]
    confidence: float = 1.0
    context: str = ""

    def to_dict(self) -> Dict:
        """Convert finding to dictionary."""
        return {
            "pii_type": self.pii_type.value,
            "value": self.value,
            "position": self.position,
            "confidence": self.confidence,
            "context": self.context,
        }


class PIIDetector:
    """Detect and redact personally identifiable information in text.

    Usage::

        detector = PIIDetector()
        findings = detector.scan_text("Contact me at john@example.com")
        redacted = detector.redact_text("Contact me at john@example.com")
    """

    # Regex patterns for PII detection
    PATTERNS: Dict[PIIType, List[str]] = {
        PIIType.EMAIL: [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        PIIType.PHONE: [
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b",
        ],
        PIIType.SSN: [
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        ],
        PIIType.CREDIT_CARD: [
            r"\b(?:\d[ -]*?){13,16}\b",
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        ],
        PIIType.IP_ADDRESS: [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        ],
        PIIType.MAC_ADDRESS: [
            r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        ],
        PIIType.URL: [
            r"https?://[^\s\"<>{}|\^`\[\]]+",
        ],
        PIIType.PASSPORT: [
            r"\b[A-Z]{1,2}\d{6,9}\b",
        ],
        PIIType.DRIVER_LICENSE: [
            r"\b[A-Z]{1,2}\d{6,8}\b",
        ],
        PIIType.DATE_OF_BIRTH: [
            r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
            r"\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])\b",
        ],
    }

    # Keywords that suggest a name or address nearby
    NAME_CONTEXT_KEYWORDS: Set[str] = {
        "name",
        "called",
        "signed by",
        "author",
        "creator",
        "annotator",
        "reviewer",
        "submitted by",
        "owner",
    }

    ADDRESS_CONTEXT_KEYWORDS: Set[str] = {
        "address",
        "street",
        "avenue",
        "road",
        "lane",
        "drive",
        "city",
        "state",
        "zip",
        "postal",
        "located at",
        "residence",
    }

    SENSITIVE_OBJECT_NAMES: Set[str] = {
        "license_plate",
        "id_card",
        "passport",
        "credit_card",
        "social_security_card",
        "face",
        "person_name",
        "license_plate_text",
        "qr_code",
        "barcode",
    }

    def __init__(
        self,
        redaction_token: str = "[REDACTED]",
        enabled_types: Optional[Set[PIIType]] = None,
        custom_patterns: Optional[Dict[PIIType, List[str]]] = None,
    ) -> None:
        """Initialize PII detector.

        Args:
            redaction_token: Token to replace detected PII with.
            enabled_types: Subset of PII types to detect. If None, all types.
            custom_patterns: Additional regex patterns to use.
        """
        self.redaction_token = redaction_token
        self.enabled_types = enabled_types or set(PIIType)
        self.custom_patterns = custom_patterns or {}

        # Compile all patterns
        self._compiled: Dict[PIIType, List[Pattern]] = {}
        for pii_type, patterns in self.PATTERNS.items():
            self._compiled[pii_type] = [re.compile(p) for p in patterns]

        for pii_type, patterns in self.custom_patterns.items():
            if pii_type not in self._compiled:
                self._compiled[pii_type] = []
            self._compiled[pii_type].extend([re.compile(p) for p in patterns])

    def scan_text(
        self,
        text: str,
        context_window: int = 20,
    ) -> List[PIIFinding]:
        """Scan text for PII occurrences.

        Args:
            text: The text to scan.
            context_window: Number of characters of context around each match.

        Returns:
            List of PIIFinding objects.
        """
        if not text:
            return []

        findings: List[PIIFinding] = []
        seen_positions: Set[Tuple[int, int]] = set()

        for pii_type in self.enabled_types:
            patterns = self._compiled.get(pii_type, [])
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()

                    # Skip overlapping matches (prefer longer ones)
                    if any(start < e and end > s for s, e in seen_positions):
                        continue

                    seen_positions.add((start, end))

                    # Extract context
                    ctx_start = max(0, start - context_window)
                    ctx_end = min(len(text), end + context_window)
                    context = text[ctx_start:ctx_end]

                    findings.append(
                        PIIFinding(
                            pii_type=pii_type,
                            value=match.group(),
                            position=(start, end),
                            confidence=self._calculate_confidence(pii_type, match.group(), context),
                            context=context,
                        )
                    )

        # Sort by position
        findings.sort(key=lambda f: f.position[0])
        return findings

    def redact_text(self, text: str) -> str:
        """Redact all detected PII from text.

        Args:
            text: The text to redact.

        Returns:
            Redacted text with PII replaced by redaction_token.
        """
        if not text:
            return text

        findings = self.scan_text(text)
        if not findings:
            return text

        # Build redacted text from right to left to preserve positions
        result = text
        for finding in reversed(findings):
            start, end = finding.position
            result = result[:start] + self.redaction_token + result[end:]

        return result

    def redact_with_type_hint(self, text: str) -> str:
        """Redact PII with type hints (e.g., [EMAIL_REDACTED]).

        Args:
            text: The text to redact.

        Returns:
            Redacted text with type-specific tokens.
        """
        if not text:
            return text

        findings = self.scan_text(text)
        if not findings:
            return text

        result = text
        for finding in reversed(findings):
            start, end = finding.position
            token = f"[{finding.pii_type.value.upper()}_REDACTED]"
            result = result[:start] + token + result[end:]

        return result

    def check_objects(self, objects: List[Dict]) -> List[Dict]:
        """Check annotation objects for sensitive object types.

        Args:
            objects: List of object dictionaries with 'name' keys.

        Returns:
            List of sensitive object dictionaries found.
        """
        found = []
        for obj in objects:
            name = obj.get("name", "").lower()
            if name in self.SENSITIVE_OBJECT_NAMES:
                found.append(obj)
        return found

    def scan_annotation_text(self, annotation: Dict) -> List[PIIFinding]:
        """Scan all text fields in an annotation dictionary for PII.

        Args:
            annotation: Annotation dictionary to scan.

        Returns:
            Combined list of PII findings across all text fields.
        """
        all_findings: List[PIIFinding] = []
        text_fields = self._extract_text_fields(annotation)

        for field_path, text in text_fields:
            findings = self.scan_text(text)
            for finding in findings:
                finding.context = f"field={field_path}; {finding.context}"
            all_findings.extend(findings)

        return all_findings

    def _extract_text_fields(
        self,
        obj: Dict,
        path: str = "",
        max_depth: int = 5,
        current_depth: int = 0,
    ) -> List[Tuple[str, str]]:
        """Recursively extract string fields from a dictionary."""
        if current_depth >= max_depth:
            return []

        results = []
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str) and len(value) > 2:
                results.append((current_path, value))
            elif isinstance(value, dict):
                results.extend(
                    self._extract_text_fields(value, current_path, max_depth, current_depth + 1)
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        results.extend(
                            self._extract_text_fields(
                                item, f"{current_path}[{i}]", max_depth, current_depth + 1
                            )
                        )
                    elif isinstance(item, str) and len(item) > 2:
                        results.append((f"{current_path}[{i}]", item))

        return results

    def _calculate_confidence(self, pii_type: PIIType, value: str, context: str) -> float:
        """Calculate confidence score for a PII match."""
        confidence = 1.0
        context_lower = context.lower()

        # Adjust based on value characteristics
        if pii_type == PIIType.EMAIL:
            if "@" in value and "." in value.split("@")[-1]:
                confidence = 0.95
            else:
                confidence = 0.5
        elif pii_type == PIIType.PHONE:
            digits = sum(c.isdigit() for c in value)
            if digits >= 10:
                confidence = 0.9
            else:
                confidence = 0.6
        elif pii_type == PIIType.SSN:
            digits = sum(c.isdigit() for c in value)
            if digits == 9:
                confidence = 0.95
            else:
                confidence = 0.7
        elif pii_type == PIIType.IP_ADDRESS:
            confidence = 0.85
        elif pii_type == PIIType.CREDIT_CARD:
            digits = sum(c.isdigit() for c in value)
            if 13 <= digits <= 16:
                confidence = 0.9
            else:
                confidence = 0.5

        # Context boosts
        if pii_type == PIIType.NAME:
            if any(kw in context_lower for kw in self.NAME_CONTEXT_KEYWORDS):
                confidence = min(confidence + 0.15, 1.0)
        elif pii_type == PIIType.ADDRESS:
            if any(kw in context_lower for kw in self.ADDRESS_CONTEXT_KEYWORDS):
                confidence = min(confidence + 0.15, 1.0)

        return confidence

    def get_stats(self, text: str) -> Dict[str, int]:
        """Get PII detection statistics for a text.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary mapping PII type names to occurrence counts.
        """
        findings = self.scan_text(text)
        stats: Dict[str, int] = {}
        for finding in findings:
            stats[finding.pii_type.value] = stats.get(finding.pii_type.value, 0) + 1
        return stats


__all__ = ["PIIDetector", "PIIFinding", "PIIType"]
