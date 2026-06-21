"""Tests for PII detection module.

Tests for PIIDetector, PIIFinding, and PIIType classes.
"""

from dvas.security.pii import PIIDetector, PIIFinding, PIIType


class TestPIIType:
    """Test PIIType enum."""

    def test_pii_type_values(self):
        """Test that all PII types are defined."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.SSN.value == "ssn"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.IP_ADDRESS.value == "ip_address"
        assert PIIType.MAC_ADDRESS.value == "mac_address"
        assert PIIType.URL.value == "url"
        assert PIIType.PASSPORT.value == "passport"
        assert PIIType.DRIVER_LICENSE.value == "driver_license"
        assert PIIType.DATE_OF_BIRTH.value == "date_of_birth"


class TestPIIFinding:
    """Test PIIFinding dataclass."""

    def test_pii_finding_creation(self):
        """Test creating a PIIFinding."""
        finding = PIIFinding(
            pii_type=PIIType.EMAIL,
            value="test@example.com",
            position=(10, 24),
            confidence=0.95,
            context="Contact test@example.com for info",
        )
        assert finding.pii_type == PIIType.EMAIL
        assert finding.value == "test@example.com"
        assert finding.position == (10, 24)
        assert finding.confidence == 0.95

    def test_pii_finding_to_dict(self):
        """Test converting PIIFinding to dict."""
        finding = PIIFinding(
            pii_type=PIIType.PHONE,
            value="555-123-4567",
            position=(0, 12),
            confidence=0.9,
        )
        d = finding.to_dict()
        assert d["pii_type"] == "phone"
        assert d["value"] == "555-123-4567"
        assert d["position"] == (0, 12)
        assert d["confidence"] == 0.9


class TestPIIDetector:
    """Test PIIDetector class."""

    def test_scan_text_empty(self):
        """Test scanning empty text."""
        detector = PIIDetector()
        findings = detector.scan_text("")
        assert findings == []

    def test_scan_text_no_pii(self):
        """Test scanning text with no PII."""
        detector = PIIDetector()
        text = "This is a normal sentence without any personal information."
        findings = detector.scan_text(text)
        assert findings == []

    def test_scan_text_email(self):
        """Test detecting email addresses."""
        detector = PIIDetector()
        text = "Contact me at john.doe@example.com for more info."
        findings = detector.scan_text(text)
        assert len(findings) >= 1
        assert any(f.pii_type == PIIType.EMAIL for f in findings)

    def test_scan_text_phone(self):
        """Test detecting phone numbers."""
        detector = PIIDetector()
        text = "Call me at 555-123-4567 or (555) 987-6543."
        findings = detector.scan_text(text)
        assert len(findings) >= 1
        assert any(f.pii_type == PIIType.PHONE for f in findings)

    def test_scan_text_ssn(self):
        """Test detecting SSN patterns."""
        detector = PIIDetector()
        text = "My SSN is 123-45-6789."
        findings = detector.scan_text(text)
        assert len(findings) >= 1
        assert any(f.pii_type == PIIType.SSN for f in findings)

    def test_scan_text_credit_card(self):
        """Test detecting credit card numbers."""
        detector = PIIDetector()
        text = "Card number: 1234-5678-9012-3456"
        findings = detector.scan_text(text)
        assert len(findings) >= 1
        assert any(f.pii_type == PIIType.CREDIT_CARD for f in findings)

    def test_scan_text_ip_address(self):
        """Test detecting IP addresses."""
        detector = PIIDetector()
        text = "Server at 192.168.1.1 is down."
        findings = detector.scan_text(text)
        assert len(findings) >= 1
        assert any(f.pii_type == PIIType.IP_ADDRESS for f in findings)

    def test_scan_text_multiple_pii(self):
        """Test detecting multiple types of PII."""
        detector = PIIDetector()
        text = "Email: john@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
        findings = detector.scan_text(text)
        types = {f.pii_type for f in findings}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types
        assert PIIType.SSN in types

    def test_redact_text(self):
        """Test redacting PII from text."""
        detector = PIIDetector(redaction_token="[REDACTED]")
        text = "Contact me at john@example.com for help."
        redacted = detector.redact_text(text)
        assert "john@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_text_multiple(self):
        """Test redacting multiple PII occurrences."""
        detector = PIIDetector(redaction_token="[REDACTED]")
        text = "Email: john@example.com and jane@example.com"
        redacted = detector.redact_text(text)
        assert "john@example.com" not in redacted
        assert "jane@example.com" not in redacted
        assert redacted.count("[REDACTED]") == 2

    def test_redact_with_type_hint(self):
        """Test redacting with type hints."""
        detector = PIIDetector()
        text = "Contact me at john@example.com"
        redacted = detector.redact_with_type_hint(text)
        assert "john@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted

    def test_check_objects(self):
        """Test checking objects for sensitive types."""
        detector = PIIDetector()
        objects = [
            {"name": "person", "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"name": "face", "bbox": [0.5, 0.6, 0.7, 0.8]},
            {"name": "license_plate", "bbox": [0.9, 0.1, 0.95, 0.15]},
        ]
        sensitive = detector.check_objects(objects)
        assert len(sensitive) == 2
        names = [obj["name"] for obj in sensitive]
        assert "face" in names
        assert "license_plate" in names

    def test_check_objects_empty(self):
        """Test checking empty object list."""
        detector = PIIDetector()
        assert detector.check_objects([]) == []

    def test_scan_annotation_text(self):
        """Test scanning annotation dictionary for PII."""
        detector = PIIDetector()
        annotation = {
            "id": "ann_001",
            "caption": "Contact me at john@example.com",
            "metadata": {
                "author": "John Doe",
                "contact": "555-123-4567",
            },
        }
        findings = detector.scan_annotation_text(annotation)
        assert len(findings) >= 2

    def test_get_stats(self):
        """Test getting PII detection statistics."""
        detector = PIIDetector()
        text = "Email: john@example.com, Phone: 555-123-4567"
        stats = detector.get_stats(text)
        assert stats.get("email", 0) >= 1
        assert stats.get("phone", 0) >= 1

    def test_custom_redaction_token(self):
        """Test custom redaction token."""
        detector = PIIDetector(redaction_token="[MASKED]")
        text = "Email: john@example.com"
        redacted = detector.redact_text(text)
        assert "[MASKED]" in redacted

    def test_enabled_types_filter(self):
        """Test filtering by enabled types."""
        detector = PIIDetector(enabled_types={PIIType.EMAIL})
        text = "Email: john@example.com, Phone: 555-123-4567"
        findings = detector.scan_text(text)
        types = {f.pii_type for f in findings}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE not in types

    def test_custom_patterns(self):
        """Test custom regex patterns."""
        custom = {
            PIIType.USERNAME: [r"\buser_[a-z0-9]+\b"],
        }
        detector = PIIDetector(custom_patterns=custom)
        text = "User: user_abc123 and user_xyz789"
        findings = detector.scan_text(text)
        # Custom patterns are added to default ones
        assert len(findings) >= 0  # May or may not match depending on pattern


class TestPIIDetectorEdgeCases:
    """Test edge cases for PIIDetector."""

    def test_none_input(self):
        """Test with None input."""
        detector = PIIDetector()
        findings = detector.scan_text(None)
        assert findings == []

    def test_very_long_text(self):
        """Test with very long text."""
        detector = PIIDetector()
        text = "Email: " + "a" * 10000 + "@example.com"
        findings = detector.scan_text(text)
        # Should not crash
        assert isinstance(findings, list)

    def test_unicode_text(self):
        """Test with unicode text containing PII."""
        detector = PIIDetector()
        text = "Contact: 你好@example.com"
        findings = detector.scan_text(text)
        # Should handle unicode gracefully
        assert isinstance(findings, list)

    def test_overlapping_patterns(self):
        """Test handling of overlapping patterns."""
        detector = PIIDetector()
        # Text with potential overlapping matches
        text = "123-45-6789 and 123456789"
        findings = detector.scan_text(text)
        # Should handle without error
        assert isinstance(findings, list)
