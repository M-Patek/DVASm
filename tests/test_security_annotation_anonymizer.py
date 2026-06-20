"""Tests for annotation anonymization module.

Tests for AnnotationAnonymizer, AnonymizationConfig, and utility functions.
"""

from unittest.mock import MagicMock

import pytest

from dvas.security.annotation_anonymizer import (
    AnonymizationConfig,
    AnnotationAnonymizer,
    anonymize_annotation_dict,
    mask_annotation_fields,
)
from dvas.security.pii import PIIType


class TestAnonymizationConfig:
    """Test AnonymizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = AnonymizationConfig()
        assert config.anonymize_captions is True
        assert config.anonymize_qa_pairs is True
        assert config.anonymize_object_names is False
        assert config.anonymize_action_verbs is False
        assert config.anonymize_metadata is True
        assert config.redaction_token == "[REDACTED]"
        assert config.use_type_hints is False
        assert config.hash_ids is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AnonymizationConfig(
            anonymize_captions=False,
            redaction_token="[MASKED]",
            hash_ids=False,
        )
        assert config.anonymize_captions is False
        assert config.redaction_token == "[MASKED]"
        assert config.hash_ids is False


class TestAnnotationAnonymizer:
    """Test AnnotationAnonymizer class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        anonymizer = AnnotationAnonymizer()
        assert anonymizer.config is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = AnonymizationConfig(redaction_token="[MASKED]")
        anonymizer = AnnotationAnonymizer(config)
        assert anonymizer.config.redaction_token == "[MASKED]"

    def test_anonymize_text(self):
        """Test anonymizing a text string."""
        anonymizer = AnnotationAnonymizer()
        text = "Contact me at john@example.com"
        result = anonymizer.anonymize_text(text)
        assert "john@example.com" not in result
        assert "[REDACTED]" in result

    def test_anonymize_text_empty(self):
        """Test anonymizing empty text."""
        anonymizer = AnnotationAnonymizer()
        assert anonymizer.anonymize_text("") == ""

    def test_anonymize_text_no_pii(self):
        """Test anonymizing text with no PII."""
        anonymizer = AnnotationAnonymizer()
        text = "This is a normal caption."
        result = anonymizer.anonymize_text(text)
        assert result == text

    def test_anonymize_text_with_type_hints(self):
        """Test anonymizing with type hints."""
        config = AnonymizationConfig(use_type_hints=True)
        anonymizer = AnnotationAnonymizer(config)
        text = "Contact me at john@example.com"
        result = anonymizer.anonymize_text(text)
        assert "john@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_scan_annotation_empty(self):
        """Test scanning empty annotation."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        annotation.segments = []
        findings = anonymizer.scan_annotation(annotation)
        assert findings == []

    def test_scan_annotation_with_pii(self):
        """Test scanning annotation with PII."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        segment = MagicMock()
        segment.caption = "Contact me at john@example.com"
        segment.caption_dense = None
        segment.qa_pairs = []
        annotation.segments = [segment]

        findings = anonymizer.scan_annotation(annotation)
        assert len(findings) > 0
        assert any(f.pii_type == PIIType.EMAIL for f in findings)

    def test_get_pii_report(self):
        """Test generating PII report."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        segment = MagicMock()
        segment.caption = "Email: john@example.com, Phone: 555-123-4567"
        segment.caption_dense = None
        segment.qa_pairs = []
        annotation.segments = [segment]

        report = anonymizer.get_pii_report(annotation)
        assert report["has_pii"] is True
        assert report["total_findings"] > 0
        assert "email" in report["findings_by_type"]

    def test_get_pii_report_no_pii(self):
        """Test PII report with no PII."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        segment = MagicMock()
        segment.caption = "This is a normal caption."
        segment.caption_dense = None
        segment.qa_pairs = []
        annotation.segments = [segment]

        report = anonymizer.get_pii_report(annotation)
        assert report["has_pii"] is False
        assert report["total_findings"] == 0


class TestAnonymizeAnnotationDict:
    """Test anonymize_annotation_dict function."""

    def test_anonymize_dict(self):
        """Test anonymizing a dictionary."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Contact me at john@example.com",
            "metadata": {
                "author": "John Doe",
            },
        }
        result = anonymize_annotation_dict(annotation_dict)
        assert "john@example.com" not in result["caption"]
        assert "[REDACTED]" in result["caption"]

    def test_anonymize_dict_nested(self):
        """Test anonymizing nested dictionary."""
        annotation_dict = {
            "id": "ann_001",
            "segments": [
                {
                    "caption": "Email: john@example.com",
                },
            ],
        }
        result = anonymize_annotation_dict(annotation_dict)
        assert "john@example.com" not in result["segments"][0]["caption"]

    def test_anonymize_dict_no_pii(self):
        """Test anonymizing dict with no PII."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Normal caption",
        }
        result = anonymize_annotation_dict(annotation_dict)
        assert result["caption"] == "Normal caption"

    def test_anonymize_dict_empty(self):
        """Test anonymizing empty dict."""
        result = anonymize_annotation_dict({})
        assert result == {}


class TestMaskAnnotationFields:
    """Test mask_annotation_fields function."""

    def test_mask_single_field(self):
        """Test masking a single field."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Sensitive caption",
            "author": "John Doe",
        }
        result = mask_annotation_fields(annotation_dict, ["caption"])
        assert result["caption"] == "[REDACTED]"
        assert result["author"] == "John Doe"

    def test_mask_multiple_fields(self):
        """Test masking multiple fields."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Sensitive",
            "author": "John",
        }
        result = mask_annotation_fields(annotation_dict, ["caption", "author"])
        assert result["caption"] == "[REDACTED]"
        assert result["author"] == "[REDACTED]"

    def test_mask_nested_field(self):
        """Test masking nested field."""
        annotation_dict = {
            "id": "ann_001",
            "metadata": {
                "author": "John Doe",
            },
        }
        result = mask_annotation_fields(annotation_dict, ["metadata.author"])
        assert result["metadata"]["author"] == "[REDACTED]"

    def test_mask_nonexistent_field(self):
        """Test masking non-existent field."""
        annotation_dict = {
            "id": "ann_001",
        }
        result = mask_annotation_fields(annotation_dict, ["nonexistent"])
        assert result == annotation_dict

    def test_mask_custom_mask(self):
        """Test masking with custom mask."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Sensitive",
        }
        result = mask_annotation_fields(annotation_dict, ["caption"], mask="[MASKED]")
        assert result["caption"] == "[MASKED]"

    def test_mask_preserves_other_fields(self):
        """Test that unmasked fields are preserved."""
        annotation_dict = {
            "id": "ann_001",
            "caption": "Sensitive",
            "tags": ["tag1", "tag2"],
        }
        result = mask_annotation_fields(annotation_dict, ["caption"])
        assert result["id"] == "ann_001"
        assert result["tags"] == ["tag1", "tag2"]


class TestAnnotationAnonymizerEdgeCases:
    """Test edge cases for AnnotationAnonymizer."""

    def test_anonymize_text_none(self):
        """Test anonymizing None text."""
        anonymizer = AnnotationAnonymizer()
        assert anonymizer.anonymize_text(None) is None

    def test_scan_annotation_none_segments(self):
        """Test scanning annotation with None segments."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        annotation.segments = None
        findings = anonymizer.scan_annotation(annotation)
        assert findings == []

    def test_get_pii_report_none_caption(self):
        """Test PII report with None caption."""
        anonymizer = AnnotationAnonymizer()
        annotation = MagicMock()
        segment = MagicMock()
        segment.caption = None
        segment.caption_dense = None
        segment.qa_pairs = []
        annotation.segments = [segment]

        report = anonymizer.get_pii_report(annotation)
        assert report["has_pii"] is False
