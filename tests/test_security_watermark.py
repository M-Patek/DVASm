"""Tests for watermark module.

Tests for Watermarker, BatchWatermarker, WatermarkConfig, and WatermarkInfo.
"""

import pytest

from dvas.security.watermark import (
    BatchWatermarker,
    WatermarkConfig,
    WatermarkInfo,
    WatermarkType,
    Watermarker,
)


class TestWatermarkType:
    """Test WatermarkType enum."""

    def test_watermark_type_values(self):
        """Test watermark type values."""
        assert WatermarkType.INVISIBLE_TEXT.value == "invisible_text"
        assert WatermarkType.VISIBLE_TEXT.value == "visible_text"
        assert WatermarkType.BINARY.value == "binary"
        assert WatermarkType.METADATA.value == "metadata"


class TestWatermarkConfig:
    """Test WatermarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = WatermarkConfig()
        assert config.watermark_type == WatermarkType.INVISIBLE_TEXT
        assert config.organization_id == "dvas"
        assert config.include_timestamp is True
        assert config.include_user_id is True
        assert config.include_export_id is True
        assert config.custom_fields == {}


class TestWatermarkInfo:
    """Test WatermarkInfo dataclass."""

    def test_watermark_info_creation(self):
        """Test creating WatermarkInfo."""
        info = WatermarkInfo(
            organization_id="test_org",
            recipient_id="user_001",
            timestamp=1234567890.0,
            user_id="user_001",
            export_id="exp_001",
            custom_fields={"key": "value"},
            is_valid=True,
            confidence=1.0,
        )
        assert info.organization_id == "test_org"
        assert info.recipient_id == "user_001"
        assert info.is_valid is True
        assert info.confidence == 1.0

    def test_watermark_info_to_dict(self):
        """Test converting WatermarkInfo to dict."""
        info = WatermarkInfo(
            organization_id="test_org",
            recipient_id="user_001",
            timestamp=1234567890.0,
            user_id="user_001",
            export_id="exp_001",
            custom_fields={"key": "value"},
            is_valid=True,
            confidence=1.0,
        )
        d = info.to_dict()
        assert d["organization_id"] == "test_org"
        assert d["recipient_id"] == "user_001"
        assert d["is_valid"] is True


class TestWatermarker:
    """Test Watermarker class."""

    def test_init(self):
        """Test initialization."""
        watermarker = Watermarker(organization_id="test_org")
        assert watermarker.organization_id == "test_org"

    def test_init_default_org(self):
        """Test initialization with default org."""
        watermarker = Watermarker()
        assert watermarker.organization_id == "dvas"

    def test_embed_watermark(self):
        """Test embedding a watermark."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        assert len(watermarked) > 0

    def test_embed_watermark_empty(self):
        """Test embedding watermark in empty text."""
        watermarker = Watermarker()
        result = watermarker.embed_watermark("", "user_001")
        assert result == ""

    def test_extract_watermark(self):
        """Test extracting a watermark."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        info = watermarker.extract_watermark(watermarked)
        assert info is not None
        assert info.is_valid is True
        assert info.recipient_id == "user_001"

    def test_extract_watermark_no_watermark(self):
        """Test extracting from text without watermark."""
        watermarker = Watermarker()
        text = "This is plain text without any watermark."
        info = watermarker.extract_watermark(text)
        assert info is None

    def test_extract_watermark_empty(self):
        """Test extracting from empty text."""
        watermarker = Watermarker()
        info = watermarker.extract_watermark("")
        assert info is None

    def test_verify_watermark(self):
        """Test verifying a watermark."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        assert watermarker.verify_watermark(watermarked, "user_001") is True

    def test_verify_watermark_wrong_recipient(self):
        """Test verifying with wrong recipient."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        assert watermarker.verify_watermark(watermarked, "user_002") is False

    def test_embed_visible_watermark(self):
        """Test embedding visible watermark."""
        watermarker = Watermarker()
        text = "This is test data."
        watermarked = watermarker.embed_visible_watermark(text, "Test Label")
        assert "EXPORTED" in watermarked
        assert "Test Label" in watermarked

    def test_embed_visible_watermark_header(self):
        """Test embedding visible watermark at header."""
        watermarker = Watermarker()
        text = "This is test data."
        watermarked = watermarker.embed_visible_watermark(text, "Test Label", position="header")
        assert watermarker.remove_watermark(watermarked) != text

    def test_embed_visible_watermark_empty(self):
        """Test embedding visible watermark in empty text."""
        watermarker = Watermarker()
        result = watermarker.embed_visible_watermark("", "Test")
        assert "EXPORTED" in result

    def test_has_watermark(self):
        """Test detecting if text has watermark."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        assert watermarker.has_watermark(watermarked) is True

    def test_has_watermark_no_watermark(self):
        """Test detecting no watermark."""
        watermarker = Watermarker()
        text = "This is plain text."
        assert watermarker.has_watermark(text) is False

    def test_remove_watermark(self):
        """Test removing watermark from text."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked = watermarker.embed_watermark(text, "user_001")
        cleaned = watermarker.remove_watermark(watermarked)
        # After removing zero-width chars, should be close to original
        assert len(cleaned) > 0

    def test_remove_watermark_plain_text(self):
        """Test removing watermark from plain text."""
        watermarker = Watermarker()
        text = "This is plain text."
        cleaned = watermarker.remove_watermark(text)
        assert cleaned == text

    def test_different_recipients_different_watermarks(self):
        """Test that different recipients get different watermarks."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked1 = watermarker.embed_watermark(text, "user_001")
        watermarked2 = watermarker.embed_watermark(text, "user_002")
        assert watermarked1 != watermarked2

    def test_same_recipient_same_watermark(self):
        """Test that same recipient gets consistent watermark."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data for watermarking."
        watermarked1 = watermarker.embed_watermark(text, "user_001")
        watermarked2 = watermarker.embed_watermark(text, "user_001")
        # Same recipient, same org -> same watermark
        assert watermarked1 == watermarked2

    def test_embed_binary_watermark(self):
        """Test embedding binary watermark."""
        watermarker = Watermarker(organization_id="test_org")
        data = b"This is binary data for watermarking."
        watermarked = watermarker.embed_binary_watermark(data, "user_001")
        assert len(watermarked) > len(data)
        assert b"DVAS_WM:" in watermarked

    def test_extract_binary_watermark(self):
        """Test extracting binary watermark."""
        watermarker = Watermarker(organization_id="test_org")
        data = b"This is binary data for watermarking."
        watermarked = watermarker.embed_binary_watermark(data, "user_001", export_id="exp_001")
        info = watermarker.extract_binary_watermark(watermarked)
        assert info is not None
        assert info.is_valid is True
        assert info.recipient_id == "user_001"

    def test_extract_binary_watermark_no_watermark(self):
        """Test extracting from data without watermark."""
        watermarker = Watermarker()
        data = b"This is plain binary data."
        info = watermarker.extract_binary_watermark(data)
        assert info is None

    def test_embed_watermark_with_export_id(self):
        """Test embedding with export ID."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data."
        watermarked = watermarker.embed_watermark(text, "user_001", export_id="exp_001")
        info = watermarker.extract_watermark(watermarked)
        assert info is not None
        assert info.export_id == "exp_001"

    def test_embed_watermark_with_user_id(self):
        """Test embedding with user ID."""
        watermarker = Watermarker(organization_id="test_org")
        text = "This is test data."
        watermarked = watermarker.embed_watermark(text, "user_001", user_id="admin_001")
        info = watermarker.extract_watermark(watermarked)
        assert info is not None
        assert info.user_id == "admin_001"

    def test_different_orgs_different_watermarks(self):
        """Test that different orgs produce different watermarks."""
        watermarker1 = Watermarker(organization_id="org1")
        watermarker2 = Watermarker(organization_id="org2")
        text = "This is test data."
        watermarked1 = watermarker1.embed_watermark(text, "user_001")
        watermarked2 = watermarker2.embed_watermark(text, "user_001")
        # Different orgs -> different secret keys -> different watermarks
        assert watermarked1 != watermarked2


class TestBatchWatermarker:
    """Test BatchWatermarker class."""

    def test_init(self):
        """Test initialization."""
        watermarker = Watermarker()
        batch = BatchWatermarker(watermarker)
        assert batch.watermarker is watermarker

    def test_watermark_annotations(self):
        """Test watermarking a batch of annotations."""
        watermarker = Watermarker(organization_id="test_org")
        batch = BatchWatermarker(watermarker)
        annotations = [
            {"id": "ann_001", "caption": "This is a long caption for watermarking."},
            {"id": "ann_002", "caption": "Another long caption for testing."},
        ]
        results = batch.watermark_annotations(annotations, "user_001", "exp_001")
        assert len(results) == 2
        assert all(r["id"] in ["ann_001", "ann_002"] for r in results)

    def test_watermark_annotations_empty(self):
        """Test watermarking empty list."""
        watermarker = Watermarker()
        batch = BatchWatermarker(watermarker)
        results = batch.watermark_annotations([], "user_001", "exp_001")
        assert results == []

    def test_watermark_annotations_nested_dict(self):
        """Test watermarking nested dictionaries."""
        watermarker = Watermarker(organization_id="test_org")
        batch = BatchWatermarker(watermarker)
        annotations = [
            {
                "id": "ann_001",
                "metadata": {
                    "description": "This is a long description for watermarking.",
                },
            },
        ]
        results = batch.watermark_annotations(annotations, "user_001", "exp_001")
        assert len(results) == 1
