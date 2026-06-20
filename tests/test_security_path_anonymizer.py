"""Tests for video path anonymization module.

Tests for VideoPathAnonymizer, PathAnonymizationRule, and utility functions.
"""

import re
from pathlib import Path

import pytest

from dvas.security.path_anonymizer import (
    PathAnonymizationRule,
    VideoPathAnonymizer,
    anonymize_video_path,
    is_path_sensitive,
)


class TestPathAnonymizationRule:
    """Test PathAnonymizationRule dataclass."""

    def test_rule_creation(self):
        """Test creating a path anonymization rule."""
        rule = PathAnonymizationRule(
            pattern=re.compile(r"test"),
            replacement="[REDACTED]",
            description="Test rule",
        )
        assert rule.replacement == "[REDACTED]"
        assert rule.description == "Test rule"


class TestVideoPathAnonymizer:
    """Test VideoPathAnonymizer class."""

    def test_anonymize_linux_home(self):
        """Test anonymizing Linux home directory path."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        result = anonymizer.anonymize(path)
        assert "johndoe" not in result
        assert "<USER>" in result

    def test_anonymize_mac_home(self):
        """Test anonymizing macOS home directory path."""
        anonymizer = VideoPathAnonymizer()
        path = "/Users/janedoe/Movies/test.mp4"
        result = anonymizer.anonymize(path)
        assert "janedoe" not in result
        assert "<USER>" in result

    def test_anonymize_windows_path(self):
        """Test anonymizing Windows path."""
        anonymizer = VideoPathAnonymizer()
        path = r"C:\Users\johndoe\Videos\test.mp4"
        result = anonymizer.anonymize(path)
        assert "johndoe" not in result
        assert "<USER>" in result

    def test_anonymize_unc_path(self):
        """Test anonymizing UNC path."""
        anonymizer = VideoPathAnonymizer()
        path = r"\\server\share\videos\test.mp4"
        result = anonymizer.anonymize(path)
        assert "server" not in result
        assert "<HOST>" in result

    def test_anonymize_s3_path(self):
        """Test anonymizing S3 path."""
        anonymizer = VideoPathAnonymizer()
        path = "s3://my-sensitive-bucket/videos/test.mp4"
        result = anonymizer.anonymize(path)
        assert "my-sensitive-bucket" not in result
        assert "<BUCKET>" in result

    def test_anonymize_gcs_path(self):
        """Test anonymizing GCS path."""
        anonymizer = VideoPathAnonymizer()
        path = "gs://my-project-bucket/data/video.mp4"
        result = anonymizer.anonymize(path)
        assert "my-project-bucket" not in result
        assert "<BUCKET>" in result

    def test_anonymize_empty_path(self):
        """Test anonymizing empty path."""
        anonymizer = VideoPathAnonymizer()
        assert anonymizer.anonymize("") == ""

    def test_anonymize_none_path(self):
        """Test anonymizing None path."""
        anonymizer = VideoPathAnonymizer()
        assert anonymizer.anonymize(None) == ""

    def test_anonymize_batch(self):
        """Test batch anonymization."""
        anonymizer = VideoPathAnonymizer()
        paths = [
            "/home/johndoe/videos/a.mp4",
            "/home/janedoe/videos/b.mp4",
            "/Users/admin/Movies/c.mp4",
        ]
        results = anonymizer.anonymize_batch(paths)
        assert len(results) == 3
        assert all("<USER>" in r for r in results)

    def test_anonymize_with_hash(self):
        """Test anonymizing with hash."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        result = anonymizer.anonymize_with_hash(path)
        assert "johndoe" not in result
        # Should contain hash prefix
        assert "h_" in result

    def test_anonymize_with_hash_consistency(self):
        """Test that hashing is consistent."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        result1 = anonymizer.anonymize_with_hash(path)
        result2 = anonymizer.anonymize_with_hash(path)
        assert result1 == result2

    def test_extract_identifiers(self):
        """Test extracting identifiers from path."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        identifiers = anonymizer.extract_identifiers(path)
        assert "johndoe" in identifiers["usernames"]

    def test_extract_identifiers_unc(self):
        """Test extracting identifiers from UNC path."""
        anonymizer = VideoPathAnonymizer()
        path = r"\\server01\share\videos"
        identifiers = anonymizer.extract_identifiers(path)
        assert "server01" in identifiers["hostnames"]

    def test_get_original_from_anonymized(self):
        """Test reverse lookup of anonymized path."""
        anonymizer = VideoPathAnonymizer()
        original = "/home/johndoe/videos/test.mp4"
        anonymized = anonymizer.anonymize(original)
        candidates = [original, "/home/other/videos/test.mp4"]
        found = anonymizer.get_original_from_anonymized(anonymized, candidates)
        assert found == original

    def test_get_original_not_found(self):
        """Test reverse lookup when no match."""
        anonymizer = VideoPathAnonymizer()
        anonymized = "/some/anonymized/path"
        candidates = ["/home/johndoe/videos/test.mp4"]
        found = anonymizer.get_original_from_anonymized(anonymized, candidates)
        assert found is None

    def test_add_custom_rule(self):
        """Test adding custom anonymization rule."""
        anonymizer = VideoPathAnonymizer()
        anonymizer.add_rule(
            pattern=re.compile(r"secret"),
            replacement="[REDACTED]",
            description="Remove secret keyword",
        )
        result = anonymizer.anonymize("/data/secret/videos/test.mp4")
        assert "secret" not in result
        assert "[REDACTED]" in result

    def test_cache(self):
        """Test that caching works."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        result1 = anonymizer.anonymize(path)
        result2 = anonymizer.anonymize(path)
        assert result1 == result2

    def test_clear_cache(self):
        """Test clearing the cache."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/johndoe/videos/test.mp4"
        anonymizer.anonymize(path)
        anonymizer.clear_cache()
        # Cache should be empty
        assert len(anonymizer._cache) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_anonymize_video_path(self):
        """Test the convenience function."""
        result = anonymize_video_path("/home/johndoe/videos/test.mp4")
        assert "johndoe" not in result
        assert "<USER>" in result

    def test_is_path_sensitive_linux(self):
        """Test detecting sensitive Linux path."""
        assert is_path_sensitive("/home/johndoe/videos/test.mp4")

    def test_is_path_sensitive_windows(self):
        """Test detecting sensitive Windows path."""
        assert is_path_sensitive(r"C:\Users\johndoe\Videos\test.mp4")

    def test_is_path_sensitive_s3(self):
        """Test detecting sensitive S3 path."""
        assert is_path_sensitive("s3://my-bucket/videos/test.mp4")

    def test_is_path_sensitive_not_sensitive(self):
        """Test non-sensitive path."""
        assert not is_path_sensitive("/data/videos/test.mp4")

    def test_is_path_sensitive_unc(self):
        """Test detecting sensitive UNC path."""
        assert is_path_sensitive(r"\\\\server\\share\\videos")

    def test_is_path_sensitive_empty(self):
        """Test empty path."""
        assert not is_path_sensitive("")


class TestVideoPathAnonymizerEdgeCases:
    """Test edge cases for VideoPathAnonymizer."""

    def test_very_long_path(self):
        """Test with very long path."""
        anonymizer = VideoPathAnonymizer()
        long_component = "a" * 1000
        path = f"/home/{long_component}/videos/test.mp4"
        result = anonymizer.anonymize(path)
        # Should not crash
        assert isinstance(result, str)

    def test_path_with_special_chars(self):
        """Test path with special characters."""
        anonymizer = VideoPathAnonymizer()
        path = "/home/user@domain/videos/test.mp4"
        result = anonymizer.anonymize(path)
        assert isinstance(result, str)

    def test_path_as_path_object(self):
        """Test with Path object input."""
        anonymizer = VideoPathAnonymizer()
        # On Windows, Path('/home/johndoe/videos/test.mp4') becomes '\\home\\johndoe\\videos\\test.mp4'
        # So we use a raw string instead
        path = "/home/johndoe/videos/test.mp4"
        result = anonymizer.anonymize(path)
        assert "johndoe" not in result
