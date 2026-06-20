"""Video path anonymization for DVAS.

Provides utilities to anonymize file paths that may contain usernames,
hostnames, or other identifying information.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Union

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PathAnonymizationRule:
    """A rule for anonymizing path components."""

    pattern: Pattern
    replacement: str
    description: str


class VideoPathAnonymizer:
    """Anonymize video file paths that may contain PII.

    Usage::

        anonymizer = VideoPathAnonymizer()
        anon_path = anonymizer.anonymize("/home/johndoe/videos/test.mp4")
        # Result: "/home/<USER>/videos/test.mp4"
    """

    # Common username/home directory patterns
    DEFAULT_RULES = [
        # Linux/macOS home directories
        PathAnonymizationRule(
            pattern=re.compile(r"^(/(?:home|Users)/)([^/]+)(/.*)?$"),
            replacement=r"\1<USER>\3",
            description="Replace username in home directory path",
        ),
        # Windows user profiles
        PathAnonymizationRule(
            pattern=re.compile(
                r"(?i)([A-Za-z]:[\\\\\\\\]|[\\\\\\\\])Users[\\\\\\\\]([^\\\\\\\\]+)([\\\\\\\\].*)?"
            ),
            replacement=r"\1Users\\<USER>\3",
            description="Replace Windows username",
        ),
        # Hostname in UNC paths
        PathAnonymizationRule(
            pattern=re.compile(r"^(\\\\)([^\\/]+)(\\.*)?$"),
            replacement=r"\1<HOST>\3",
            description="Replace hostname in UNC path",
        ),
        # IP address in path
        PathAnonymizationRule(
            pattern=re.compile(r"([\\/])(?:\d{1,3}\.){3}\d{1,3}([\\/])"),
            replacement=r"\1<IP>\2",
            description="Replace IP address in path",
        ),
        # S3 bucket with potentially sensitive name
        PathAnonymizationRule(
            pattern=re.compile(r"^(s3://)([^/]+)(/.*)?$"),
            replacement=r"\1<BUCKET>\3",
            description="Replace S3 bucket name",
        ),
        # GCP bucket
        PathAnonymizationRule(
            pattern=re.compile(r"^(gs://)([^/]+)(/.*)?$"),
            replacement=r"\1<BUCKET>\3",
            description="Replace GCS bucket name",
        ),
        # Azure blob
        PathAnonymizationRule(
            pattern=re.compile(r"^(https?://[^/]+\.blob\.core\.windows\.net/)([^/]+)(/.*)?$"),
            replacement=r"\1<CONTAINER>\3",
            description="Replace Azure container name",
        ),
    ]

    def __init__(
        self,
        rules: Optional[List[PathAnonymizationRule]] = None,
        hash_salt: Optional[str] = None,
    ) -> None:
        """Initialize the path anonymizer.

        Args:
            rules: Custom anonymization rules. If None, uses default rules.
            hash_salt: Optional salt for consistent hashing of path components.
        """
        self.rules = rules or self.DEFAULT_RULES.copy()
        self.hash_salt = hash_salt or "dvas_path_anon"
        self._cache: Dict[str, str] = {}

    def anonymize(self, path: Union[str, Path]) -> str:
        """Anonymize a video file path.

        Replaces usernames, hostnames, and other identifying components
        with placeholder tokens.

        Args:
            path: The path to anonymize.

        Returns:
            Anonymized path string.
        """
        if not path:
            return str(path) if path else ""

        path_str = str(path)

        # Check cache
        if path_str in self._cache:
            return self._cache[path_str]

        result = path_str

        # Apply each rule
        for rule in self.rules:
            try:
                result = rule.pattern.sub(rule.replacement, result)
            except re.error:
                logger.warning(f"Invalid regex in path anonymization rule: {rule.description}")
                continue

        # Additional processing
        result = self._anonymize_remaining_components(result)

        self._cache[path_str] = result
        return result

    def anonymize_with_hash(self, path: Union[str, Path]) -> str:
        """Anonymize path by hashing sensitive components.

        Instead of replacing with placeholders, this hashes the sensitive
        parts to create a consistent but non-reversible mapping.

        Args:
            path: The path to anonymize.

        Returns:
            Path with hashed sensitive components.
        """
        if not path:
            return str(path) if path else ""

        path_str = str(path)
        parsed = Path(path_str)
        parts = parsed.parts

        if len(parts) <= 1:
            return path_str

        new_parts = [parts[0]] if parsed.is_absolute() else []

        for i, part in enumerate(parts[1:]):
            if self._is_sensitive_component(part, i):
                hashed = self._hash_component(part)
                new_parts.append(f"h_{hashed}")
            else:
                new_parts.append(part)

        return str(Path(*new_parts))

    def anonymize_batch(self, paths: List[Union[str, Path]]) -> List[str]:
        """Anonymize multiple paths.

        Args:
            paths: List of paths to anonymize.

        Returns:
            List of anonymized paths.
        """
        return [self.anonymize(p) for p in paths]

    def get_original_from_anonymized(
        self,
        anonymized: str,
        candidates: List[str],
    ) -> Optional[str]:
        """Attempt to find the original path from an anonymized one.

        This is a best-effort reverse lookup that matches the anonymized
        path against a list of candidate paths.

        Args:
            anonymized: The anonymized path.
            candidates: List of candidate original paths.

        Returns:
            The matching original path, or None if no match found.
        """
        for candidate in candidates:
            if self.anonymize(candidate) == anonymized:
                return candidate
        return None

    def extract_identifiers(self, path: Union[str, Path]) -> Dict[str, List[str]]:
        """Extract potentially identifying components from a path.

        Args:
            path: The path to analyze.

        Returns:
            Dictionary mapping identifier types to lists of values.
        """
        path_str = str(path)
        identifiers: Dict[str, List[str]] = {
            "usernames": [],
            "hostnames": [],
            "ip_addresses": [],
            "buckets": [],
        }

        # Check for username patterns
        home_match = re.search(r"/(?:home|Users)/([^/]+)", path_str)
        if home_match:
            identifiers["usernames"].append(home_match.group(1))

        # Check for hostname in UNC
        unc_match = re.search(r"^\\\\([^\\/]+)", path_str)
        if unc_match:
            identifiers["hostnames"].append(unc_match.group(1))

        # Check for IP addresses
        ip_matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", path_str)
        identifiers["ip_addresses"].extend(ip_matches)

        # Check for S3/GCS buckets
        bucket_match = re.search(r"^(?:s3|gs)://([^/]+)", path_str)
        if bucket_match:
            identifiers["buckets"].append(bucket_match.group(1))

        return identifiers

    def _anonymize_remaining_components(self, path: str) -> str:
        """Apply additional anonymization to remaining components."""
        # Anonymize any remaining email-like patterns in path
        path = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", "<EMAIL>", path)

        # Anonymize phone number-like patterns
        path = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "<PHONE>", path)

        return path

    def _is_sensitive_component(self, part: str, index: int) -> bool:
        """Check if a path component is likely sensitive."""
        # First component after root on Unix is often username
        if index == 0 and part not in ("home", "Users", "tmp", "var", "opt", "usr"):
            return True

        # Check for common username indicators
        if part.lower() in ("documents", "desktop", "downloads", "pictures", "videos"):
            return False

        # Check for patterns that look like usernames
        if re.match(r"^[a-z][a-z0-9_]{2,}$", part):
            return True

        return False

    def _hash_component(self, component: str) -> str:
        """Create a consistent hash of a path component."""
        data = f"{self.hash_salt}:{component}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def add_rule(self, pattern: Pattern, replacement: str, description: str) -> None:
        """Add a custom anonymization rule.

        Args:
            pattern: Compiled regex pattern to match.
            replacement: Replacement string with capture groups.
            description: Human-readable description of the rule.
        """
        self.rules.append(
            PathAnonymizationRule(
                pattern=pattern,
                replacement=replacement,
                description=description,
            )
        )

    def clear_cache(self) -> None:
        """Clear the path anonymization cache."""
        self._cache.clear()


def anonymize_video_path(path: Union[str, Path]) -> str:
    """Convenience function to anonymize a single video path.

    Args:
        path: The path to anonymize.

    Returns:
        Anonymized path string.
    """
    anonymizer = VideoPathAnonymizer()
    return anonymizer.anonymize(path)


def is_path_sensitive(path: Union[str, Path]) -> bool:
    """Check if a path contains potentially sensitive information.

    Args:
        path: The path to check.

    Returns:
        True if the path contains sensitive components.
    """
    path_str = str(path).lower()

    # Check for home directory patterns
    if re.search(r"/(?:home|users)/[^/]+", path_str):
        return True

    # Check for Windows user profiles
    if re.search(r"[A-Za-z]:[\\\\/]users[\\\\/][^\\\\/]+", path_str, re.IGNORECASE):
        return True

    # Check for UNC paths with potential hostnames
    if re.match(r"^\\\\", path_str):
        return True

    # Check for S3/GCS with bucket names
    if re.match(r"^(s3|gs)://", path_str):
        return True

    return False


__all__ = [
    "VideoPathAnonymizer",
    "PathAnonymizationRule",
    "anonymize_video_path",
    "is_path_sensitive",
]
