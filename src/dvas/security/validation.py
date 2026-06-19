"""Input validation and sanitization for DVAS.

Provides string sanitization, SQL injection detection, XSS detection,
path traversal prevention, and filename sanitization.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validate and sanitize user inputs.

    Usage::

        validator = InputValidator()
        clean = validator.sanitize_string(user_input)
        validator.validate_video_id(video_id)
    """

    # Patterns for common attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
        r"(\b(OR|AND)\s+\d+=\d+)",
        r"(--|#|\/\*)",
        r"(\bEXEC\b|\bEXECUTE\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.",
        r"^/",
        r"^\\\\",
    ]

    def __init__(self) -> None:
        self._sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self._xss_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS]
        self._path_patterns = [re.compile(p) for p in self.PATH_TRAVERSAL_PATTERNS]

    def sanitize_string(self, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize a string input.

        Removes control characters, trims whitespace, and optionally escapes HTML.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")

        # Trim whitespace
        value = value.strip()

        # Remove control characters except newlines and tabs
        value = "".join(c for c in value if c == "\n" or c == "\t" or ord(c) >= 32)

        # Escape HTML if not allowed
        if not allow_html:
            value = html.escape(value)

        # Limit length
        if len(value) > max_length:
            value = value[:max_length]

        return value

    def validate_video_id(self, video_id: str) -> str:
        """Validate a video ID.

        Args:
            video_id: The video ID to validate

        Returns:
            The validated video ID

        Raises:
            ValueError: If the video ID is invalid
        """
        if not video_id or len(video_id) < 3:
            raise ValueError("Video ID must be at least 3 characters")

        if len(video_id) > 256:
            raise ValueError("Video ID must be at most 256 characters")

        # Allow alphanumeric, underscore, hyphen, and dots
        if not re.match(r"^[a-zA-Z0-9_.-]+$", video_id):
            raise ValueError("Video ID contains invalid characters")

        # Check for path traversal
        if ".." in video_id or "/" in video_id or "\\" in video_id:
            raise ValueError("Video ID cannot contain path separators")

        return video_id

    def validate_file_path(self, path: str, allowed_prefixes: Optional[List[str]] = None) -> str:
        """Validate a file path to prevent path traversal.

        Args:
            path: The path to validate
            allowed_prefixes: List of allowed path prefixes

        Returns:
            The validated path

        Raises:
            ValueError: If the path is invalid or outside allowed prefixes
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Normalize the path
        normalized = Path(path).resolve()

        # Check for path traversal
        for pattern in self._path_patterns:
            if pattern.search(str(normalized)):
                raise ValueError(f"Path contains traversal characters: {path}")

        # Check allowed prefixes
        if allowed_prefixes:
            allowed = False
            for prefix in allowed_prefixes:
                prefix_path = Path(prefix).resolve()
                try:
                    normalized.relative_to(prefix_path)
                    allowed = True
                    break
                except ValueError:
                    continue

            if not allowed:
                raise ValueError(f"Path outside allowed directories: {path}")

        return str(normalized)

    def check_sql_injection(self, value: str) -> List[str]:
        """Check for SQL injection patterns.

        Returns:
            List of matched patterns (empty if clean)
        """
        matches = []
        for pattern in self._sql_patterns:
            if pattern.search(value):
                matches.append(pattern.pattern)
        return matches

    def check_xss(self, value: str) -> List[str]:
        """Check for XSS patterns.

        Returns:
            List of matched patterns (empty if clean)
        """
        matches = []
        for pattern in self._xss_patterns:
            if pattern.search(value):
                matches.append(pattern.pattern)
        return matches

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal and injection.

        Args:
            filename: The filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators
        filename = filename.replace("/", "_").replace("\\", "_")

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Remove control characters
        filename = "".join(c for c in filename if ord(c) >= 32)

        # Limit length
        if len(filename) > 255:
            filename = filename[:255]

        # Ensure not empty
        if not filename:
            filename = "unnamed"

        return filename


__all__ = ["InputValidator"]
