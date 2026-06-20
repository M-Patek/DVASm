"""Watermarking for exported DVAS data.

Provides invisible and visible watermarking capabilities to detect
data leakage and trace the origin of exported annotations.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class WatermarkType(str, Enum):
    """Types of watermarks."""

    INVISIBLE_TEXT = "invisible_text"
    VISIBLE_TEXT = "visible_text"
    BINARY = "binary"
    METADATA = "metadata"


@dataclass
class WatermarkConfig:
    """Configuration for watermark embedding."""

    watermark_type: WatermarkType = WatermarkType.INVISIBLE_TEXT
    organization_id: str = "dvas"
    include_timestamp: bool = True
    include_user_id: bool = True
    include_export_id: bool = True
    custom_fields: Dict[str, str] = None

    def __post_init__(self):
        if self.custom_fields is None:
            self.custom_fields = {}


@dataclass
class WatermarkInfo:
    """Information extracted from a watermark."""

    organization_id: str
    recipient_id: str
    timestamp: Optional[float]
    user_id: Optional[str]
    export_id: Optional[str]
    custom_fields: Dict[str, str]
    is_valid: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "organization_id": self.organization_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "export_id": self.export_id,
            "custom_fields": self.custom_fields,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
        }


class Watermarker:
    """Add watermarks to exported data for leak detection.

    Usage::

        watermarker = Watermarker(organization_id="my_org")
        watermarked = watermarker.embed_watermark(
            text="Export data here",
            recipient_id="user_001",
            export_id="export_123",
        )

        # Later, detect the source
        info = watermarker.extract_watermark(watermarked)
        if info.is_valid:
            print(f"Data originated from: {info.recipient_id}")
    """

    # Zero-width Unicode characters for invisible watermarking
    ZW_SPACE = "​"  # Zero width space
    ZW_NON_JOINER = "‌"  # Zero width non-joiner
    ZW_JOINER = "‍"  # Zero width joiner
    BOM = "﻿"  # Byte order mark

    def __init__(
        self,
        organization_id: str = "dvas",
        secret_key: Optional[str] = None,
    ) -> None:
        """Initialize the watermarker.

        Args:
            organization_id: Organization identifier for watermarking.
            secret_key: Secret key for HMAC-based watermarking.
        """
        self.organization_id = organization_id
        self.secret_key = (
            secret_key
            or hashlib.sha256(f"dvas_watermark_{organization_id}".encode()).hexdigest()[:32]
        )

    def embed_watermark(
        self,
        text: str,
        recipient_id: str,
        export_id: Optional[str] = None,
        user_id: Optional[str] = None,
        custom_fields: Optional[Dict[str, str]] = None,
    ) -> str:
        """Embed an invisible watermark in text.

        Args:
            text: The text to watermark.
            recipient_id: ID of the recipient (for tracing leaks).
            export_id: Optional export batch ID.
            user_id: Optional user ID who performed the export.
            custom_fields: Optional custom fields to include.

        Returns:
            Watermarked text with invisible markers.
        """
        if not text:
            return text

        # Build watermark payload
        payload = {
            "org": self.organization_id,
            "rec": recipient_id,
            "ts": int(time.time()) if export_id else None,
            "exp": export_id,
            "usr": user_id,
            "custom": custom_fields or {},
        }

        # Create HMAC signature
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        signature = hmac.new(
            self.secret_key.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]

        # Encode payload + signature as invisible characters
        encoded = self._encode_invisible(payload_str + ":" + signature)

        # Insert at a deterministic position
        insert_pos = len(text) // 2
        watermarked = text[:insert_pos] + encoded + text[insert_pos:]

        return watermarked

    def extract_watermark(self, text: str) -> Optional[WatermarkInfo]:
        """Extract watermark information from text.

        Args:
            text: The text to analyze.

        Returns:
            WatermarkInfo if a valid watermark is found, None otherwise.
        """
        if not text:
            return None

        # Extract zero-width characters
        zw_chars = self._extract_zw_chars(text)
        if not zw_chars:
            return None

        try:
            # Decode payload
            decoded = self._decode_invisible(zw_chars)
            if ":" not in decoded:
                return None

            payload_str, signature = decoded.rsplit(":", 1)
            payload = json.loads(payload_str)

            # Verify signature
            expected_sig = hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256,
            ).hexdigest()[:16]

            is_valid = hmac.compare_digest(signature, expected_sig)

            return WatermarkInfo(
                organization_id=payload.get("org", ""),
                recipient_id=payload.get("rec", ""),
                timestamp=payload.get("ts"),
                user_id=payload.get("usr"),
                export_id=payload.get("exp"),
                custom_fields=payload.get("custom", {}),
                is_valid=is_valid,
                confidence=1.0 if is_valid else 0.5,
            )

        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return None

    def embed_visible_watermark(
        self,
        text: str,
        label: str,
        position: str = "footer",
    ) -> str:
        """Embed a visible watermark in text.

        Args:
            text: The text to watermark.
            label: The visible label to add.
            position: Where to place the watermark ("header", "footer").

        Returns:
            Text with visible watermark.
        """
        timestamp = datetime.now().isoformat()
        watermark_line = f"\n---\n[EXPORTED: {label} | {timestamp}]\n---\n"

        if not text:
            return watermark_line

        if position == "header":
            return watermark_line + text
        else:
            return text + watermark_line

        if position == "header":
            return watermark_line + text
        else:
            return text + watermark_line

    def embed_binary_watermark(
        self,
        data: bytes,
        recipient_id: str,
        export_id: Optional[str] = None,
    ) -> bytes:
        """Embed a watermark in binary data.

        Args:
            data: Binary data to watermark.
            recipient_id: ID of the recipient.
            export_id: Optional export batch ID.

        Returns:
            Watermarked binary data.
        """
        # Create watermark header
        payload = {
            "org": self.organization_id,
            "rec": recipient_id,
            "ts": int(time.time()),
            "exp": export_id,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]

        # Create watermark block
        watermark = b"DVAS_WM:" + payload_str.encode() + b":" + signature.encode()

        # Append to data
        return data + watermark

    def extract_binary_watermark(self, data: bytes) -> Optional[WatermarkInfo]:
        """Extract watermark from binary data.

        Args:
            data: Binary data to analyze.

        Returns:
            WatermarkInfo if found, None otherwise.
        """
        marker = b"DVAS_WM:"
        idx = data.rfind(marker)
        if idx == -1:
            return None

        try:
            watermark_data = data[idx + len(marker) :].decode()
            payload_str, signature = watermark_data.rsplit(":", 1)
            payload = json.loads(payload_str)

            expected_sig = hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256,
            ).hexdigest()[:16]

            is_valid = hmac.compare_digest(signature, expected_sig)

            return WatermarkInfo(
                organization_id=payload.get("org", ""),
                recipient_id=payload.get("rec", ""),
                timestamp=payload.get("ts"),
                user_id=None,
                export_id=payload.get("exp"),
                custom_fields={},
                is_valid=is_valid,
                confidence=1.0 if is_valid else 0.5,
            )
        except (ValueError, json.JSONDecodeError):
            return None

    def verify_watermark(self, text: str, expected_recipient: str) -> bool:
        """Verify that text contains a watermark for the expected recipient.

        Args:
            text: The text to verify.
            expected_recipient: The expected recipient ID.

        Returns:
            True if the watermark matches the expected recipient.
        """
        info = self.extract_watermark(text)
        if info is None:
            return False
        return info.is_valid and info.recipient_id == expected_recipient

    def _encode_invisible(self, data: str) -> str:
        """Encode string as invisible Unicode characters."""
        # Convert to binary
        binary = "".join(format(ord(c), "016b") for c in data)

        # Map to zero-width characters
        encoded = ""
        for bit in binary:
            if bit == "0":
                encoded += self.ZW_SPACE
            else:
                encoded += self.ZW_NON_JOINER

        return encoded

    def _decode_invisible(self, zw_chars: str) -> str:
        """Decode invisible Unicode characters back to string."""
        binary = ""
        for char in zw_chars:
            if char == self.ZW_SPACE:
                binary += "0"
            elif char == self.ZW_NON_JOINER:
                binary += "1"

        if len(binary) % 16 != 0:
            # Pad to multiple of 16
            binary = binary[: len(binary) - (len(binary) % 16)]

        # Convert binary back to string
        chars = []
        for i in range(0, len(binary), 16):
            chunk = binary[i : i + 16]
            if len(chunk) == 16:
                char_code = int(chunk, 2)
                if 0 <= char_code <= 0x10FFFF:
                    chars.append(chr(char_code))

        return "".join(chars)

    def _extract_zw_chars(self, text: str) -> str:
        """Extract zero-width characters from text."""
        zw_set = {self.ZW_SPACE, self.ZW_NON_JOINER, self.ZW_JOINER, self.BOM}
        return "".join(c for c in text if c in zw_set)

    def has_watermark(self, text: str) -> bool:
        """Check if text contains a watermark.

        Args:
            text: The text to check.

        Returns:
            True if a watermark is detected.
        """
        zw_chars = self._extract_zw_chars(text)
        return len(zw_chars) > 0

    def remove_watermark(self, text: str) -> str:
        """Remove any watermarks from text.

        Args:
            text: The text to clean.

        Returns:
            Text with watermarks removed.
        """
        zw_set = {self.ZW_SPACE, self.ZW_NON_JOINER, self.ZW_JOINER, self.BOM}
        return "".join(c for c in text if c not in zw_set)


class BatchWatermarker:
    """Watermark multiple items in a batch export."""

    def __init__(self, watermarker: Watermarker) -> None:
        """Initialize batch watermarker.

        Args:
            watermarker: The Watermarker instance to use.
        """
        self.watermarker = watermarker

    def watermark_annotations(
        self,
        annotations: List[Dict[str, Any]],
        recipient_id: str,
        export_id: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Watermark a batch of annotations.

        Args:
            annotations: List of annotation dictionaries.
            recipient_id: ID of the recipient.
            export_id: Export batch ID.
            user_id: Optional user ID.

        Returns:
            Watermarked annotations.
        """
        results = []
        for i, annotation in enumerate(annotations):
            # Watermark text fields
            watermarked = self._watermark_dict(
                annotation,
                recipient_id,
                f"{export_id}:{i}",
                user_id,
            )
            results.append(watermarked)

        return results

    def _watermark_dict(
        self,
        data: Dict[str, Any],
        recipient_id: str,
        export_id: str,
        user_id: Optional[str],
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Recursively watermark string fields in a dictionary."""
        if depth > 10:
            return data

        result = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10:
                result[key] = self.watermarker.embed_watermark(
                    value,
                    recipient_id,
                    export_id,
                    user_id,
                )
            elif isinstance(value, dict):
                result[key] = self._watermark_dict(
                    value, recipient_id, export_id, user_id, depth + 1
                )
            elif isinstance(value, list):
                result[key] = [
                    self._watermark_dict(item, recipient_id, export_id, user_id, depth + 1)
                    if isinstance(item, dict)
                    else self.watermarker.embed_watermark(item, recipient_id, export_id, user_id)
                    if isinstance(item, str) and len(item) > 10
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result


__all__ = [
    "Watermarker",
    "BatchWatermarker",
    "WatermarkConfig",
    "WatermarkInfo",
    "WatermarkType",
]
