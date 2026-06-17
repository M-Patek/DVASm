"""Security and privacy protection for annotations."""

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class PIIDetector:
    """Detect and redact personally identifiable information."""

    # Patterns for PII detection
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    }

    SENSITIVE_OBJECTS = {
        "license_plate",
        "id_card",
        "passport",
        "credit_card",
        "social_security_card",
        "face",
        "person_name",
    }

    def __init__(self, redaction_token: str = "[REDACTED]"):
        self.redaction_token = redaction_token

    def scan_text(self, text: str) -> List[Dict]:
        """Scan text for PII."""
        findings = []

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append({
                    "type": pii_type,
                    "value": match.group(),
                    "position": (match.start(), match.end()),
                })

        return findings

    def redact_text(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text

        for pii_type, pattern in self.PATTERNS.items():
            redacted = re.sub(pattern, self.redaction_token, redacted)

        return redacted

    def check_objects(self, objects: List[Dict]) -> List[str]:
        """Check for sensitive objects."""
        found = []
        for obj in objects:
            if obj.get("name", "").lower() in self.SENSITIVE_OBJECTS:
                found.append(obj["name"])
        return found


class DataAnonymizer:
    """Anonymize annotation data."""

    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or "dvas_default_salt"

    def hash_id(self, original_id: str) -> str:
        """Create consistent hash of ID."""
        return hashlib.sha256(
            f"{original_id}{self.salt}".encode()
        ).hexdigest()[:16]

    def anonymize_annotation(self, annotation: Annotation) -> Annotation:
        """Create anonymized version of annotation."""
        import copy

        anon = copy.deepcopy(annotation)

        # Hash sensitive IDs
        anon.id = self.hash_id(annotation.id)
        anon.video_id = self.hash_id(annotation.video_id)

        # Anonymize video path
        if anon.video_path:
            path = Path(anon.video_path)
            anon.video_path = str(path.parent / self.hash_id(path.name))

        return anon


@dataclass
class SecurityAuditLog:
    """Security audit event."""

    timestamp: str
    event_type: str
    user_id: Optional[str]
    resource_id: str
    action: str
    access_granted: bool
    ip_address: Optional[str]
    user_agent: Optional[str]
    metadata: Dict


class SecurityAuditor:
    """Audit logging for security events."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_access(
        self,
        event_type: str,
        resource_id: str,
        action: str,
        access_granted: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **metadata,
    ) -> None:
        """Log an access event."""
        log_entry = SecurityAuditLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            access_granted=access_granted,
            ip_address=ip_address,
            user_agent=metadata.get("user_agent"),
            metadata=metadata,
        )

        # Append to log file
        with open(self.log_path, "a") as f:
            f.write(log_entry.__dict__.__str__() + "\n")

        # Also log to structured logger
        logger.info(
            "security_audit",
            event_type=event_type,
            resource=resource_id,
            action=action,
            granted=access_granted,
            user=user_id,
        )

    def get_access_history(
        self, resource_id: str, limit: int = 100
    ) -> List[Dict]:
        """Get access history for a resource."""
        history = []

        if not self.log_path.exists():
            return history

        with open(self.log_path) as f:
            for line in reversed(f.readlines()):
                if resource_id in line:
                    # Parse log entry (simplified)
                    history.append({"raw": line.strip()})
                    if len(history) >= limit:
                        break

        return history


class Watermarker:
    """Add invisible watermarks to detect data leakage."""

    def __init__(self, organization_id: str = "dvas"):
        self.org_id = organization_id

    def embed_watermark(self, text: str, recipient_id: str) -> str:
        """Embed invisible watermark in text."""
        # Simple zero-width character watermarking
        # In production, use more sophisticated steganography

        # Create hash-based watermark
        watermark = hmac.new(
            self.org_id.encode(),
            recipient_id.encode(),
            hashlib.sha256,
        ).hexdigest()[:8]

        # Encode as zero-width characters
        encoded = ""
        for char in watermark:
            val = int(char, 16)
            if val < 8:
                encoded += "​"  # Zero width space
            else:
                encoded += "‌"  # Zero width non-joiner

        # Insert at random position
        pos = len(text) // 2
        watermarked = text[:pos] + encoded + text[pos:]

        return watermarked

    def extract_watermark(self, text: str) -> Optional[str]:
        """Extract watermark if present."""
        # Extract zero-width characters
        zw_chars = [c for c in text if c in ("​", "‌")]

        if not zw_chars:
            return None

        # Decode (simplified)
        decoded = "".join("0" if c == "​" else "1" for c in zw_chars)

        return decoded[:8] if decoded else None


class AccessControl:
    """Role-based access control for annotations."""

    ROLES = {
        "admin": {"read", "write", "delete", "export", "manage_users"},
        "annotator": {"read", "write"},
        "reviewer": {"read", "write", "approve"},
        "viewer": {"read"},
        "api_client": {"read", "export"},
    }

    def __init__(self):
        self.user_roles: Dict[str, str] = {}
        self.resource_owners: Dict[str, str] = {}

    def assign_role(self, user_id: str, role: str) -> None:
        """Assign role to user."""
        if role not in self.ROLES:
            raise ValueError(f"Invalid role: {role}")
        self.user_roles[user_id] = role

    def set_owner(self, resource_id: str, user_id: str) -> None:
        """Set resource owner."""
        self.resource_owners[resource_id] = user_id

    def has_permission(
        self, user_id: str, resource_id: str, action: str
    ) -> bool:
        """Check if user has permission for action."""
        # Owner always has full access
        if self.resource_owners.get(resource_id) == user_id:
            return True

        # Check role permissions
        role = self.user_roles.get(user_id, "viewer")
        permissions = self.ROLES.get(role, set())

        return action in permissions

    def require_permission(self, user_id: str, resource_id: str, action: str):
        """Decorator to require permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.has_permission(user_id, resource_id, action):
                    raise PermissionError(
                        f"User {user_id} lacks permission {action} for {resource_id}"
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator
