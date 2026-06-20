"""Security and compliance module for DVAS.

Provides security utilities including:
- PII detection and redaction
- Video path anonymization
- Annotation anonymization
- Watermarking for data leakage detection
- Role-based access control (RBAC)
- Comprehensive audit logging
- Encryption at rest
- Secret management
- Tenant-level data isolation
- Export approval audit
- Data retention policy
- Deletion request flow (GDPR)
- Security regression tests
- Bandit/SAST gate integration
"""

from __future__ import annotations

# Re-export from specialized modules
from dvas.security.annotation_anonymizer import (
    AnonymizationConfig,
    AnnotationAnonymizer,
    anonymize_annotation_dict,
    mask_annotation_fields,
)
from dvas.security.audit_comprehensive import (
    AuditDecorator,
    AuditEvent,
    AuditEventType,
    AuditLog,
    AuditSeverity,
)
from dvas.security.audit_log import AuditEvent as AuditEventLegacy
from dvas.security.audit_log import AuditEventType as AuditEventTypeLegacy
from dvas.security.audit_log import AuditLogger
from dvas.security.crypto import DataEncryptor, PasswordHasher, secure_hex, secure_token
from dvas.security.deletion import (
    DeletionRequest,
    DeletionRequestFlow,
    DeletionScope,
    DeletionStatus,
)
from dvas.security.encryption import (
    EncryptionAtRest,
    EncryptionConfig,
    FieldEncryption,
    KeyManager,
)
from dvas.security.export_audit import (
    ExportApproval,
    ExportApprovalAudit,
    ExportStatus,
)
from dvas.security.headers import (
    APIKeyManager,
    ContentSecurityPolicy,
    CSRFProtection,
    SecurityHeaders,
)
from dvas.security.path_anonymizer import (
    PathAnonymizationRule,
    VideoPathAnonymizer,
    anonymize_video_path,
    is_path_sensitive,
)
from dvas.security.pii import PIIDetector, PIIFinding, PIIType
from dvas.security.privacy import (
    AccessControl,
    DataAnonymizer,
    PIIDetector as PIIDetectorLegacy,
    SecurityAuditor,
    SecurityAuditLog,
    Watermarker as WatermarkerLegacy,
)
from dvas.security.rbac import (
    ROLE_PERMISSIONS,
    AccessPolicy,
    Permission,
    PermissionChecker,
    RBAC,
    Role,
)
from dvas.security.regression import (
    SecurityRegressionRunner,
    SecurityRegressionTestSuite,
    SecurityTestResult,
)
from dvas.security.retention import (
    DataRetentionPolicy,
    DataType,
    RetentionAction,
    RetentionRecord,
    RetentionRule,
)
from dvas.security.sast import (
    BanditScanner,
    SASTGate,
    SASTGateConfig,
    SASTFinding,
    SecurityScanner,
    Severity,
    Confidence,
)
from dvas.security.secrets import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    Secret,
    SecretManager,
)
from dvas.security.tenant import (
    Tenant,
    TenantIsolationError,
    TenantManager,
    TenantScopedStorage,
)
from dvas.security.validation import InputValidator
from dvas.security.watermark import (
    BatchWatermarker,
    WatermarkConfig,
    WatermarkInfo,
    WatermarkType,
    Watermarker,
)

__all__ = [
    # PII Detection
    "PIIDetector",
    "PIIFinding",
    "PIIType",
    # Path Anonymization
    "VideoPathAnonymizer",
    "PathAnonymizationRule",
    "anonymize_video_path",
    "is_path_sensitive",
    # Annotation Anonymization
    "AnnotationAnonymizer",
    "AnonymizationConfig",
    "anonymize_annotation_dict",
    "mask_annotation_fields",
    # Watermark
    "Watermarker",
    "BatchWatermarker",
    "WatermarkConfig",
    "WatermarkInfo",
    "WatermarkType",
    # RBAC
    "RBAC",
    "Permission",
    "Role",
    "AccessPolicy",
    "PermissionChecker",
    "ROLE_PERMISSIONS",
    # Audit Log
    "AuditLog",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditDecorator",
    # Encryption
    "EncryptionAtRest",
    "EncryptionConfig",
    "FieldEncryption",
    "KeyManager",
    # Secret Management
    "SecretManager",
    "Secret",
    "EnvironmentSecretProvider",
    "CompositeSecretProvider",
    # Tenant Isolation
    "TenantManager",
    "Tenant",
    "TenantScopedStorage",
    "TenantIsolationError",
    # Export Approval
    "ExportApprovalAudit",
    "ExportApproval",
    "ExportStatus",
    # Data Retention
    "DataRetentionPolicy",
    "RetentionRule",
    "RetentionRecord",
    "RetentionAction",
    "DataType",
    # Deletion Request
    "DeletionRequestFlow",
    "DeletionRequest",
    "DeletionStatus",
    "DeletionScope",
    # Security Regression Tests
    "SecurityRegressionTestSuite",
    "SecurityRegressionRunner",
    "SecurityTestResult",
    # SAST Gate
    "BanditScanner",
    "SASTGate",
    "SASTGateConfig",
    "SASTFinding",
    "SecurityScanner",
    "Severity",
    "Confidence",
    # Legacy / Backward compatibility
    "AuditLogger",
    "AuditEventLegacy",
    "AuditEventTypeLegacy",
    "DataEncryptor",
    "PasswordHasher",
    "secure_token",
    "secure_hex",
    "CSRFProtection",
    "ContentSecurityPolicy",
    "SecurityHeaders",
    "APIKeyManager",
    "InputValidator",
    "DataAnonymizer",
    "AccessControl",
    "SecurityAuditor",
    "SecurityAuditLog",
    "PIIDetectorLegacy",
    "WatermarkerLegacy",
]
