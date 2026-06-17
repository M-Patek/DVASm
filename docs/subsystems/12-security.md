---
id: 12-security
title: "12-Security — Privacy & Access Control"
status: stable
applies_to:
  - "src/dvas/security/**"
code_anchors:
  - "src/dvas/security/privacy.py:PIIDetector"
  - "src/dvas/security/privacy.py:DataAnonymizer"
  - "src/dvas/security/privacy.py:SecurityAuditor"
  - "src/dvas/security/privacy.py:Watermarker"
  - "src/dvas/security/privacy.py:AccessControl"
agent_hints:
  - "WARNING: PII detector uses regex - not 100% accurate, add manual review"
  - "WARNING: Watermark is simple zero-width chars - use steganography library for production"
  - "WARNING: AccessControl is in-memory only - persist to database in production"
  - "WARNING: Always use HTTPS for API in production"
---

# §12 Security

Privacy protection, PII detection, access control, and audit logging.

---

## §0 — One-liner

Protect sensitive data through PII detection, anonymization, watermarking, and role-based access control.

## §1 — Core concepts

- **PIIDetector**: Detect PII in text (email, phone, SSN, credit card)
- **DataAnonymizer**: Hash IDs and anonymize paths
- **SecurityAuditor**: Audit log for access events
- **Watermarker**: Embed invisible watermarks for leak detection
- **AccessControl**: Role-based permissions (RBAC)

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `privacy.py:PIIDetector` | Scan for PII | Before storage/export |
| `privacy.py:DataAnonymizer` | Anonymize data | Public datasets |
| `privacy.py:SecurityAuditor` | Log access | Security compliance |
| `privacy.py:Watermarker` | Leak detection | External sharing |
| `privacy.py:AccessControl` | Permissions | Multi-user setup |

## §3 — Key behaviors & contracts

### Behavior 1: PII Detection

```python
detector = PIIDetector(redaction_token="[REDACTED]")

# Scan text
findings = detector.scan_text(text)
# Returns: [{"type": "email", "value": "...", "position": (start, end)}]

# Redact
clean = detector.redact_text(text)
```

**Detection patterns**:
- Email addresses
- Phone numbers
- SSN (XXX-XX-XXXX)
- Credit card numbers

### Behavior 2: Data Anonymization

```python
anonymizer = DataAnonymizer(salt="secret_salt")

# Hash identifiers
new_id = anonymizer.hash_id("original_id_123")

# Anonymize full annotation
anon_annotation = anonymizer.anonymize_annotation(annotation)
```

### Behavior 3: Security Audit

```python
auditor = SecurityAuditor(log_path=Path("logs/security.log"))

auditor.log_access(
    event_type="annotation_read",
    resource_id="ann_123",
    action="read",
    access_granted=True,
    user_id="user@example.com",
    ip_address="192.168.1.1",
)

# Query history
history = auditor.get_access_history("ann_123")
```

### Behavior 4: Watermarking

```python
watermarker = Watermarker(organization_id="company_xyz")

# Embed watermark (zero-width characters)
watermarked = watermarker.embed_watermark(text, recipient_id="client_001")

# Detect watermark
found = watermarker.extract_watermark(leaked_text)
```

### Behavior 5: Access Control

```python
ac = AccessControl()

# Assign roles
ac.assign_role("user1", "annotator")
ac.assign_role("admin1", "admin")

# Set ownership
ac.set_owner("ann_123", "user1")

# Check permission
if ac.has_permission("user1", "ann_123", "write"):
    # Allow write
    pass

# Available roles: admin, annotator, reviewer, viewer, api_client
```

## §4 — Integration with other subsystems

- **Upstream**: Applied to `Annotation` from `01-data`
- **Downstream**: Controls `07-api` access
- **Related**: Used before `06-export` for external sharing

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| PII detection | Complete | Regex-based |
| Anonymization | Complete | Hash-based IDs |
| Audit logging | Complete | File-based |
| Watermarking | Basic | Zero-width chars |
| RBAC | Complete | 5 roles |
| Encryption at rest | Missing | Use filesystem encryption |
| Rate limiting | Missing | Add to API layer |

## §6 — Testing

```bash
# Test PII detection
python -c "
from dvas.security.privacy import PIIDetector

detector = PIIDetector()
text = 'Contact me at test@email.com or call 555-1234'
findings = detector.scan_text(text)
print(f'Found {len(findings)} PII items')
"
```

---

*Subsystem doc: 12-security | Updated: 2024-06-17*
