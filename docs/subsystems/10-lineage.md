---
id: 10-lineage
title: "10-Lineage — Schema Version & Data Provenance"
status: stable
applies_to:
  - "src/dvas/lineage/**"
code_anchors:
  - "src/dvas/lineage/lineage_tracker.py:LineageTracker"
  - "src/dvas/lineage/lineage_tracker.py:SchemaVersion"
agent_hints:
  - "WARNING: LineageTracker is in-memory only — persist externally if needed"
  - "WARNING: Schema compatibility checks are advisory — they don't block operations"
---

# §10 Lineage

Tracks annotation lifecycle and schema version compatibility.

---

## §0 — One-liner

In-memory tracker for annotation provenance and schema compatibility between v1.0 (EPIC) and v2.0 (VLA enhanced).

## §1 — Core concepts

- **LineageTracker**: Records processing steps for each annotation
- **SchemaVersion**: Enum of supported versions (v1.0, v2.0, v3.0)
- **SchemaCompatibility**: Result of compatibility check with warnings/errors
- **LineageStep**: Single step in an annotation's lifecycle

## §2 — Entry points

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `lineage_tracker.py:LineageTracker` | Track provenance | After each pipeline step |
| `lineage_tracker.py:SchemaVersion` | Version enum | Checking schema version |

## §3 — Key behaviors

### Behavior 1: Record Steps

```python
from dvas.lineage import LineageTracker

tracker = LineageTracker()
tracker.record_step("ann_001", "pipeline_annotation", {"model": "gpt-5.5"})
tracker.record_step("ann_001", "human_review", {"reviewer": "alice"})

provenance = tracker.get_provenance("ann_001")
```

### Behavior 2: Check Compatibility

```python
from dvas.lineage import LineageTracker

tracker = LineageTracker()
compat = tracker.check_compatibility(annotation, target_version="2.0")

if not compat.compatible:
    print("Errors:", compat.errors)
if compat.warnings:
    print("Warnings:", compat.warnings)
```

Compatibility matrix:

| From | To | Compatible | Notes |
|------|-----|-----------|-------|
| 1.0 | 1.0 | ✅ | Identity |
| 1.0 | 2.0 | ✅ | v2.0 backward compatible |
| 2.0 | 1.0 | ❌ | Enhanced fields lost |
| 2.0 | 2.0 | ✅ | Identity |

## §4 — Integration

- **Upstream**: Called by `04-pipeline` after each step
- **Downstream**: Used by `16-governance` for standard conversion validation

## §5 — Current state

| Aspect | Status | Notes |
|--------|--------|-------|
| Step tracking | Complete | In-memory only |
| Compatibility check | Complete | v1.0 ↔ v2.0 |
| Persistence | Missing | Add SQLite/Redis storage |

---

*Subsystem doc: 10-lineage | Updated: 2026-06-19*
