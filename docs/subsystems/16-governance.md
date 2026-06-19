---
id: 16-governance
title: "16-Governance — Annotation Standard Management"
status: stable
applies_to:
  - "src/dvas/governance/**"
code_anchors:
  - "src/dvas/governance/adapters.py:StandardAdapter"
  - "src/dvas/governance/adapters.py:EPICAdapter"
  - "src/dvas/governance/adapters.py:Ego4DAdapter"
  - "src/dvas/governance/adapters.py:OpenXAdapter"
agent_hints:
  - "WARNING: Custom standard data may lose fields when converting to EPIC"
  - "WARNING: Open X adapter requires embodiment fields to be populated"
---

# §16 Governance

Annotation standard management and multi-standard adapters.

---

## §0 — One-liner

Convert annotations between standards (EPIC-KITCHENS, Ego4D, Open X-Embodiment) with field-aware adapters.

## §1 — Core concepts

- **StandardAdapter**: Base class for all standard adapters
- **EPICAdapter**: Minimal verb+noun format (v1.0 compatible)
- **Ego4DAdapter**: Rich narration with instrument and state changes
- **OpenXAdapter**: Robot embodiment with gripper poses and joint targets

## §2 — Entry points

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `adapters.py:get_adapter` | Get adapter by standard | Converting annotations |
| `adapters.py:list_standards` | List supported standards | UI/discovery |

## §3 — Key behaviors

### Behavior 1: Convert to Standard

```python
from dvas.governance import get_adapter
from dvas.data.schemas import AnnotationStandard

adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
epic_data = adapter.to_standard(annotation)
```

### Behavior 2: Convert from Standard

```python
adapter = get_adapter(AnnotationStandard.EGO4D)
annotation = adapter.from_standard(ego4d_data)
```

### Behavior 3: Field Loss Awareness

| Standard | Preserves | Loses |
|----------|----------|-------|
| EPIC | verb, noun, hand | instrument, physical, embodiment, temporal |
| Ego4D | verb, noun, hand, instrument, source_state, target_state | physical, embodiment |
| Open X | verb, noun, embodiment | caption, qa_pairs, temporal |

## §4 — Integration

- **Upstream**: Consumes `Annotation` from `01-data`
- **Downstream**: Used by `06-export` for standard-specific export
- **Cross-cutting**: Used by `10-lineage` for compatibility validation

## §5 — Current state

| Aspect | Status | Notes |
|--------|--------|-------|
| EPIC adapter | Complete | Full round-trip support |
| Ego4D adapter | Complete | Narration + state changes |
| Open X adapter | Complete | Embodiment action space |
| Custom standard | Complete | Passthrough (no conversion) |

---

*Subsystem doc: 16-governance | Updated: 2026-06-19*
