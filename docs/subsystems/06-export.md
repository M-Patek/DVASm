---
id: 06-export
title: "06-Export — Format Conversion & CLI"
status: stable
applies_to:
  - "src/dvas/export/**"
code_anchors:
  - "src/dvas/export/adapters.py:export_annotations"
  - "src/dvas/export/adapters.py:LLaVAAdapter"
  - "src/dvas/export/adapters.py:OpenAIAdapter"
  - "src/dvas/export/cli.py:export"
agent_hints:
  - "WARNING: Use export_annotations() for programmatic export"
  - "WARNING: Use CLI for interactive export and inspection"
  - "WARNING: Add new formats by implementing ExportAdapter base class"
  - "WARNING: JSONL format is standard for training data"
---

# §06 Export & Formatting

Convert internal annotations to various training formats (LLaVA, OpenAI, ShareGPT).

---

## §0 — One-liner

Adapter pattern for converting annotations to LLaVA, OpenAI, and ShareGPT training formats, with CLI for interactive use.

## §1 — Core concepts

- **ExportAdapter**: Abstract base class for format converters
- **LLaVAAdapter**: Conversations format (human/gpt turns)
- **OpenAIAdapter**: Messages format for fine-tuning API
- **ShareGPTAdapter**: Vicuna/ShareGPT compatible format
- **export_annotations()**: Programmatic export function
- **CLI**: Interactive export with statistics and inspection

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `adapters.py:export_annotations` | Programmatic export | Pipeline scripts |
| `adapters.py:LLaVAAdapter` | LLaVA format | LLaVA training |
| `adapters.py:OpenAIAdapter` | OpenAI format | GPT fine-tuning |
| `cli.py:export` | CLI export | Interactive use |
| `cli.py:inspect` | Inspect annotation | Debugging |

## §3 — Key behaviors & contracts

### Behavior 1: Adapter Pattern

Implement `ExportAdapter` to add new formats:
```python
class MyAdapter(ExportAdapter):
    def export(self, annotations: List[Annotation]) -> List[Dict]:
        # Convert to your format
        return data
```

Register in `ADAPTERS` dict.

### Behavior 2: CLI Commands

- `dvas-export list-formats` - Show available formats
- `dvas-export export` - Export annotations
- `dvas-export stats` - Show storage statistics
- `dvas-export inspect` - View single annotation

### Behavior 3: Format Details

Programmatic exports should use `export_annotations()`. CLI/API callers materialize `AnnotationStore.load_all()` generators before emptiness checks or statistics; targeted exports resolve direct annotation IDs, `{video_id}_annotated`, and `video_id` filters.

**LLaVA format**:
```json
{
  "id": "...",
  "video": "...",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

**OpenAI format**:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**World Model format** (v2.0):
```json
{
  "id": "...",
  "video_id": "...",
  "episodes": [
    {
      "observation": {...},
      "actions": [...],
      "next_observation": {...}
    }
  ],
  "dynamics": {
    "physical_constraints": [...],
    "causal_links": [...]
  }
}
```

## §4 — Integration with other subsystems

- **Upstream**: Consumes `Annotation` objects from `01-data`
- **Downstream**: Training frameworks (LLaVA, OpenAI, World Model)
- **Cross-cutting**: Uses `16-governance` adapters for standard-specific export

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Adapter pattern | Complete | Extensible base class |
| LLaVA format | Complete | Conversations schema |
| OpenAI format | Complete | Messages schema |
| ShareGPT format | Complete | Vicuna compatible |
| World Model format | Complete | Observation-action-next_observation tuples |
| CLI tool | Complete | Full-featured; handles generator-backed stores and targeted video IDs |
| Custom formats | Easy to add | Just implement adapter |

**Active known_gaps**: none

## §6 — Testing

```bash
# List formats
python -m dvas.export.cli list-formats

# Export all gold annotations
python -m dvas.export.cli export -o train.jsonl -f llava -s gold

# Inspect specific annotation
python -m dvas.export.cli inspect vid_xxx

# Show stats
python -m dvas.export.cli stats
```

---

*Subsystem doc: 06-export | Updated: 2026-06-19*
