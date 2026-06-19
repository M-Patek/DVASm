---
id: 01-data
title: "01-Data — Video Loading, Preprocessing, Storage"
status: active-dev
applies_to:
  - "src/dvas/data/**"
code_anchors:
  - "src/dvas/data/schemas.py:Annotation"
  - "src/dvas/data/schemas.py:Segment"
  - "src/dvas/data/schemas.py:Action"
  - "src/dvas/data/video_loader.py:VideoLoader"
  - "src/dvas/data/video_loader.py:EPICKitchensLoader"
  - "src/dvas/data/video_reader.py:VideoReader"
  - "src/dvas/data/storage.py:AnnotationStore"
agent_hints:
  - "WARNING: Always use VideoLoader as context manager: `with VideoLoader(path) as loader:`"
  - "WARNING: EPICKitchensLoader expects specific directory structure—check path exists"
  - "WARNING: Annotation model uses Pydantic v2—use model_dump() not dict()"
  - "WARNING: Storage uses orjson for speed—use .model_dump() then orjson.dumps()"
  - "WARNING: Large videos may cause memory issues—use read_frames() generator, don't convert to list"
  - "WARNING: Supported formats are listed in dvas.data.SUPPORTED_VIDEO_FORMATS (mp4/mov/avi/mkv/webm/m4v/flv/3gp via OpenCV+FFmpeg); VideoReader raises ValueError on other extensions"
---

# §01 Data Layer

Core data models, video loading, and annotation storage. Defines the central `Annotation` schema that all other subsystems depend on.

---

## §0 — One-liner

Load videos, extract frames, store annotations in structured format with multi-format export.

## §1 — Core concepts

- **Annotation**: Root document for a video. Contains metadata, segments, quality scores.
- **Segment**: Temporal slice of video (start_time, end_time) with caption, QA pairs, objects, actions.
- **Action**: Verb-noun pair with hand (left/right) for robotic manipulation tracking. **v2.0 adds**: instrument, source/target state, physical properties, embodiment action space.
- **VideoLoader**: Context-managed video reader with frame sampling and scene detection.
- **EPICKitchensLoader**: Specialized loader for EPIC-KITCHENS dataset with action label mapping.
- **AnnotationStore**: File-based storage with version control and format export.
- **Schema v2.0**: Backward compatible with v1.0 (EPIC). All new fields are Optional.

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `schemas.py:Annotation` | Root data model | Creating/modifying annotation structures |
| `schemas.py:Segment` | Temporal segment model | Adding/updating video segments |
| `video_loader.py:VideoLoader` | Generic video loading | Loading any video file |
| `video_loader.py:EPICKitchensLoader` | EPIC dataset loader | Working with EPIC-KITCHENS |
| `storage.py:AnnotationStore` | Persistence layer | Saving/loading annotations |

## §3 — Key behaviors & contracts

### Behavior 1: Video Loading

- VideoLoader is **context manager only**—always use `with` statement
- Frame sampling: uniform sampling via `num_frames` parameter
- Scene detection: uses histogram comparison, configurable threshold
- Optical flow: available for motion intensity estimation

### Behavior 2: EPIC-KITCHENS Loading

- Expects structure: `{root}/{participant}/videos/{video_id}.MP4`
- Loads verb/noun class mappings from CSV files
- `get_actions_for_video()` returns action labels with timestamps

### Behavior 3: Storage Schema

```
data/annotations/
├── gold/        # Teacher model outputs (ground truth)
├── model/       # Student model outputs
├── reviewed/    # Human-reviewed final data
└── versions/    # Snapshots
```

- IDs are sharded: first 2 chars become subdirectory (e.g., `ab123` → `ab/ab123.json`)
- Uses orjson for fast serialization
- Supports LLaVA, OpenAI, and **World Model** format export

- `AnnotationStore.load_all()` returns a generator; convert to `list(...)` before using `len()`, sampling, duplicate detection, or repeated iteration.
- Storage source aliases: `gold` and `teacher` both map to the gold annotation directory; `reviewed` and `human` both map to reviewed annotations.

### Behavior 4: Schema v2.0 Compatibility

All v2.0 fields are **Optional** — existing v1.0 data loads without migration:

| Layer | Fields | Status |
|-------|--------|--------|
| EPIC v1.0 | verb, noun, hand | Stable |
| VLA v2.0 | instrument, physical, source_state, target_state, embodiment | Optional |
| WM v2.0 | state_predictions, dynamics | Optional, placeholder |

Use `annotation.is_v2_enhanced()` to check if v2.0 fields are populated.

### Behavior 4: Cross-Cutting Schema Changes

If modifying `Annotation`, `Segment`, `Action`, or `Object`:
- This is T5 change (schema change)
- Must update all subsystems that consume these
- Check `storage.py`, `export/` adapters, teacher prompts

## §4 — Integration with other subsystems

- **Upstream**: Raw video files (EPIC-KITCHENS or customer uploads)
- **Downstream**: Used by `04-pipeline` for annotation generation
- **Cross-cutting**: `06-export` depends on schema for format conversion

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Core schemas | Complete | Annotation, Segment, Action, Object defined |
| VideoLoader | Complete | Scene detection, frame sampling, optical flow |
| EPIC loader | Complete | Action label parsing, metadata loading |
| Storage | Complete | JSON with versioning |
| Format support | Complete | MP4/MOV/AVI/MKV/WebM/M4V/FLV/3GP via OpenCV+FFmpeg |
| Distributed processing | Missing | Single-process only |

**Active known_gaps** (from `status.yaml`):
- No distributed video processing (severity: low)

## §6 — Testing

```bash
# Unit tests for data layer
pytest tests/test_schemas.py -v
pytest tests/test_video_loader.py -v
pytest tests/test_storage.py -v

# Multi-format coverage (TestVideoFormatSupport)
pytest tests/test_video_loader.py::TestVideoFormatSupport -v

# Integration test with sample video
pytest tests/test_data_integration.py -v
```

---

*Subsystem doc: 01-data | Updated: 2026-06-19*
