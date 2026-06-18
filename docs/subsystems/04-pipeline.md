---
id: 04-pipeline
title: "04-Pipeline — End-to-End Annotation Pipeline"
status: active-dev
applies_to:
  - "src/dvas/pipeline/**"
code_anchors:
  - "src/dvas/pipeline/core.py:AnnotationPipeline"
  - "src/dvas/pipeline/core.py:EPICAnnotationPipeline"
  - "src/dvas/pipeline/core.py:create_training_data_from_gold"
agent_hints:
  - "WARNING: Pipeline uses scene detection to create segments—adjust threshold for granularity"
  - "WARNING: EPIC pipeline combines automatic scene detection with existing action labels"
  - "WARNING: Always use async process_batch()—don't loop annotate_video() synchronously"
  - "WARNING: Results auto-saved to AnnotationStore—check store for progress on failure"
  - "DEPRECATED: src/dvas/core/pipeline.py was renamed to _deprecated_pipeline.py (2026-06-18). Use dvas.pipeline.core:AnnotationPipeline for production."
---

# §04 Annotation Pipeline

Orchestrates video loading, teacher model inference, parsing, and storage. Main entry point for batch annotation workflows.

---

## §0 — One-liner

Async pipeline that loads videos, detects scenes, calls teacher models, parses structured output, and saves annotations.

## §1 — Core concepts

- **AnnotationPipeline**: Generic pipeline for any video source
- **EPICAnnotationPipeline**: Specialized for EPIC-KITCHENS with action label integration
- **Scene Detection**: Automatic temporal segmentation using histogram comparison
- **Response Parsing**: Extracts JSON/structured data from teacher model outputs
- **Training Data Export**: Converts gold annotations to SFT format for student training

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `core.py:AnnotationPipeline` | Generic pipeline | Custom video sources |
| `core.py:AnnotationPipeline.annotate_video()` | Single video | Testing, small batches |
| `core.py:AnnotationPipeline.process_batch()` | Multiple videos | Production batch jobs |
| `core.py:EPICAnnotationPipeline` | EPIC-specific | EPIC-KITCHENS annotation |
| `core.py:create_training_data_from_gold()` | Export for training | After gold data collection |

## §3 — Key behaviors & contracts

### Behavior 1: Pipeline Flow

```
Input: video_path, video_id
├── Load with VideoLoader
├── Detect scenes → segments [(start, end), ...]
├── For each segment:
│   ├── Extract frames (uniform sample)
│   ├── Call teacher.annotate()
│   ├── Parse response → Segment object
│   └── Collect segments
├── Build Annotation object
├── Save to AnnotationStore
└── Return Annotation
```

### Behavior 2: Scene Detection

- Uses histogram comparison with configurable threshold
- Minimum scene duration: 1 second (configurable)
- Finer granularity = more API calls = higher cost

### Behavior 3: Response Parsing

Current parsing is heuristic-based:
1. Look for JSON block in response with regex
2. Extract key fields: `scene_description`, `hand_actions`, `objects`, `steps`
3. Fallback to plain text if parsing fails

**Warning**: This is brittle. Future improvement: use structured output / JSON mode when APIs support it consistently.

### Behavior 4: Batch Processing

- Uses `asyncio.Semaphore` for concurrency control
- Failed items are caught as exceptions and recorded in failed list
- Check `isinstance(result, Exception)` to identify failures
- Successful results are `Annotation` objects
- **Note**: `process_batch()` now uses `asyncio.gather(return_exceptions=True)` for robust error handling
- **Performance**: Batch processing includes garbage collection between chunks to free frame memory

### Behavior 5: Performance Optimizations

Recent performance improvements:

| Optimization | Location | Impact |
|-------------|----------|--------|
| Frame seeking | `video_reader.py` | Skip frames via `CAP_PROP_POS_FRAMES` instead of reading each one |
| Key frame heap | `frame_sampler.py` | Min-heap for top-N selection instead of storing all frames |
| Concurrent encoding | `teacher/base.py` | ThreadPool for base64 encoding of large frame batches |
| Metadata caching | `video_loader.py` | Module-level cache avoids re-reading video headers |
| GC between chunks | `core.py` | Explicit garbage collection in batch processing |
| Async streaming | `video_loader.py` | Background thread + asyncio queue for frame streaming |

## §4 — Integration with other subsystems

- **Upstream**: Depends on `01-data` for video loading and storage
- **Upstream**: Depends on `02-teacher` for inference
- **Downstream**: Produces annotations consumed by `05-evaluation` and `06-export`

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Generic pipeline | Complete | Scene detect → annotate → store |
| EPIC pipeline | Complete | Integrates with EPIC action labels |
| Response parsing | Partial | Heuristic-based, needs JSON mode |
| Batch processing | Complete | With semaphore concurrency + GC between chunks |
| Checkpoint/resume | Missing | No recovery on failure |
| Progress tracking | Partial | Via AnnotationStore counts |
| Performance | Optimized | Frame seeking, heap sampling, concurrent encoding |

**Active known_gaps** (from `status.yaml`):
- No batch retry logic for API failures (severity: medium)

## §6 — Testing

```bash
# Test generic pipeline with sample video
pytest tests/test_pipeline.py::test_annotate_video -v

# Test EPIC pipeline (requires EPIC-KITCHENS data)
pytest tests/test_pipeline_epic.py -v --epic-root /path/to/epic

# Test batch processing
pytest tests/test_pipeline_batch.py -v

# Run performance benchmarks
python benchmarks/perf_benchmark.py
```

---

*Subsystem doc: 04-pipeline | Updated: 2024-06-18*
