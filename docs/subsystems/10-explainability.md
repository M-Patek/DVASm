---
id: 10-explainability
title: "10-Explainability — Visualization & XAI"
status: stable
applies_to:
  - "src/dvas/explainability/**"
code_anchors:
  - "src/dvas/explainability/visualizer.py:KeyFrameExtractor"
  - "src/dvas/explainability/visualizer.py:AttentionVisualizer"
  - "src/dvas/explainability/visualizer.py:AnnotationVisualizer"
  - "src/dvas/explainability/visualizer.py:ExplainabilityReport"
agent_hints:
  - "WARNING: AttentionVisualizer heatmaps are synthetic - requires model attention weights"
  - "WARNING: KeyFrameExtractor importance is heuristic-based"
  - "WARNING: Attention animation requires large storage space"
  - "WARNING: Frame overlays may obscure video content"
---

# §10 Explainability

Visual explanations of annotations through attention maps, key frames, and reasoning traces.

---

## §0 — One-liner

Generate attention heatmaps, extract key frames, and create visual explanations for annotation decisions.

## §1 — Core concepts

- **KeyFrameExtractor**: Find most important frames per segment
- **AttentionVisualizer**: Generate attention heatmaps (placeholder for real attention)
- **AnnotationVisualizer**: Overlay annotations on frames
- **ExplainabilityReport**: Comprehensive XAI report
- **AttentionHeatmap**: 2D attention + max attention point

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `visualizer.py:KeyFrameExtractor` | Extract key frames | Dataset review |
| `visualizer.py:AttentionVisualizer` | Create heatmaps | Model debugging |
| `visualizer.py:AnnotationVisualizer` | Overlay annotations | Human review |
| `visualizer.py:ExplainabilityReport` | Generate XAI report | Documentation |

## §3 — Key behaviors & contracts

### Behavior 1: Key Frame Extraction

```python
extractor = KeyFrameExtractor(num_keyframes=5)
keyframes = extractor.extract(video_path, annotation)

# Each KeyFrameInfo contains:
# - timestamp: Video position
# - importance_score: 0-1 importance
# - description: Caption at that point
# - objects, actions: Present in frame
```

**Importance factors**:
- Segment duration
- Number of actions
- Number of objects
- Caption detail level

### Behavior 2: Attention Visualization

```python
viz = AttentionVisualizer()

# Static heatmap on frame
heatmap = viz.generate_heatmap(frame, attention_weights)

# Temporal attention timeline
timeline = viz.create_temporal_attention_map(annotation, video_path)
```

### Behavior 3: Frame Annotation Overlay

```python
viz = AnnotationVisualizer()
annotated = viz.visualize_segment(
    frame,
    segment,
    draw_objects=True,
    draw_actions=True,
    draw_caption=True
)
```

### Behavior 4: Explanation Generation

```python
report = ExplainabilityReport()
result = report.generate_report(annotation, video_path, output_dir)

# Includes:
# - Keyframes with importance
# - Segment explanations
# - Reasoning traces
# - Summary image
```

## §4 — Integration with other subsystems

- **Upstream**: Consumes `Annotation` from `01-data`
- **Upstream**: Uses `01-data/video_loader` for frames
- **Downstream**: Visualizations for human review
- **Related**: Can explain `02-teacher` and `03-student` outputs

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Key frame extraction | Complete | Importance scoring |
| Attention heatmaps | Partial | Placeholder (needs model) |
| Temporal maps | Complete | Timeline visualization |
| Frame overlays | Complete | Text rendering |
| Summary images | Complete | Grid layout |
| Real attention weights | Missing | Requires model integration |
| Interactive visualization | Missing | Future: web interface |

## §6 — Testing

```bash
# Create sample visualization
python -c "
from dvas.explainability.visualizer import KeyFrameExtractor
# Note: requires actual video and annotation
print('Visualizer imported successfully')
"
```

---

*Subsystem doc: 10-explainability | Updated: 2024-06-17*
