---
id: 08-routing
title: "08-Routing — Smart Model Selection"
status: stable
applies_to:
  - "src/dvas/routing/**"
code_anchors:
  - "src/dvas/routing/smart_router.py:SmartRouter"
  - "src/dvas/routing/smart_router.py:VideoComplexityAnalyzer"
  - "src/dvas/routing/ensemble.py:MultiTeacherEnsemble"
  - "src/dvas/routing/ensemble.py:TeacherVote"
agent_hints:
  - "WARNING: SmartRouter estimates costs - actual API costs may vary"
  - "WARNING: VideoComplexityAnalyzer samples frames for speed, not full scan"
  - "WARNING: MultiTeacherEnsemble multiplies API costs by number of teachers"
  - "WARNING: Use IncrementalConsensus to reduce cost when early consensus reached"
---

# §08 Smart Routing

Adaptive model selection and ensemble voting for cost-quality optimization.

---

## §0 — One-liner

Route videos to optimal model (Teacher/Student) based on complexity analysis, with multi-teacher ensemble voting for high-stakes annotations.

## §1 — Core concepts

- **VideoComplexityAnalyzer**: Motion, scene, object density analysis
- **SmartRouter**: 4 strategies (cost/quality/balanced/adaptive)
- **RoutingStrategy**: ENUM defining selection approach
- **MultiTeacherEnsemble**: Parallel teacher queries with consensus
- **TeacherVote**: Individual teacher response with confidence
- **ConsensusMethods**: Different voting algorithms

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `smart_router.py:SmartRouter` | Route videos to models | Cost optimization |
| `smart_router.py:VideoComplexityAnalyzer` | Analyze complexity | Before routing |
| `ensemble.py:MultiTeacherEnsemble` | Multi-teacher voting | High-stakes data |
| `ensemble.py:TeacherVote` | Single vote structure | Vote aggregation |

## §3 — Key behaviors & contracts

### Behavior 1: Adaptive Routing

```python
router = SmartRouter(strategy=RoutingStrategy.ADAPTIVE)
decision = await router.route(video_path)
# decision.model_type → Teacher/Student selection
# decision.estimated_cost → cost in USD
# decision.confidence_threshold → quality threshold
```

**Complexity factors**:
- Motion score (0-1)
- Scene changes (0-20)
- Object density (0-1)
- Hand interaction density (0-1)

### Behavior 2: Ensemble Voting

```python
ensemble = MultiTeacherEnsemble(consensus_method="confidence_aware")
result = await ensemble.annotate_with_ensemble(frames)  # teachers return GenerationResult
# result.agreement_score → inter-teacher agreement
# result.dissenting_opinions → where teachers disagree
```

Teacher responses are read from `GenerationResult.text` before confidence scoring or consensus.

**Consensus methods**:
- `majority`: Simple vote
- `weighted`: Weight by confidence
- `confidence_aware`: Merge responses by confidence
- `best_of_n`: Return best if above threshold

### Behavior 3: Cost Estimation

| Model | Estimated Cost |
|-------|---------------|
| GPT-4V Teacher | $0.05/video |
| Claude Teacher | $0.04/video |
| Together | $0.02/video |
| Student Local | $0.001/video |
| Student Edge | $0.0005/video |

## §4 — Integration with other subsystems

- **Upstream**: Uses `01-data/video_loader` for complexity analysis
- **Upstream**: Uses `02-teacher` models for ensemble voting
- **Downstream**: Replaces single teacher in `04-pipeline`
- **Related**: Works with `03-student` for fallback decisions

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Complexity analysis | Complete | 5-factor scoring |
| Adaptive routing | Complete | 4 strategies |
| Ensemble voting | Complete | 4 consensus methods; uses GenerationResult.text |
| Cost estimation | Complete | Based on frame count |
| Incremental consensus | Complete | Early stopping |
| Disagreement resolution | Partial | 3 strategies implemented |

## §6 — Testing

```bash
# Run routing tests
pytest tests/test_routing.py -v

# Test with actual video
python -c "
from dvas.routing.smart_router import SmartRouter
import asyncio

async def test():
    router = SmartRouter()
    # Note: requires actual video file
    # decision = await router.route('video.mp4')
    print('Router initialized successfully')

asyncio.run(test())
"
```

---

*Subsystem doc: 08-routing | Updated: 2026-06-19*
