---
id: 05-evaluation
title: "05-Evaluation — Quality Assessment"
status: stable
applies_to:
  - "src/dvas/models/evaluator/**"
code_anchors:
  - "src/dvas/models/evaluator/metrics.py:MetricsCalculator"
  - "src/dvas/models/evaluator/metrics.py:compare_annotations"
  - "src/dvas/models/evaluator/llm_judge.py:LLMJudge"
  - "src/dvas/models/evaluator/llm_judge.py:ConsistencyChecker"
agent_hints:
  - "WARNING: Use MetricsCalculator for fast automatic metrics (BLEU, ROUGE, CIDEr, METEOR)"
  - "WARNING: Use LLMJudge for semantic quality evaluation (slower but more accurate)"
  - "WARNING: Use ConsistencyChecker for temporal coherence across segments"
  - "WARNING: LLM-as-Judge costs API tokens—use sparingly on sample batches"
---

# §05 Quality Evaluation

Multi-level quality assessment: automatic n-gram metrics, LLM semantic evaluation, and temporal consistency checks.

---

## §0 — One-liner

Compute BLEU/ROUGE/CIDEr/METEOR scores, use GPT-4 as semantic judge, and verify temporal consistency across segments.

## §1 — Core concepts

- **MetricsCalculator**: Fast n-gram metrics (BLEU-1/2/3/4, ROUGE-L, CIDEr, METEOR)
- **LLMJudge**: Use GPT-4 to evaluate semantic quality across 5 dimensions
- **ConsistencyChecker**: Verify temporal coherence and action sequence consistency
- **compare_annotations()**: Utility to compare prediction vs reference

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `metrics.py:MetricsCalculator` | Automatic metrics | Batch evaluation, regression tests |
| `metrics.py:compare_annotations` | Quick comparison | Unit tests, validation scripts |
| `llm_judge.py:LLMJudge` | Semantic evaluation | Quality assurance, human-level judgment |
| `llm_judge.py:ConsistencyChecker` | Temporal checks | Post-processing validation |

## §3 — Key behaviors & contracts

### Behavior 1: Automatic Metrics

Fast metrics based on n-gram overlap:
- BLEU: Precision-focused, good for grammatical correctness
- ROUGE: Recall-focused, good for coverage
- CIDEr: Consensus-based, weights by TF-IDF
- METEOR: Accounts for synonyms and word order

### Behavior 2: LLM-as-Judge

Uses a `TeacherModel` judge and consumes the standardized `GenerationResult.text` response to score 5 dimensions:
- Accuracy: Factual correctness
- Completeness: Coverage of important aspects
- Clarity: Understandability
- Relevance: Alignment with video
- Structure: Logical organization

### Behavior 3: Consistency Checks

- Temporal: Detects overlapping segments, checks narrative flow
- Action: Identifies redundant actions, validates action sequences

## §4 — Integration with other subsystems

- **Upstream**: Consumes annotations from `01-data`
- **Upstream**: Can use `02-teacher` models as judge
- **Downstream**: Results feed into quality filtering pipeline

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| BLEU/ROUGE/CIDEr/METEOR | Complete | Full implementation |
| LLM-as-Judge | Complete | 5-dimension evaluation; honors GenerationResult contract |
| Consistency checks | Complete | Temporal + action |
| Batch evaluation | Complete | Async with concurrency |
| Human correlation study | Missing | Validate LLM scores vs human |
| Review Workbench | Complete | Phase 11: dataset browser, annotation editor, diff viewer, reviewer assignment, approval workflow |

**Active known_gaps**: none

## §6 — Review Workbench (Phase 11)

The review workbench (`src/dvas/review/`) provides comprehensive tools for human review of annotations:

- **DatasetBrowser**: Browse, filter, sort, and paginate annotation datasets
- **SegmentViewer**: View video segments with frame-level annotation overlays
- **FrameStripViewer**: Display keyframe strips with selection and comparison
- **AnnotationEditor**: Edit annotations with change tracking and undo/redo
- **AnnotationDiff**: Compare two annotations with visual diff highlighting
- **TeacherOutputViewer**: View raw teacher model outputs and fallback chains
- **QualityScoreViewer**: Display per-dimension quality score breakdowns
- **ReviewerAssignment**: Assign reviews with workload balancing and skill matching
- **ReviewQueue**: Priority-based queue with batch assignment
- **DisagreementReview**: Handle teacher/student disagreements with resolution workflow
- **ApprovalWorkflow**: Multi-stage approval with rejection tracking and export gate
- **ReviewerMetrics**: Track reviewer performance with leaderboard

### Usage

```python
from dvas.review import DatasetBrowser, ApprovalWorkflow, ReviewerAssignment

# Browse annotations with filtering
browser = DatasetBrowser(annotations)
results = browser.browse(
    dataset_filter=DatasetFilter(min_quality_score=0.7),
    sort_field=SortField.QUALITY_SCORE,
    page=1,
    page_size=20,
)

# Manage approval workflow
workflow = ApprovalWorkflow()
workflow.register_annotation("ann1")
workflow.transition("ann1", WorkflowStage.HUMAN_REVIEW, "system")
workflow.approve("ann1", "reviewer1")
```

## §6 — Testing

```bash
# Run metrics tests
pytest tests/test_metrics.py -v

# Run LLM judge tests (requires API key)
pytest tests/test_llm_judge.py -v --openai-key $OPENAI_API_KEY
```

---

*Subsystem doc: 05-evaluation | Updated: 2026-06-20*
