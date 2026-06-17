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

Uses GPT-4 to score 5 dimensions:
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
| LLM-as-Judge | Complete | 5-dimension evaluation |
| Consistency checks | Complete | Temporal + action |
| Batch evaluation | Complete | Async with concurrency |
| Human correlation study | Missing | Validate LLM scores vs human |

**Active known_gaps**: none

## §6 — Testing

```bash
# Run metrics tests
pytest tests/test_metrics.py -v

# Run LLM judge tests (requires API key)
pytest tests/test_llm_judge.py -v --openai-key $OPENAI_API_KEY
```

---

*Subsystem doc: 05-evaluation | Updated: 2024-06-17*
