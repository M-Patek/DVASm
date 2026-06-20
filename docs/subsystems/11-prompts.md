---
id: 11-prompts
title: "11-Prompts — Prompt System Upgrade (Phase 9)"
applies_to:
  - "src/dvas/prompts/**"
code_anchors:
  - "src/dvas/prompts/registry.py:PromptRegistry"
  - "src/dvas/prompts/versioning.py:PromptVersion"
  - "src/dvas/prompts/ab_testing.py:ABTestRunner"
  - "src/dvas/prompts/attribution.py:PromptAttributionTracker"
  - "src/dvas/prompts/auto_select.py:AutoSelector"
  - "src/dvas/prompts/few_shot.py:SemanticExampleIndex"
  - "src/dvas/prompts/regression.py:PromptRegressionTest"
agent_hints:
  - "Registry is in-memory only - persist for production"
  - "A/B test assignments are deterministic per entity_id"
  - "SemanticExampleIndex uses simple hash embeddings - replace with real embeddings"
  - "Regression tests use mock scoring - replace with LLM evaluation"
---

# §11 Prompts — Prompt System Upgrade

Comprehensive prompt management system with versioning, A/B testing,
auto-selection, few-shot retrieval, and regression testing.

---

## §0 — One-liner

Manage, version, test, and automatically select the best prompt templates
for video annotation across multiple domains.

## §1 — Core concepts

- **PromptRegistry**: CRUD operations for prompt templates with lineage tracking
- **PromptVersion**: Semantic versioning with compatibility and diff support
- **ABTestRunner**: Random assignment and statistical comparison of prompt variants
- **PromptAttributionTracker**: Track which prompts produced which annotations
- **AutoSelector**: Automatically select prompts based on video characteristics
- **SemanticExampleIndex**: Vector-based few-shot example retrieval
- **PromptRegressionTest**: Baseline comparison and golden set validation
- **Domain-specific Packs**: VLA, World Model, and Human Review prompt templates

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `registry.py:PromptRegistry` | Store and manage prompt templates | Creating, updating, deleting prompts |
| `versioning.py:PromptVersion` | Version management | Tracking prompt changes |
| `ab_testing.py:ABTestRunner` | Compare prompt variants | Testing new prompt versions |
| `attribution.py:PromptAttributionTracker` | Track prompt performance | Quality analysis |
| `auto_select.py:AutoSelector` | Auto-select prompts | Before annotation |
| `few_shot.py:SemanticExampleIndex` | Retrieve examples | Prompt augmentation |
| `regression.py:PromptRegressionTest` | Validate prompt changes | Before deploying |

## §3 — Key behaviors & contracts

### Behavior 1: Prompt Registry CRUD

```python
from dvas.prompts.registry import PromptRegistry, PromptDomain

registry = PromptRegistry()

# Create a prompt
prompt = registry.create(
    name="kitchen_detailed",
    template="Describe this kitchen video...",
    domain=PromptDomain.KITCHEN,
    version="1.0.0",
    description="Detailed kitchen annotation prompt",
    tags=["kitchen", "detailed"],
    variables=["video_path"],
)

# Retrieve
retrieved = registry.get(prompt.id)

# Update
registry.update(prompt.id, template="Updated template...")

# Fork (create new version)
child = registry.fork(prompt.id, "kitchen_v2", "2.0.0")
```

**Lineage tracking**: Each prompt stores its ancestor IDs. Use `get_lineage()` and `get_children()` to traverse the family tree.

### Behavior 2: Version Management

```python
from dvas.prompts.versioning import PromptVersion, compute_diff

# Parse and compare versions
v1 = PromptVersion.parse("1.2.3")
v2 = PromptVersion.parse("1.3.0")
assert v1 < v2

# Compute diff between templates versions
diff = compute_diff(old_template, new_template, "v1.0", "v2.0")
print(f"Similarity: {diff.similarity_ratio:.2%}")
print(f"Added: {len(diff.added_lines)} lines")
```

### Behavior 3: A/B Testing

```python
from dvas.prompts.ab_testing import ABTestConfig, ABTestRunner, AssignmentMethod

runner = ABTestRunner()
config = ABTestConfig(
    test_name="kitchen_prompt_test",
    variant_a_id="prompt_v1",
    variant_b_id="prompt_v2",
    traffic_split=0.5,
    assignment_method=AssignmentMethod.HASH,
)
runner.register_test(config)

# Assign variant
variant_id = runner.assign_variant("kitchen_prompt_test", "video_123")

# Record metrics
runner.record_metric("kitchen_prompt_test", variant_id, quality=0.85, latency_ms=120)

# Compare results
result = runner.compare("kitchen_prompt_test", metric="quality")
winner = runner.get_winner("kitchen_prompt_test")
```

### Behavior 4: Quality Attribution

```python
from dvas.prompts.attribution import PromptAttributionTracker

tracker = PromptAttributionTracker()

# Record which prompt produced an annotation
record = tracker.record_attribution(
    annotation=annotation,
    prompt_id="prompt_1",
    prompt_version="1.0.0",
    quality_scores=quality_scores,
    latency_ms=120.0,
    cost=0.01,
)

# Get performance summary
summary = tracker.compute_performance_summary("prompt_1", "1.0.0")
print(f"Avg quality: {summary.avg_quality_score:.3f}")
```

### Behavior 5: Auto-Selection

```python
from dvas.prompts.auto_select import AutoSelector, VideoCharacteristics
from dvas.prompts.registry import PromptRegistry

registry = PromptRegistry()
# ... register prompts ...

selector = AutoSelector(registry=registry)

# Select based on video path
prompt = selector.select(
    video_path=Path("cooking_video.mp4"),
    task_type="caption",
)

# Or based on characteristics
chars = VideoCharacteristics(
    duration_seconds=120,
    scene_count=8,
    motion_score=0.8,
)
prompt = selector.select_for_characteristics(chars)
```

### Behavior 6: Few-Shot Example Retrieval

```python
from dvas.prompts.few_shot import SemanticExampleIndex, Example, create_domain_example_packs

# Using the semantic index
index = SemanticExampleIndex()
index.add_example(Example(
    id="ex1",
    input_text="A person chopping vegetables",
    output_text="The person uses a knife to chop vegetables.",
    domain=PromptDomain.KITCHEN,
))

results = index.search("chopping", domain=PromptDomain.KITCHEN, top_k=3)

# Using domain packs
packs = create_domain_example_packs()
kitchen_pack = packs[PromptDomain.KITCHEN]
examples = kitchen_pack.get_examples(query="chopping", top_k=2)
```

### Behavior 7: Regression Testing

```python
from dvas.prompts.regression import PromptRegressionTest, GoldenAnnotation, RegressionStatus

test = PromptRegressionTest()

# Add golden annotations
test.add_golden_annotation(GoldenAnnotation(
    id="gold_1",
    video_id="vid_1",
    expected_output="The person picks up the cup.",
    expected_quality=0.9,
))

# Set baseline
test.set_baseline("prompt_1", 0.85)

# Run regression test
results = test.run_test(prompt, test_set="default")
for result in results:
    if result.status == RegressionStatus.FAIL:
        print(f"Regression detected: {result.details}")
```

## §4 — Domain-Specific Prompt Packs

### VLA Pack (`packs/vla_pack.py`)

Templates for Vision-Language-Action model training:
- `vla_grasp_analysis`: Grasp type and contact analysis
- `vla_trajectory`: End effector trajectory description
- `vla_action_sequence`: Step-by-step action sequences
- `vla_affordance`: Object affordance identification
- `vla_fine_motor`: Fine-grained motor control description

### World Model Pack (`packs/world_model_pack.py`)

Templates for world model training:
- `wm_state_prediction`: Predict next scene state
- `wm_dynamics`: Physical dynamics description
- `wm_counterfactual`: Generate counterfactual scenarios
- `wm_causal_chain`: Identify causal event chains
- `wm_scene_graph`: Generate structured scene graphs
- `wm_temporal_reasoning`: Temporal relationship analysis

### Human Review Pack (`packs/human_review_pack.py`)

Templates for human review and quality assessment:
- `review_overall_quality`: Comprehensive quality review
- `review_factuality`: Fact-check annotations
- `review_vla_specific`: VLA training suitability review
- `review_comparison`: Compare two annotations
- `review_correction`: Provide corrected annotations
- `review_consensus`: Evaluate reviewer consensus

## §5 — Integration with other subsystems

- **Upstream**: Uses `01-data/video_loader` for video characteristics
- **Downstream**: Prompts feed into `02-teacher` and `03-student`
- **Quality**: Integrates with `09-quality` for score attribution
- **Existing**: Extends `adaptive.py` with registry, versioning, and testing

## §6 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Prompt Registry | Complete | In-memory CRUD with lineage |
| Version Management | Complete | Semantic versioning, diffs |
| A/B Testing | Complete | Random/hash/round-robin assignment |
| Quality Attribution | Complete | Per-prompt performance tracking |
| Auto-Selection | Complete | Domain detection + exploration |
| Few-Shot Retrieval | Complete | Hash-based embeddings (replace with real) |
| Regression Testing | Complete | Golden set validation |
| Domain Packs | Complete | VLA, World Model, Human Review |
| Persistence | Gap | Registry is in-memory only |
| Real Embeddings | Gap | SemanticExampleIndex uses hash embeddings |
| LLM Scoring | Gap | Regression uses mock scoring |

---

*Subsystem doc: 11-prompts | Updated: 2026-06-20*
