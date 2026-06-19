---
id: 13-prompts
title: "13-Prompts — Adaptive Prompt Engineering"
status: stable
applies_to:
  - "src/dvas/prompts/**"
code_anchors:
  - "src/dvas/prompts/adaptive.py:AdaptivePromptEngine"
  - "src/dvas/prompts/adaptive.py:PromptLibrary"
  - "src/dvas/prompts/adaptive.py:VideoTypeClassifier"
  - "src/dvas/prompts/adaptive.py:DynamicPromptOptimizer"
agent_hints:
  - "WARNING: Video classification is heuristic-based - may misclassify"
  - "WARNING: Prompt performance tracking is in-memory - persist for production"
  - "WARNING: Complex videos may be misclassified as simple"
  - "WARNING: Few-shot examples are hardcoded - use dynamic retrieval in production"
---

# §13 Prompts

Adaptive prompt engineering based on video characteristics and task requirements.

---

## §0 — One-liner

Generate optimal prompts based on video type, complexity, and historical performance.

## §1 — Core concepts

- **AdaptivePromptEngine**: Main prompt generation orchestrator
- **PromptLibrary**: Specialized templates for different video types
- **VideoTypeClassifier**: Categorize videos (kitchen, robot, medical, etc.)
- **DynamicPromptOptimizer**: Improve prompts based on feedback
- **PromptTemplate**: Template with metadata and performance tracking

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `adaptive.py:AdaptivePromptEngine` | Generate prompts | Before annotation |
| `adaptive.py:PromptLibrary` | Access templates | Custom prompts |
| `adaptive.py:VideoTypeClassifier` | Classify video | Routing decisions |
| `adaptive.py:DynamicPromptOptimizer` | Optimize prompts | Continuous improvement |

## §3 — Key behaviors & contracts

### Behavior 1: Video Classification

```python
classifier = VideoTypeClassifier()
category = classifier.classify(video_path)

# Categories:
# - KITCHEN_COOKING
# - ROBOT_MANIPULATION
# - ASSEMBLY
# - MEDICAL
# - SPORTS
# - GENERAL
```

**Classification factors**:
- Filename keywords
- Color analysis over sampled `Frame.data` arrays from `VideoLoader.read_frames()`
- Motion patterns

### Behavior 2: Prompt Generation

```python
engine = AdaptivePromptEngine()
prompt = engine.generate_prompt(
    video_path=video_path,
    task_type="caption",
    preferred_complexity=ComplexityLevel.DETAILED
)
```

**Complexity levels**:
- SIMPLE: Quick description
- MODERATE: Standard detail
- COMPLEX: Fine-grained analysis

### Behavior 3: Template Selection

```python
# Templates available for each category
library = PromptLibrary()

templates = library.get_by_category(
    VideoCategory.KITCHEN_COOKING,
    complexity=ComplexityLevel.COMPLEX
)
```

**Built-in templates**:
- `kitchen_detailed`: Hand actions, ingredients, tools
- `kitchen_quick`: Main activity summary
- `robot_fine_grained`: Grasp types, trajectories
- `general_detailed`: Standard annotation
- `qa_generation`: Q&A pair generation

### Behavior 4: Feedback Loop

```python
# Record quality feedback
engine.feedback_loop(
    template_name="kitchen_detailed",
    video_path=video_path,
    quality_score=0.85,
)

# Automatic template selection uses historical performance
```

### Behavior 5: Prompt Optimization

```python
optimizer = DynamicPromptOptimizer()
optimizer.record_performance(prompt, category, metrics)

# Get improvement suggestions
suggestions = optimizer.suggest_improvements()
```

## §4 — Integration with other subsystems

- **Upstream**: Uses `01-data/video_loader` for classification
- **Downstream**: Prompts feed into `02-teacher` and `03-student`
- **Related**: Used by `08-routing` for complexity estimation

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Video classification | Complete | Heuristic-based; consumes Frame dataclass objects correctly |
| Prompt templates | Complete | 5 specialized templates |
| Complexity estimation | Complete | Duration + scene count |
| Feedback loop | Complete | Moving average scoring |
| Prompt optimization | Partial | Basic suggestions |
| Semantic retrieval | Missing | Use embeddings for examples |
| A/B prompt testing | Missing | Integrate with monitoring |

## §6 — Testing

```bash
# Test prompt generation
python -c "
from dvas.prompts.adaptive import AdaptivePromptEngine, VideoCategory

engine = AdaptivePromptEngine()

# List available templates
from dvas.prompts.adaptive import PromptLibrary
lib = PromptLibrary()
templates = lib.get_by_category(VideoCategory.KITCHEN_COOKING)
print(f'Found {len(templates)} kitchen templates')
"
```

---

*Subsystem doc: 13-prompts | Updated: 2026-06-19*
