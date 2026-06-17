---
id: 02-teacher
title: "02-Teacher — GPT-4V, Claude, Together API Wrappers"
status: active-dev
applies_to:
  - "src/dvas/models/teacher/**"
code_anchors:
  - "src/dvas/models/teacher/base.py:TeacherModel"
  - "src/dvas/models/teacher/gpt4v.py:GPT4VTeacher"
  - "src/dvas/models/teacher/claude.py:ClaudeTeacher"
  - "src/dvas/models/teacher/together.py:TogetherTeacher"
agent_hints:
  - "WARNING: Always use async annotate()—batch processing uses asyncio.gather"
  - "WARNING: GPT-4V supports 32 frames max—sampler will downsample if more provided"
  - "WARNING: Claude supports 20 frames max—most restrictive"
  - "WARNING: Together/Qwen supports 8 frames max—cheapest option"
  - "WARNING: Base class prompt templates are hardcoded—update base.py for now"
---

# §02 Teacher Models

API wrappers for commercial and hosted VLM models. Generate gold-standard annotations for distillation.

---

## §0 — One-liner

Unified async interface to GPT-4V, Claude 3, and Together AI (Qwen2-VL) for batch video annotation.

## §1 — Core concepts

- **TeacherModel**: Abstract base class defining the interface
- **GPT4VTeacher**: OpenAI GPT-4V/GPT-4o with 32-frame support
- **ClaudeTeacher**: Anthropic Claude 3 Sonnet/Opus with 20-frame support
- **TogetherTeacher**: Together.ai API for open-source models (Qwen2-VL)
- **Fine-grained Prompt**: Specialized prompt for robotic manipulation scenes
- **Frame Encoding**: BGR→RGB conversion, JPEG compression, base64 encoding

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `base.py:TeacherModel` | Abstract interface | Implementing new teacher |
| `gpt4v.py:GPT4VTeacher` | OpenAI models | Best quality, higher cost |
| `claude.py:ClaudeTeacher` | Anthropic models | Alternative quality option |
| `together.py:TogetherTeacher` | OSS via API | Cheapest, good for validation |

## §3 — Key behaviors & contracts

### Behavior 1: Async Pattern

All teacher models are **async**:
```python
teacher = GPT4VTeacher()
result = await teacher.annotate(frames=frames, task="fine_grained")
```

Use `annotate_batch()` for concurrent processing with semaphore-based rate limiting.

### Behavior 2: Frame Limits & Sampling

| Model | Max Frames | Detail Level |
|-------|-----------|--------------|
| GPT-4V/4o | 32 | high (if ≤8) / low (if >8) |
| Claude 3 | 20 | medium |
| Qwen2-VL (Together) | 8 | medium |

If more frames provided, uniform sampling reduces to max.

### Behavior 3: Prompt Templates

Base class provides default prompts:
- `caption`: Simple description
- `dense_caption`: Detailed with temporal info
- `qa`: Question-answer generation
- `temporal`: Time segment localization
- `fine_grained`: **Main use case—robotic manipulation focus**

### Behavior 4: Error Handling

- Uses `tenacity` for retry with exponential backoff
- 3 retries max for API failures
- Batch processing returns exceptions in result list—check with `isinstance(result, Exception)`

## §4 — Integration with other subsystems

- **Upstream**: Instantiates in `04-pipeline` with API keys from environment
- **Downstream**: Output feeds into `Annotation` schema creation
- **Cross-cutting**: Import in `05-evaluation` for LLM-as-Judge

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Base class | Complete | Abstract interface with utilities |
| GPT-4V | Complete | Tested with OpenAI SDK |
| Claude 3 | Complete | Tested with Anthropic SDK |
| Together | Complete | Together.ai SDK with OpenAI compat |
| Prompt templates | Hardcoded | TD-002 to externalize |
| Batch retry logic | Missing | Must add checkpoint/resume |

**Active known_gaps**: none (all low priority)

## §6 — Testing

```bash
# Test individual teachers (requires API keys)
pytest tests/test_teacher_gpt4v.py -v --openai-key $OPENAI_API_KEY
pytest tests/test_teacher_claude.py -v --anthropic-key $ANTHROPIC_API_KEY

# Test base class and utilities
pytest tests/test_teacher_base.py -v
```

**Note**: Use Together API for cheaper testing of prompt variations.

---

*Subsystem doc: 02-teacher | Updated: 2024-06-17*
