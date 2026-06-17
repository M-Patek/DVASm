# DVAS Architecture Constitution

> Design rules and invariants for the DVAS system.
> Read this when implementing new features or resolving design conflicts.

---

## Invariants (Never Violate)

### INV-1: Schema Immutability

Once an `Annotation`, `Segment`, or `Action` field is used in production, it cannot be removed or repurposed. Only additive changes allowed.

**Why**: Backward compatibility for stored annotations.

### INV-2: Async-First API

All I/O operations must be async. Blocking calls are forbidden in core modules.

**Why**: Concurrent processing of video batches requires async/await throughput.

### INV-3: No Hardcoded Secrets

API keys, passwords, credentials must come from environment via `pydantic-settings`.

**Why**: Security. Prevents credential leakage in version control.

### INV-4: Pydantic for All Schemas

All data structures use Pydantic v2 for validation and serialization.

**Why**: Type safety, automatic validation, clear error messages.

### INV-5: Machine-First Documentation

Status, bugs, and tech debt live in `docs/_machine/*.yaml`. Human prose is secondary.

**Why**: Agents can read and update structured YAML. Prose rots.

---

## Design Principles (Prefer These)

### PRP-1: Teacher-Student Decoupling

Teacher and student models share no code except the base interface. No implicit dependencies.

### PRP-2: Frame Budget Awareness

All video processing respects frame limits of target models. Auto-downsample, never fail.

### PRP-3: Storage Agnostic

Pipeline logic doesn't care if storage is local filesystem, S3, or database. Use `AnnotationStore` abstraction.

### PRP-4: Format Late Binding

Internal representation is canonical. Export to customer formats (LLaVA, OpenAI) at boundary.

### PRP-5: Fail Fast, Retry with Context

API failures raise immediately. Batch processor handles retry with checkpoint context.

---

## Patterns (Use These Solutions)

### PTT-1: Context Managers for Resources

```python
# Good
with VideoLoader(path) as loader:
    frames = list(loader.read_frames())

# Bad
loader = VideoLoader(path)  # Resource leak risk
```

### PTT-2: Semaphore Concurrency Control

```python
# Good
semaphore = asyncio.Semaphore(10)
async with semaphore:
    await api.call()

# Bad
await asyncio.gather(*[api.call() for _ in range(1000)])  # Thundering herd
```

### PTT-3: Result Union for Batch Operations

```python
# Good
results = await batch_process(items)
for item, result in zip(items, results):
    if isinstance(result, Exception):
        handle_error(item, result)
    else:
        handle_success(item, result)
```

### PTT-4: YAML Frontmatter for Subsystem Docs

```markdown
---
id: NN-name
code_anchors:
  - "path/to/file.py:ClassName"
agent_hints:
  - "Critical warning about this subsystem"
---
```

---

## Anti-Patterns (Never Do These)

### ANT-1: Sync in Async Context

```python
# Forbidden
def blocking_call(): ...
await asyncio.to_thread(blocking_call)  # Only if unavoidable
```

### ANT-2: String Concatenation for Paths

```python
# Forbidden
path = f"{root}/{subdir}/{file}"

# Required
path = Path(root) / subdir / file
```

### ANT-3: Bare Except

```python
# Forbidden
except:
    pass

# Required
except SpecificException as e:
    structured_log(e)
```

### ANT-4: Mutable Default Args

```python
# Forbidden
def func(items=[]): ...

# Required
def func(items=None):
    items = items or []
```

---

## Decision Records

For major architectural decisions, see `docs/adr/NNNN-*.md`:

| ADR | Topic | Status |
|-----|-------|--------|
| 0001 | Use Pydantic v2 for schemas | accepted |
| 0002 | Teacher-Student distillation approach | accepted |
| 0003 | Async-first architecture | accepted |

---

*Constitution version: 0.1.0 | Last updated: 2024-06-17*
