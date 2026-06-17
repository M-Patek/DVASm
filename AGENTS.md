# AGENTS.md — DVAS Agent Boot Protocol

> For Claude Code, Cursor, Copilot, Gemini CLI, and compatible AI agents.
> This file is auto-loaded on session start. Do not rename.
>
> **DVAS** = Distilled Video Annotation Specialist

---

## PROTOCOL

### START — Session Initialization

```
READ docs/INDEX.md
├── Quick Map (§1) → route to subsystem
├── Reading Protocol (§4) → token budget if needed
└── Subsystem IDs (§3) → note your target ID

IF touching subsystem(s):
    READ docs/subsystems/NN-*.md FOR EACH ID:
    ├── frontmatter.code_anchors → entry points
    ├── frontmatter.agent_hints → known gotchas / overrides
    ├── §0 or §1 → infrastructure/concepts
    └── §2+ → ONLY IF task requires depth

    APPLY agent_hints protocol (see EXEC section)

IF task spans 3+ subsystems:
    SPAWN Explore sub-agent
```

### EXEC — During Work

**Hints Protocol**: After reading subsystem doc, check `frontmatter.agent_hints`:
- `WARNING: <message>` → Display the message, then continue with root AGENTS.md rules
- `OVERRIDE: <rule> → <description>` → Apply the override instead of root rule for this subsystem

| Trigger | Action |
|---------|--------|
| Doc vs code conflict | Trust code, update doc |
| Touch `code_anchors:` file | RUN `python scripts/check_doc_anchors.py --quiet` |
| Modify data schemas (Annotation/Segment) | Mark cross-cutting (ALL layers affected) |
| Add new teacher model | Update `llms.txt` subsystems |
| Fix a bug | Append to `docs/_machine/bugs.yaml` |
| Add tech debt | Append to `docs/_machine/tech-debt.yaml` |

### EXIT — Session Termination

**Step 1: Classify**

| Change Pattern | Type | Checklist | Validate |
|---------------|------|-----------|----------|
| `.md`, `.txt`, comments, typo, format | T1 | Nothing | V0 — skip |
| Rename var/func, extract, simplify | T2 | CHANGELOG + doc-anchors + tests | V1 — unit tests |
| Fix bug, edge case, wrong output | T3 | CHANGELOG + STATUS + doc-anchors + tests | V2 — integration tests |
| New func/class, API extend, config | T4 | CHANGELOG + STATUS + doc-anchors + tests | V2 — integration tests |
| Schema change (Annotation/Segment/Action) | T5 | FULL + tech-debt + subsystem docs | V3 — E2E pipeline test |

**Step 2: Execute Checklist**

| Checklist | Do |
|-----------|-----|
| LIGHT (T1) | Nothing |
| STANDARD (T2) | CHANGELOG + doc-anchors + tests |
| FULL (T3/T4) | CHANGELOG + STATUS + doc-anchors + tests |
| FULL+ (T5) | FULL + tech-debt review |

**Step 3: CHANGELOG Entry**

Template:
```markdown
### Session — <summary>

- **Type**: T<N>
- **Goal**: <why>
- **Done**:
  - <change 1>
  - <change 2>
- **Files**: <paths>
- **Validation**: V<N> — <evidence>
- **Left for next time**: <if any>
```

---

## REFERENCE

### Navigation

| Need | Location |
|------|----------|
| Subsystem routing | `docs/INDEX.md` |
| Architecture decisions | `docs/adr/NNNN-*.md` |
| Subsystem details | `docs/subsystems/NN-*.md` |
| Subsystem-specific rules | `docs/subsystems/NN-*.md` `agent_hints` frontmatter |
| Recent changes | `docs/changelog/CHANGELOG.md` |
| Tech debt status | `python scripts/generate_reports.py tech-debt` |
| Known gaps | `python scripts/check_known_gaps.py` |

### Facts

- **Language**: Python 3.10+
- **Architecture**: Layered (Data → Models → Pipeline → API)
- **Core Schema**: `Annotation`, `Segment`, `Action`, `Object`
- **Validation**: V0=none / V1=unit / V2=integration / V3=E2E

### Forbidden

| Action | Why |
|--------|-----|
| Edit accepted ADRs | Immutable; supersede with new ADR |
| Add ✅/🚧/❌ to prose docs | Use `status.yaml` |
| Restate code in prose | Doc rots |
| Claim T1 to skip validation | Misclassification |
| Skip changelog for T2-T5 | Audit gap |
| Hardcode API keys | Use `pydantic-settings` + env vars |
| Skip `_machine/*.yaml` updates | Machine state becomes stale |
