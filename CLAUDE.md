# DVAS — Claude Context

Thin import of `@AGENTS.md` protocol.

## Quick Start

1. **Read**: `llms.txt` → `docs/INDEX.md` → target subsystem doc
2. **Code**: `src/dvas/<subsystem>/`
3. **Status**: `docs/_machine/status.yaml`

## Current Priorities (from status.yaml)

| ID | Subsystem | Status | Next Action |
|--|--|--|--|
| 01-data | Data Layer | active-dev | Migrate video_loader.py, schemas.py |
| 02-teacher | Teacher Models | active-dev | Migrate GPT-4V, Claude, Together clients |
| 04-pipeline | Pipeline | active-dev | Migrate annotation pipeline |
| 03-student | Student Models | draft | Not started |
| 05-evaluation | Evaluation | draft | Not started |
| 06-export | Export | draft | Not started |
| 07-api | API | draft | Not started |

## Active Gaps

- 01-data: No distributed processing, limited video formats
- 04-pipeline: No batch retry logic

See `docs/_machine/status.yaml` for full details.

---

*This file exists for compatibility. Main protocol: `AGENTS.md`*
