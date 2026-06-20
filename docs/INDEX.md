# docs/INDEX.md — DVAS Navigation Hub

> **Token budget**: ~500 tokens to read this file.
> Purpose: Route to the right document. Do not store domain knowledge here.

---

## NAVIGATE — Where To Go

```
NEED project scope / non-goals?
└── READ: docs/architecture/positioning.md

NEED architecture overview?
└── READ: docs/architecture/top-level-design.md

NEED design rules / invariants?
└── READ: docs/architecture/constitution.md

NEED why a decision was made?
└── READ: docs/adr/NNNN-*.md

NEED subsystem details?
└── READ: docs/subsystems/NN-*.md

NEED current health/gaps?
└── READ: docs/_machine/status.yaml

NEED recent changes?
└── READ: docs/changelog/CHANGELOG.md

NEED deprecated paths?
└── READ: docs/deprecated.md

NEED ops commands?
└── READ: docs/operations/cheatsheet.md
```

---

## SUBSYSTEMS — Quick Index

| ID | Name | Status | Health | Doc | Code |
|--|--|--|--|--|--|
| 01-data | Data Layer | stable | green | [01-data.md](subsystems/01-data.md) | `src/dvas/data/` |
| 02-teacher | Teacher Models | stable | green | [02-teacher.md](subsystems/02-teacher.md) | `src/dvas/models/teacher/` |
| 03-student | Student Models | stable | green | [03-student.md](subsystems/03-student.md) | `src/dvas/models/student/` |
| 04-pipeline | Annotation Pipeline | stable | green | [04-pipeline.md](subsystems/04-pipeline.md) | `src/dvas/pipeline/` |
| 05-evaluation | Quality Evaluation | stable | green | [05-evaluation.md](subsystems/05-evaluation.md) | `src/dvas/models/evaluator/` |
| 06-export | Export & Formatting | stable | green | [06-export.md](subsystems/06-export.md) | `src/dvas/export/` |
| 07-api | API Service | stable | green | [07-api.md](subsystems/07-api.md) | `src/dvas/api/` |
| 08-routing | Smart Routing | stable | green | [08-routing.md](subsystems/08-routing.md) | `src/dvas/routing/` |
| 09-quality | Data Quality | stable | green | [09-quality.md](subsystems/09-quality.md) | `src/dvas/quality/` |
| 10-explainability | Explainability | stable | green | [10-explainability.md](subsystems/10-explainability.md) | `src/dvas/explainability/` |
| 11-monitoring | Monitoring & A/B Testing | stable | green | [11-monitoring.md](subsystems/11-monitoring.md) | `src/dvas/monitoring/` |
| 12-security | Security & Privacy | stable | green | [12-security.md](subsystems/12-security.md) | `src/dvas/security/` |
| 13-prompts | Adaptive Prompts | stable | green | [13-prompts.md](subsystems/13-prompts.md) | `src/dvas/prompts/` |
| 14-deployment | Edge Deployment | draft | green | [14-deployment.md](subsystems/14-deployment.md) | `src/dvas/deployment/` |
| 15-infrastructure | Infrastructure | stable | green | [15-infrastructure.md](subsystems/15-infrastructure.md) | `src/dvas/infrastructure/` |
| 16-governance | Governance | stable | green | [16-governance.md](subsystems/16-governance.md) | `src/dvas/governance/` |

**Status values**: `stable`, `active-dev`, `experimental`, `deprecated`
**Health values**: `green`, `yellow`, `red`

---

## TOKEN BUDGET

| Step | File | ~Tokens |
|--|--|--|
| 1 | `AGENTS.md` (auto-loaded) | 1500 |
| 2 | `llms.txt` (this) | 300 |
| 3 | `docs/INDEX.md` | 400 |
| 4 | `docs/_machine/status.yaml` | 500 |
| 5 | `docs/subsystems/NN-*.md` | 1500-2500 |
| 6 | Source files | varies |

**Total overhead: ~3000 tokens**

IF task spans 3+ subsystems: **SPAWN Explore sub-agent**

---

## EXIT — Session End Checklist

Per `AGENTS.md`:

| Change Type | Check |
|--|--|
| T1 (Docs/Typo) | Nothing |
| T2+ (Code) | Run `scripts/check_doc_anchors.py` |
| T3+ (Bug/Feature) | Update `docs/_machine/status.yaml` + `bugs.yaml` |
| T5 (Schema) | Update all affected subsystem docs |
| T2+ (Code) | Update `docs/changelog/CHANGELOG.md` |

---

## EXTERNAL — Key Files

| Path | Purpose |
|--|--|
| `AGENTS.md` (root) | Boot protocol |
| `llms.txt` (root) | Machine-readable index |
| `docs/_machine/status.yaml` | Subsystem health & gaps |
| `docs/_machine/bugs.yaml` | Bug index |
| `docs/_machine/tech-debt.yaml` | Tech debt tracking |
| `scripts/check_doc_anchors.py` | Drift detector |
| `scripts/check_known_gaps.py` | Gap validator |

---

*Updated: 2026-06-18*
