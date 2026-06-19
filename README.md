# DVAS — Distilled Video Annotation Specialist

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-900%2B%20passing-brightgreen.svg)]()

> AI-powered video annotation platform with teacher-student distillation for robotic manipulation and egocentric vision.

## Overview

DVAS generates high-quality temporal annotations for videos using a **teacher-student distillation** architecture:

- **Teacher Models** (GPT-5.5, Claude, Together) generate gold-standard annotations  
- **Student Models** (Qwen2-VL) are fine-tuned on distilled annotations for cost-efficient inference
- **Pipeline** handles scene detection, frame sampling, retry logic, and checkpointing

### Key Features

- **Streaming Video Processing** — Async frame streaming with `asyncio.Queue`, no full video load
- **Performance Optimized** — Frame seeking, min-heap sampling, concurrent encoding, metadata caching
- **Teacher-Student Distillation** — GPT/Claude → fine-tuned Qwen2-VL for cost-efficient inference
- **Quality Loop** — 9-dimension analysis (factuality, grounding, temporal consistency) with LLM-as-judge
- **VLA/Robot Data** — Hand pose, gripper state, affordance, counterfactual annotations for robotics
- **World Model** — State prediction, dynamics annotation, causal relation extraction
- **Multi-Format Export** — LLaVA, OpenAI, ShareGPT, Open X-Embodiment, RLDS, Ego4D formats
- **Fault Tolerance** — Retry with exponential backoff, checkpoint persistence, batch recovery
- **Security** — PII detection, data anonymization, watermarking, RBAC
- **Developer Experience** — Hot reload, code scaffolding, DB migrations, lint/test runners
- **Advanced Algorithms** — Adaptive sampling, keyframe extraction, video summarization, semantic segmentation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/M-Patek/DVASm.git
cd DVASm

# Install with all dependencies
pip install -e ".[all]"

# Or install core only
pip install -e .
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

```env
OPENAI_API_KEY=sk-...
# Optional: ANTHROPIC_API_KEY, TOGETHER_API_KEY
```

### Annotate a Video

```bash
# Using CLI
python -m dvas annotate path/to/video.mp4 --model gpt-5.5 -o output.json

# Using Python
import asyncio
from pathlib import Path
from dvas.pipeline.core import AnnotationPipeline
from dvas.models.teacher import TeacherModel

async def main():
    teacher = TeacherModel(model_name="gpt-5.5")  # Auto-detects OpenAI API
    pipeline = AnnotationPipeline(teacher_model=teacher)
    annotation = await pipeline.annotate_video(
        video_path=Path("video.mp4"),
        video_id="my_video"
    )
    print(f"Generated {len(annotation.segments)} segments")

asyncio.run(main())
```

### Run End-to-End Test (No API Key Required)

```bash
# Create mini synthetic dataset
python examples/create_mini_dataset.py --num-videos 3

# Export to LLaVA training format
python examples/create_mini_dataset.py --export-llava data/training/mini.jsonl

# Verify training pipeline is ready
python tests/test_train_dryrun.py
```

### Annotate EPIC-KITCHENS Dataset

```bash
# Download annotation files only
python examples/download_epic.py --annotations-only

# Annotate with Claude (requires ANTHROPIC_API_KEY)
python examples/annotate_epic.py --split train --num 10 --model claude-opus-4-8

# Export gold annotations for student training
python examples/export_training.py --output data/training/sft.jsonl --format llava
```

## Architecture

```
DVAS
├── Teacher Models      (GPT-5.5, Claude, Together API - Unified Interface)
│   ├── Unified TeacherModel with auto-provider detection
│   ├── Model/Provider Registry with capabilities & pricing
│   ├── Quota Manager & Rate Limit Scheduler
│   └── Multi-teacher Consensus & Adaptive Fallback Chains
├── Pipeline            (Scene detection → Frame sampling → Annotation)
│   ├── Checkpoint persistence for resumable processing
│   ├── Saga pattern for distributed transactions
│   ├── Quality Gates (factuality, grounding, temporal consistency)
│   ├── Parser with confidence scoring & fallback audit
│   └── Batch processing with concurrency control
├── Quality Loop        (Automatic quality analysis & feedback)
│   ├── Automatic Quality Analyzer (9 quality dimensions)
│   ├── LLM-as-Judge Pipeline for quality evaluation
│   ├── Human Review & Disagreement Queues
│   └── Acceptance Criteria & Trend Dashboard
├── Student Models      (Qwen2-VL fine-tuning with closed-loop evaluation)
│   ├── LoRA Adapter Registry with data version binding
│   ├── SFT/DPO training on distilled annotations
│   ├── Confidence Calibration & Active Learning
│   └── Teacher vs Student Evaluation & Regression Benchmark
├── World Model         (State prediction & dynamics annotation)
│   ├── State before/after generation
│   ├── Physical dynamics annotation (mass, friction, contact)
│   ├── Causal relation extraction
│   └── Counterfactual generation & training export
├── Export              (LLaVA, OpenAI, ShareGPT, Open X-Embodiment, RLDS)
├── API                 (FastAPI REST endpoints with auth)
└── Security            (PII detection, anonymization, RBAC)
```

## Subsystems

| ID | Subsystem | Status | Description |
|----|-----------|--------|-------------|
| 01 | [Data Layer](docs/subsystems/01-data.md) | ✅ Enhanced | Pluggable backends (LocalFS/SQLite/PostgreSQL/S3), index management |
| 02 | [Teacher Models](docs/subsystems/02-teacher.md) | ✅ Enhanced | Registry, capabilities, pricing, quota, consensus, fallback chains |
| 03 | [Student (VLA)](docs/subsystems/03-student.md) | ✅ Enhanced | LoRA registry, calibration, active learning, evaluation, benchmark |
| 04 | [Pipeline](docs/subsystems/04-pipeline.md) | ✅ Stable | Annotation pipeline with Saga pattern & checkpointing |
| 05 | [Evaluation](docs/subsystems/05-evaluation.md) | ✅ Stable | BLEU/CIDEr + LLM-as-Judge |
| 06 | [Export](docs/subsystems/06-export.md) | ✅ Enhanced | Multi-format: LLaVA, OpenAI, Open X-Embodiment, RLDS, Ego4D |
| 07 | [API](docs/subsystems/07-api.md) | ✅ Stable | FastAPI REST endpoints with auth |
| 08 | [Routing](docs/subsystems/08-routing.md) | ✅ Stable | Smart router with complexity-based model selection |
| 09 | [Quality Loop](docs/subsystems/09-quality.md) | ✅ Enhanced | Auto analysis, LLM-as-judge, review queues, acceptance criteria |
| 10 | [Lineage](docs/subsystems/10-lineage.md) | ✅ Stable | Schema version management, data provenance tracking |
| 11 | [Prompts](docs/subsystems/13-prompts.md) | ✅ Stable | Adaptive prompt engineering, video classification |
| 12 | [Security](docs/subsystems/12-security.md) | ✅ Stable | PII detection, anonymization, watermarking, RBAC |
| 13 | [Monitoring](docs/subsystems/11-monitoring.md) | ✅ Stable | A/B testing, drift detection, performance monitoring |
| 14 | [World Model](docs/subsystems/14-world-model.md) | ✅ Enhanced | State prediction, dynamics annotation, causal extraction, benchmarks |
| 15 | [Infrastructure](docs/subsystems/15-infrastructure.md) | 📝 Draft | Data platform deployment, ops |
| 16 | [Governance](docs/subsystems/16-governance.md) | ✅ Stable | Annotation standard management, multi-standard adapters |

**Test Coverage:** 900+ tests passing (~95% pass rate, failures are Windows/env-specific)

## Development

### Run Tests

```bash
# All tests (use --basetemp on Windows to avoid permission issues)
python -m pytest tests/ -v --basetemp=./tmp/pytest

# Specific test file
python -m pytest tests/test_video_loader.py -v

# With coverage
python -m pytest tests/ --cov=src/dvas --cov-report=html
```

### Code Quality

```bash
# Using the built-in CLI (recommended)
python -m dvas lint --fix
python -m dvas test --cov

# Or manually
ruff check src/ --fix
mypy src/dvas
black src/ tests/
```

### Project Structure

```
DVASm/
├── src/dvas/              # Core source code
│   ├── api/               # FastAPI REST endpoints with auth & rate limiting
│   ├── cli/               # Developer tools (scaffold, migrate, dev mode)
│   ├── config/            # Settings, prompts, constants
│   ├── core/              # Event bus, circuit breaker, algorithms, actors, Saga
│   ├── data/              # Video loading, schemas, storage
│   │   └── robot_schemas/ # VLA/Robot annotations (hand pose, affordance, counterfactual)
│   ├── models/            # Teacher & student models
│   │   ├── teacher/base.py    # Unified TeacherModel (all providers)
│   │   ├── teacher/           # Registry, capabilities, pricing, quota, consensus
│   │   ├── student/           # LoRA registry, SFT/DPO, calibration, selection
│   │   └── evaluator/         # Metrics & LLM-as-Judge
│   ├── persistence/       # Pluggable backends (LocalFS, SQLite, PostgreSQL, S3)
│   ├── pipeline/          # Annotation pipeline
│   │   ├── core.py        # Main orchestrator
│   │   ├── builder.py     # Annotation construction
│   │   ├── checkpoint.py  # Resume persistence
│   │   ├── parser.py      # Response parsing
│   │   ├── quality_gate.py    # Quality validation gates
│   │   └── state_machine.py   # Annotation lifecycle management
│   ├── prompts/           # Adaptive prompt engineering
│   ├── quality/           # Quality loop (analyzer, llm_judge, review_queue, trend_dashboard)
│   ├── routing/           # SmartRouter, Ensemble, CostOptimizer
│   ├── security/          # Privacy & access control, audit logging
│   ├── world_model/       # State prediction, dynamics, causal extraction, training export
│   └── utils/             # Logging, retry, caching, observability
├── tests/                 # Test suite (900+ tests)
├── examples/              # Usage examples
│   ├── annotate_epic.py
│   ├── download_epic.py
│   ├── create_mini_dataset.py
│   ├── export_training.py
│   ├── train_student_sft.py     # Student SFT training
│   ├── train_student_dpo.py     # Student DPO training
│   └── eval_teacher_vs_student.py  # Teacher vs Student evaluation
├── scripts/               # Utility scripts
├── benchmarks/            # Performance benchmarks
└── docs/                  # Documentation
    ├── subsystems/        # Per-subsystem docs (16 subsystems)
    ├── architecture/      # Design docs & constitution
    └── _machine/          # Status & tech debt tracking
```

## CLI Reference

### Core Commands

```bash
# Annotate a single video
python -m dvas annotate video.mp4 --model gpt-5.5 -o annotation.json

# Annotate EPIC-KITCHENS dataset split
python -m dvas annotate_epic --split train --num 10

# Export annotations to training format
python -m dvas export training_data.jsonl --format llava --source gold

# Show statistics
python -m dvas stats --source gold
```

### Developer Commands

```bash
# Development mode with hot reload
python -m dvas dev --server --port 8000

# Generate code scaffolding
python -m dvas scaffold module my_feature --output src/dvas/
python -m dvas scaffold model my_teacher --output src/dvas/models/teacher/
python -m dvas scaffold test my_feature --output tests/

# Database migrations
python -m dvas migrate status
python -m dvas migrate create --name add_users_table
python -m dvas migrate up

# Generate API documentation
python -m dvas docs --output docs/api --format markdown

# Lint and format code
python -m dvas lint --fix

# Run tests with coverage
python -m dvas test --cov --fail-fast

# Validate environment
python -m dvas validate

# Show project info
python -m dvas info
```

## Performance

DVAS implements multiple performance optimizations across the video processing pipeline:

| Optimization | Impact |
|-------------|--------|
| **Frame Seeking** | Direct jump to target frames instead of sequential read when `step > 1` |
| **Min-Heap Sampling** | KeyFrameSampler uses O(K) memory instead of O(N) |
| **Concurrent Encoding** | ThreadPoolExecutor for parallel base64 encoding of frame batches |
| **Metadata Caching** | Module-level cache avoids re-reading video headers |
| **GC Between Chunks** | Explicit garbage collection in batch processing to free frame memory |
| **Async Streaming** | Background thread + asyncio queue for true frame streaming |
| **Adaptive Sampling** | Content-aware frame allocation using motion/edge/entropy metrics |
| **Min-Max Heap** | O(log n) insertion with O(1) median query for importance tracking |
| **Sliding Window Buffer** | Circular buffer for streaming frame analysis without reallocation |
| **Semantic Segmentation** | Multi-feature boundary detection for coherent shot grouping |

Run the benchmark:

```bash
python benchmarks/perf_benchmark.py
```

## Tech Debt & Status

See [`docs/_machine/`](docs/_machine/) for:

- [`status.yaml`](docs/_machine/status.yaml) — Subsystem health & gaps
- [`tech-debt.yaml`](docs/_machine/tech-debt.yaml) — Tracked technical debt
- [`bugs.yaml`](docs/_machine/bugs.yaml) — Bug index

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [OpenAI](https://openai.com/) / [Anthropic](https://anthropic.com/) / [Together](https://together.ai/) for teacher models
- [OpenCV](https://opencv.org/) for video processing
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [FastAPI](https://fastapi.tiangolo.com/) for API layer
