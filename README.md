# DVAS — Distilled Video Annotation Specialist

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-130%2F130%20passing-brightgreen.svg)]()

> AI-powered video annotation platform with teacher-student distillation for robotic manipulation and egocentric vision.

## Overview

DVAS generates high-quality temporal annotations for videos using a **teacher-student distillation** architecture:

- **Teacher Models** (GPT-4V, Claude, Together) generate gold-standard annotations
- **Student Models** (Qwen2-VL) are fine-tuned on distilled annotations for cost-efficient inference
- **Pipeline** handles scene detection, frame sampling, retry logic, and checkpointing

### Key Features

- **Streaming Video Processing** — Async frame streaming with `asyncio.Queue`, no full video load
- **Performance Optimized** — Frame seeking, min-heap sampling, concurrent encoding, metadata caching
- **Adaptive Prompts** — Video-type classification with specialized prompt templates
- **Fault Tolerance** — Retry with exponential backoff, checkpoint persistence, batch recovery
- **Multi-Format Export** — LLaVA, OpenAI, ShareGPT training formats
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
python -m dvas annotate path/to/video.mp4 --model gpt-4o -o output.json

# Using Python
import asyncio
from pathlib import Path
from dvas.pipeline.core import AnnotationPipeline
from dvas.models.teacher.gpt4v import GPT4VTeacher

async def main():
    teacher = GPT4VTeacher(model_name="gpt-4o")
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
# Generate synthetic test video
python scripts/generate_test_video.py

# Run full pipeline with MockTeacher
python scripts/run_e2e.py
```

Output:
```
[1/5] Initializing pipeline with MockTeacher...
[2/5] Running annotation pipeline...
[3/5] Inspecting segments...
[4/5] Saving annotation...
[5/5] Exporting to training format...

End-to-End Test Complete!
Teacher API calls: 1
Total frames processed: 8
```

## Architecture

```
DVAS
├── Teacher Models      (GPT-4V, Claude, Together API)
│   └── Generate gold annotations with retry + connection pooling
├── Pipeline            (Scene detection → Frame sampling → Annotation)
│   ├── Checkpoint persistence for resumable processing
│   └── Batch processing with concurrency control
├── Student Models      (Qwen2-VL fine-tuning)
│   └── SFT/DPO training on distilled annotations
├── Evaluation          (BLEU, ROUGE, CIDEr, LLM-as-Judge)
├── Export              (LLaVA, OpenAI, ShareGPT formats)
└── API                 (FastAPI REST endpoints)
```

## Subsystems

| ID | Subsystem | Status | Description |
|----|-----------|--------|-------------|
| 01 | [Data Layer](docs/subsystems/01-data.md) | Active | Video loading, preprocessing, storage |
| 02 | [Teacher Models](docs/subsystems/02-teacher.md) | Active | GPT-4V, Claude, Together API wrappers |
| 03 | [Student Models](docs/subsystems/03-student.md) | Draft | Qwen2-VL fine-tuning (SFT/DPO) |
| 04 | [Pipeline](docs/subsystems/04-pipeline.md) | Active | Annotation pipeline with retry |
| 05 | [Evaluation](docs/subsystems/05-evaluation.md) | Stable | BLEU/CIDEr + LLM-as-Judge |
| 06 | [Export](docs/subsystems/06-export.md) | Stable | Multi-format training data export |
| 07 | [API](docs/subsystems/07-api.md) | Stable | FastAPI REST endpoints |
| 08-14 | *(see docs/subsystems/)* | | |

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
│   ├── api/               # FastAPI REST endpoints with rate limiting & health checks
│   ├── cli/               # Developer tools (scaffold, migrate, dev mode)
│   ├── core/              # Event bus, circuit breaker, algorithms, actors
│   ├── data/              # Video loading, schemas, storage
│   ├── models/            # Teacher & student models
│   ├── pipeline/          # Annotation pipeline
│   │   ├── core.py        # Main orchestrator
│   │   ├── builder.py     # Annotation construction
│   │   ├── checkpoint.py  # Resume persistence
│   │   └── parser.py      # Response parsing
│   ├── prompts/           # Adaptive prompt engineering
│   ├── quality/           # Quality evaluation
│   ├── security/          # Privacy & access control, audit logging, encryption
│   └── utils/             # Logging, retry, caching
├── tests/                 # Test suite (130 tests)
│   ├── test_security.py     # Security utilities (56 tests)
│   ├── test_properties.py   # Property-based tests (10 tests)
│   ├── test_load.py         # Load & benchmark tests (12 tests)
│   ├── test_algorithms.py   # Algorithm & data structure tests (52 tests)
│   └── test_cli.py          # CLI developer tools tests (23 tests)
├── scripts/               # Utility scripts
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
│   ├── subsystems/        # Per-subsystem docs
│   ├── architecture/      # Design docs
│   └── _machine/          # Status & tech debt tracking
└── pyproject.toml         # Project configuration
```

## CLI Reference

### Core Commands

```bash
# Annotate a single video
python -m dvas annotate video.mp4 --model gpt-4o -o annotation.json

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
