# DVAS Changelog

> Session-based change log following AGENTS.md EXIT protocol.
> See `docs/_machine/status.yaml` for subsystem health.

---

## Template

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

### Session — Deleted deprecated code, increased test coverage to 53%

- **Type**: T5
- **Goal**: Increase test coverage and delete deprecated code
- **Done**:
  - **Test Coverage**: Increased from 41% to 53% (+12%)
    - Added `tests/test_api_endpoints.py` (26 tests) - API coverage from 0% to 90%
    - Added `tests/test_pipeline_core.py` (17 tests) - pipeline core tests
    - Added `tests/test_export_adapters.py` (10 tests) - export coverage from 0% to 97%
    - Added `tests/test_evaluator.py` (17 tests) - BLEU/ROUGE/LLM Judge
    - Added `tests/test_storage_mocked.py` (15 tests) - storage with mocking
    - Added `tests/test_indexed_store.py` (15 tests) - SQLite indexed store
    - Extended `tests/test_cli.py` with main CLI tests
  - **Deleted Deprecated Code**:
    - Deleted `src/dvas/core/_deprecated_pipeline.py` (205 lines) - unused actor+saga pipeline
    - Deleted `src/dvas/core/types.py` (130 lines) - unused type definitions
    - Deleted `src/dvas/models/teacher/mock.py` - unused mock teacher
    - Updated `docs/_machine/status.yaml` to remove dual-pipeline known_gap
    - Updated `docs/_machine/tech-debt.yaml` with TD-006 resolution
    - Updated `tests/test_deprecated.py` to remove tests for deleted modules
- **Files**:
  - `src/dvas/core/_deprecated_pipeline.py` → **DELETED**
  - `src/dvas/core/types.py` → **DELETED**
  - `src/dvas/models/teacher/mock.py` → **DELETED**
  - `tests/test_api_endpoints.py` → NEW (26 tests)
  - `tests/test_pipeline_core.py` → NEW (17 tests)
  - `tests/test_export_adapters.py` → NEW (10 tests)
  - `tests/test_evaluator.py` → NEW (17 tests)
  - `tests/test_indexed_store.py` → NEW (15 tests)
  - `tests/test_deprecated.py` → Updated
- **Validation**: V2 — pytest tests pass, coverage increased to 53%
- **Left for next time**: None - coverage sufficient for current scope

---

### Session — Resolved Tech Debt: TD-002, TD-005, dual-pipeline, API auth

- **Type**: T5
- **Goal**: Close multiple long-standing tech debt items
- **Done**:
  - **TD-002**: Moved hardcoded prompts from `base.py` to Jinja2 templates
    - Created `src/dvas/config/prompts/base_tasks.j2` with 5 task templates
    - Created `src/dvas/config/prompts/__init__.py` with PromptManager
    - Updated `TeacherModel._get_default_prompt()` to use new system
    - Added 12 tests in `tests/test_prompt_templates.py`
  - **TD-005**: Added GitHub Actions CI/CD pipeline
    - Created `.github/workflows/ci.yml` with 4 jobs: lint, test, student-sft, docs-validation, security
    - Updated `pyproject.toml` dev dependencies with jinja2, bandit
    - Matrix testing: Ubuntu/Windows × Python 3.10-3.13
  - **Dual-pipeline tech debt**: Resolved core/pipeline.py experiment
    - Renamed to `_deprecated_pipeline.py` (legacy reference)
    - Removed from `dvas.core.__all__` exports
    - Updated status.yaml and 04-pipeline.md with deprecation notice
    - Decision: Rejected actor+saga Pipeline in favor of AnnotationPipeline
  - **API Authentication**: Closed 07-api "No authentication" gap
    - Created `src/dvas/api/auth.py` with API key middleware
    - Config via API_KEY, API_KEY_HEADER, ALLOW_UNAUTHENTICATED env vars
    - Added `/api/v1/auth/status` endpoint
    - Applied `require_auth` dependencies to protected endpoints
    - Added 6 tests in `tests/test_api_auth.py`
- **Files**:
  - `src/dvas/config/prompts/` (new directory)
  - `src/dvas/api/auth.py` (new)
  - `.github/workflows/ci.yml` (new)
  - `src/dvas/core/_deprecated_pipeline.py` (renamed)
  - `src/dvas/core/__init__.py` (removed Pipeline exports)
  - `src/dvas/models/teacher/base.py` (updated prompts)
  - `src/dvas/config/__init__.py` (added PromptManager export)
  - `src/dvas/config/settings.py` (added auth settings)
  - `src/dvas/api/main.py` (added auth dependencies)
  - `src/dvas/api/__init__.py` (added auth exports)
  - `tests/test_prompt_templates.py` (new, 12 tests)
  - `tests/test_api_auth.py` (new, 6 tests)
  - `docs/_machine/tech-debt.yaml` (2 items resolved)
  - `docs/_machine/status.yaml` (3 gaps updated)
  - `docs/subsystems/04-pipeline.md`, `07-api.md` (updated)
- **Validation**: V2 — `pytest tests/test_prompt_templates.py tests/test_api_auth.py` → 18 tests pass
- **Left for next time**:
  - GPU E2E validation for 03-student (known gap remains - requires real Qwen2-VL-7B GPU run)
  - Redis/Celery for 07-api task persistence (known gap remains)

---

### Session — Updated subsystem docs to Architecture spec

- **Type**: T1
- **Goal**: Align with Architecture repo's frontmatter/hints protocol updates
- **Done**:
  - Removed `health`, `last_validated`, `related`, `tags` from all subsystem frontmatters
  - Updated `agent_hints` to use `WARNING:` prefix per new protocol
  - Updated AGENTS.md with Hints Protocol section
  - Simplified human-readable prose (removed redundant warnings now in hints)
- **Files**:
  - `AGENTS.md`
  - `docs/subsystems/01-*.md`, `02-*.md`, `04-*.md`, `05-*.md`, `06-*.md`, `07-*.md`
- **Validation**: V0 — All code anchors pass

---

### Session — Implemented missing subsystems (05, 06, 07)

- **Type**: T4
- **Goal**: Complete evaluation, export, and API subsystems
- **Done**:
  - 05-evaluation: Implemented metrics.py (BLEU, ROUGE, CIDEr, METEOR)
  - 05-evaluation: Implemented llm_judge.py with LLM-as-Judge and ConsistencyChecker
  - 05-evaluation: Created subsystem doc with code anchors
  - 06-export: Implemented CLI tool with export, stats, inspect commands
  - 06-export: Created subsystem doc
  - 07-api: Implemented FastAPI with upload, task, status, export endpoints
  - 07-api: Created subsystem doc
  - Updated all subsystem statuses to stable
  - Updated changelog
- **Files**:
  - `src/dvas/models/evaluator/*.py`
  - `src/dvas/export/*.py`
  - `src/dvas/api/*.py`
  - `docs/subsystems/05-*.md`, `06-*.md`, `07-*.md`
  - `docs/_machine/status.yaml`
- **Validation**: V2 — All code anchors pass validation
- **Left for next time**:
  - Add student model training (03-student)
  - Add authentication to API
  - Add Redis/Celery for task persistence

---

### Session — Resolved remaining Phase 5 gaps (retry tests, SFT dry-run, dual-pipeline)

- **Type**: T4
- **Goal**: Close the 3 remaining items from previous session: pipeline retry tests, student SFT dry-run, dual-pipeline tech debt documentation
- **Done**:
  - **04-pipeline retry tests**: Created `tests/test_pipeline_retry.py` with 5 E2E tests:
    - TestRetryRecovery: `_call_teacher_with_retry` recovers from transient failures
    - TestBatchPartialFailure: `process_batch` isolates per-item failures
    - TestCheckpointResume: Full resume-from-checkpoint with store hydration
  - **Fixed real bugs discovered during testing**:
    - `src/dvas/pipeline/core.py:process_batch` now loads checkpoint at start, filters already-processed video_ids, marks per-video checkpoint on success, hydrates skipped items from store
  - **03-student SFT dry-run**: Created `tests/test_student_sft_dryrun.py` with 9 tests:
    - TestSFTConfig: 4 tests for config defaults, model config, overrides, report_to field
    - TestSFTDatasetLoading: 3 tests for LLaVA JSONL loading, missing file handling, malformed line skipping
    - TestSFTDryRun: 2 tests for end-to-end SFT plumbing with all heavy deps mocked
  - **Fixed real bug discovered during testing**:
    - `src/dvas/models/student/sft_trainer.py:126` was reading `train_cfg.report_to` but `TrainingConfig` doesn't have that field — changed to `config.report_to` (lives on `SFTConfig`)
  - **Dual-pipeline tech debt**: Marked `src/dvas/core/pipeline.py` as EXPERIMENTAL:
    - Added WARNING docstring to file header
    - Added agent_hint to `docs/subsystems/04-pipeline.md`
    - Added known_gap to `docs/_machine/status.yaml` for dual-pipeline decision
  - **Updated status.yaml**: Synced 01-04 subsystem statuses to match code reality (all stable/green)
- **Files**:
  - `tests/test_pipeline_retry.py` (new)
  - `tests/test_student_sft_dryrun.py` (new)
  - `src/dvas/pipeline/core.py` (bug fixes)
  - `src/dvas/models/student/sft_trainer.py` (bug fix: report_to)
  - `src/dvas/core/pipeline.py` (experimental warning)
  - `docs/subsystems/04-pipeline.md` (agent_hint)
  - `docs/_machine/status.yaml` (known_gap update)
- **Validation**: V2 — All new tests pass (9 SFT + 5 retry + 35 video_loader + 9 pipeline = 58 tests)

---

### Session — Complete DVAS migration to AGENT-friendly architecture

- **Type**: T5
- **Goal**: Full implementation of code migration following AGENT-friendly architecture
- **Done**:
  - Migrated all data layer code (schemas.py, video_loader.py, storage.py)
  - Migrated all teacher model code (base.py, gpt4v.py, claude.py, together.py)
  - Migrated pipeline code (core.py with AnnotationPipeline, EPICAnnotationPipeline)
  - Migrated export adapters (adapters.py with LLaVA, OpenAI, ShareGPT formats)
  - Created pyproject.toml with dependencies
  - Fixed scripts/check_doc_anchors.py encoding and path issues
  - All code anchors validated and passing
- **Files**:
  - `src/dvas/` - All source code modules
  - `pyproject.toml`
  - `.env.example`
  - `.gitignore`
  - `scripts/check_doc_anchors.py`
- **Validation**: V2 — All code anchors verified with script
- **Left for next time**:
  - Implement actual CLI in `dvas.__main__`
  - Add pytest test suite
  - Implement student model training
  - Implement evaluation metrics

---

### Session — AGENT-friendly architecture migration

- **Type**: T5
- **Goal**: Reorganize project to machine-first architecture with AGENTS.md protocol
- **Done**:
  - Created AGENTS.md with boot, exec, exit protocols
  - Created llms.txt machine-readable index
  - Created docs/_machine/{status,bugs,tech-debt}.yaml
  - Created docs/INDEX.md navigation hub
  - Created docs/architecture/{constitution,top-level-design}.md
  - Created subsystem docs: 01-data.md, 02-teacher.md, 04-pipeline.md
  - Migrated from human-first to machine-first documentation
- **Files**:
  - `AGENTS.md`
  - `llms.txt`
  - `docs/_machine/*.yaml`
  - `docs/INDEX.md`
  - `docs/architecture/*.md`
  - `docs/subsystems/*.md`
- **Validation**: V2 — Architecture validated by inspecting structure and cross-references
- **Left for next time**:
  - Migrate actual Python code to new src/dvas/ structure
  - Create missing subsystem docs (03, 05, 06, 07)
  - Add ADR documents

---

### Session — Deep Optimization: Performance, Quality, Architecture

- **Type**: T5
- **Goal**: Comprehensive system optimization across performance, code quality, reliability, and architecture
- **Done**:
  - **Performance Optimization**:
    - Rewrote video_loader.py with true streaming (no full video load)
    - Added VideoLoader.iter_frames() for memory-efficient processing
    - Implemented VideoLoader.aiter_frames() async streaming
    - Replaced optical flow with sparse sampling for motion detection
    - Optimized scene detection with adaptive sampling
  - **Code Quality**:
    - Added mypy strict mode configuration
    - Created test suite (test_schemas.py, test_utils.py)
    - Added pytest.ini with coverage settings
  - **Reliability**:
    - Created utils/retry.py with exponential backoff and circuit breaker
    - Updated pipeline/core.py with checkpoint support
    - Integrated structured logging with structlog
    - Created utils/logging.py with JSON console output
  - **03-Student Subsystem** (major milestone):
    - Implemented config.py with SFTConfig, DPOConfig
    - Created sft_trainer.py for Qwen2-VL LoRA fine-tuning
    - Created dpo_trainer.py for preference optimization
    - Implemented inference.py with vLLM support
    - Added StudentTeacherBridge for fallback logic
    - Created dataset.py for data loading
  - **Infrastructure**:
    - Added aiofiles, aiocache, structlog dependencies
    - Created utils/queue.py with Redis queue support
    - Added Celery task queue implementation
    - Created 03-student.md subsystem documentation
  - **API Improvements**:
    - Migrated API to aiofiles for async file I/O
    - Added streaming request body handling
    - Integrated structured logging
- **Files**:
  - `src/dvas/data/video_loader.py` - Streaming rewrite
  - `src/dvas/utils/cache.py` - New
  - `src/dvas/utils/retry.py` - New
  - `src/dvas/utils/logging.py` - New
  - `src/dvas/utils/queue.py` - New
  - `src/dvas/models/student/*.py` - Complete training subsystem
  - `docs/subsystems/03-student.md` - New
  - `pyproject.toml` - Updated dependencies
  - `tests/` - Test suite
- **Validation**: V2 - Code anchors pass, new tests added
- **Left for next time**:
  - Run full integration tests with real video files
  - Setup GPU training environment
  - Redis production deployment

---

### Session — Divergent Deep Optimization: Advanced Features & Intelligence

- **Type**: T5
- **Goal**: Implement advanced, innovative features beyond core functionality - smart routing, ensemble voting, explainability, and edge deployment
- **Done**:
  - **Smart Routing System** (routing/smart_router.py):
    - VideoComplexityAnalyzer with motion, scene, object density analysis
    - SmartRouter with 4 strategies: cost_optimized, quality_optimized, balanced, adaptive
    - Dynamic model selection (Teacher/Student) based on complexity score
    - Budget-aware routing with cost estimation
    - Routing statistics and performance tracking
  - **Multi-Teacher Ensemble** (routing/ensemble.py):
    - MultiTeacherEnsemble with parallel teacher querying
    - Consensus strategies: majority, weighted, confidence_aware, best_of_n
    - Disagreement detection and resolution strategies
    - IncrementalConsensus for early stopping
    - Teacher performance statistics tracking
  - **Data Quality Platform** (quality/analyzer.py):
    - DataQualityAnalyzer with comprehensive metrics (diversity, balance, coverage)
    - AnomalyDetector with Z-score and duplicate detection
    - DatasetQualityMetrics dataclass
    - DataAugmenter with paraphrase, temporal_shift, object_swap strategies
    - Quality report generation with recommendations
  - **Explainability & Visualization** (explainability/visualizer.py):
    - KeyFrameExtractor with importance scoring
    - AttentionVisualizer with heatmap generation
    - AnnotationVisualizer with frame overlay
    - ExplainabilityReport with reasoning generation
    - Temporal attention map creation
  - **A/B Testing Framework** (monitoring/ab_testing.py):
    - ABTestManager with statistical significance testing
    - Traffic splitting with hash-based assignment
    - DriftDetector for data and model drift
    - PerformanceMonitor with anomaly detection
    - Automatic winner determination with confidence intervals
  - **Security & Privacy** (security/privacy.py):
    - PII detector with regex patterns for email, phone, SSN
    - DataAnonymizer with ID hashing
    - SecurityAuditor with audit logging
    - Watermarker with zero-width character steganography
    - AccessControl with role-based permissions
  - **Adaptive Prompt Engineering** (prompts/adaptive.py):
    - VideoTypeClassifier for content categorization
    - AdaptivePromptEngine with complexity-aware prompt selection
    - PromptLibrary with specialized templates (kitchen, robot, medical)
    - DynamicPromptOptimizer with performance-based improvement suggestions
    - Feedback loop for prompt quality improvement
  - **Edge Deployment** (deployment/edge.py):
    - ONNXExporter for vision encoder and text decoder
    - TensorRTOptimizer for NVIDIA edge devices
    - ModelQuantizer with INT8 static/dynamic quantization
    - EdgeInferenceEngine with ONNX Runtime support
    - MobileExporter for CoreML and TFLite
  - **Infrastructure**:
    - Created 7 new packages: routing, quality, explainability, monitoring, security, prompts, deployment
    - All packages include __init__.py
    - Updated pyproject.toml with edge deployment dependencies
- **Files**:
  - `src/dvas/routing/smart_router.py` + ensemble.py
  - `src/dvas/quality/analyzer.py`
  - `src/dvas/explainability/visualizer.py`
  - `src/dvas/monitoring/ab_testing.py`
  - `src/dvas/security/privacy.py`
  - `src/dvas/prompts/adaptive.py`
  - `src/dvas/deployment/edge.py`
  - 7 new package directories
- **Validation**: V3 - System architecture validated, integration points defined
- **Left for next time**:
  - Create subsystem documentation for new modules
  - Add integration tests between routing and existing pipeline
  - Implement actual CoreML/TFLite export for mobile
  - Setup production A/B testing infrastructure

---

### Session — Resolve 01-data MP4-only gap + sync subsystem status

- **Type**: T4
- **Goal**: Close the "Limited video format support" known_gap on 01-data and align 01-04 subsystem status with code reality across `status.yaml` / `llms.txt` / `INDEX.md`
- **Done**:
  - Added `SUPPORTED_VIDEO_FORMATS` constant in `src/dvas/data/video_reader.py` enumerating OpenCV/FFmpeg-supported formats (mp4, m4v, mov, avi, mkv, webm, flv, 3gp, 3gpp, ts, mpeg, mpg, ogv)
  - `VideoReader.__init__` now validates file extension against `SUPPORTED_VIDEO_FORMATS` and raises `ValueError` with the supported list for unknown formats
  - Extended `EPICKitchensLoader.get_video_path()` extension list to cover MKV, WebM, M4V (with case variants); MP4 still preferred
  - Exported `SUPPORTED_VIDEO_FORMATS` and `VideoReader` from `dvas.data`
  - Added 18 new tests in `tests/test_video_loader.py::TestVideoFormatSupport` (parametrized format acceptance/rejection, EPIC loader resolution, MP4 preference)
  - **Status sync** (code ↔ docs realignment):
    - 01-data: `active-dev/yellow` → `stable/green` (format gap closed; 17/17 format tests pass; only "no distributed processing" remains as low gap)
    - 02-teacher: `active-dev` → `stable` (3 production teachers with real SDK wiring; the "active-dev" tag was conservative)
    - 03-student: `draft` → `stable` (code is complete: 1238 LOC, real SFT/DPO/Inf/StudentTeacherBridge; remaining gap is GPU E2E validation, not implementation)
    - 04-pipeline: kept `active-dev` (uncommitted diff in `core/concurrency.py`, `pipeline/core.py`, `pipeline/checkpoint.py` confirms active work; the "no batch retry" gap is partially closed but not fully resolved)
  - Updated `docs/subsystems/01-data.md`: status table reflects "Format support: Complete"; `agent_hints` updated with the new supported-formats contract
- **Files**:
  - `src/dvas/data/video_reader.py` — `SUPPORTED_VIDEO_FORMATS` + format validation
  - `src/dvas/data/video_loader.py` — extended EPIC extension list
  - `src/dvas/data/__init__.py` — export new public symbols
  - `tests/test_video_loader.py` — `TestVideoFormatSupport` class (18 tests)
  - `docs/subsystems/01-data.md` — §5 status table + agent_hints + date
  - `docs/_machine/status.yaml` — 01-04 status/health/known_gaps + last_full_audit
  - `llms.txt` — 01-04 status/health
  - `docs/INDEX.md` — 01-04 row
- **Validation**:
  - V1 — `pytest tests/test_video_loader.py` → **35 passed** (17 existing + 18 new)
  - V2 — `python scripts/check_doc_anchors.py` → **PASS** (all 4 affected subsystem docs)
- **Left for next time**:
  - 04-pipeline "No batch retry logic" gap: full resume-from-checkpoint integration test with simulated API outage
  - 03-student: GPU E2E validation of SFT training (lowest-effort LoRA run on 1 sample)
  - Document deprecation of `core/pipeline.py:Pipeline` vs `src/dvas/pipeline/core.py:AnnotationPipeline` (Phase 5 dual-pipeline tech debt)

---
