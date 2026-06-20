# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-20

### Added - Phase 4: Teacher System Enhancement

#### Provider & Model Registry (`src/dvas/models/teacher/registry.py`)
- `Provider` enum for OpenAI, Anthropic, Together
- `ModelSpec` dataclass with frame limits, token limits, features
- `ModelRegistry` for centralized model management
- Auto-detection of provider from model name
- Model recommendations by task type

#### Capability Registry (`src/dvas/models/teacher/capabilities.py`)
- `Capability` enum (VISION, JSON_MODE, STRUCTURED_OUTPUT, etc.)
- `CapabilityRegistry` for provider capability contracts
- Structured output support detection
- Rate limit tracking per provider
- Batch size limits

#### Pricing Registry (`src/dvas/models/teacher/pricing.py`)
- `PricingInfo` with per-token and per-image pricing
- `PricingRegistry` for model cost tracking
- Cost estimation with batch discounts
- Cached token pricing support
- Cross-model cost comparison

#### Latency & Quality Profiles (`src/dvas/models/teacher/profiles.py`)
- `LatencyProfile` with p50/p95/p99 percentiles
- `QualityProfile` with task-specific scores
- `ProfileManager` for performance tracking
- Cost-quality efficiency analysis
- Baseline profiles for GPT-5.5, Claude Opus/Sonnet, Llama models

#### Quota Manager (`src/dvas/models/teacher/quota.py`)
- Daily/monthly quota tracking
- Request, token, and cost limits
- Async-safe quota checking
- Usage recording and reporting
- Automatic quota reset on day/month change

#### Rate Limit Scheduler (`src/dvas/models/teacher/scheduler.py`)
- Token bucket rate limiting
- Adaptive backoff on rate limit errors
- Multi-provider scheduling
- Latency tracking per provider
- Configurable burst sizes

#### Multi-Teacher Consensus (`src/dvas/models/teacher/consensus.py`)
- `ConsensusEngine` with similarity-based clustering
- Text and JSON output comparison
- Weighted consensus voting
- Confidence scoring
- `DisagreementMiner` for analyzing model disagreements

#### Fallback Chain (`src/dvas/models/teacher/fallback.py`)
- `FallbackChain` with configurable rules
- `AdaptiveFallbackChain` with success rate learning
- Automatic fallback on failure/timeout/rate limit
- Exponential backoff with jitter
- Failure history tracking

#### Tests
- 163 tests covering all new modules
- Registry, capabilities, pricing, profiles, quota, scheduler, consensus, fallback

### Changed
- Updated `src/dvas/models/teacher/__init__.py` to export all new modules

## [0.1.0] - 2026-06-18

### Added
- Initial DVAS release
- Teacher models: GPT-5.5, Claude, Together API wrappers
- Student models: Qwen2-VL finetuning via SFT/DPO
- Pipeline: End-to-end video annotation
- Evaluation: BLEU/CIDEr + LLM-as-Judge
- Export: LLaVA, OpenAI, ShareGPT formats
- API: FastAPI REST endpoints
- Routing: Smart model selection
