---
id: 07-api
title: "07-API — REST API Service"
status: stable
applies_to:
  - "src/dvas/api/**"
code_anchors:
  - "src/dvas/api/main.py:VideoUploadResponse"
  - "src/dvas/api/main.py:AnnotationTaskRequest"
  - "src/dvas/api/main.py:main"
  - "src/dvas/api/auth.py:verify_api_key"
agent_hints:
  - "WARNING: Tasks are stored in memory—restart clears them. Use Redis in production."
  - "INFO: API key auth available via API_KEY env var. Set ALLOW_UNAUTHENTICATED=false to require auth."
  - "WARNING: Export generates temp files—ensure cleanup in production."
  - "WARNING: Video uploads saved to data/raw/uploads/"
---

# §07 API Service

FastAPI-based REST API for video upload, annotation tasks, and result export.

---

## §0 — One-liner

REST API for video upload, async annotation tasks, status polling, and training data export.

## §1 — Core concepts

- **Video Upload**: POST /videos/upload - Accepts MP4/AVI/MOV, returns video_id
- **Annotation Task**: POST /annotations/tasks - Queues annotation job
- **Task Status**: GET /annotations/tasks/{task_id} - Poll for completion
- **Get Annotation**: GET /annotations/{video_id} - Retrieve completed annotation
- **Export**: POST /export - Download annotations as JSONL
- **Authentication**: API key via X-API-Key header (optional by default)

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `main.py:VideoUploadResponse` | Response model | Client code |
| `main.py:AnnotationTaskRequest` | Request model | Client code |
| `main.py:main` | Run server | Starting API |
| `auth.py:verify_api_key` | Auth dependency | Securing endpoints |

## §3 — Key behaviors & contracts

### Behavior 1: Authentication

API key authentication is controlled via environment variables:

```bash
# Enable auth (optional - allows unauthenticated if key missing)
export API_KEY="your-secret-key"
export API_KEY_HEADER="X-API-Key"  # Default
export ALLOW_UNAUTHENTICATED="false"  # Require key
```

Protected endpoints (require auth when enabled):
- POST /api/v1/videos/upload
- POST /api/v1/annotations/tasks
- GET /api/v1/annotations/tasks/{task_id}
- GET /api/v1/annotations/{video_id}
- POST /api/v1/export
- GET /api/v1/search

Strict endpoints (always require key if configured):
- GET /api/v1/stats

### Behavior 2: Async Task Processing

- Task created immediately with status "pending"
- Background task runs annotation pipeline
- Poll GET /annotations/tasks/{task_id} for completion

### Behavior 3: Video Upload

- Saved to data/raw/uploads/{video_id}_{filename}
- 50MB limit (configurable in uvicorn)
- MIME type validation

### Behavior 4: Export Formats

Supported formats: llava, openai, sharegpt
Exported as JSONL file with appropriate schema. When `video_ids` is provided, export now writes only the requested annotations by using the loaded/filtered annotations and `dvas.export.adapters.export_annotations()`, rather than re-exporting the whole source.

---

## §4 — Phase 10: API and Task System (New)

### Task Storage (`task_store.py`)

Abstract TaskStore interface with three implementations:

| Store | Backend | Use Case |
|-------|---------|----------|
| `InMemoryTaskStore` | dict | Development, testing |
| `RedisTaskStore` | Redis | Production single-node |
| `PostgresTaskStore` | PostgreSQL | Production multi-node |

Task model fields: id, status, type, payload, result, error, progress, created_at, updated_at, tenant_id, priority, retry_count, max_retries

### Task Queue (`task_queue.py`)

Abstract TaskQueue with Celery/RQ/Arq-compatible wrapper:
- `enqueue()` - Add task to queue with priority
- `dequeue()` - Get next available task
- `ack()` - Acknowledge completion
- `retry()` - Retry with exponential backoff
- `cancel()` - Cancel pending task
- Priority support (1-10, lower = higher priority)

### Task Lifecycle (`task_lifecycle.py`)

TaskLifecycleManager provides:
- **Retry**: Exponential backoff, fixed delay, linear backoff, or no retry
- **Cancellation**: Cancel pending/queued/retrying tasks
- **Resume**: Resume from checkpoint with progress restoration
- **Progress Streaming**: SSE/WebSocket-ready async generator

### Batch Job API (`batch_api.py`)

FastAPI endpoints under `/api/v1/batch`:
- `POST /jobs` - Create batch annotation job
- `GET /jobs/{batch_id}` - Get batch status
- `GET /jobs/{batch_id}/results` - Get results
- `GET /jobs/{batch_id}/progress` - Get progress
- `POST /jobs/{batch_id}/cancel` - Cancel batch
- `GET /jobs` - List batch jobs with filtering

### Dataset API (`dataset_api.py`)

FastAPI endpoints under `/api/v1/datasets`:
- `POST /` - Create dataset
- `GET /{dataset_id}` - Get dataset
- `PUT /{dataset_id}` - Update dataset
- `DELETE /{dataset_id}` - Delete dataset
- `GET /` - List with filtering (source, tag)
- `GET /{dataset_id}/statistics` - Dataset statistics
- `GET /{dataset_id}/annotations` - Get annotations

### Annotation Review API (`review_api.py`)

FastAPI endpoints under `/api/v1/review`:
- `GET /queue` - Review queue with filtering
- `POST /queue` - Add to review queue
- `POST /assign` - Assign to reviewer
- `POST /decision` - Submit approve/reject
- `GET /stats` - Review statistics
- `GET /{annotation_id}` - Get review details

### Export Job API (`export_api.py`)

FastAPI endpoints under `/api/v1/export`:
- `POST /jobs` - Create export job
- `GET /jobs/{job_id}` - Get export status
- `POST /jobs/{job_id}/process` - Process export
- `GET /jobs/{job_id}/download` - Download file
- `GET /formats` - List supported formats (llava, openai, sharegpt, ego4d, rlds)

### Model Registry API (`model_api.py`)

FastAPI endpoints under `/api/v1/models`:
- `POST /register` - Register model
- `GET /{model_id}` - Get model
- `GET /` - List with filtering (provider, vision, video)
- `GET /{model_id}/versions` - Get versions
- `POST /{model_id}/versions` - Add version
- `POST /query` - Query by capabilities
- `GET /{model_id}/capabilities` - Get capabilities

### Prompt Registry API (`prompt_api.py`)

FastAPI endpoints under `/api/v1/prompts`:
- `POST /` - Create prompt
- `GET /{prompt_id}` - Get prompt
- `GET /{prompt_id}/content` - Get prompt content
- `PUT /{prompt_id}` - Update prompt
- `DELETE /{prompt_id}` - Delete prompt
- `GET /` - List with filtering
- `POST /{prompt_id}/versions` - Create version
- `GET /{prompt_id}/versions` - List versions
- `POST /{prompt_id}/versions/{version}/activate` - Activate version

### Tenant Isolation (`tenant.py`)

Tenant middleware for multi-tenant support:
- `TenantMiddleware` - Extract tenant from X-Tenant-ID header
- `TenantStore` - In-memory tenant registry
- `TenantScopedAccess` - Scoped data access helper
- Per-tenant rate limits, quotas, and feature flags

### Rate Limiting (`rate_limit.py`)

Per-tenant rate limiting with token bucket:
- `TenantRateLimiter` - Per-tenant token buckets
- `TenantRateLimitMiddleware` - Automatic rate limiting
- Path-based categorization (videos, annotations, export, etc.)
- Daily quota tracking

### Audit Log (`audit_log.py`)

Request/response logging:
- `AuditLogEntry` - Structured log entry
- `AuditLogStore` - In-memory storage (bounded)
- `AuditLogger` - Log requests and actions
- Query by tenant, time range, path, status code

---

## §5 — Integration with other subsystems

- **Upstream**: HTTP clients (curl, Python requests)
- **Downstream**: Uses `04-pipeline` for annotation
- **Downstream**: Uses `01-data` for storage
- **Task System**: Uses `task_store`, `task_queue`, `task_lifecycle` for async processing

## §6 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Upload endpoint | Complete | With type validation |
| Task creation | Complete | Async background |
| Status polling | Complete | |
| Export | Complete | Multiple formats; honors `video_ids` filtering |
| **Authentication** | **Complete** | API key via X-API-Key header (2026-06-18) |
| Task persistence | Complete | InMemory, Redis, PostgreSQL backends |
| Rate limiting | Complete | Per-tenant token bucket |
| Batch API | Complete | Phase 10 |
| Dataset API | Complete | Phase 10 |
| Review API | Complete | Phase 10 |
| Model Registry | Complete | Phase 10 |
| Prompt Registry | Complete | Phase 10 |
| Tenant Isolation | Complete | Phase 10 |
| Audit Log | Complete | Phase 10 |

**Active known_gaps**:
- None

---

*Subsystem doc: 07-api | Updated: 2026-06-20*
