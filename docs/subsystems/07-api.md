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
Exported as JSONL file with appropriate schema.

## §4 — Integration with other subsystems

- **Upstream**: HTTP clients (curl, Python requests)
- **Downstream**: Uses `04-pipeline` for annotation
- **Downstream**: Uses `01-data` for storage

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Upload endpoint | Complete | With type validation |
| Task creation | Complete | Async background |
| Status polling | Complete | |
| Export | Complete | Multiple formats |
| **Authentication** | **Complete** | API key via X-API-Key header (2026-06-18) |
| Task persistence | Missing | Need Redis |
| Rate limiting | Complete | Token bucket middleware |

**Active known_gaps** (from `status.yaml`):
- ~~No authentication~~ **RESOLVED 2026-06-18**: API key auth via `api/auth.py`
- In-memory task storage (severity: medium)
