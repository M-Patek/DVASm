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
agent_hints:
  - "WARNING: Tasks are stored in memory—restart clears them. Use Redis in production."
  - "WARNING: No authentication currently—add API key middleware for production."
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

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `main.py:VideoUploadResponse` | Response model | Client code |
| `main.py:AnnotationTaskRequest` | Request model | Client code |
| `main.py:main` | Run server | Starting API |

## §3 — Key behaviors & contracts

### Behavior 1: Async Task Processing

- Task created immediately with status "pending"
- Background task runs annotation pipeline
- Poll GET /annotations/tasks/{task_id} for completion

### Behavior 2: Video Upload

- Saved to data/raw/uploads/{video_id}_{filename}
- 50MB limit (configurable in uvicorn)
- MIME type validation

### Behavior 3: Export Formats

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
| Authentication | Missing | Need API key auth |
| Task persistence | Missing | Need Redis |
| Rate limiting | Missing | Need middleware |

**Active known_gaps** (from `status.yaml`):
- No authentication (severity: medium)
- In-memory task storage (severity: medium)

## §6 — Testing

```bash
# Start server
python -m dvas.api

# Upload video
curl -X POST -F "file=@video.mp4" http://localhost:8000/videos/upload

# Create task
curl -X POST http://localhost:8000/annotations/tasks \
  -H "Content-Type: application/json" \
  -d '{"video_id": "vid_xxx", "teacher_model": "gpt-4o"}'

# Check status
curl http://localhost:8000/annotations/tasks/task_xxx

# Export
curl -X POST http://localhost:8000/export \
  -H "Content-Type: application/json" \
  -d '{"format": "llava"}' \
  --output export.jsonl
```

---

*Subsystem doc: 07-api | Updated: 2024-06-17*
