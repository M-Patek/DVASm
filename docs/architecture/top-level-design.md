# DVAS Top-Level Design

> System overview for architecture understanding.
> Read `docs/INDEX.md` first for navigation.

---

## System Context

DVAS generates gold-standard video-text annotations for training Vision-Language Models. It uses a **teacher-student distillation** architecture:

- **Teacher**: Expensive commercial models (GPT-4V, Claude) generate ground truth
- **Student**: Fine-tuned open-source models (Qwen2-VL) replicate teacher at 10-100x lower cost
- **Pipeline**: Orchestrates video processing, annotation, and quality control
- **Export**: Delivers data in customer's required format

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  API Layer (07-api)                                             │
│  REST endpoints for external integration                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  Pipeline Layer (04-pipeline)                                   │
│  Orchestration: scene detection → annotation → storage          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  Model Layer                                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │ 02-teacher    │  │ 03-student    │  │ 05-evaluation │       │
│  │ GPT-4V/Claude │  │ Qwen2-VL SFT  │  │ Quality judge │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  Data Layer (01-data)                                           │
│  Video loading, preprocessing, storage, schemas                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
EPIC-KITCHENS Video
        │
        ▼
┌───────────────────┐
│  VideoLoader      │  Load video, extract metadata
│  (01-data)        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Scene Detection  │  Split into temporal segments
│  (01-data)        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐      ┌──────────────────┐
│  Teacher Model    │◄────►│  GPT-4V/Claude   │
│  (02-teacher)     │      │  API             │
└─────────┬─────────┘      └──────────────────┘
          │
          ▼
┌───────────────────┐
│  Response Parse   │  Extract structured data
│  (04-pipeline)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  AnnotationStore  │  Save gold annotation
│  (01-data)        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Export Adapter   │  Convert to LLaVA/OpenAI format
│  (06-export)      │
└───────────────────┘
```

---

## Key Interfaces

### TeacherModel (02-teacher)

```python
class TeacherModel(UnifiedModel):
    @abstractmethod
    async def annotate(
        self,
        video_path: Optional[Path] = None,
        frames: Optional[List[np.ndarray]] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult: ...
```

**Key**: All models now return `GenerationResult` with standardized fields: `text`, `model_type`, `status`, `latency_ms`, `token_usage`, `cost_usd`.

### UnifiedModel (Base)

```python
class UnifiedModel(ABC):
    @abstractmethod
    async def generate(self, **kwargs) -> GenerationResult: ...
    @abstractmethod
    async def generate_batch(self, items: List[Dict]) -> List[GenerationResult]: ...
    @property
    @abstractmethod
    def model_type(self) -> ModelType: ...
    @property
    def model_version(self) -> str: ...
    def supports(self, capability: str) -> bool: ...
    def estimate_cost(self, **kwargs) -> float: ...
```

### AnnotationStore (01-data)

```python
class AnnotationStore:
    def save(self, annotation: Annotation, source: str) -> Path: ...
    def load(self, annotation_id: str, source: str) -> Optional[Annotation]: ...
    def load_all(self, source: str) -> Iterator[Annotation]: ...
```

### Pipeline (04-pipeline)

```python
class AnnotationPipeline:
    async def annotate_video(self, video_path: Path, video_id: str) -> Annotation: ...
    async def process_batch(
        self, items: List[Dict], max_concurrent: int = 5
    ) -> Tuple[List[Annotation], List[Dict]]: ...
```

---

## Scalability Considerations

| Component | Current | Target |
|-----------|---------|--------|
| Video processing | Single-process | Ray/Dask distributed |
| API calls | 10 concurrent | 100 with quota management |
| Storage | Local filesystem | S3 + PostgreSQL |
| Inference | Together API | Self-hosted vLLM |

---

## Security Model

1. **API Keys**: Environment variables only, never in code
2. **Video Data**: Assume customer-confidential, access logging
3. **Annotations**: Encrypt at rest, audit trail for exports
4. **API Service**: Authentication required, rate limiting per customer

---

*Design version: 0.1.0 | Last updated: 2024-06-17*
