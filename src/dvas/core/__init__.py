"""Core module initialization.

Exposes the key architectural components for event-driven,
saga-based, and pipeline-based architectures.
"""

from dvas.core.concurrency import (
    AsyncBatchProcessor,
    AsyncIteratorBridge,
    ConcurrencyLimiter,
    FrameEncoderPool,
    PoolRegistry,
    ProcessPoolWrapper,
    WorkStealingPool,
    run_in_thread,
)
from dvas.core.event_bus import (
    AnnotationCompletedEvent,
    AnnotationFailedEvent,
    AnnotationStartedEvent,
    Event,
    EventBus,
    EventPriority,
    LoggingMiddleware,
    MetricsMiddleware,
    PipelineStageCompletedEvent,
    SceneDetectedEvent,
    VideoLoadedEvent,
    get_event_bus,
)
from dvas.core.outbox import OutboxPublisher, OutboxStatus, OutboxStore
from dvas.core.pipeline import (
    BaseStage,
    DetectScenesStage,
    FunctionStage,
    LoadVideoStage,
    Pipeline,
    PipelineContext,
    PipelineStage,
    SaveAnnotationStage,
    StageRegistry,
    StageResult,
)
from dvas.core.saga import (
    AnnotationSagaBuilder,
    FunctionSagaStep,
    Saga,
    SagaContext,
    SagaOrchestrator,
    SagaStep,
    SagaStepResult,
)

__all__ = [
    # Concurrency
    "AsyncIteratorBridge",
    "AsyncBatchProcessor",
    "ConcurrencyLimiter",
    "FrameEncoderPool",
    "WorkStealingPool",
    "ProcessPoolWrapper",
    "PoolRegistry",
    "run_in_thread",
    # Event Bus
    "EventBus",
    "Event",
    "EventPriority",
    "get_event_bus",
    "LoggingMiddleware",
    "MetricsMiddleware",
    # Events
    "VideoLoadedEvent",
    "SceneDetectedEvent",
    "AnnotationStartedEvent",
    "AnnotationCompletedEvent",
    "AnnotationFailedEvent",
    "PipelineStageCompletedEvent",
    # Saga
    "Saga",
    "SagaStep",
    "SagaStepResult",
    "SagaContext",
    "SagaOrchestrator",
    "FunctionSagaStep",
    "AnnotationSagaBuilder",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    "PipelineContext",
    "StageResult",
    "BaseStage",
    "FunctionStage",
    "StageRegistry",
    "LoadVideoStage",
    "DetectScenesStage",
    "SaveAnnotationStage",
    # Outbox
    "OutboxStore",
    "OutboxPublisher",
    "OutboxStatus",
]
