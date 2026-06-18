"""Saga pattern implementation for distributed transaction management.

Provides a Saga orchestrator that manages multi-step transactions
with automatic compensation on failure.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Saga types
# ---------------------------------------------------------------------------

class SagaStatus(Enum):
    """Status of a saga execution."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    FAILED = auto()


class SagaStepStatus(Enum):
    """Status of an individual saga step."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()


@dataclass
class SagaStepResult:
    """Result of executing a saga step."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class SagaContext:
    """Shared context passed between saga steps."""

    saga_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    results: List[SagaStepResult] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)


# ---------------------------------------------------------------------------
# Saga Step
# ---------------------------------------------------------------------------

class SagaStep(ABC):
    """Abstract base for a saga step.

    Each step defines an action and an compensation action.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = SagaStepStatus.PENDING
        self.result: Optional[SagaStepResult] = None

    @abstractmethod
    async def execute(self, context: SagaContext) -> SagaStepResult:
        """Execute the step. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def compensate(self, context: SagaContext) -> SagaStepResult:
        """Compensate (undo) the step. Must be implemented by subclasses."""
        pass


class FunctionSagaStep(SagaStep):
    """A saga step backed by callable functions."""

    def __init__(
        self,
        name: str,
        action: Callable[[SagaContext], Coroutine[Any, Any, SagaStepResult]],
        compensation: Callable[[SagaContext], Coroutine[Any, Any, SagaStepResult]],
    ) -> None:
        super().__init__(name)
        self._action = action
        self._compensation = compensation

    async def execute(self, context: SagaContext) -> SagaStepResult:
        return await self._action(context)

    async def compensate(self, context: SagaContext) -> SagaStepResult:
        return await self._compensation(context)


# ---------------------------------------------------------------------------
# Saga Definition
# ---------------------------------------------------------------------------

class Saga:
    """A saga defines a sequence of steps that form a distributed transaction.

    Usage::

        saga = Saga("annotate_video")
        saga.add_step(load_video_step)
        saga.add_step(detect_scenes_step)
        saga.add_step(annotate_segments_step)
        saga.add_step(save_annotation_step)

        result = await orchestrator.execute(saga, context)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: List[SagaStep] = []
        self._current_index = 0

    def add_step(self, step: SagaStep) -> "Saga":
        """Add a step to the saga."""
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Saga Orchestrator
# ---------------------------------------------------------------------------

class SagaOrchestrator:
    """Orchestrates saga execution with compensation support.

    Manages the lifecycle of sagas including execution, failure handling,
    and compensation.
    """

    def __init__(self) -> None:
        self._active_sagas: Dict[str, Saga] = {}
        self._saga_status: Dict[str, SagaStatus] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        saga: Saga,
        context: Optional[SagaContext] = None,
        timeout: Optional[float] = None,
        on_failure: Optional[callable] = None,
    ) -> SagaContext:
        """Execute a saga.

        Args:
            saga: The saga to execute
            context: Optional initial context
            timeout: Optional timeout in seconds
            on_failure: Optional callback called when saga fails.
                       Receives (saga_id, failed_step_name, context) as args.

        Returns:
            Final saga context

        Raises:
            SagaExecutionError: If saga execution fails
        """
        saga_id = str(uuid.uuid4())[:8]
        ctx = context or SagaContext(saga_id=saga_id)
        ctx.saga_id = saga_id

        self._active_sagas[saga_id] = saga
        self._saga_status[saga_id] = SagaStatus.RUNNING

        logger.info(
            "saga_started",
            saga_id=saga_id,
            saga_name=saga.name,
            steps=len(saga),
        )

        completed_steps: List[SagaStep] = []

        try:
            for step in saga.steps:
                step.status = SagaStepStatus.RUNNING

                start = time.perf_counter()
                try:
                    if timeout:
                        result = await asyncio.wait_for(step.execute(ctx), timeout=timeout)
                    else:
                        result = await step.execute(ctx)
                except asyncio.TimeoutError:
                    result = SagaStepResult(
                        success=False,
                        error=f"Step '{step.name}' timed out after {timeout}s",
                    )

                step_latency = (time.perf_counter() - start) * 1000
                result.latency_ms = step_latency
                step.result = result
                ctx.results.append(result)

                if not result.success:
                    step.status = SagaStepStatus.FAILED
                    logger.error(
                        "saga_step_failed",
                        saga_id=saga_id,
                        step=step.name,
                        error=result.error,
                    )

                    # Compensate completed steps in reverse order
                    await self._compensate(completed_steps, ctx)
                    self._saga_status[saga_id] = SagaStatus.FAILED

                    # Call failure callback if provided (for checkpoint coordination)
                    if on_failure:
                        try:
                            await on_failure(saga_id, step.name, ctx)
                        except Exception as callback_err:
                            logger.error(
                                "saga_failure_callback_error",
                                saga_id=saga_id,
                                error=str(callback_err),
                            )

                    raise SagaExecutionError(
                        f"Saga '{saga.name}' failed at step '{step.name}': {result.error}",
                        saga_id=saga_id,
                        failed_step=step.name,
                    )

                step.status = SagaStepStatus.COMPLETED
                completed_steps.append(step)
                logger.info(
                    "saga_step_completed",
                    saga_id=saga_id,
                    step=step.name,
                    latency_ms=step_latency,
                )

            self._saga_status[saga_id] = SagaStatus.COMPLETED
            logger.info("saga_completed", saga_id=saga_id, saga_name=saga.name)
            return ctx

        except SagaExecutionError:
            raise
        except Exception as e:
            logger.error("saga_unexpected_error", saga_id=saga_id, error=str(e))
            await self._compensate(completed_steps, ctx)
            self._saga_status[saga_id] = SagaStatus.FAILED

            # Call failure callback if provided
            if on_failure:
                try:
                    await on_failure(saga_id, "unexpected", ctx)
                except Exception as callback_err:
                    logger.error(
                        "saga_failure_callback_error",
                        saga_id=saga_id,
                        error=str(callback_err),
                    )

            raise SagaExecutionError(
                f"Saga '{saga.name}' failed unexpectedly: {e}",
                saga_id=saga_id,
            ) from e

    async def _compensate(
        self,
        steps: List[SagaStep],
        context: SagaContext,
    ) -> None:
        """Run compensation for completed steps in reverse order."""
        self._saga_status[context.saga_id] = SagaStatus.COMPENSATING

        for step in reversed(steps):
            if step.status != SagaStepStatus.COMPLETED:
                continue

            step.status = SagaStepStatus.COMPENSATING
            logger.info("saga_compensating", saga_id=context.saga_id, step=step.name)

            try:
                result = await step.compensate(context)
                if result.success:
                    step.status = SagaStepStatus.COMPENSATED
                    logger.info(
                        "saga_compensated",
                        saga_id=context.saga_id,
                        step=step.name,
                    )
                else:
                    logger.error(
                        "saga_compensation_failed",
                        saga_id=context.saga_id,
                        step=step.name,
                        error=result.error,
                    )
            except Exception as e:
                logger.error(
                    "saga_compensation_error",
                    saga_id=context.saga_id,
                    step=step.name,
                    error=str(e),
                )

        self._saga_status[context.saga_id] = SagaStatus.COMPENSATED

    def get_status(self, saga_id: str) -> Optional[SagaStatus]:
        """Get the status of a saga by ID."""
        return self._saga_status.get(saga_id)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SagaExecutionError(Exception):
    """Raised when saga execution fails."""

    def __init__(
        self,
        message: str,
        saga_id: Optional[str] = None,
        failed_step: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.saga_id = saga_id
        self.failed_step = failed_step


# ---------------------------------------------------------------------------
# Annotation Pipeline Saga Builder
# ---------------------------------------------------------------------------

class AnnotationSagaBuilder:
    """Builder for creating annotation pipeline sagas.

    Encapsulates the saga definition for the annotation pipeline,
    making it easy to create and customize.
    """

    @staticmethod
    def create_annotation_saga(
        video_path: str,
        video_id: str,
        teacher_model: Any,
        store: Any,
    ) -> Saga:
        """Create a saga for the annotation pipeline.

        Args:
            video_path: Path to the video file
            video_id: Unique video identifier
            teacher_model: Teacher model instance
            store: Annotation store instance

        Returns:
            Configured saga ready for execution
        """
        saga = Saga(f"annotate_{video_id}")

        # Step 1: Load video
        saga.add_step(
            FunctionSagaStep(
                name="load_video",
                action=AnnotationSagaBuilder._load_video_action(video_path, video_id),
                compensation=AnnotationSagaBuilder._noop_compensation(),
            )
        )

        # Step 2: Detect scenes
        saga.add_step(
            FunctionSagaStep(
                name="detect_scenes",
                action=AnnotationSagaBuilder._detect_scenes_action(),
                compensation=AnnotationSagaBuilder._noop_compensation(),
            )
        )

        # Step 3: Annotate segments
        saga.add_step(
            FunctionSagaStep(
                name="annotate_segments",
                action=AnnotationSagaBuilder._annotate_action(teacher_model),
                compensation=AnnotationSagaBuilder._noop_compensation(),
            )
        )

        # Step 4: Save annotation
        saga.add_step(
            FunctionSagaStep(
                name="save_annotation",
                action=AnnotationSagaBuilder._save_action(store),
                compensation=AnnotationSagaBuilder._delete_annotation_action(store),
            )
        )

        return saga

    @staticmethod
    def _load_video_action(video_path: str, video_id: str) -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            from dvas.data.video_loader import VideoLoader

            try:
                loader = VideoLoader(video_path)
                ctx.set("loader", loader)
                ctx.set("video_id", video_id)
                ctx.set("metadata", loader.metadata)
                return SagaStepResult(success=True, data={"video_path": video_path})
            except Exception as e:
                return SagaStepResult(success=False, error=str(e))

        return action

    @staticmethod
    def _detect_scenes_action() -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            try:
                loader = ctx.get("loader")
                if loader is None:
                    return SagaStepResult(success=False, error="No video loader available")

                scenes = loader.detect_scenes()
                ctx.set("scenes", scenes)
                return SagaStepResult(success=True, data={"num_scenes": len(scenes)})
            except Exception as e:
                return SagaStepResult(success=False, error=str(e))

        return action

    @staticmethod
    def _annotate_action(teacher_model: Any) -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            try:
                scenes = ctx.get("scenes", [])
                if not scenes:
                    return SagaStepResult(success=False, error="No scenes to annotate")

                # This would call the teacher model for each scene
                # Simplified for illustration
                annotations = []
                ctx.set("annotations", annotations)
                return SagaStepResult(success=True, data={"annotations": len(annotations)})
            except Exception as e:
                return SagaStepResult(success=False, error=str(e))

        return action

    @staticmethod
    def _save_action(store: Any) -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            try:
                annotations = ctx.get("annotations")
                if annotations is None:
                    return SagaStepResult(success=False, error="No annotations to save")

                # Save logic here
                return SagaStepResult(success=True)
            except Exception as e:
                return SagaStepResult(success=False, error=str(e))

        return action

    @staticmethod
    def _delete_annotation_action(store: Any) -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            # Delete the saved annotation if it exists
            return SagaStepResult(success=True)

        return action

    @staticmethod
    def _noop_compensation() -> Callable:
        async def action(ctx: SagaContext) -> SagaStepResult:
            return SagaStepResult(success=True)

        return action
