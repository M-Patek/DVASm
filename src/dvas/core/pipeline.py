"""Pipeline stage pattern for extensible, composable processing pipelines.

Provides a plugin-based architecture where pipeline stages can be
registered, composed, and executed with support for branching,
parallel execution, and conditional processing.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Pipeline types
# ---------------------------------------------------------------------------

class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    SKIPPED = auto()
    FAILED = auto()


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""

    success: bool
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.success


class PipelineContext:
    """Shared context passed through pipeline stages.

    Stages can read from and write to this context to share data.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = initial_data or {}
        self._stage_results: Dict[str, StageResult] = {}
        self._execution_order: List[str] = []

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self._data.get(key, default)

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with a dictionary."""
        self._data.update(data)

    def record_result(self, stage_name: str, result: StageResult) -> None:
        """Record the result of a stage execution."""
        self._stage_results[stage_name] = result
        self._execution_order.append(stage_name)

    def get_result(self, stage_name: str) -> Optional[StageResult]:
        """Get the result of a specific stage."""
        return self._stage_results.get(stage_name)

    @property
    def all_results(self) -> Dict[str, StageResult]:
        """Get all stage results."""
        return self._stage_results.copy()

    @property
    def execution_order(self) -> List[str]:
        """Get the order in which stages were executed."""
        return self._execution_order.copy()


# ---------------------------------------------------------------------------
# Stage Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for pipeline stages.

    All pipeline stages must implement this protocol.
    """

    @property
    def name(self) -> str:
        """Unique name for this stage."""
        ...

    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage."""
        ...

    def should_run(self, context: PipelineContext) -> bool:
        """Check if this stage should run given the current context."""
        ...


# ---------------------------------------------------------------------------
# Base implementations
# ---------------------------------------------------------------------------

class BaseStage(ABC):
    """Abstract base class for pipeline stages.

    Provides common functionality like timing, logging, and error handling.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self.status = StageStatus.PENDING

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def _execute(self, context: PipelineContext) -> StageResult:
        """Override this method to implement stage logic."""
        pass

    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage with timing and error handling."""
        self.status = StageStatus.RUNNING
        start = time.perf_counter()

        try:
            result = await self._execute(context)
            self.status = StageStatus.COMPLETED if result.success else StageStatus.FAILED
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result

        except Exception as e:
            self.status = StageStatus.FAILED
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("stage_execution_failed", stage=self.name, error=str(e))
            return StageResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def should_run(self, context: PipelineContext) -> bool:
        """Override to add conditional execution logic."""
        return True


class FunctionStage(BaseStage):
    """A pipeline stage backed by a function.

    Usage::

        async def my_stage(ctx: PipelineContext) -> StageResult:
            # Do work
            return StageResult(success=True, data={"key": "value"})

        stage = FunctionStage("my_stage", my_stage)
    """

    def __init__(
        self,
        name: str,
        func: Callable[[PipelineContext], Coroutine[Any, Any, StageResult]],
    ) -> None:
        super().__init__(name)
        self._func = func

    async def _execute(self, context: PipelineContext) -> StageResult:
        return await self._func(context)


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

class Pipeline:
    """A pipeline composed of stages.

    Stages are executed sequentially by default, but branching and
    parallel execution are also supported.

    Usage::

        pipeline = Pipeline("annotate_video")
        pipeline.add_stage(load_stage)
        pipeline.add_stage(detect_stage)
        pipeline.add_stage(annotate_stage)
        pipeline.add_stage(save_stage)

        result = await pipeline.execute(context)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._stages: List[PipelineStage] = []
        self._error_handler: Optional[Callable[[Exception, PipelineContext], None]] = None

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add a stage to the pipeline."""
        self._stages.append(stage)
        return self

    def on_error(
        self,
        handler: Callable[[Exception, PipelineContext], None],
    ) -> "Pipeline":
        """Set an error handler for the pipeline."""
        self._error_handler = handler
        return self

    async def execute(self, context: Optional[PipelineContext] = None) -> PipelineContext:
        """Execute all stages in the pipeline.

        Args:
            context: Optional initial context

        Returns:
            Final pipeline context with all results
        """
        ctx = context or PipelineContext()
        logger.info("pipeline_started", name=self.name)

        for stage in self._stages:
            # Check if stage should run
            if not stage.should_run(ctx):
                logger.debug("stage_skipped", stage=stage.name)
                ctx.record_result(stage.name, StageResult(success=True, metadata={"skipped": True}))
                continue

            # Execute stage
            result = await stage.execute(ctx)
            ctx.record_result(stage.name, result)

            # Handle failure
            if not result.success:
                logger.error(
                    "pipeline_stage_failed",
                    pipeline=self.name,
                    stage=stage.name,
                    error=result.error,
                )
                if self._error_handler:
                    try:
                        self._error_handler(Exception(result.error or "Stage failed"), ctx)
                    except Exception as e:
                        logger.error("error_handler_failed", error=str(e))
                break

        logger.info("pipeline_completed", name=self.name, stages_executed=len(ctx.execution_order))
        return ctx

    async def execute_parallel(
        self,
        context: Optional[PipelineContext] = None,
        max_concurrent: int = 4,
    ) -> PipelineContext:
        """Execute stages in parallel where possible.

        Stages that don't depend on each other can run concurrently.
        This is a simplified version - full dependency graph support
        would require a DAG executor.
        """
        ctx = context or PipelineContext()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_stage(stage: PipelineStage) -> None:
            async with semaphore:
                result = await stage.execute(ctx)
                ctx.record_result(stage.name, result)

        # Execute all stages concurrently (simplified)
        tasks = [asyncio.create_task(_run_stage(stage)) for stage in self._stages]
        await asyncio.gather(*tasks, return_exceptions=True)

        return ctx


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------

class StageRegistry:
    """Registry for pipeline stages.

    Allows stages to be registered and retrieved by name,
    enabling plugin-based architectures.
    """

    _stages: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, stage_class: type) -> None:
        """Register a stage class."""
        cls._stages[name] = stage_class
        logger.debug("stage_registered", name=name, class_=stage_class.__name__)

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a registered stage class."""
        return cls._stages.get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Optional[PipelineStage]:
        """Create an instance of a registered stage."""
        stage_class = cls._stages.get(name)
        if stage_class:
            return stage_class(**kwargs)
        return None

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all registered stage names."""
        return list(cls._stages.keys())


# ---------------------------------------------------------------------------
# Pre-built stages for annotation pipeline
# ---------------------------------------------------------------------------

class LoadVideoStage(BaseStage):
    """Pipeline stage for loading video."""

    def __init__(self, video_path: str) -> None:
        super().__init__("load_video")
        self.video_path = video_path

    async def _execute(self, context: PipelineContext) -> StageResult:
        from dvas.data.video_loader import VideoLoader

        try:
            loader = VideoLoader(self.video_path)
            context.set("loader", loader)
            context.set("metadata", loader.metadata)
            return StageResult(
                success=True,
                data={"video_path": self.video_path, "duration": loader.metadata.duration},
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))


class DetectScenesStage(BaseStage):
    """Pipeline stage for scene detection."""

    def __init__(self, threshold: float = 30.0, max_scenes: int = 50) -> None:
        super().__init__("detect_scenes")
        self.threshold = threshold
        self.max_scenes = max_scenes

    async def _execute(self, context: PipelineContext) -> StageResult:
        try:
            loader = context.get("loader")
            if loader is None:
                return StageResult(success=False, error="No video loader available")

            scenes = loader.detect_scenes(
                threshold=self.threshold,
                max_scenes=self.max_scenes,
            )
            context.set("scenes", scenes)
            return StageResult(success=True, data={"num_scenes": len(scenes)})
        except Exception as e:
            return StageResult(success=False, error=str(e))


class AnnotateSegmentsStage(BaseStage):
    """Pipeline stage for annotating video segments."""

    def __init__(self, teacher_model: Any, num_frames: int = 16) -> None:
        super().__init__("annotate_segments")
        self.teacher_model = teacher_model
        self.num_frames = num_frames

    async def _execute(self, context: PipelineContext) -> StageResult:
        try:
            scenes = context.get("scenes", [])
            if not scenes:
                return StageResult(success=False, error="No scenes to annotate")

            annotations = []
            for scene in scenes:
                # Simplified: actual implementation would call teacher model
                annotations.append({"scene": scene, "annotation": "placeholder"})

            context.set("annotations", annotations)
            return StageResult(success=True, data={"annotations": len(annotations)})
        except Exception as e:
            return StageResult(success=False, error=str(e))


class SaveAnnotationStage(BaseStage):
    """Pipeline stage for saving annotations."""

    def __init__(self, store: Any) -> None:
        super().__init__("save_annotation")
        self.store = store

    async def _execute(self, context: PipelineContext) -> StageResult:
        try:
            annotations = context.get("annotations")
            if annotations is None:
                return StageResult(success=False, error="No annotations to save")

            # Save logic here
            return StageResult(success=True)
        except Exception as e:
            return StageResult(success=False, error=str(e))
