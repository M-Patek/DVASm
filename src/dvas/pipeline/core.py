"""Annotation pipeline - thin orchestration layer.

Delegates all work to focused components:
- CheckpointManager: checkpoint persistence
- StructuredParser: response parsing
- AnnotationBuilder: annotation construction
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dvas.core.concurrency import AsyncBatchProcessor
from dvas.data.schemas import Annotation, Segment, VideoMetadata
from dvas.data.storage import AnnotationStore
from dvas.data.video_loader import Frame, VideoLoader
from dvas.exceptions import PipelineError
from dvas.models.base import GenerationResult
from dvas.pipeline.builder import AnnotationBuilder
from dvas.pipeline.checkpoint import CheckpointManager
from dvas.pipeline.parser import StructuredParser
from dvas.utils.logging import get_logger
from dvas.utils.retry import with_retry

if TYPE_CHECKING:
    from dvas.models.teacher.base import TeacherModel

logger = get_logger(__name__)


class AnnotationPipeline:
    """Thin orchestrator for video annotation.

    Pipeline does not contain business logic. It only coordinates
    between specialized components.
    """

    def __init__(
        self,
        teacher_model: Optional["TeacherModel"] = None,
        store: Optional[AnnotationStore] = None,
        num_frames: int = 16,
        segment_duration: float = 5.0,
        checkpoint_path: Optional[Path] = None,
    ):
        """Initialize the annotation pipeline.

        Args:
            teacher_model: Teacher model for generating annotations
            store: Annotation storage backend
            num_frames: Number of frames to sample per segment
            segment_duration: Duration of each segment in seconds
            checkpoint_path: Path for checkpoint persistence
        """
        self._teacher = teacher_model
        self.store = store or AnnotationStore()
        self.num_frames = num_frames
        self.segment_duration = segment_duration
        self.checkpoint = CheckpointManager(checkpoint_path) if checkpoint_path else None
        self.parser = StructuredParser()
        self.builder = AnnotationBuilder(
            model_version=getattr(teacher_model, "model_name", "unknown")
        )

    @property
    def teacher(self) -> "TeacherModel":
        """Lazy-load teacher model if not provided."""
        if self._teacher is None:
            from dvas.config import settings
            from dvas.models.teacher import TeacherModel

            self._teacher = TeacherModel(model_name=settings.DEFAULT_TEACHER_MODEL)
            self.builder.model_version = self._teacher.model_name
        return self._teacher

    @teacher.setter
    def teacher(self, value: "TeacherModel") -> None:
        self._teacher = value
        self.builder.model_version = getattr(value, "model_name", "unknown")

    @with_retry(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        exceptions=(ConnectionError, TimeoutError, OSError),
    )
    async def annotate_video(
        self,
        video_path: Path,
        video_id: str,
    ) -> Annotation:
        """Generate complete annotation for a video.

        Args:
            video_path: Path to the video file
            video_id: Unique identifier for the video

        Returns:
            Complete annotation with segments and metadata
        """
        logger.info("annotation_starting", video_id=video_id, path=str(video_path))

        # Load checkpoint if exists
        if self.checkpoint:
            self.checkpoint.load()

        # Check checkpoint - verify storage exists
        if self.checkpoint and self.checkpoint.is_processed(video_id):
            logger.info("video_already_processed", video_id=video_id)
            existing = self.store.load(f"{video_id}_annotated", source="gold")
            if existing:
                return existing
            # Checkpoint says processed but storage missing - inconsistency detected
            logger.warning(
                "checkpoint_storage_inconsistency",
                video_id=video_id,
                message="Checkpoint indicates processed but annotation not found in storage. Reprocessing.",
            )
            self.checkpoint.processed_ids.discard(video_id)

        # Step 1: Load video and detect scenes
        segments: List[Segment] = []
        metadata: Optional[VideoMetadata] = None

        with VideoLoader(video_path) as loader:
            metadata = loader.metadata
            scenes = loader.detect_scenes(min_duration=1.0, max_scenes=20)
            logger.info("scenes_detected", video_id=video_id, num_scenes=len(scenes))

            # Step 2: Annotate each scene
            for i, (start, end) in enumerate(scenes):
                try:
                    segment = await self._annotate_segment(
                        loader=loader,
                        start_time=start,
                        end_time=end,
                    )
                    segments.append(segment)
                except Exception as e:
                    logger.error(
                        "segment_annotation_failed",
                        video_id=video_id,
                        segment=i,
                        error=str(e),
                    )
                    segments.append(self.builder.build_empty_segment(start, end, str(e)))

        if not metadata:
            raise RuntimeError("Failed to extract video metadata")

        # Step 3: Build and save annotation
        annotation = self.builder.build_annotation(
            video_id=video_id,
            video_path=str(video_path),
            segments=segments,
            metadata=metadata,
        )

        # Atomic save + checkpoint update
        try:
            self.store.save(annotation, source="gold")
            # Only mark checkpoint after successful save
            if self.checkpoint:
                self.checkpoint.mark_processed(video_id)
                self.checkpoint.save()
        except Exception as e:
            logger.error(
                "annotation_save_failed",
                video_id=video_id,
                error=str(e),
            )
            # If save failed, don't mark checkpoint
            raise PipelineError(
                f"Failed to save annotation for {video_id}",
                stage="save",
                details={"original_error": str(e)},
            ) from e

        logger.info(
            "annotation_completed",
            video_id=video_id,
            num_segments=len(segments),
        )

        return annotation

    async def _annotate_segment(
        self,
        loader: VideoLoader,
        start_time: float,
        end_time: float,
    ) -> Segment:
        """Annotate a single segment."""
        # Collect frames
        frames: List[Frame] = []
        async for frame in loader.aiter_frames(
            start_time=start_time,
            end_time=end_time,
            num_frames=self.num_frames,
        ):
            frames.append(frame)

        if not frames:
            return self.builder.build_empty_segment(start_time, end_time, "no_frames_extracted")

        frame_arrays = [f.data for f in frames]

        # Call teacher with retry
        result = await self._call_teacher_with_retry(frame_arrays)

        # Handle different result statuses
        if result.is_recoverable():
            # Timeout or rate limit - should trigger circuit breaker
            error_msg = result.error_message or f"teacher_{result.status.value}"
            logger.warning(
                "teacher_recoverable_error",
                status=result.status.value,
                latency_ms=result.latency_ms,
            )
            return self.builder.build_empty_segment(start_time, end_time, error_msg)

        if result.is_failure():
            return self.builder.build_empty_segment(
                start_time, end_time, result.error_message or "teacher_failed"
            )

        # TIMEOUT/RATE_LIMITED already handled, but check for other non-success
        if not result.is_success() and not result.is_fallback():
            return self.builder.build_empty_segment(
                start_time, end_time, f"teacher_{result.status.value}"
            )

        response_text = result.text

        # Parse response
        parsed = self._parse_response(response_text)

        # Build segment
        return self.builder.build_segment(
            start_time=start_time,
            end_time=end_time,
            response_text=response_text,
            parsed=parsed,
        )

    @with_retry(
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        exceptions=(ConnectionError, TimeoutError, OSError),
    )
    async def _call_teacher_with_retry(self, frame_arrays: List[Any]) -> GenerationResult:
        """Call teacher model with retry logic."""
        return await self.teacher.generate(frames=frame_arrays, task="fine_grained")

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse model response using structured parser."""
        parsed = self.parser.parse(text)
        return self.parser.to_legacy_dict(parsed)

    async def process_batch(
        self,
        video_items: List[Dict[str, Any]],
        max_concurrent: int = 5,
        checkpoint_every: int = 10,
    ) -> Tuple[List[Annotation], List[Dict]]:
        """Process multiple videos with concurrency control and streaming results.

        Uses AsyncBatchProcessor for controlled concurrency with proper
        backpressure and error isolation.

        Args:
            video_items: List of dicts with 'video_path' and 'video_id'
            max_concurrent: Max concurrent annotation tasks
            checkpoint_every: Save checkpoint every N items

        Returns:
            Tuple of (successful_annotations, failed_items)
        """
        from dvas.utils.retry import BatchProcessor

        # Load checkpoint once at the start of the batch so resume works.
        if self.checkpoint:
            self.checkpoint.load()

        # Filter out videos we've already processed in a previous run.
        skipped: List[Dict] = []
        items_to_run: List[Dict] = []
        for item in video_items:
            video_id = item.get("video_id", "")
            if self.checkpoint and self.checkpoint.is_processed(video_id):
                skipped.append(item)
            else:
                items_to_run.append(item)

        if skipped:
            logger.info(
                "process_batch_skipping_processed",
                skipped=len(skipped),
                remaining=len(items_to_run),
            )

        successful: List[Annotation] = []
        failed: List[Dict] = []
        processed_count = 0

        batch_processor = None
        if self.checkpoint:
            batch_processor = BatchProcessor(
                checkpoint_path=self.checkpoint.checkpoint_path.parent / "batch_checkpoint.json",
                batch_size=checkpoint_every,
            )

        async def process_one(item: Dict) -> Optional[Annotation]:
            try:
                return await self.annotate_video(
                    video_path=Path(item["video_path"]),
                    video_id=item["video_id"],
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error(
                    "batch_processing_failed",
                    video_id=item.get("video_id"),
                    error=str(e),
                )
                if self.checkpoint:
                    self.checkpoint.mark_failed(item.get("video_id", "unknown"), str(e))
                return None

        # Use AsyncBatchProcessor for controlled concurrency with backpressure
        batch = AsyncBatchProcessor[Dict](max_concurrent=max_concurrent)
        results = await batch.process(
            items_to_run,
            process_one,
            on_error=lambda item, err: logger.error(
                "batch_item_failed",
                video_id=item.get("video_id"),
                error=str(err),
            ),
        )

        # Collect results
        for i, result in enumerate(results):
            processed_count += 1
            video_id = (
                items_to_run[i].get("video_id", "unknown") if i < len(items_to_run) else "unknown"
            )
            if isinstance(result, Annotation):
                successful.append(result)
                # Mark the per-video checkpoint so resume skips it next run.
                # annotate_video also marks it on the success path, but that's
                # only reached on the inner save success — a video_id that
                # bypasses annotate_video (e.g. mocked in tests, or short-circuited
                # by a future orchestrator) would otherwise be silently lost.
                if self.checkpoint:
                    self.checkpoint.mark_processed(video_id)
                if batch_processor:
                    batch_processor.mark_processed(f"item_{processed_count}")
            elif result is None:
                failed_item = items_to_run[i] if i < len(items_to_run) else {}
                failed.append({"item": failed_item, "error": "processing_failed"})
                if batch_processor:
                    batch_processor.mark_failed(f"item_{processed_count}", "processing_failed")

            # Save checkpoint periodically
            if processed_count % checkpoint_every == 0 and self.checkpoint:
                self.checkpoint.save()
                if batch_processor:
                    batch_processor._save_checkpoint()
                logger.info(
                    "batch_checkpoint_saved",
                    processed=processed_count,
                    total=len(video_items),
                )

        # Final flush: ensure the per-video checkpoint reflects this batch's
        # successes even if the periodic save didn't fire.
        if self.checkpoint:
            self.checkpoint.save()

        # Hydrate skipped items from the store so callers receive the full
        # set of annotations matching the input items.
        for item in skipped:
            video_id = item.get("video_id", "")
            existing = self.store.load(f"{video_id}_annotated", source="gold")
            if existing is not None:
                successful.append(existing)

        return successful, failed


class EPICAnnotationPipeline(AnnotationPipeline):
    """Specialized pipeline for EPIC-KITCHENS dataset."""

    def __init__(self, epic_root: Path, **kwargs: Any) -> None:
        """Initialize EPIC-KITCHENS pipeline.

        Args:
            epic_root: Root directory of EPIC-KITCHENS dataset
            **kwargs: Passed to AnnotationPipeline.__init__
        """
        super().__init__(**kwargs)
        from dvas.data.video_loader import EPICKitchensLoader

        self.epic_loader = EPICKitchensLoader(epic_root)

    async def annotate_split(
        self,
        split: str = "train",
        max_videos: int = 1000,
        participant: Optional[str] = None,
    ) -> Tuple[List[Annotation], List[Dict]]:
        """Annotate videos from an EPIC-KITCHENS split."""
        import pandas as pd

        split_file = self.epic_loader.root_path / f"EPIC_100_{split}.csv"
        df = pd.read_csv(split_file)

        if participant:
            df = df[df["participant_id"] == participant]

        video_ids = df["video_id"].unique()[:max_videos]

        if self.checkpoint:
            video_ids = [vid for vid in video_ids if not self.checkpoint.is_processed(vid)]
            logger.info("filtered_processed_videos", remaining=len(video_ids))

        items = []
        for vid in video_ids:
            video_path = self.epic_loader.get_video_path(vid)
            if video_path and video_path.exists():
                items.append({"video_id": vid, "video_path": video_path})

        return await self.process_batch(items)


def create_training_data_from_gold(
    store: AnnotationStore,
    output_path: Path,
    format: str = "llava",
) -> int:
    """Export gold annotations as training dataset."""
    annotations = store.load_all(source="gold")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            if format == "llava":
                data = ann.to_llava_format()
            elif format == "openai":
                data = ann.to_openai_format()
            else:
                import json as _json

                data = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a video understanding AI.",
                        },
                        {
                            "role": "user",
                            "content": f"<video> {ann.video_path}\nAnalyze.",
                        },
                        {
                            "role": "assistant",
                            "content": ann.segments[0].caption_dense if ann.segments else "",
                        },
                    ]
                }

            import json as _json

            f.write(_json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    logger.info("training_data_exported", count=count, format=format, path=str(output_path))
    return count
