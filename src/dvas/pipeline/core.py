"""Annotation pipeline with retry and checkpoint support."""

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dvas.data.schemas import (
    Action,
    Annotation,
    Hand,
    Object,
    QAPair,
    Segment,
    VideoMetadata,
)
from dvas.data.storage import AnnotationStore
from dvas.data.video_loader import Frame, VideoLoader
from dvas.utils.logging import get_logger
from dvas.utils.retry import with_retry

if TYPE_CHECKING:
    from dvas.models.teacher.base import TeacherModel

logger = get_logger(__name__)


class PipelineCheckpoint:
    """Checkpoint for resumable pipeline processing."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_ids: set = set()
        self.failed_items: List[Dict] = []

    def load(self) -> bool:
        """Load checkpoint if exists."""
        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            self.processed_ids = set(data.get("processed", []))
            self.failed_items = data.get("failed", [])
            logger.info(
                "checkpoint_loaded",
                processed=len(self.processed_ids),
                failed=len(self.failed_items),
            )
            return True
        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            return False

    def save(self) -> None:
        """Save current checkpoint."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(
                {
                    "processed": list(self.processed_ids),
                    "failed": self.failed_items,
                },
                f,
            )

    def mark_processed(self, video_id: str) -> None:
        """Mark video as processed."""
        self.processed_ids.add(video_id)

    def mark_failed(self, video_id: str, error: str) -> None:
        """Mark video as failed."""
        self.failed_items.append({"video_id": video_id, "error": error})


class AnnotationPipeline:
    """Pipeline for generating fine-grained temporal annotations with retry."""

    def __init__(
        self,
        teacher_model: Optional["TeacherModel"] = None,
        store: Optional[AnnotationStore] = None,
        num_frames: int = 16,
        segment_duration: float = 5.0,
        checkpoint_path: Optional[Path] = None,
    ):
        self._teacher = teacher_model
        self.store = store or AnnotationStore()
        self.num_frames = num_frames
        self.segment_duration = segment_duration
        self.checkpoint = (
            PipelineCheckpoint(checkpoint_path) if checkpoint_path else None
        )

    @property
    def teacher(self) -> "TeacherModel":
        if self._teacher is None:
            from dvas.models.teacher.gpt4v import GPT4VTeacher
            self._teacher = GPT4VTeacher()
        return self._teacher

    @teacher.setter
    def teacher(self, value: "TeacherModel") -> None:
        self._teacher = value

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
        """Generate complete annotation for a video with retry."""
        logger.info(
            "annotation_starting",
            video_id=video_id,
            path=str(video_path),
        )

        # Check if already processed
        if self.checkpoint and video_id in self.checkpoint.processed_ids:
            logger.info("video_already_processed", video_id=video_id)
            existing = self.store.load(f"{video_id}_annotated", source="gold")
            if existing:
                return existing

        segments: List[Segment] = []
        metadata: Optional[VideoMetadata] = None

        with VideoLoader(video_path) as loader:
            metadata = loader.metadata

            # Detect scenes using optimized method
            scenes = loader.detect_scenes(min_duration=1.0, max_scenes=20)
            logger.info(
                "scenes_detected",
                video_id=video_id,
                num_scenes=len(scenes),
            )

            # Process each scene
            for i, (start, end) in enumerate(scenes):
                try:
                    segment = await self._annotate_segment_streaming(
                        loader=loader,
                        start_time=start,
                        end_time=end,
                        segment_idx=i,
                    )
                    segments.append(segment)
                except Exception as e:
                    logger.error(
                        "segment_annotation_failed",
                        video_id=video_id,
                        segment=i,
                        error=str(e),
                    )
                    # Continue with other segments

        if not metadata:
            raise RuntimeError("Failed to extract video metadata")

        # Build complete annotation
        annotation = Annotation(
            id=f"{video_id}_annotated",
            video_id=video_id,
            video_path=str(video_path),
            segments=segments,
            metadata=metadata,
            source="teacher",
            model_version=self.teacher.model_name,
        )

        # Save to store
        self.store.save(annotation, source="gold")

        # Update checkpoint
        if self.checkpoint:
            self.checkpoint.mark_processed(video_id)
            self.checkpoint.save()

        logger.info(
            "annotation_completed",
            video_id=video_id,
            num_segments=len(segments),
        )

        return annotation

    async def _annotate_segment_streaming(
        self,
        loader: VideoLoader,
        start_time: float,
        end_time: float,
        segment_idx: int = 0,
    ) -> Segment:
        """Annotate a single segment using streaming frames."""
        # Use streaming instead of loading all frames into memory
        frames: List[Frame] = []
        async for frame in loader.aiter_frames(
            start_time=start_time,
            end_time=end_time,
            num_frames=self.num_frames,
        ):
            frames.append(frame)

        if not frames:
            return Segment(
                start_time=start_time,
                end_time=end_time,
                caption="",
            )

        frame_arrays = [f.data for f in frames]

        # Get teacher response with retry
        @with_retry(
            max_attempts=3,
            base_delay=2.0,
            max_delay=60.0,
            exceptions=(ConnectionError, TimeoutError, OSError),
        )
        async def _call_teacher():
            return await self.teacher.annotate(
                frames=frame_arrays,
                task="fine_grained",
            )

        result = await _call_teacher()
        response_text = result.get("text", "")

        # Parse structured output
        parsed = self._parse_response(response_text)

        return Segment(
            start_time=start_time,
            end_time=end_time,
            caption=parsed.get("scene_description", ""),
            caption_dense=response_text,
            qa_pairs=parsed.get("qa_pairs", []),
            objects=parsed.get("objects", []),
            actions=parsed.get("actions", []),
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse structured output from model response."""
        result = {
            "scene_description": "",
            "qa_pairs": [],
            "objects": [],
            "actions": [],
        }

        # Try to extract JSON-like structure
        try:
            # Look for JSON block
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Extract scene description
                result["scene_description"] = data.get(
                    "scene_description", text[:500]
                )

                # Extract QA pairs
                if "steps" in data:
                    for i, step in enumerate(data["steps"][:5]):
                        result["qa_pairs"].append(
                            QAPair(
                                question=f"Step {i+1}: What action is performed?",
                                answer=step.get("action", "")
                                + " "
                                + step.get("details", ""),
                            )
                        )

                # Extract objects
                if "objects" in data:
                    for obj in data["objects"]:
                        result["objects"].append(
                            Object(
                                name=obj.get("name", "unknown"),
                                attributes={"state": obj.get("state", "")},
                            )
                        )

                # Extract actions
                if "hand_actions" in data:
                    for ha in data["hand_actions"]:
                        result["actions"].append(
                            Action(
                                verb=ha.get("action", "").split()[0],
                                noun=ha.get("target", ""),
                                hand=Hand(ha.get("hand", "unknown")),
                            )
                        )
            else:
                # No JSON block found, use plain text
                result["scene_description"] = text[:500]

        except json.JSONDecodeError:
            # If JSON parsing fails, use heuristics
            result["scene_description"] = text[:500]

        return result

    async def process_batch(
        self,
        video_items: List[Dict[str, Any]],
        max_concurrent: int = 5,
        checkpoint_every: int = 10,
    ) -> Tuple[List[Annotation], List[Dict]]:
        """Process multiple videos with concurrency control and checkpointing.

        Uses BatchProcessor for automatic checkpoint persistence and retry.

        Returns:
            Tuple of (successful annotations, failed items)
        """
        from dvas.utils.retry import BatchProcessor

        semaphore = asyncio.Semaphore(max_concurrent)
        successful: List[Annotation] = []
        failed: List[Dict] = []
        processed_count = 0

        # Initialize BatchProcessor for checkpoint persistence
        batch_processor = None
        if self.checkpoint:
            batch_processor = BatchProcessor(
                checkpoint_path=self.checkpoint.checkpoint_path.parent / "batch_checkpoint.json",
                batch_size=checkpoint_every,
            )

        async def process_one(item: Dict) -> Optional[Annotation]:
            async with semaphore:
                try:
                    result = await self.annotate_video(
                        video_path=Path(item["video_path"]),
                        video_id=item["video_id"],
                    )
                    return result
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.error(
                        "batch_processing_failed",
                        video_id=item.get("video_id"),
                        error=str(e),
                    )
                    if self.checkpoint:
                        self.checkpoint.mark_failed(
                            item.get("video_id", "unknown"), str(e)
                        )
                    return None

        # Process in chunks to save checkpoints
        for i in range(0, len(video_items), checkpoint_every):
            chunk = video_items[i : i + checkpoint_every]
            tasks = [process_one(item) for item in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in chunk_results:
                processed_count += 1
                if isinstance(result, Exception):
                    failed.append({"error": str(result)})
                    if batch_processor:
                        batch_processor.mark_failed(
                            f"item_{processed_count}", str(result)
                        )
                elif result is not None:
                    successful.append(result)
                    if batch_processor:
                        batch_processor.mark_processed(f"item_{processed_count}")

            # Save checkpoint after each chunk
            if self.checkpoint:
                self.checkpoint.save()
                if batch_processor:
                    batch_processor._save_checkpoint()
                logger.info(
                    "batch_checkpoint_saved",
                    processed=processed_count,
                    total=len(video_items),
                )

        return successful, failed


class EPICAnnotationPipeline(AnnotationPipeline):
    """Specialized pipeline for EPIC-KITCHENS dataset."""

    def __init__(self, epic_root: Path, **kwargs):
        super().__init__(**kwargs)
        from dvas.data.video_loader import EPICKitchensLoader

        self.epic_loader = EPICKitchensLoader(epic_root)

    async def annotate_split(
        self,
        split: str = "train",
        max_videos: int = 1000,
        participant: Optional[str] = None,
    ) -> Tuple[List[Annotation], List[Dict]]:
        """Annotate videos from an EPIC-KITCHENS split with checkpointing."""
        # Get video list
        import pandas as pd

        split_file = self.epic_loader.root_path / f"EPIC_100_{split}.csv"
        df = pd.read_csv(split_file)

        if participant:
            df = df[df["participant_id"] == participant]

        video_ids = df["video_id"].unique()[:max_videos]

        # Filter out already processed
        if self.checkpoint:
            video_ids = [
                vid
                for vid in video_ids
                if vid not in self.checkpoint.processed_ids
            ]
            logger.info(
                "filtered_processed_videos",
                remaining=len(video_ids),
            )

        # Build video items
        items = []
        for vid in video_ids:
            video_path = self.epic_loader.get_video_path(vid)
            if video_path and video_path.exists():
                items.append({
                    "video_id": vid,
                    "video_path": video_path,
                })

        # Process batch
        return await self.process_batch(items)


def create_training_data_from_gold(
    store: AnnotationStore,
    output_path: Path,
    format: str = "llava",
) -> int:
    """Export gold annotations as training dataset for student model."""
    annotations = store.load_all(source="gold")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            if format == "llava":
                data = ann.to_llava_format()
            elif format == "openai":
                data = ann.to_openai_format()
            else:
                # Conversational format for SFT
                data = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a video understanding AI that generates detailed annotations for robotic manipulation tasks.",
                        },
                        {
                            "role": "user",
                            "content": f"<video> {ann.video_path}\nProvide a detailed analysis of this first-person video, including hand actions, object interactions, and temporal sequence.",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                ann.segments[0].caption_dense
                                if ann.segments
                                else ""
                            ),
                        },
                    ]
                }

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    logger.info(
        "training_data_exported",
        count=count,
        format=format,
        path=str(output_path),
    )

    return count
