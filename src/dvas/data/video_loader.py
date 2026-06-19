"""Video loading and processing utilities - now a thin coordinator over focused components.

Components:
- VideoReader: raw frame reading
- FrameSampler: sampling strategies
- SceneDetector: scene boundary detection
- MotionEstimator: motion intensity estimation
"""

import logging
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import cv2

from dvas.config import settings
from dvas.data.frame_sampler import (
    FrameSampler,
    SamplerConfig,
    UniformSampler,
)
from dvas.data.motion_estimator import (
    MotionEstimator,
    MotionEstimatorRegistry,
)
from dvas.data.scene_detector import (
    SceneDetector,
    SceneDetectorRegistry,
)
from dvas.core.concurrency import AsyncIteratorBridge, run_in_thread
from dvas.data.schemas import VideoMetadata
from dvas.data.video_reader import Frame, VideoReader

logger = logging.getLogger(__name__)

# Module-level cache for video metadata with size limit
# Using simple OrderedDict-like approach for LRU eviction
_metadata_cache: Dict[Path, VideoMetadata] = {}
_metadata_cache_order: List[Path] = []
_metadata_cache_max_size: int = 1000


def _lazy_import_pandas():
    """Lazy import pandas to reduce startup time."""
    try:
        import pandas as pd

        return pd
    except ImportError:
        return None


def clear_metadata_cache() -> None:
    """Clear the video metadata cache. Call when files change on disk."""
    _metadata_cache.clear()
    _metadata_cache_order.clear()


def get_metadata_cache_size() -> int:
    """Return the number of cached metadata entries."""
    return len(_metadata_cache)


def _add_to_metadata_cache(path: Path, metadata: VideoMetadata) -> None:
    """Add entry to cache with LRU eviction."""
    global _metadata_cache, _metadata_cache_order

    if path in _metadata_cache:
        # Move to end (most recently used)
        _metadata_cache_order.remove(path)
        _metadata_cache_order.append(path)
        _metadata_cache[path] = metadata
        return

    # Evict oldest if at capacity
    if len(_metadata_cache) >= _metadata_cache_max_size:
        oldest = _metadata_cache_order.pop(0)
        del _metadata_cache[oldest]
        logger.debug("metadata_cache_evicted", path=str(oldest))

    _metadata_cache[path] = metadata
    _metadata_cache_order.append(path)


class VideoLoader:
    """Thin coordinator over focused video processing components.

    Delegates all work to specialized components:
    - VideoReader: raw frame access
    - FrameSampler: frame sampling strategies
    - SceneDetector: scene boundary detection
    - MotionEstimator: motion intensity estimation
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        sampler: Optional[FrameSampler] = None,
        scene_detector: Optional[SceneDetector] = None,
        motion_estimator: Optional[MotionEstimator] = None,
    ):
        self.video_path = Path(video_path)
        self._reader = VideoReader(video_path)
        self._sampler = sampler or UniformSampler(
            SamplerConfig(target_fps=target_fps, resize=resize)
        )
        self._scene_detector = scene_detector or SceneDetectorRegistry.create("histogram")
        self._motion_estimator = motion_estimator or MotionEstimatorRegistry.create("optical_flow")

    def __enter__(self) -> "VideoLoader":
        self._reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._reader.__exit__(exc_type, exc_val, exc_tb)

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata (cached at module level to avoid re-reading headers)."""
        # Check module-level cache first
        if self.video_path in _metadata_cache:
            # Update LRU order
            _metadata_cache_order.remove(self.video_path)
            _metadata_cache_order.append(self.video_path)
            return _metadata_cache[self.video_path]

        # Get from reader and cache
        meta = self._reader.metadata
        _add_to_metadata_cache(self.video_path, meta)
        return meta

    # --- Frame reading (delegated to VideoReader + Sampler) ---

    def iter_frames(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Iterator[Frame]:
        """Stream frames from video.

        For backward compatibility. New code should use VideoReader directly.
        """
        meta = self.metadata
        fps = meta.fps
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or meta.duration) * fps)
        end_frame = min(end_frame, meta.total_frames)
        step = step or 1

        for frame in self._reader.read_frames(start_frame, end_frame, step):
            yield frame

    def read_frames(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: Optional[int] = None,
    ) -> Iterator[Frame]:
        """Read frames with optional uniform sampling.

        For backward compatibility. New code should use FrameSampler directly.
        """
        if num_frames is None:
            yield from self.iter_frames(start_time, end_time)
            return

        # Use the configured sampler with num_frames override
        config = SamplerConfig(
            num_frames=num_frames,
            resize=self._sampler.config.resize,
        )
        sampler = UniformSampler(config)
        yield from sampler.sample(self._reader, start_time, end_time)

    def extract_frames(
        self,
        output_dir: Union[str, Path],
        num_frames: int = 8,
        format: str = "jpg",
        quality: int = 95,
    ) -> List[Path]:
        """Extract frames and save to disk efficiently."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        encode_params = []

        if format.lower() in ("jpg", "jpeg"):
            encode_ext = ".jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            encode_ext = ".png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        for frame in self.read_frames(num_frames=num_frames):
            frame_path = output_dir / f"frame_{frame.idx:06d}{encode_ext}"
            cv2.imwrite(str(frame_path), frame.data, encode_params)
            frame_paths.append(frame_path)

        return frame_paths

    # --- Scene detection (delegated to SceneDetector) ---

    def detect_scenes(
        self,
        threshold: float = 30.0,
        min_duration: float = 1.0,
        max_scenes: int = 50,
    ) -> List[Tuple[float, float]]:
        """Detect scene changes.

        For backward compatibility. New code should use SceneDetector directly.
        """
        # Create detector with current parameters
        detector = SceneDetectorRegistry.create(
            "histogram",
            threshold=threshold,
            min_duration=min_duration,
            max_scenes=max_scenes,
        )
        boundaries = detector.detect(self._reader)
        return [(b.start_time, b.end_time) for b in boundaries]

    # --- Motion estimation (delegated to MotionEstimator) ---

    def compute_motion_score(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_frames: int = 10,
    ) -> float:
        """Compute motion intensity score (0-1).

        For backward compatibility. New code should use MotionEstimator directly.
        """
        result = self._motion_estimator.estimate(self._reader, start_time, end_time, sample_frames)
        return result.score

    # --- Async streaming ---

    async def read_frames_async(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: Optional[int] = None,
    ) -> AsyncIterator[Frame]:
        """Async generator that yields frames without blocking the event loop.

        Wraps the synchronous :py:meth:`read_frames` in ``asyncio.to_thread``
        so that OpenCV I/O does not stall the async event loop.  Use this in
        async pipelines (e.g. FastAPI endpoints) instead of the sync APIs.

        Usage::

            async for frame in loader.read_frames_async(num_frames=8):
                await process(frame)
        """
        # Build the synchronous generator once; we will drive it from a
        # background thread and yield each frame asynchronously.
        sync_iter = self.read_frames(start_time, end_time, num_frames)

        # Use a sentinel that cannot be confused with a real Frame object.
        sentinel = object()

        def _next() -> Union[Frame, object]:
            """Return the next frame or *sentinel* on StopIteration."""
            try:
                return next(sync_iter)
            except StopIteration:
                return sentinel

        while True:
            frame = await run_in_thread(_next)
            if frame is sentinel:
                break
            yield frame  # type: ignore[misc]

    async def aiter_frames(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        num_frames: Optional[int] = None,
    ) -> AsyncIterator[Frame]:
        """Async iterator for frames with true streaming.

        Uses AsyncIteratorBridge to properly bridge sync frame reading
        with async consumption, avoiding the asyncio.run_coroutine_threadsafe
        anti-pattern.
        """
        # Create sync iterator
        sync_iter = self.read_frames(start_time, end_time, num_frames)

        # Use the bridge for proper async iteration
        bridge = AsyncIteratorBridge[Frame](sync_iter, queue_size=16)
        async for frame in bridge:
            yield frame

    async def extract_frames_async(
        self,
        output_dir: Union[str, Path],
        num_frames: int = 8,
        format: str = "jpg",
        quality: int = 95,
    ) -> List[Path]:
        """Extract frames asynchronously using thread pool for encoding."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        encode_params = []

        if format.lower() in ("jpg", "jpeg"):
            encode_ext = ".jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            encode_ext = ".png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        async for frame in self.aiter_frames(num_frames=num_frames):
            frame_path = output_dir / f"frame_{frame.idx:06d}{encode_ext}"
            # Offload blocking I/O to thread pool
            await run_in_thread(
                cv2.imwrite,
                str(frame_path),
                frame.data,
                encode_params,
            )
            frame_paths.append(frame_path)

        return frame_paths


class EPICKitchensLoader:
    """Optimized loader for EPIC-KITCHENS dataset."""

    def __init__(self, root_path: Optional[Union[str, Path]] = None):
        self.root_path = Path(root_path or settings.EPIC_KITCHENS_ROOT)
        if not self.root_path.exists():
            raise ValueError(f"EPIC-KITCHENS root not found: {self.root_path}")

        self._annotations: Optional[Dict] = None
        self._action_df: Optional[Any] = None

    @property
    def annotations(self) -> Dict:
        """Lazy-loaded annotations."""
        if self._annotations is None:
            self._annotations = self._load_annotations()
        return self._annotations

    def _load_annotations(self) -> Dict:
        """Load EPIC-KITCHENS annotation files (cached)."""
        pd = _lazy_import_pandas()
        if pd is None:
            return {"verbs": {}, "nouns": {}, "actions": []}

        annotations = {"verbs": {}, "nouns": {}, "actions": []}

        # Load verb classes
        verb_file = self.root_path / "EPIC_100_verb_classes.csv"
        if verb_file.exists():
            verbs_df = pd.read_csv(verb_file)
            annotations["verbs"] = dict(zip(verbs_df["id"], verbs_df["key"]))

        # Load noun classes
        noun_file = self.root_path / "EPIC_100_noun_classes.csv"
        if noun_file.exists():
            nouns_df = pd.read_csv(noun_file)
            annotations["nouns"] = dict(zip(nouns_df["id"], nouns_df["key"]))

        return annotations

    def _get_actions_df(self) -> Optional[Any]:
        """Get actions DataFrame (lazy-loaded)."""
        if self._action_df is None:
            pd = _lazy_import_pandas()
            if pd is None:
                return None

            action_file = self.root_path / "EPIC_100_train.csv"
            if action_file.exists():
                self._action_df = pd.read_csv(action_file)

        return self._action_df

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Get path to video file with extension auto-detection.

        Tries common MP4-style extensions first (most common in EPIC-KITCHENS),
        then other formats supported by OpenCV/FFmpeg. Case-insensitive
        alternates are included because EPIC-KITCHENS uses uppercase '.MP4'.
        """
        participant = video_id.split("_")[0]
        base_path = self.root_path / participant / "videos" / video_id

        # Ordered by likelihood for EPIC-KITCHENS. Each entry can have
        # a case variant because some sources mix casing.
        for ext in [
            ".MP4",
            ".mp4",
            ".mov",
            ".MOV",
            ".avi",
            ".mkv",
            ".MKV",
            ".webm",
            ".m4v",
            ".M4V",
        ]:
            path = base_path.with_suffix(ext)
            if path.exists():
                return path

        return None

    def load_video(self, video_id: str, **kwargs) -> VideoLoader:
        """Load a specific video."""
        video_path = self.get_video_path(video_id)
        if video_path is None:
            raise FileNotFoundError(f"Video not found: {video_id}")
        return VideoLoader(video_path, **kwargs)

    def get_actions_for_video(self, video_id: str) -> List[Dict]:
        """Get all actions for a specific video efficiently."""
        df = self._get_actions_df()
        if df is None:
            return []

        # Use pandas query for efficiency
        matches = df[df["video_id"] == video_id]
        return matches.to_dict("records")

    def iter_videos(
        self,
        split: str = "train",
        max_videos: Optional[int] = None,
    ) -> Iterator[Tuple[str, Path]]:
        """Iterate over all videos in a split."""
        df = self._get_actions_df()
        if df is None:
            return

        video_ids = df["video_id"].unique()
        if max_videos:
            video_ids = video_ids[:max_videos]

        for vid in video_ids:
            video_path = self.get_video_path(vid)
            if video_path and video_path.exists():
                yield vid, video_path

    def create_segment_annotations(self, video_id: str) -> List[Dict]:
        """Create segment annotations from EPIC-KITCHENS action labels."""
        actions = self.get_actions_for_video(video_id)
        verbs = self.annotations["verbs"]
        nouns = self.annotations["nouns"]

        segments = []
        for action in actions:
            verb = verbs.get(action.get("verb_class"), "unknown")
            noun = nouns.get(action.get("noun_class"), "unknown")

            segment = {
                "start_time": action.get("start_timestamp", 0),
                "end_time": action.get("stop_timestamp", 0),
                "caption": f"{verb} {noun}",
                "actions": [
                    {
                        "verb": verb,
                        "noun": noun,
                        "hand": action.get("hand", "unknown"),
                    }
                ],
            }
            segments.append(segment)

        return segments
