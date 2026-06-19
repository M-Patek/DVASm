"""Pipeline state machine for annotation lifecycle management.

Provides explicit state transitions for annotation tasks:
- PENDING → PROCESSING → COMPLETED/FAILED
- Tracks per-segment and per-video state
- Supports checkpoint resume at any state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AnnotationState(Enum):
    """States in the annotation lifecycle."""

    PENDING = auto()  # Waiting to start
    LOADING = auto()  # Loading video
    DETECTING_SCENES = auto()  # Scene detection
    PROCESSING_SEGMENT = auto()  # Annotating a segment
    PARSING_RESPONSE = auto()  # Parsing teacher response
    VALIDATING = auto()  # Running quality validation
    SAVING = auto()  # Persisting annotation
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Failed with error
    PARTIAL = auto()  # Some segments succeeded, some failed


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


# Valid state transitions
_VALID_TRANSITIONS: Dict[AnnotationState, set] = {
    AnnotationState.PENDING: {
        AnnotationState.LOADING,
        AnnotationState.FAILED,
    },
    AnnotationState.LOADING: {
        AnnotationState.DETECTING_SCENES,
        AnnotationState.FAILED,
    },
    AnnotationState.DETECTING_SCENES: {
        AnnotationState.PROCESSING_SEGMENT,
        AnnotationState.FAILED,
    },
    AnnotationState.PROCESSING_SEGMENT: {
        AnnotationState.PARSING_RESPONSE,
        AnnotationState.PROCESSING_SEGMENT,  # Next segment
        AnnotationState.FAILED,
    },
    AnnotationState.PARSING_RESPONSE: {
        AnnotationState.VALIDATING,
        AnnotationState.PARSING_RESPONSE,  # Retry with fallback
        AnnotationState.FAILED,
    },
    AnnotationState.VALIDATING: {
        AnnotationState.SAVING,
        AnnotationState.FAILED,
    },
    AnnotationState.SAVING: {
        AnnotationState.COMPLETED,
        AnnotationState.FAILED,
    },
    AnnotationState.COMPLETED: set(),  # Terminal state
    AnnotationState.FAILED: set(),  # Terminal state
    AnnotationState.PARTIAL: set(),  # Terminal state
}


@dataclass
class SegmentState:
    """State for a single segment annotation."""

    segment_idx: int
    start_time: float
    end_time: float
    state: AnnotationState = AnnotationState.PENDING
    error: Optional[str] = None
    attempts: int = 0
    teacher_latency_ms: float = 0.0
    parse_confidence: float = 0.0
    validation_score: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize segment state."""
        return {
            "segment_idx": self.segment_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "state": self.state.name,
            "error": self.error,
            "attempts": self.attempts,
            "teacher_latency_ms": self.teacher_latency_ms,
            "parse_confidence": self.parse_confidence,
            "validation_score": self.validation_score,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class VideoAnnotationState:
    """Complete state for video annotation task."""

    video_id: str
    video_path: str
    state: AnnotationState = AnnotationState.PENDING
    segment_states: List[SegmentState] = field(default_factory=list)
    current_segment_idx: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_segments: int = 0
    completed_segments: int = 0
    failed_segments: int = 0

    # Metrics
    total_teacher_latency_ms: float = 0.0
    total_parsing_time_ms: float = 0.0
    total_validation_time_ms: float = 0.0

    def transition_to(self, new_state: AnnotationState, error: Optional[str] = None) -> None:
        """Transition to a new state with validation."""
        if new_state not in _VALID_TRANSITIONS.get(self.state, set()):
            raise StateTransitionError(
                f"Invalid transition from {self.state.name} to {new_state.name}"
            )

        old_state = self.state
        self.state = new_state

        if error:
            self.error = error

        # Update timestamps
        if new_state == AnnotationState.LOADING and self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
        elif new_state in (AnnotationState.COMPLETED, AnnotationState.FAILED, AnnotationState.PARTIAL):
            self.completed_at = datetime.now(timezone.utc)

        logger.debug(
            "state_transition",
            video_id=self.video_id,
            from_state=old_state.name,
            to_state=new_state.name,
        )

    def start_segment(self, segment_idx: int, start_time: float, end_time: float) -> SegmentState:
        """Start annotating a new segment."""
        seg_state = SegmentState(
            segment_idx=segment_idx,
            start_time=start_time,
            end_time=end_time,
            state=AnnotationState.PENDING,
            started_at=datetime.now(timezone.utc),
        )

        # Extend or update segment_states list
        if segment_idx < len(self.segment_states):
            self.segment_states[segment_idx] = seg_state
        else:
            self.segment_states.append(seg_state)

        self.current_segment_idx = segment_idx
        return seg_state

    def complete_segment(
        self,
        segment_idx: int,
        teacher_latency_ms: float,
        parse_confidence: float,
        validation_score: float,
    ) -> None:
        """Mark a segment as successfully completed."""
        if segment_idx >= len(self.segment_states):
            return

        seg = self.segment_states[segment_idx]
        seg.state = AnnotationState.COMPLETED
        seg.teacher_latency_ms = teacher_latency_ms
        seg.parse_confidence = parse_confidence
        seg.validation_score = validation_score
        seg.completed_at = datetime.now(timezone.utc)

        self.completed_segments += 1
        self.total_teacher_latency_ms += teacher_latency_ms

    def fail_segment(self, segment_idx: int, error: str) -> None:
        """Mark a segment as failed."""
        if segment_idx >= len(self.segment_states):
            return

        seg = self.segment_states[segment_idx]
        seg.state = AnnotationState.FAILED
        seg.error = error
        seg.completed_at = datetime.now(timezone.utc)

        self.failed_segments += 1

    def calculate_final_state(self) -> AnnotationState:
        """Calculate final state based on segment outcomes."""
        if self.failed_segments == 0 and self.completed_segments == self.total_segments:
            return AnnotationState.COMPLETED
        elif self.completed_segments > 0:
            return AnnotationState.PARTIAL
        else:
            return AnnotationState.FAILED

    def get_progress_percentage(self) -> float:
        """Get annotation progress as percentage."""
        if self.total_segments == 0:
            return 0.0
        return (self.completed_segments + self.failed_segments) / self.total_segments * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize video annotation state."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "state": self.state.name,
            "segment_states": [s.to_dict() for s in self.segment_states],
            "current_segment_idx": self.current_segment_idx,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "failed_segments": self.failed_segments,
            "progress_percentage": self.get_progress_percentage(),
            "metrics": {
                "total_teacher_latency_ms": self.total_teacher_latency_ms,
                "total_parsing_time_ms": self.total_parsing_time_ms,
                "total_validation_time_ms": self.total_validation_time_ms,
            },
        }


class AnnotationStateMachine:
    """State machine manager for annotation tasks.

    Tracks state for multiple concurrent annotation tasks and
    provides checkpoint-compatible state serialization.
    """

    def __init__(self):
        self._tasks: Dict[str, VideoAnnotationState] = {}

    def create_task(self, video_id: str, video_path: str) -> VideoAnnotationState:
        """Create a new annotation task."""
        if video_id in self._tasks:
            logger.warning("task_already_exists", video_id=video_id)
            return self._tasks[video_id]

        task = VideoAnnotationState(
            video_id=video_id,
            video_path=video_path,
        )
        self._tasks[video_id] = task
        logger.info("task_created", video_id=video_id)
        return task

    def get_task(self, video_id: str) -> Optional[VideoAnnotationState]:
        """Get task state by video ID."""
        return self._tasks.get(video_id)

    def transition_task(
        self,
        video_id: str,
        new_state: AnnotationState,
        error: Optional[str] = None,
    ) -> Optional[VideoAnnotationState]:
        """Transition a task to a new state."""
        task = self._tasks.get(video_id)
        if not task:
            logger.warning("task_not_found_for_transition", video_id=video_id)
            return None

        try:
            task.transition_to(new_state, error)
        except StateTransitionError as e:
            logger.error("invalid_state_transition", video_id=video_id, error=str(e))
            raise

        return task

    def remove_task(self, video_id: str) -> None:
        """Remove a task from tracking."""
        if video_id in self._tasks:
            del self._tasks[video_id]
            logger.info("task_removed", video_id=video_id)

    def get_all_tasks(self) -> Dict[str, VideoAnnotationState]:
        """Get all tracked tasks."""
        return self._tasks.copy()

    def get_tasks_by_state(self, state: AnnotationState) -> List[VideoAnnotationState]:
        """Get all tasks in a specific state."""
        return [t for t in self._tasks.values() if t.state == state]

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpoint."""
        return {
            "tasks": {vid: task.to_dict() for vid, task in self._tasks.items()},
            "task_count": len(self._tasks),
            "completed_count": len(self.get_tasks_by_state(AnnotationState.COMPLETED)),
            "failed_count": len(self.get_tasks_by_state(AnnotationState.FAILED)),
        }

    @classmethod
    def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "AnnotationStateMachine":
        """Restore state from checkpoint."""
        sm = cls()
        # Note: Full restoration would require deserializing segment states
        # This is a simplified version for checkpoint compatibility
        tasks_data = data.get("tasks", {})
        for video_id, task_data in tasks_data.items():
            task = VideoAnnotationState(
                video_id=video_id,
                video_path=task_data.get("video_path", ""),
                state=AnnotationState[task_data.get("state", "PENDING")],
                total_segments=task_data.get("total_segments", 0),
                completed_segments=task_data.get("completed_segments", 0),
                failed_segments=task_data.get("failed_segments", 0),
            )
            sm._tasks[video_id] = task

        return sm
