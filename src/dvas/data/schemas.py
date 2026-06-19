"""Data schemas for DVAS — Schema v2.0 (VLA + World Model compatible).

This module defines the central Annotation schema used by all subsystems.
Schema v2.0 adds VLA-enhanced fields (instrument, physical properties,
temporal relations) and World Model extension fields (state predictions,
dynamics) while maintaining backward compatibility with v1.0 EPIC data.

All new fields are Optional — existing v1.0 data loads without migration.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Hand(str, Enum):
    """Hand used for action."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    UNKNOWN = "unknown"


class AnnotationStandard(str, Enum):
    """Annotation standard this data follows."""

    EPIC_KITCHENS = "epic"
    EGO4D = "ego4d"
    OPEN_X_EMBODIMENT = "open_x"
    CUSTOM = "custom"


# ── Spatial / Detection ──────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Bounding box in normalized coordinates [x1, y1, x2, y2]."""

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class Object(BaseModel):
    """Object detected in video segment."""

    name: str
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    # VLA v2.0: material, color, state for richer object descriptions
    material: Optional[str] = None
    color: Optional[str] = None
    state: Optional[str] = None  # e.g. "open", "closed", "full", "empty"


# ── Physical / Embodiment (VLA enhancement) ──────────────────────────


class PhysicalProperties(BaseModel):
    """Physical properties of an action — used for robotic manipulation."""

    force: Optional[str] = None  # "gentle", "firm", "none"
    trajectory: Optional[str] = None  # "downward", "upward", "circular"
    contact_type: Optional[str] = None  # "grasp", "push", "slide", "release"
    tool: Optional[str] = None  # instrument used


class EmbodimentAction(BaseModel):
    """Robot embodiment action space (Open X-Embodiment style)."""

    gripper_pose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz]
    joint_target: Optional[List[float]] = None  # 7-DOF or per-robot
    action_space: Optional[Literal["absolute", "delta"]] = None
    gripper_state: Optional[Literal["open", "close"]] = None


# ── Action (EPIC compatible + VLA enhanced) ──────────────────────────


class Action(BaseModel):
    """Action performed in video segment.

    Backward compatible with EPIC-KITCHENS verb+noun.
    VLA v2.0 adds: instrument, source/target state, physical properties,
    embodiment action space.
    """

    # === EPIC v1.0 compatible layer ===
    verb: str
    noun: str
    hand: Hand = Hand.UNKNOWN
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # === VLA v2.0 enhancement layer ===
    instrument: Optional[str] = None  # tool / medium (Ego4D style)
    source_state: Optional[str] = None  # initial state before action
    target_state: Optional[str] = None  # expected state after action
    physical: Optional[PhysicalProperties] = None
    embodiment: Optional[EmbodimentAction] = None


# ── Temporal Relations (VLA enhancement) ─────────────────────────────


class TemporalRelation(BaseModel):
    """Temporal relation between segments."""

    relation: Literal["before", "after", "during", "overlaps", "contains"]
    target_segment_id: str
    description: Optional[str] = None


# ── QA Pairs ─────────────────────────────────────────────────────────


class QAPair(BaseModel):
    """Question-answer pair for video segment."""

    question: str
    answer: str
    question_type: Literal["what", "how", "why", "when", "where", "who", "other"] = "other"
    confidence: Optional[float] = None


# ── Segment ──────────────────────────────────────────────────────────


class Segment(BaseModel):
    """Temporal segment of a video with annotations."""

    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    caption: str
    caption_dense: Optional[str] = None  # Detailed description
    qa_pairs: List[QAPair] = Field(default_factory=list)
    objects: List[Object] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    key_frames: List[int] = Field(default_factory=list)  # Frame indices

    # VLA v2.0 enhancements
    temporal_relations: List[TemporalRelation] = Field(default_factory=list)
    scene_type: Optional[str] = None  # "kitchen", "office", "outdoor"
    lighting: Optional[str] = None  # "bright", "dim", "natural"

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if "start_time" in info.data and v < info.data["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ── World Model Extensions (Optional, reserved) ──────────────────────


class StatePrediction(BaseModel):
    """State prediction annotation for World Model training.

    Currently a placeholder — fields populated by future world model pipeline.
    """

    predicted_next_frame_desc: Optional[str] = None
    expected_state_change: Optional[str] = None
    preconditions: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)


class DynamicsAnnotation(BaseModel):
    """Physical dynamics annotation for World Model training.

    Currently a placeholder — fields populated by future world model pipeline.
    """

    physical_constraints: List[str] = Field(default_factory=list)
    causal_links: List[Dict[str, str]] = Field(default_factory=list)
    counterfactuals: List[Dict[str, str]] = Field(default_factory=list)


# ── Video Metadata ───────────────────────────────────────────────────


class VideoMetadata(BaseModel):
    """Metadata for a video."""

    fps: float
    resolution: List[int]  # [width, height]
    duration: float
    total_frames: int
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    has_audio: bool = False
    # VLA v2.0 enhancements
    camera_type: Optional[str] = None  # "egocentric", "static", "moving"
    environment: Optional[str] = None  # "indoor", "outdoor", "simulation"


# ── Root Annotation ──────────────────────────────────────────────────


class Annotation(BaseModel):
    """Complete annotation for a video — Schema v2.0.

    Backward compatible with v1.0: all new fields are Optional.
    Old data loads without migration; new fields default to None.
    """

    id: str
    video_id: str
    video_path: str
    segments: List[Segment] = Field(default_factory=list)
    metadata: VideoMetadata
    source: Literal["teacher", "student", "human", "hybrid"] = "teacher"
    model_version: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    parent_id: Optional[str] = None  # For tracking revisions
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

    # Schema v2.0: version and standard tracking
    schema_version: str = "2.0"
    annotation_standard: AnnotationStandard = AnnotationStandard.CUSTOM

    # World Model extensions (optional, reserved for future)
    state_predictions: Optional[StatePrediction] = None
    dynamics: Optional[DynamicsAnnotation] = None

    # Data lineage (optional)
    lineage: Optional[Dict[str, Any]] = None

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
        }
    }

    # ── Export Methods ───────────────────────────────────────────────

    def to_llava_format(self) -> Dict[str, Any]:
        """Convert to LLaVA training format."""
        conversations = []

        for i, segment in enumerate(self.segments):
            user_prompt = f"<video>\n详细描述视频中第{i + 1}个片段的动作。"
            if segment.objects:
                objects_str = ", ".join([obj.name for obj in segment.objects[:5]])
                user_prompt += f" 特别关注以下物体：{objects_str}。"

            conversations.append({"from": "human", "value": user_prompt})

            response = segment.caption
            if segment.actions:
                actions_str = "; ".join(
                    [f"{a.hand.value}手{a.verb}{a.noun}" for a in segment.actions[:3]]
                )
                response += f"\n\n动作分解：{actions_str}"

            conversations.append({"from": "gpt", "value": response})

        return {
            "id": self.id,
            "video": self.video_path,
            "conversations": conversations,
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        messages = []

        for segment in self.segments:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": self.video_path},
                        {
                            "type": "text",
                            "text": f"描述这个视频片段（{segment.start_time:.1f}s - {segment.end_time:.1f}s）",
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": segment.caption,
                }
            )

        return {
            "id": self.id,
            "messages": messages,
        }

    # ── Utility Methods ──────────────────────────────────────────────

    def get_total_duration(self) -> float:
        """Get total annotated duration."""
        return sum(seg.duration for seg in self.segments)

    def get_action_verbs(self) -> List[str]:
        """Get all unique action verbs."""
        verbs = set()
        for seg in self.segments:
            for action in seg.actions:
                verbs.add(action.verb)
        return sorted(list(verbs))

    def get_object_names(self) -> List[str]:
        """Get all unique object names."""
        objects = set()
        for seg in self.segments:
            for obj in seg.objects:
                objects.add(obj.name)
        return sorted(list(objects))

    def get_instruments(self) -> List[str]:
        """Get all unique instruments (VLA v2.0)."""
        instruments = set()
        for seg in self.segments:
            for action in seg.actions:
                if action.instrument:
                    instruments.add(action.instrument)
        return sorted(list(instruments))

    def get_physical_actions(self) -> List[Dict[str, Any]]:
        """Get actions with physical properties (VLA v2.0)."""
        result = []
        for seg in self.segments:
            for action in seg.actions:
                if action.physical:
                    result.append(
                        {
                            "verb": action.verb,
                            "noun": action.noun,
                            "physical": action.physical.model_dump(exclude_none=True),
                        }
                    )
        return result

    def is_v2_enhanced(self) -> bool:
        """Check if this annotation uses v2.0 enhanced fields."""
        for seg in self.segments:
            for action in seg.actions:
                if action.instrument or action.physical or action.source_state:
                    return True
            if seg.temporal_relations or seg.scene_type:
                return True
        if self.state_predictions or self.dynamics:
            return True
        return False
