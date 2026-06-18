"""Data schemas for DVAS."""

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


class Action(BaseModel):
    """Action performed in video segment."""

    verb: str
    noun: str
    hand: Hand = Hand.UNKNOWN
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class QAPair(BaseModel):
    """Question-answer pair for video segment."""

    question: str
    answer: str
    question_type: Literal["what", "how", "why", "when", "where", "who", "other"] = "other"
    confidence: Optional[float] = None


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

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if "start_time" in info.data and v < info.data["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VideoMetadata(BaseModel):
    """Metadata for a video."""

    fps: float
    resolution: List[int]  # [width, height]
    duration: float
    total_frames: int
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    has_audio: bool = False


class Annotation(BaseModel):
    """Complete annotation for a video."""

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

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
        }
    }

    def to_llava_format(self) -> Dict[str, Any]:
        """Convert to LLaVA training format."""
        conversations = []

        for i, segment in enumerate(self.segments):
            # User prompt with video placeholder
            user_prompt = f"<video>\n详细描述视频中第{i+1}个片段的动作。"
            if segment.objects:
                objects_str = ", ".join([obj.name for obj in segment.objects[:5]])
                user_prompt += f" 特别关注以下物体：{objects_str}。"

            conversations.append({"from": "human", "value": user_prompt})

            # Assistant response
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
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": self.video_path},
                    {"type": "text", "text": f"描述这个视频片段（{segment.start_time:.1f}s - {segment.end_time:.1f}s）"},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": segment.caption,
            })

        return {
            "id": self.id,
            "messages": messages,
        }

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
