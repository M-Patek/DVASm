"""Type-safe utilities and runtime type validation.

Provides NewType aliases, TypedDict, Protocol, and runtime validation
to enforce type safety beyond static analysis.
"""

from __future__ import annotations

import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# NewType aliases for domain-specific types
# ---------------------------------------------------------------------------

from typing import NewType

# Video-related types
VideoId = NewType("VideoId", str)
AnnotationId = NewType("AnnotationId", str)
FrameIndex = NewType("FrameIndex", int)
Timestamp = NewType("Timestamp", float)
Duration = NewType("Duration", float)
FPS = NewType("FPS", float)

# Model-related types
ModelName = NewType("ModelName", str)
Prompt = NewType("Prompt", str)
Confidence = NewType("Confidence", float)
CostUSD = NewType("CostUSD", float)
LatencyMS = NewType("LatencyMS", float)

# Content types
Caption = NewType("Caption", str)
Verb = NewType("Verb", str)
Noun = NewType("Noun", str)
BoundingBoxCoords = NewType("BoundingBoxCoords", List[float])


# ---------------------------------------------------------------------------
# TypedDict for structured data
# ---------------------------------------------------------------------------

class VideoMetadataDict(TypedDict):
    """TypedDict for video metadata."""

    fps: float
    resolution: List[int]
    duration: float
    total_frames: int
    codec: Optional[str]
    bitrate: Optional[int]
    has_audio: bool


class SegmentDict(TypedDict):
    """TypedDict for video segment."""

    start_time: float
    end_time: float
    caption: str
    caption_dense: Optional[str]
    qa_pairs: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    key_frames: List[int]


class AnnotationDict(TypedDict):
    """TypedDict for annotation."""

    id: str
    video_id: str
    video_path: str
    segments: List[SegmentDict]
    metadata: VideoMetadataDict
    source: str
    model_version: Optional[str]
    quality_score: Optional[float]
    created_at: str


class GenerationResultDict(TypedDict):
    """TypedDict for generation result."""

    text: str
    model_type: str
    model_version: str
    status: str
    confidence: float
    latency_ms: float
    token_usage: Dict[str, int]
    cost_usd: float
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsAnnotate(Protocol):
    """Protocol for objects that can annotate."""

    async def annotate(self, frames: List[np.ndarray], prompt: Optional[str] = None) -> Any:
        ...


@runtime_checkable
class SupportsGenerate(Protocol):
    """Protocol for objects that can generate."""

    async def generate(self, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class SupportsCostEstimate(Protocol):
    """Protocol for objects that can estimate cost."""

    def estimate_cost(self, **kwargs: Any) -> float:
        ...


@runtime_checkable
class SupportsSerialize(Protocol):
    """Protocol for objects that can be serialized."""

    def to_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        ...


# ---------------------------------------------------------------------------
# Runtime type validation
# ---------------------------------------------------------------------------

T = TypeVar("T")


def validate_type(value: Any, expected_type: type, field_name: str = "") -> None:
    """Validate that a value matches the expected type at runtime.

    Args:
        value: The value to validate
        expected_type: The expected type
        field_name: Optional field name for error messages

    Raises:
        TypeError: If the value doesn't match the expected type
    """
    if not isinstance(value, expected_type):
        field = f" for field '{field_name}'" if field_name else ""
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(value).__name__}{field}"
        )


def validate_positive(value: Union[int, float], field_name: str = "") -> None:
    """Validate that a numeric value is positive.

    Args:
        value: The value to validate
        field_name: Optional field name for error messages

    Raises:
        ValueError: If the value is not positive
    """
    if value <= 0:
        field = f" for field '{field_name}'" if field_name else ""
        raise ValueError(f"Value must be positive{field}: {value}")


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    field_name: str = "",
) -> None:
    """Validate that a numeric value is within a range.

    Args:
        value: The value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Optional field name for error messages

    Raises:
        ValueError: If the value is out of range
    """
    if not min_val <= value <= max_val:
        field = f" for field '{field_name}'" if field_name else ""
        raise ValueError(
            f"Value must be between {min_val} and {max_val}{field}: {value}"
        )


def validate_non_empty(value: Union[str, List, Dict], field_name: str = "") -> None:
    """Validate that a collection is not empty.

    Args:
        value: The value to validate
        field_name: Optional field name for error messages

    Raises:
        ValueError: If the value is empty
    """
    if not value:
        field = f" for field '{field_name}'" if field_name else ""
        raise ValueError(f"Value must not be empty{field}")


# ---------------------------------------------------------------------------
# Type validation decorator
# ---------------------------------------------------------------------------

def typed(**type_hints: type) -> Callable:
    """Decorator that validates function arguments at runtime.

    Usage::

        @typed(name=str, age=int)
        def greet(name, age):
            print(f"Hello {name}, you are {age} years old")

        greet("Alice", 30)  # OK
        greet("Bob", "thirty")  # TypeError
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, expected_type in type_hints.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    validate_type(value, expected_type, param_name)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def returns(expected_type: type) -> Callable:
    """Decorator that validates the return type at runtime.

    Usage::

        @returns(int)
        def add(a, b):
            return a + b
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if result is not None and not isinstance(result, expected_type):
                raise TypeError(
                    f"Expected return type {expected_type.__name__}, "
                    f"got {type(result).__name__}"
                )
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Generic constraints
# ---------------------------------------------------------------------------

class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: Any) -> bool:
        ...

    def __gt__(self, other: Any) -> bool:
        ...


Numeric = Union[int, float]


# ---------------------------------------------------------------------------
# Type aliases for common patterns
# ---------------------------------------------------------------------------

# Callback types
Callback = Callable[..., Any]
AsyncCallback = Callable[..., Any]

# Result types
Result = Union[T, Exception]
Maybe = Optional[T]

# Configuration types
ConfigDict = Dict[str, Any]
