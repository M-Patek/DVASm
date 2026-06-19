"""Tests for governance adapters — pytest function style.

Covers EPIC, Ego4D, and Open X adapters with round-trip
(to_standard → from_standard) verification.
"""

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    AnnotationStandard,
    EmbodimentAction,
    Hand,
    Object,
    Segment,
    VideoMetadata,
)
from dvas.governance import get_adapter, list_standards
from dvas.governance.adapters import EPICAdapter, Ego4DAdapter, OpenXAdapter


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def epic_annotation() -> Annotation:
    """Minimal EPIC-KITCHENS annotation."""
    return Annotation(
        id="ann_001",
        video_id="vid_001",
        video_path="/tmp/test.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=1.0,
                caption="pick cup",
                actions=[
                    Action(
                        verb="pick",
                        noun="cup",
                        hand=Hand.RIGHT,
                        start_time=0.0,
                        end_time=1.0,
                    )
                ],
            )
        ],
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=1.0,
            total_frames=30,
        ),
        annotation_standard=AnnotationStandard.EPIC_KITCHENS,
    )


@pytest.fixture
def ego4d_annotation() -> Annotation:
    """Ego4D annotation with objects and instruments."""
    return Annotation(
        id="ann_002",
        video_id="vid_002",
        video_path="/tmp/test.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=2.0,
                caption="cut tomato with knife",
                actions=[
                    Action(
                        verb="cut",
                        noun="tomato",
                        hand=Hand.RIGHT,
                        instrument="knife",
                        source_state="whole",
                        target_state="sliced",
                    )
                ],
                objects=[
                    Object(
                        name="tomato",
                        attributes={"color": "red"},
                        state="whole",
                    ),
                    Object(
                        name="knife",
                        attributes={"material": "steel"},
                        state="sharp",
                    ),
                ],
            )
        ],
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=2.0,
            total_frames=60,
            camera_type="egocentric",
            environment="kitchen",
        ),
        annotation_standard=AnnotationStandard.EGO4D,
    )


@pytest.fixture
def openx_annotation() -> Annotation:
    """Open X-Embodiment annotation with embodiment data."""
    return Annotation(
        id="ann_003",
        video_id="vid_003",
        video_path="/tmp/test.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=1.0,
                caption="grasp object",
                actions=[
                    Action(
                        verb="grasp",
                        noun="object",
                        hand=Hand.RIGHT,
                        embodiment=EmbodimentAction(
                            gripper_pose=[0.1, 0.2, 0.3],
                            joint_target=[0.0, 0.0, 0.0],
                            action_space="absolute",
                            gripper_state="close",
                        ),
                    )
                ],
            )
        ],
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[640, 480],
            duration=1.0,
            total_frames=30,
        ),
        annotation_standard=AnnotationStandard.OPEN_X_EMBODIMENT,
    )


# ── Registry tests ────────────────────────────────────────────────────────


def test_list_standards() -> None:
    standards = list_standards()
    assert len(standards) == 3
    assert AnnotationStandard.EPIC_KITCHENS in standards
    assert AnnotationStandard.EGO4D in standards
    assert AnnotationStandard.OPEN_X_EMBODIMENT in standards


def test_get_adapter_epic() -> None:
    adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
    assert isinstance(adapter, EPICAdapter)
    assert adapter.standard == AnnotationStandard.EPIC_KITCHENS


def test_get_adapter_ego4d() -> None:
    adapter = get_adapter(AnnotationStandard.EGO4D)
    assert isinstance(adapter, Ego4DAdapter)


def test_get_adapter_openx() -> None:
    adapter = get_adapter(AnnotationStandard.OPEN_X_EMBODIMENT)
    assert isinstance(adapter, OpenXAdapter)


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported annotation standard"):
        get_adapter(AnnotationStandard.CUSTOM)


# ── EPIC Adapter tests ────────────────────────────────────────────────────


def test_epic_to_standard(epic_annotation: Annotation) -> None:
    adapter = EPICAdapter()
    data = adapter.to_standard(epic_annotation)

    assert data["id"] == "ann_001"
    assert data["video_id"] == "vid_001"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["actions"][0]["verb"] == "pick"
    assert data["segments"][0]["actions"][0]["hand"] == "right"


def test_epic_from_standard_roundtrip(epic_annotation: Annotation) -> None:
    adapter = EPICAdapter()
    data = adapter.to_standard(epic_annotation)
    restored = adapter.from_standard(data)

    assert restored.id == epic_annotation.id
    assert restored.video_id == epic_annotation.video_id
    assert len(restored.segments) == 1
    assert restored.segments[0].actions[0].verb == "pick"
    assert restored.annotation_standard == AnnotationStandard.EPIC_KITCHENS


def test_epic_from_standard_defaults() -> None:
    adapter = EPICAdapter()
    data = {
        "id": "ann_default",
        "video_id": "vid_default",
        "segments": [],
    }
    restored = adapter.from_standard(data)
    assert restored.metadata.fps == 30.0
    assert restored.metadata.resolution == [1920, 1080]


# ── Ego4D Adapter tests ───────────────────────────────────────────────────


def test_ego4d_to_standard(ego4d_annotation: Annotation) -> None:
    adapter = Ego4DAdapter()
    data = adapter.to_standard(ego4d_annotation)

    assert len(data["narrations"]) == 1
    narration = data["narrations"][0]
    assert narration["narration"] == "cut tomato with knife"
    assert narration["actions"][0]["instrument"] == "knife"
    assert len(narration["objects"]) == 2


def test_ego4d_from_standard_roundtrip(ego4d_annotation: Annotation) -> None:
    adapter = Ego4DAdapter()
    data = adapter.to_standard(ego4d_annotation)
    restored = adapter.from_standard(data)

    assert restored.id == ego4d_annotation.id
    assert len(restored.segments) == 1
    assert restored.segments[0].actions[0].instrument == "knife"
    assert restored.segments[0].objects[0].name == "tomato"


# ── Open X Adapter tests ──────────────────────────────────────────────────


def test_openx_to_standard(openx_annotation: Annotation) -> None:
    adapter = OpenXAdapter()
    data = adapter.to_standard(openx_annotation)

    assert len(data["steps"]) == 1
    step = data["steps"][0]
    assert step["verb"] == "grasp"
    assert step["language_instruction"] == "grasp object"
    assert step["action"]["gripper_state"] == "close"


def test_openx_from_standard_roundtrip(openx_annotation: Annotation) -> None:
    adapter = OpenXAdapter()
    data = adapter.to_standard(openx_annotation)
    restored = adapter.from_standard(data)

    assert restored.id == openx_annotation.id
    assert len(restored.segments) == 1
    action = restored.segments[0].actions[0]
    assert action.verb == "grasp"
    assert action.embodiment is not None
    assert action.embodiment.gripper_state == "close"


def test_openx_from_standard_no_embodiment() -> None:
    adapter = OpenXAdapter()
    data = {
        "id": "ann_no_emb",
        "video_id": "vid_no_emb",
        "steps": [
            {"verb": "move", "noun": "arm"},
        ],
    }
    restored = adapter.from_standard(data)
    assert restored.segments[0].actions[0].embodiment is None
