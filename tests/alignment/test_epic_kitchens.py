"""Alignment tests for EPIC-KITCHENS format compatibility.

Verifies that DVAS annotations can be correctly exported to and
are compatible with EPIC-KITCHENS format specifications.
"""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    AnnotationStandard,
    BoundingBox,
    Hand,
    Object,
    Segment,
    VideoMetadata,
)
from dvas.export.formats import Ego4DFormatter  # EPIC is compatible with Ego4D
from dvas.governance import get_adapter


@pytest.fixture
def sample_epic_annotation():
    """Create a sample annotation with EPIC-compatible data."""
    return Annotation(
        id="epic_test_001",
        video_id="P01_01",
        video_path="/data/epic/P01/videos/P01_01.MP4",
        metadata=VideoMetadata(
            fps=60.0,
            resolution=[1920, 1080],
            duration=300.0,
            total_frames=18000,
            has_audio=False,
            camera_type="egocentric",
            environment="kitchen",
        ),
        segments=[
            Segment(
                start_time=10.5,
                end_time=15.2,
                caption="Take knife",
                actions=[
                    Action(
                        verb="take",
                        noun="knife",
                        hand=Hand.RIGHT,
                        start_time=11.0,
                        end_time=14.0,
                        confidence=0.95,
                    )
                ],
                objects=[
                    Object(
                        name="knife",
                        bbox=BoundingBox(x1=0.4, y1=0.3, x2=0.6, y2=0.5),
                        confidence=0.92,
                    ),
                ],
                key_frames=[660, 720, 780, 840],
            ),
            Segment(
                start_time=20.0,
                end_time=28.0,
                caption="Cut bread",
                actions=[
                    Action(
                        verb="cut",
                        noun="bread",
                        hand=Hand.RIGHT,
                        start_time=21.0,
                        end_time=26.0,
                        confidence=0.88,
                    )
                ],
                objects=[
                    Object(
                        name="bread",
                        bbox=BoundingBox(x1=0.3, y1=0.4, x2=0.7, y2=0.8),
                        confidence=0.90,
                    ),
                    Object(
                        name="knife",
                        bbox=BoundingBox(x1=0.4, y1=0.3, x2=0.6, y2=0.5),
                        confidence=0.85,
                    ),
                ],
                key_frames=[1200, 1320, 1440, 1560],
            ),
        ],
        source="teacher",
        annotation_standard=AnnotationStandard.EPIC_KITCHENS,
    )


@pytest.fixture
def epic_format_data():
    """Sample data in EPIC-KITCHENS format."""
    return {
        "id": "epic_import_test",
        "video_id": "P02_05",
        "segments": [
            {
                "start_time": 5.0,
                "end_time": 10.0,
                "actions": [
                    {
                        "verb": "open",
                        "noun": "fridge",
                        "hand": "left",
                        "start_time": 5.5,
                        "end_time": 8.0,
                    }
                ],
            },
            {
                "start_time": 12.0,
                "end_time": 18.0,
                "actions": [
                    {
                        "verb": "take",
                        "noun": "milk",
                        "hand": "right",
                        "start_time": 13.0,
                        "end_time": 15.0,
                    }
                ],
            },
        ],
        "metadata": {
            "fps": 60.0,
            "resolution": [1920, 1080],
            "duration": 25.0,
        },
    }


class TestEPICKitchensGovernanceAdapter:
    """Test EPIC-KITCHENS governance adapter."""

    def test_epic_adapter_to_standard(self, sample_epic_annotation):
        """Test converting annotation to EPIC format."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
        result = adapter.to_standard(sample_epic_annotation)

        # Check structure
        assert "id" in result
        assert "video_id" in result
        assert "segments" in result
        assert "metadata" in result

        # Check content
        assert result["id"] == "epic_test_001"
        assert result["video_id"] == "P01_01"
        assert len(result["segments"]) == 2

    def test_epic_adapter_from_standard(self, epic_format_data):
        """Test converting EPIC format to annotation."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
        annotation = adapter.from_standard(epic_format_data)

        assert annotation.id == "epic_import_test"
        assert annotation.video_id == "P02_05"
        assert len(annotation.segments) == 2
        assert annotation.annotation_standard == AnnotationStandard.EPIC_KITCHENS

    def test_epic_action_format(self, sample_epic_annotation):
        """Test that EPIC actions are correctly formatted."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
        result = adapter.to_standard(sample_epic_annotation)

        segment = result["segments"][0]
        assert "actions" in segment

        action = segment["actions"][0]
        assert "verb" in action
        assert "noun" in action
        assert "hand" in action
        assert action["verb"] == "take"
        assert action["noun"] == "knife"
        assert action["hand"] == "right"

    def test_epic_metadata(self, sample_epic_annotation):
        """Test EPIC metadata conversion."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
        result = adapter.to_standard(sample_epic_annotation)

        meta = result["metadata"]
        assert meta["fps"] == 60.0
        assert meta["resolution"] == [1920, 1080]
        assert meta["duration"] == 300.0

    def test_epic_round_trip(self, epic_format_data):
        """Test round-trip conversion (EPIC -> Annotation -> EPIC)."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        # Import
        annotation = adapter.from_standard(epic_format_data)

        # Export
        result = adapter.to_standard(annotation)

        # Verify core fields preserved
        assert result["id"] == epic_format_data["id"]
        assert result["video_id"] == epic_format_data["video_id"]
        assert len(result["segments"]) == len(epic_format_data["segments"])

    def test_epic_hand_values(self):
        """Test various hand value conversions."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        test_cases = [
            ("left", Hand.LEFT),
            ("right", Hand.RIGHT),
            ("both", Hand.BOTH),
            ("unknown", Hand.UNKNOWN),
        ]

        for hand_str, expected_hand in test_cases:
            data = {
                "id": "test",
                "video_id": "test",
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "actions": [
                            {
                                "verb": "test",
                                "noun": "test",
                                "hand": hand_str,
                            }
                        ],
                    }
                ],
                "metadata": {"fps": 30.0, "resolution": [1920, 1080], "duration": 1.0},
            }

            annotation = adapter.from_standard(data)
            assert annotation.segments[0].actions[0].hand == expected_hand


class TestEPICEgo4DCompatibility:
    """Test EPIC-KITCHENS compatibility with Ego4D formatter."""

    def test_epic_via_ego4d_formatter(self, sample_epic_annotation):
        """Test exporting EPIC data via Ego4D formatter."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_epic_annotation)

        # Basic structure should work
        assert "video_id" in result
        assert "narrations" in result
        assert result["video_id"] == "P01_01"

    def test_epic_verb_noun_preservation(self, sample_epic_annotation):
        """Test that verb/noun are preserved in export."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_epic_annotation)

        for i, narr in enumerate(result["narrations"]):
            assert len(narr["actions"]) > 0
            actions = narr["actions"]

            if i == 0:
                assert actions[0]["verb"] == "take"
                assert actions[0]["noun"] == "knife"
            elif i == 1:
                assert actions[0]["verb"] == "cut"
                assert actions[0]["noun"] == "bread"

    def test_epic_hand_preservation(self, sample_epic_annotation):
        """Test that hand information is preserved."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_epic_annotation)

        for narr in result["narrations"]:
            for action in narr["actions"]:
                assert action["hand"] == "right"


class TestEPICKitchensFormatFeatures:
    """Test EPIC-KITCHENS specific format features."""

    def test_epic_v1_compatibility(self):
        """Test compatibility with EPIC v1.0 format."""
        # EPIC v1.0 only has verb, noun, hand
        annotation = Annotation(
            id="v1_test",
            video_id="P01_101",
            video_path="/data/P01_101.MP4",
            metadata=VideoMetadata(
                fps=60.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=600,
            ),
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=2.0,
                    caption="Simple action",
                    actions=[
                        Action(
                            verb="take",
                            noun="cup",
                            hand=Hand.RIGHT,
                        )
                    ],
                )
            ],
            annotation_standard=AnnotationStandard.EPIC_KITCHENS,
        )

        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)
        result = adapter.to_standard(annotation)

        # Should only have basic fields
        action = result["segments"][0]["actions"][0]
        assert set(action.keys()) == {"verb", "noun", "hand", "start_time", "end_time"}

    def test_epic_60fps_handling(self):
        """Test handling of EPIC's 60fps videos."""
        annotation = Annotation(
            id="fps_test",
            video_id="P01_60fps",
            video_path="/data/test.MP4",
            metadata=VideoMetadata(
                fps=60.0,
                resolution=[1920, 1080],
                duration=60.0,
                total_frames=3600,
            ),
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Test",
                    key_frames=[0, 30, 60],  # Frame indices at 60fps
                )
            ],
        )

        formatter = Ego4DFormatter()
        result = formatter.format_annotation(annotation)

        assert result["video_metadata"]["fps"] == 60.0
        assert result["narrations"][0]["key_frames"] == [0, 30, 60]

    def test_epic_video_id_format(self):
        """Test EPIC video ID format (Participant_Video)."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        data = {
            "id": "test",
            "video_id": "P01_123",
            "segments": [],
            "metadata": {"fps": 60.0, "resolution": [1920, 1080], "duration": 10.0},
        }

        annotation = adapter.from_standard(data)
        assert annotation.video_id == "P01_123"

        # Check export preserves format
        result = adapter.to_standard(annotation)
        assert result["video_id"] == "P01_123"

    def test_epic_directory_structure(self):
        """Test handling of EPIC directory structure."""
        annotation = Annotation(
            id="path_test",
            video_id="P03_45",
            video_path="/data/epic-kitchens/P03/videos/P03_45.MP4",
            metadata=VideoMetadata(
                fps=60.0,
                resolution=[1920, 1080],
                duration=100.0,
                total_frames=6000,
            ),
            segments=[],
        )

        formatter = Ego4DFormatter()
        result = formatter.format_annotation(annotation)

        assert result["video_metadata"]["video_path"] == "/data/epic-kitchens/P03/videos/P03_45.MP4"


class TestEPICKitchensEdgeCases:
    """Test EPIC-KITCHENS edge cases."""

    def test_epic_empty_segments(self):
        """Test handling of empty segments."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        data = {
            "id": "empty_test",
            "video_id": "P01_empty",
            "segments": [],
            "metadata": {"fps": 60.0, "resolution": [1920, 1080], "duration": 5.0},
        }

        annotation = adapter.from_standard(data)
        assert len(annotation.segments) == 0

    def test_epic_no_hand_specified(self):
        """Test handling of actions without hand specified."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        data = {
            "id": "hand_test",
            "video_id": "P01_hand",
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "actions": [
                        {
                            "verb": "look",
                            "noun": "recipe",
                            # No hand specified
                        }
                    ],
                }
            ],
            "metadata": {"fps": 60.0, "resolution": [1920, 1080], "duration": 1.0},
        }

        annotation = adapter.from_standard(data)
        # Should default to unknown
        assert annotation.segments[0].actions[0].hand == Hand.UNKNOWN

    def test_epic_temporal_order(self):
        """Test that temporal order is preserved."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        data = {
            "id": "order_test",
            "video_id": "P01_order",
            "segments": [
                {
                    "start_time": 10.0,
                    "end_time": 12.0,
                    "actions": [{"verb": "first", "noun": "action"}],
                },
                {
                    "start_time": 15.0,
                    "end_time": 17.0,
                    "actions": [{"verb": "second", "noun": "action"}],
                },
                {
                    "start_time": 20.0,
                    "end_time": 22.0,
                    "actions": [{"verb": "third", "noun": "action"}],
                },
            ],
            "metadata": {"fps": 60.0, "resolution": [1920, 1080], "duration": 30.0},
        }

        annotation = adapter.from_standard(data)

        assert annotation.segments[0].start_time == 10.0
        assert annotation.segments[1].start_time == 15.0
        assert annotation.segments[2].start_time == 20.0

    def test_epic_caption_invention(self):
        """Test that EPIC adapter invents captions (EPIC has none)."""
        adapter = get_adapter(AnnotationStandard.EPIC_KITCHENS)

        data = {
            "id": "caption_test",
            "video_id": "P01_cap",
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "actions": [{"verb": "take", "noun": "plate"}],
                }
            ],
            "metadata": {"fps": 60.0, "resolution": [1920, 1080], "duration": 1.0},
        }

        annotation = adapter.from_standard(data)
        # EPIC doesn't have captions, so adapter should set empty string
        assert annotation.segments[0].caption == ""
