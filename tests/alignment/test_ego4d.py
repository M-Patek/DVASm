"""Alignment tests for Ego4D format compatibility.

Verifies that DVAS annotations can be correctly exported to and
are compatible with Ego4D format specifications.
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
    PhysicalProperties,
    Segment,
    TemporalRelation,
    VideoMetadata,
)
from dvas.export.formats import Ego4DFormatter


@pytest.fixture
def sample_annotation():
    """Create a sample annotation for testing."""
    return Annotation(
        id="ego4d_test_001",
        video_id="video_001",
        video_path="/data/videos/video_001.mp4",
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=60.0,
            total_frames=1800,
            has_audio=True,
            camera_type="egocentric",
            environment="kitchen",
        ),
        segments=[
            Segment(
                start_time=5.0,
                end_time=10.0,
                caption="Open the refrigerator",
                caption_dense="The person opens the refrigerator door with their right hand",
                actions=[
                    Action(
                        verb="open",
                        noun="refrigerator",
                        hand=Hand.RIGHT,
                        start_time=5.5,
                        end_time=8.0,
                        instrument="hand",
                        source_state="closed",
                        target_state="open",
                        physical=PhysicalProperties(
                            force="gentle",
                            trajectory="outward",
                            contact_type="grasp",
                        ),
                    )
                ],
                objects=[
                    Object(
                        name="refrigerator",
                        bbox=BoundingBox(x1=0.2, y1=0.1, x2=0.8, y2=0.9),
                        confidence=0.95,
                        state="open",
                        material="metal",
                    ),
                    Object(
                        name="hand",
                        bbox=BoundingBox(x1=0.6, y1=0.4, x2=0.7, y2=0.6),
                        confidence=0.88,
                    ),
                ],
                key_frames=[150, 180, 210, 240],
                scene_type="kitchen",
                lighting="indoor",
                temporal_relations=[
                    TemporalRelation(
                        relation="before",
                        target_segment_id="segment_002",
                        description="Happens before taking out ingredients",
                    )
                ],
            ),
            Segment(
                start_time=12.0,
                end_time=18.0,
                caption="Take out ingredients",
                actions=[
                    Action(
                        verb="take",
                        noun="milk",
                        hand=Hand.LEFT,
                        start_time=13.0,
                        end_time=15.0,
                    ),
                    Action(
                        verb="take",
                        noun="eggs",
                        hand=Hand.RIGHT,
                        start_time=15.5,
                        end_time=17.0,
                    ),
                ],
                objects=[
                    Object(name="milk", state="held"),
                    Object(name="eggs", state="held"),
                ],
                key_frames=[360, 400, 450, 500],
            ),
        ],
        source="teacher",
        model_version="gpt-4v-2024-04-09",
        quality_score=0.92,
        annotation_standard=AnnotationStandard.EGO4D,
    )


@pytest.fixture
def sample_annotations():
    """Create multiple sample annotations."""
    return [
        Annotation(
            id=f"ego4d_test_{i:03d}",
            video_id=f"video_{i:03d}",
            video_path=f"/data/videos/video_{i:03d}.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=30.0 + i * 10,
                total_frames=900 + i * 300,
                has_audio=True,
                camera_type="egocentric",
            ),
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=5.0,
                    caption=f"Activity {i}",
                    actions=[Action(verb="do", noun="task", hand=Hand.RIGHT)],
                )
            ],
        )
        for i in range(3)
    ]


class TestEgo4DFormatter:
    """Test Ego4D formatter compatibility."""

    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = Ego4DFormatter()
        assert formatter.include_narrations is True
        assert formatter.include_hand_tracking is True

        formatter_no_narration = Ego4DFormatter(include_narrations=False)
        assert formatter_no_narration.include_narrations is False

    def test_basic_structure(self, sample_annotation):
        """Test basic Ego4D output structure."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        # Check required fields
        assert "video_id" in result
        assert "video_metadata" in result
        assert "narrations" in result
        assert "annotation_standard" in result

        # Check standard
        assert result["annotation_standard"] == "ego4d"

    def test_video_metadata(self, sample_annotation):
        """Test video metadata conversion."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        meta = result["video_metadata"]
        assert meta["video_id"] == "video_001"
        assert meta["fps"] == 30.0
        assert meta["resolution"] == [1920, 1080]
        assert meta["has_audio"] is True
        assert meta["camera_type"] == "egocentric"
        assert meta["environment"] == "kitchen"

    def test_narration_structure(self, sample_annotation):
        """Test narration structure."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narrations = result["narrations"]
        assert len(narrations) == 2

        # Check first narration
        narr = narrations[0]
        assert narr["start_time"] == 5.0
        assert narr["end_time"] == 10.0
        assert narr["narration_text"] == "Open the refrigerator"
        assert (
            narr["narration_dense"]
            == "The person opens the refrigerator door with their right hand"
        )

    def test_action_conversion(self, sample_annotation):
        """Test action conversion to Ego4D format."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        actions = narr["actions"]
        assert len(actions) == 1

        action = actions[0]
        assert action["verb"] == "open"
        assert action["noun"] == "refrigerator"
        assert action["hand"] == "right"
        assert action["instrument"] == "hand"

        # Check state changes
        assert action["state_change"]["from"] == "closed"
        assert action["state_change"]["to"] == "open"

    def test_object_conversion(self, sample_annotation):
        """Test object conversion to Ego4D format."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        objects = narr["objects"]
        assert len(objects) == 2

        # Check refrigerator object
        fridge = objects[0]
        assert fridge["name"] == "refrigerator"
        assert fridge["state"] == "open"
        assert fridge["material"] == "metal"
        assert "bbox" in fridge
        assert fridge["confidence"] == 0.95

    def test_temporal_relations(self, sample_annotation):
        """Test temporal relation conversion."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        relations = narr["temporal_relations"]
        assert len(relations) == 1

        rel = relations[0]
        assert rel["relation"] == "before"
        assert rel["target_segment_id"] == "segment_002"
        assert "description" in rel

    def test_scene_context(self, sample_annotation):
        """Test scene context conversion."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        assert "scene_context" in narr
        assert narr["scene_context"]["scene_type"] == "kitchen"
        assert narr["scene_context"]["lighting"] == "indoor"

    def test_key_frames(self, sample_annotation):
        """Test key frame preservation."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        assert narr["key_frames"] == [150, 180, 210, 240]

    def test_object_interactions(self, sample_annotation):
        """Test object interaction extraction."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        assert "object_interactions" in narr
        interactions = narr["object_interactions"]
        assert len(interactions) == 1

        interaction = interactions[0]
        assert interaction["action"] == "open"
        assert interaction["object"] == "refrigerator"
        assert interaction["hand"] == "right"

    def test_hand_tracking_placeholder(self, sample_annotation):
        """Test hand tracking structure."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        assert "hand_tracking" in narr
        assert "left_hand" in narr["hand_tracking"]
        assert "right_hand" in narr["hand_tracking"]

    def test_source_info(self, sample_annotation):
        """Test source information."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        assert "source" in result
        assert result["source"]["type"] == "teacher"
        assert result["source"]["model_version"] == "gpt-4v-2024-04-09"

    def test_quality_score(self, sample_annotation):
        """Test quality score preservation."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        assert result["quality_score"] == 0.92

    def test_batch_formatting(self, sample_annotations):
        """Test formatting multiple annotations."""
        formatter = Ego4DFormatter()
        results = formatter.format_annotations(sample_annotations)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["video_id"] == f"video_{i:03d}"

    def test_export_to_jsonl(self, sample_annotations):
        """Test exporting to JSONL format."""
        formatter = Ego4DFormatter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            count = formatter.export_to_file(sample_annotations, temp_path, format="jsonl")
            assert count == 3

            with open(temp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 3

                for line in lines:
                    data = json.loads(line)
                    assert "video_id" in data
                    assert "narrations" in data
        finally:
            temp_path.unlink(missing_ok=True)

    def test_export_to_json(self, sample_annotations):
        """Test exporting to JSON format."""
        formatter = Ego4DFormatter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            count = formatter.export_to_file(sample_annotations, temp_path, format="json")
            assert count == 3

            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "videos" in data
                assert len(data["videos"]) == 3
                assert data["metadata"]["format"] == "ego4d"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validation_success(self, sample_annotation):
        """Test validation of valid output."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        is_valid, errors = formatter.validate_output(result)
        assert is_valid
        assert len(errors) == 0

    def test_validation_failure(self):
        """Test validation catches missing fields."""
        formatter = Ego4DFormatter()

        invalid_data = {"video_metadata": {}}
        is_valid, errors = formatter.validate_output(invalid_data)
        assert not is_valid
        assert any("video_id" in e or "narrations" in e for e in errors)

    def test_multiple_actions(self, sample_annotation):
        """Test handling of multiple actions in segment."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        # Second narration has two actions
        narr = result["narrations"][1]
        actions = narr["actions"]
        assert len(actions) == 2

        assert actions[0]["verb"] == "take"
        assert actions[0]["noun"] == "milk"
        assert actions[1]["verb"] == "take"
        assert actions[1]["noun"] == "eggs"

    def test_empty_optional_fields(self):
        """Test handling of empty optional fields."""
        annotation = Annotation(
            id="minimal_test",
            video_id="video_min",
            video_path="/data/min.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=2.0,
                    caption="Minimal",
                )
            ],
        )

        formatter = Ego4DFormatter()
        result = formatter.format_annotation(annotation)

        # Should not raise errors
        assert result["video_id"] == "video_min"
        assert len(result["narrations"]) == 1


class TestEgo4DFormatCompatibility:
    """Test compatibility with Ego4D specification."""

    def test_narration_timestamp(self, sample_annotation):
        """Test that narration timestamp is calculated correctly."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        narr = result["narrations"][0]
        expected_timestamp = (5.0 + 10.0) / 2  # (start + end) / 2
        assert narr["timestamp"] == expected_timestamp

    def test_physical_properties_in_actions(self, sample_annotation):
        """Test physical properties are included in actions."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        action = result["narrations"][0]["actions"][0]
        assert "physical" in action
        assert action["physical"]["force"] == "gentle"
        assert action["physical"]["trajectory"] == "outward"

    def test_object_bbox_format(self, sample_annotation):
        """Test object bounding box format."""
        formatter = Ego4DFormatter()
        result = formatter.format_annotation(sample_annotation)

        obj = result["narrations"][0]["objects"][0]
        bbox = obj["bbox"]

        assert "x1" in bbox
        assert "y1" in bbox
        assert "x2" in bbox
        assert "y2" in bbox
        assert bbox["x1"] == 0.2
        assert bbox["y1"] == 0.1
