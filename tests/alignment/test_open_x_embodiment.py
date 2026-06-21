"""Alignment tests for Open X-Embodiment format compatibility.

Verifies that DVAS annotations can be correctly exported to and
imported from Open X-Embodiment format.
"""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    AnnotationStandard,
    EmbodimentAction,
    Hand,
    PhysicalProperties,
    Segment,
    VideoMetadata,
)
from dvas.export.formats import OpenXFormatter


@pytest.fixture
def sample_annotation():
    """Create a sample annotation for testing."""
    return Annotation(
        id="test_001",
        video_id="video_001",
        video_path="/data/videos/video_001.mp4",
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.5,
            total_frames=315,
            camera_type="egocentric",
            environment="kitchen",
        ),
        segments=[
            Segment(
                start_time=1.0,
                end_time=3.0,
                caption="Pick up the cup",
                actions=[
                    Action(
                        verb="pick",
                        noun="cup",
                        hand=Hand.RIGHT,
                        start_time=1.5,
                        end_time=2.5,
                        embodiment=EmbodimentAction(
                            gripper_pose=[0.5, 0.3, 0.2, 0.1, 0.0, 0.0],
                            joint_target=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            action_space="absolute",
                            gripper_state="close",
                        ),
                        instrument="hand",
                        physical=PhysicalProperties(
                            force="gentle",
                            contact_type="grasp",
                        ),
                    )
                ],
                scene_type="kitchen",
                lighting="natural",
            )
        ],
        annotation_standard=AnnotationStandard.OPEN_X_EMBODIMENT,
    )


@pytest.fixture
def sample_annotations():
    """Create a list of sample annotations."""
    return [
        Annotation(
            id=f"test_{i:03d}",
            video_id=f"video_{i:03d}",
            video_path=f"/data/videos/video_{i:03d}.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0 + i,
                total_frames=300 + i * 30,
            ),
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=2.0,
                    caption=f"Action {i}",
                    actions=[
                        Action(
                            verb="pick",
                            noun="object",
                            hand=Hand.RIGHT,
                        )
                    ],
                )
            ],
        )
        for i in range(3)
    ]


class TestOpenXEmbodimentFormatter:
    """Test Open X-Embodiment formatter compatibility."""

    def test_formatter_initialization(self):
        """Test formatter can be initialized with different configs."""
        formatter = OpenXFormatter(action_space="absolute")
        assert formatter.action_space == "absolute"

        formatter_delta = OpenXFormatter(action_space="delta")
        assert formatter_delta.action_space == "delta"

    def test_single_annotation_formatting(self, sample_annotation):
        """Test formatting a single annotation."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        # Check required fields
        assert "steps" in result
        assert "metadata" in result

        # Check metadata
        assert result["metadata"]["video_id"] == "video_001"
        assert result["metadata"]["camera_type"] == "egocentric"
        assert result["metadata"]["environment"] == "kitchen"

        # Check steps
        assert len(result["steps"]) == 1
        step = result["steps"][0]
        assert step["language_instruction"] == "Pick up the cup"

    def test_step_structure(self, sample_annotation):
        """Test that steps have proper Open X structure."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        step = result["steps"][0]

        # Check observation
        assert "observation" in step
        assert "state" in step["observation"]

        # Check action
        assert "action" in step
        assert isinstance(step["action"], list)

        # Check standard fields
        assert "discount" in step
        assert "reward" in step
        assert "timestamp" in step

    def test_embodiment_action_conversion(self, sample_annotation):
        """Test that embodiment actions are preserved."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        step = result["steps"][0]
        assert len(step["action"]) == 1

        action = step["action"][0]
        assert action["verb"] == "pick"
        assert action["noun"] == "cup"

        # Check embodiment data
        assert "embodiment_action" in action
        emb = action["embodiment_action"]
        assert emb["gripper_pose"] == [0.5, 0.3, 0.2, 0.1, 0.0, 0.0]
        assert emb["action_space"] == "absolute"
        assert emb["gripper_state"] == "close"

    def test_batch_formatting(self, sample_annotations):
        """Test formatting multiple annotations."""
        formatter = OpenXFormatter()
        results = formatter.format_annotations(sample_annotations)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["metadata"]["video_id"] == f"video_{i:03d}"

    def test_export_to_jsonl(self, sample_annotations):
        """Test exporting to JSONL format."""
        formatter = OpenXFormatter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            count = formatter.export_to_file(sample_annotations, temp_path, format="jsonl")
            assert count == 3

            # Verify file contents
            with open(temp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 3

                # Verify each line is valid JSON
                for line in lines:
                    data = json.loads(line)
                    assert "steps" in data
                    assert "metadata" in data
        finally:
            temp_path.unlink(missing_ok=True)

    def test_export_to_json(self, sample_annotations):
        """Test exporting to JSON format."""
        formatter = OpenXFormatter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            count = formatter.export_to_file(sample_annotations, temp_path, format="json")
            assert count == 3

            # Verify file contents
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "episodes" in data
                assert len(data["episodes"]) == 3
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validation_success(self, sample_annotation):
        """Test validation of valid output."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        is_valid, errors = formatter.validate_output(result)
        assert is_valid
        assert len(errors) == 0

    def test_validation_failure(self):
        """Test validation catches missing fields."""
        formatter = OpenXFormatter()

        # Invalid data - missing steps
        invalid_data = {"metadata": {}}
        is_valid, errors = formatter.validate_output(invalid_data)
        assert not is_valid
        assert any("steps" in e for e in errors)

    def test_step_validation(self, sample_annotation):
        """Test validation of step structure."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        step = result["steps"][0]
        required_fields = [
            "observation",
            "action",
            "language_instruction",
            "timestamp",
            "discount",
            "reward",
        ]
        for field in required_fields:
            assert field in step, f"Missing field: {field}"

    def test_action_space_configuration(self, sample_annotation):
        """Test different action space configurations."""
        formatter_abs = OpenXFormatter(action_space="absolute")
        formatter_delta = OpenXFormatter(action_space="delta")

        result_abs = formatter_abs.format_annotation(sample_annotation)
        result_delta = formatter_delta.format_annotation(sample_annotation)

        # Check action space is reflected in output
        assert (
            result_abs["steps"][0]["action"][0]["embodiment_action"]["action_space"] == "absolute"
        )
        assert result_delta["steps"][0]["action"][0]["embodiment_action"]["action_space"] == "delta"

    def test_empty_actions_handling(self):
        """Test handling of segments with no actions."""
        annotation = Annotation(
            id="empty_test",
            video_id="video_empty",
            video_path="/data/empty.mp4",
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
                    caption="Empty segment",
                    actions=[],  # No actions
                )
            ],
        )

        formatter = OpenXFormatter()
        result = formatter.format_annotation(annotation)

        step = result["steps"][0]
        assert step["action"] == []

    def test_physical_properties_conversion(self, sample_annotation):
        """Test that physical properties are converted."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        action = result["steps"][0]["action"][0]
        assert "physical" in action
        assert action["physical"]["force"] == "gentle"
        assert action["physical"]["contact_type"] == "grasp"


class TestOpenXEmbodimentRoundTrip:
    """Test round-trip conversion (export -> import concept)."""

    def test_format_preserves_core_fields(self, sample_annotation):
        """Test that core fields are preserved in formatting."""
        formatter = OpenXFormatter()
        result = formatter.format_annotation(sample_annotation)

        # Core fields should be preserved
        assert result["metadata"]["video_id"] == sample_annotation.video_id
        assert result["metadata"]["fps"] == sample_annotation.metadata.fps
        assert result["steps"][0]["language_instruction"] == sample_annotation.segments[0].caption

    def test_format_structural_integrity(self, sample_annotations):
        """Test structural integrity of formatted output."""
        formatter = OpenXFormatter()

        for annotation in sample_annotations:
            result = formatter.format_annotation(annotation)

            is_valid, errors = formatter.validate_output(result)
            assert is_valid, f"Validation failed: {errors}"
