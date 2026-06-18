"""Integration tests for DVAS.

End-to-end tests that verify multiple components work together.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from dvas.testing import create_test_annotation, assert_dict_subset

from dvas.data.schemas import (
    Action,
    Annotation,
    BoundingBox,
    Hand,
    Object,
    QAPair,
    Segment,
    VideoMetadata,
)


class TestEndToEndAnnotation:
    """End-to-end annotation workflow tests."""

    def test_full_annotation_workflow(self):
        """Test complete annotation workflow."""
        # 1. Create video metadata
        metadata = VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=30.0,
            total_frames=900,
        )

        # 2. Create segments with actions and objects
        segments = [
            Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Person enters kitchen",
                actions=[
                    Action(verb="enter", noun="kitchen", hand=Hand.UNKNOWN),
                ],
                objects=[
                    Object(name="person", confidence=0.95),
                ],
            ),
            Segment(
                start_time=5.0,
                end_time=15.0,
                caption="Person prepares food",
                actions=[
                    Action(verb="cut", noun="vegetables", hand=Hand.RIGHT),
                    Action(verb="pick", noun="knife", hand=Hand.RIGHT),
                ],
                objects=[
                    Object(name="knife", confidence=0.92),
                    Object(name="vegetables", confidence=0.88),
                ],
            ),
            Segment(
                start_time=15.0,
                end_time=25.0,
                caption="Person cooks food",
                actions=[
                    Action(verb="stir", noun="pot", hand=Hand.LEFT),
                ],
                objects=[
                    Object(name="pot", confidence=0.90),
                ],
            ),
        ]

        # 3. Create annotation
        annotation = Annotation(
            id="workflow_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=segments,
            metadata=metadata,
            source="teacher",
            tags=["cooking", "kitchen"],
        )

        # 4. Verify annotation
        assert annotation.id == "workflow_test"
        assert len(annotation.segments) == 3
        assert annotation.get_total_duration() == 25.0

        # 5. Convert to LLaVA format
        llava = annotation.to_llava_format()
        assert "conversations" in llava
        assert len(llava["conversations"]) == 6  # 3 segments * 2 (human + gpt)

        # 6. Convert to OpenAI format
        openai = annotation.to_openai_format()
        assert "messages" in openai
        assert len(openai["messages"]) == 6

        # 7. Extract unique verbs
        verbs = annotation.get_action_verbs()
        assert "cut" in verbs
        assert "pick" in verbs
        assert "stir" in verbs

    def test_annotation_with_qa_pairs(self):
        """Test annotation with Q&A pairs."""
        segment = Segment(
            start_time=0.0,
            end_time=10.0,
            caption="A cooking scene",
            qa_pairs=[
                QAPair(
                    question="What is the person doing?",
                    answer="Cooking food",
                    question_type="what",
                ),
                QAPair(
                    question="How is the food prepared?",
                    answer="By cutting and stirring",
                    question_type="how",
                ),
            ],
        )

        annotation = Annotation(
            id="qa_test",
            video_id="vid_qa",
            video_path="/path/to/video.mp4",
            segments=[segment],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        assert len(annotation.segments[0].qa_pairs) == 2
        assert annotation.segments[0].qa_pairs[0].question_type == "what"

    def test_annotation_with_bounding_boxes(self):
        """Test annotation with bounding boxes."""
        bbox1 = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        bbox2 = BoundingBox(x1=0.3, y1=0.4, x2=0.7, y2=0.8)

        segment = Segment(
            start_time=0.0,
            end_time=5.0,
            caption="Objects in scene",
            objects=[
                Object(name="knife", bbox=bbox1, confidence=0.95),
                Object(name="plate", bbox=bbox2, confidence=0.88),
            ],
        )

        annotation = Annotation(
            id="bbox_test",
            video_id="vid_bbox",
            video_path="/path/to/video.mp4",
            segments=[segment],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        assert annotation.segments[0].objects[0].bbox.to_list() == [0.1, 0.2, 0.5, 0.6]
        assert annotation.segments[0].objects[1].bbox.to_list() == [0.3, 0.4, 0.7, 0.8]


class TestAnnotationExport:
    """Test annotation export workflows."""

    def test_annotation_to_storage(self, tmp_path):
        """Test saving and loading annotation from storage."""
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=tmp_path / "annotations")

        annotation = Annotation(
            id="export_test",
            video_id="vid_export",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(
                    start_time=0.0,
                    end_time=5.0,
                    caption="Test segment",
                ),
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        # Save
        store.save(annotation, source="gold")

        # Load
        loaded = store.load("export_test", source="gold")
        assert loaded is not None
        assert loaded.id == "export_test"
        assert loaded.video_id == "vid_export"
        assert len(loaded.segments) == 1
        assert loaded.segments[0].caption == "Test segment"

    def test_annotation_export_to_jsonl(self, tmp_path):
        """Test exporting annotations to JSONL."""
        from dvas.data.storage import AnnotationStore

        store = AnnotationStore(root_path=tmp_path / "annotations")

        # Create multiple annotations
        for i in range(3):
            annotation = Annotation(
                id=f"export_{i}",
                video_id=f"vid_{i}",
                video_path=f"/path/to/video_{i}.mp4",
                segments=[
                    Segment(
                        start_time=0.0,
                        end_time=5.0,
                        caption=f"Segment {i}",
                    ),
                ],
                metadata=VideoMetadata(
                    fps=30.0,
                    resolution=[1920, 1080],
                    duration=5.0,
                    total_frames=150,
                ),
            )
            store.save(annotation, source="gold")

        # Export to JSONL
        output_path = tmp_path / "export.jsonl"
        count = store.export_to_jsonl(output_path, source="gold", format="llava")
        assert count == 3
        assert output_path.exists()

        # Verify content
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3


class TestPipelineIntegration:
    """Integration tests for annotation pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_with_mock_teacher(self):
        """Test pipeline with mock teacher model."""
        from dvas.pipeline.core import AnnotationPipeline

        mock_teacher = MagicMock()
        mock_teacher.model_name = "gpt-5.5"
        mock_teacher.annotate = AsyncMock(return_value={
            "text": "A person is cooking in the kitchen.",
            "model": "gpt-5.5",
        })

        pipeline = AnnotationPipeline(
            teacher_model=mock_teacher,
            num_frames=8,
        )

        assert pipeline.teacher == mock_teacher
        assert pipeline.num_frames == 8

    def test_pipeline_checkpoint_integration(self, tmp_path):
        """Test pipeline with checkpoint manager."""
        from dvas.pipeline.checkpoint import CheckpointManager

        checkpoint = CheckpointManager(tmp_path / "checkpoint.json")

        # Mark some items as processed
        checkpoint.mark_processed("video_001")
        checkpoint.mark_processed("video_002")
        checkpoint.mark_failed("video_003", "API error")

        # Save and reload
        checkpoint.save()

        new_checkpoint = CheckpointManager(tmp_path / "checkpoint.json")
        loaded = new_checkpoint.load()
        assert loaded is True
        assert "video_001" in new_checkpoint.processed_ids
        assert "video_002" in new_checkpoint.processed_ids
        assert len(new_checkpoint.failed_items) == 1


class TestDataFlow:
    """Test data flow through the system."""

    def test_video_metadata_to_annotation(self):
        """Test video metadata flows correctly to annotation."""
        metadata = VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=60.0,
            total_frames=1800,
            codec="h264",
            bitrate=5000000,
            has_audio=True,
        )

        annotation = Annotation(
            id="flow_test",
            video_id="vid_flow",
            video_path="/path/to/video.mp4",
            metadata=metadata,
        )

        # Verify metadata is preserved
        assert annotation.metadata.fps == 30.0
        assert annotation.metadata.resolution == [1920, 1080]
        assert annotation.metadata.duration == 60.0
        assert annotation.metadata.total_frames == 1800
        assert annotation.metadata.codec == "h264"
        assert annotation.metadata.has_audio is True

    def test_segment_actions_flow(self):
        """Test actions flow through segments."""
        actions = [
            Action(verb="cut", noun="vegetables", hand=Hand.RIGHT),
            Action(verb="pick", noun="knife", hand=Hand.RIGHT),
            Action(verb="stir", noun="pot", hand=Hand.LEFT),
        ]

        segment = Segment(
            start_time=0.0,
            end_time=10.0,
            caption="Cooking scene",
            actions=actions,
        )

        # Verify all actions are preserved
        assert len(segment.actions) == 3
        assert segment.actions[0].verb == "cut"
        assert segment.actions[1].verb == "pick"
        assert segment.actions[2].verb == "stir"

        # Verify hands
        assert segment.actions[0].hand == Hand.RIGHT
        assert segment.actions[2].hand == Hand.LEFT


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_bounding_box_rejected(self):
        """Test that invalid bounding boxes are rejected."""
        with pytest.raises(ValueError):
            BoundingBox(x1=-0.1, y1=0.5, x2=0.5, y2=0.5)

        with pytest.raises(ValueError):
            BoundingBox(x1=0.5, y1=0.5, x2=1.5, y2=0.5)

    def test_invalid_segment_rejected(self):
        """Test that invalid segments are rejected."""
        with pytest.raises(ValueError):
            Segment(
                start_time=10.0,
                end_time=5.0,
                caption="Invalid",
            )

    def test_empty_annotation_valid(self):
        """Test that empty annotation is valid."""
        annotation = Annotation(
            id="empty",
            video_id="vid_empty",
            video_path="/path/to/video.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=5.0,
                total_frames=150,
            ),
        )

        assert annotation.get_total_duration() == 0.0
        assert annotation.get_action_verbs() == []
        assert annotation.get_object_names() == []

    def test_annotation_with_no_segments(self):
        """Test annotation with no segments."""
        annotation = Annotation(
            id="no_segments",
            video_id="vid_none",
            video_path="/path/to/video.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        llava = annotation.to_llava_format()
        assert llava["conversations"] == []


class TestDictSubsetAssertion:
    """Test assert_dict_subset helper."""

    def test_subset_match(self):
        """Test matching subset."""
        actual = {
            "status": "success",
            "data": {"id": "123", "name": "test"},
            "meta": {"page": 1},
        }
        expected = {
            "status": "success",
            "data": {"id": "123"},
        }
        assert_dict_subset(actual, expected)

    def test_subset_mismatch(self):
        """Test non-matching subset."""
        actual = {
            "status": "error",
            "data": {"id": "123"},
        }
        expected = {
            "status": "success",
        }
        with pytest.raises(AssertionError):
            assert_dict_subset(actual, expected)

    def test_subset_missing_key(self):
        """Test missing key in actual."""
        actual = {
            "status": "success",
        }
        expected = {
            "status": "success",
            "data": {"id": "123"},
        }
        with pytest.raises(AssertionError):
            assert_dict_subset(actual, expected)
