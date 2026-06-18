"""Tests for export adapters.

Tests conversion of Annotation objects to different training formats:
- LLaVA format
- OpenAI format
- ShareGPT format
"""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import Action, Annotation, QAPair, Segment, VideoMetadata
from dvas.export.adapters import (
    LLaVAAdapter,
    OpenAIAdapter,
    ShareGPTAdapter,
    ADAPTERS,
    export_annotations,
)


@pytest.fixture
def sample_annotation():
    """Create a sample annotation for testing."""
    return Annotation(
        id="test_ann_001",
        video_id="video_001",
        video_path="/data/videos/cooking.mp4",
        segments=[
            Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Picking up a knife",
                caption_dense="The person picks up a sharp kitchen knife from the counter",
                actions=[Action(verb="pick", noun="knife")],
                qa_pairs=[QAPair(question="What is being picked up?", answer="A knife")],
            ),
            Segment(
                start_time=5.0,
                end_time=10.0,
                caption="Cutting a carrot",
                caption_dense="The person cuts a carrot into small pieces",
                actions=[Action(verb="cut", noun="carrot")],
            ),
        ],
        metadata=VideoMetadata(
            video_id="video_001",
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.0,
            total_frames=300,
        ),
        source="teacher",
        model_version="gpt-5.5-2024-05-13",
    )


@pytest.fixture
def sample_annotations(sample_annotation):
    """Create a list of sample annotations."""
    ann2 = sample_annotation.model_copy()
    ann2.id = "test_ann_002"
    ann2.video_id = "video_002"
    return [sample_annotation, ann2]


class TestLLaVAAdapter:
    """Test LLaVA export format."""

    def test_export_single_annotation(self, sample_annotation):
        """Export a single annotation to LLaVA format."""
        adapter = LLaVAAdapter()

        # Use actual Annotation method (not mocked due to Pydantic complexity)
        result = adapter.export([sample_annotation])

        assert len(result) == 1
        assert "id" in result[0]
        assert result[0]["id"] == "test_ann_001"
        assert "video" in result[0]
        assert "conversations" in result[0]
        assert len(result[0]["conversations"]) > 0

    def test_export_multiple_annotations(self, sample_annotations):
        """Export multiple annotations to LLaVA format."""
        adapter = LLaVAAdapter()

        result = adapter.export(sample_annotations)

        assert len(result) == 2
        assert all("id" in item for item in result)
        assert all("conversations" in item for item in result)


class TestOpenAIAdapter:
    """Test OpenAI export format."""

    def test_export_single_annotation(self, sample_annotation):
        """Export a single annotation to OpenAI format."""
        adapter = OpenAIAdapter()

        result = adapter.export([sample_annotation])

        assert len(result) == 1
        assert "messages" in result[0]
        assert len(result[0]["messages"]) > 0
        # Check structure of messages
        for msg in result[0]["messages"]:
            assert "role" in msg
            assert "content" in msg


class TestShareGPTAdapter:
    """Test ShareGPT export format."""

    def test_export_format_structure(self, sample_annotation):
        """Verify ShareGPT format structure."""
        adapter = ShareGPTAdapter()
        result = adapter.export([sample_annotation])

        assert len(result) == 1
        exported = result[0]

        # Check top-level fields
        assert "id" in exported
        assert "video" in exported
        assert "conversations" in exported

        # Verify video path
        assert exported["video"] == "/data/videos/cooking.mp4"

        # Verify conversations structure
        conversations = exported["conversations"]
        assert len(conversations) == 4  # 2 segments * 2 messages each

        # Check first human message
        assert conversations[0]["from"] == "human"
        assert "<video>" in conversations[0]["value"]

        # Check first gpt response
        assert conversations[1]["from"] == "gpt"
        assert conversations[1]["value"] == "Picking up a knife"

    def test_export_empty_segments(self):
        """Export annotation with no segments."""
        adapter = ShareGPTAdapter()

        ann = Annotation(
            id="empty_ann",
            video_id="vid_empty",
            video_path="/data/empty.mp4",
            segments=[],
            metadata=VideoMetadata(
                video_id="vid_empty",
                fps=30.0,
                resolution=[224, 224],
                duration=1.0,
                total_frames=30,
            ),
        )

        result = adapter.export([ann])

        assert len(result) == 1
        assert result[0]["conversations"] == []


class TestExportAnnotations:
    """Test the main export_annotations function."""

    def test_export_to_llava(self, sample_annotations):
        """Export annotations to file in LLaVA format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            count = export_annotations(
                sample_annotations,
                Path(tmp.name),
                format="llava",
            )

        assert count == 2

        # Verify file contents
        with open(tmp.name, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Parse and verify structure
        for line in lines:
            data = json.loads(line)
            assert "id" in data
            assert "video" in data
            assert "conversations" in data

        # Cleanup
        Path(tmp.name).unlink()

    def test_export_unknown_format(self, sample_annotations):
        """Export to unknown format should raise error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            with pytest.raises(ValueError) as exc_info:
                export_annotations(
                    sample_annotations,
                    Path(tmp.name),
                    format="unknown_format",
                )

            assert "Unknown format" in str(exc_info.value)
            assert "llava" in str(exc_info.value)

        Path(tmp.name).unlink()

    def test_export_to_sharegpt(self, sample_annotation):
        """Export annotations to file in ShareGPT format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            count = export_annotations(
                [sample_annotation],
                Path(tmp.name),
                format="sharegpt",
            )

        assert count == 1

        # Verify file contents
        with open(tmp.name) as f:
            data = json.loads(f.readline())

        assert data["id"] == "test_ann_001"
        assert "conversations" in data

        # Cleanup
        Path(tmp.name).unlink()


class TestAdapterRegistry:
    """Test adapter registry."""

    def test_available_adapters(self):
        """Verify all expected adapters are registered."""
        assert "llava" in ADAPTERS
        assert "openai" in ADAPTERS
        assert "sharegpt" in ADAPTERS

    def test_adapter_classes(self):
        """Verify registered adapters are correct types."""
        assert ADAPTERS["llava"] is LLaVAAdapter
        assert ADAPTERS["openai"] is OpenAIAdapter
        assert ADAPTERS["sharegpt"] is ShareGPTAdapter
