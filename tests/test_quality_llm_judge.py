"""Tests for LLM judge pipeline."""

import pytest

from dvas.data.schemas import (
    Action,
    Annotation,
    Hand,
    Segment,
    VideoMetadata,
)
from dvas.quality.llm_judge import LLMJudgeConfig, LLMJudgePipeline, LLMJudgePrompts
from dvas.quality.schema import QualityDimension, QualityScores


@pytest.fixture
def sample_annotation():
    """Create a sample annotation for testing."""
    return Annotation(
        id="ann_001",
        video_id="vid_001",
        video_path="/path/to/video.mp4",
        metadata=VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=60.0,
            total_frames=1800,
        ),
        segments=[
            Segment(
                start_time=0.0,
                end_time=5.0,
                caption="Person picks up a cup from the table",
                actions=[
                    Action(verb="pick", noun="cup", hand=Hand.RIGHT),
                ],
                objects=[],
            ),
        ],
    )


class TestLLMJudgeConfig:
    """Test LLMJudgeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = LLMJudgeConfig()
        assert config.model_name == "gpt-5.5"
        assert config.temperature == 0.2
        assert config.max_tokens == 2048
        assert config.use_structured_output is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMJudgeConfig(
            model_name="claude-opus-4-8",
            temperature=0.5,
            max_tokens=1024,
        )
        assert config.model_name == "claude-opus-4-8"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024


class TestLLMJudgePrompts:
    """Test LLMJudgePrompts."""

    def test_format_annotation(self, sample_annotation):
        """Test annotation formatting."""
        text = LLMJudgePrompts.format_annotation(sample_annotation)
        assert "ann_001" in text
        assert "Person picks up" in text
        assert "pick" in text
        assert "cup" in text

    def test_format_segments(self, sample_annotation):
        """Test segment formatting."""
        text = LLMJudgePrompts.format_segments(sample_annotation.segments)
        assert "0.0s - 5.0s" in text or "Segment 1:" in text

    def test_format_captions(self, sample_annotation):
        """Test caption formatting."""
        text = LLMJudgePrompts.format_captions(sample_annotation.segments)
        assert "Person picks up" in text

    def test_format_actions(self, sample_annotation):
        """Test action formatting."""
        text = LLMJudgePrompts.format_actions(sample_annotation.segments)
        assert "pick cup" in text

    def test_format_objects_empty(self, sample_annotation):
        """Test object formatting with no objects."""
        text = LLMJudgePrompts.format_objects(sample_annotation.segments)
        assert "No objects" in text

    def test_format_affordances(self, sample_annotation):
        """Test affordance formatting."""
        text = LLMJudgePrompts.format_affordances(sample_annotation.segments)
        assert "pick cup" in text


class TestLLMJudgePipeline:
    """Test LLMJudgePipeline."""

    def test_pipeline_creation(self):
        """Test creating pipeline."""
        config = LLMJudgeConfig()
        pipeline = LLMJudgePipeline(config=config)
        assert pipeline.config == config

    def test_extract_json(self):
        """Test JSON extraction from text."""
        pipeline = LLMJudgePipeline()

        # Test with code block
        text = '```json\n{"score": 0.8, "confidence": 0.9}\n```'
        result = pipeline._extract_json(text)
        assert result == {"score": 0.8, "confidence": 0.9}

        # Test without code block
        text = '{"score": 0.7, "confidence": 0.8}'
        result = pipeline._extract_json(text)
        assert result == {"score": 0.7, "confidence": 0.8}

        # Test invalid JSON
        text = "not valid json"
        result = pipeline._extract_json(text)
        assert result is None

    def test_set_dimension_score(self):
        """Test setting dimension scores."""
        pipeline = LLMJudgePipeline()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        from dvas.quality.schema import DimensionScore

        score = DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=0.85,
        )
        pipeline._set_dimension_score(scores, QualityDimension.FACTUALITY, score)
        assert scores.factuality_score.score == 0.85


class TestPromptTemplates:
    """Test that prompt templates are valid."""

    def test_factuality_template(self):
        """Test factuality prompt template."""
        text = LLMJudgePrompts.FACTUALITY_TEMPLATE.format(
            video_id="vid_001",
            annotation_text="Test annotation",
        )
        assert "FACTUALITY" in text
        assert "vid_001" in text
        assert "JSON format" in text

    def test_temporal_consistency_template(self):
        """Test temporal consistency prompt template."""
        text = LLMJudgePrompts.TEMPORAL_CONSISTENCY_TEMPLATE.format(
            video_id="vid_001",
            segment_times="Segment 1: 0-5s",
        )
        assert "TEMPORAL CONSISTENCY" in text
        assert "JSON format" in text

    def test_object_grounding_template(self):
        """Test object grounding prompt template."""
        text = LLMJudgePrompts.OBJECT_GROUNDING_TEMPLATE.format(
            video_id="vid_001",
            objects_text="Object: cup",
        )
        assert "OBJECT GROUNDING" in text

    def test_action_grounding_template(self):
        """Test action grounding prompt template."""
        text = LLMJudgePrompts.ACTION_GROUNDING_TEMPLATE.format(
            video_id="vid_001",
            actions_text="Action: pick cup",
        )
        assert "ACTION GROUNDING" in text

    def test_affordance_template(self):
        """Test affordance prompt template."""
        text = LLMJudgePrompts.AFFORDANCE_TEMPLATE.format(
            video_id="vid_001",
            affordance_text="pick cup",
        )
        assert "AFFORDANCE" in text

    def test_robotic_usefulness_template(self):
        """Test robotic usefulness prompt template."""
        text = LLMJudgePrompts.ROBOTIC_USEFULNESS_TEMPLATE.format(
            video_id="vid_001",
            annotation_text="Test annotation",
        )
        assert "ROBOTIC USEFULNESS" in text

    def test_language_clarity_template(self):
        """Test language clarity prompt template."""
        text = LLMJudgePrompts.LANGUAGE_CLARITY_TEMPLATE.format(
            video_id="vid_001",
            captions_text="Test caption",
        )
        assert "LANGUAGE CLARITY" in text

    def test_comprehensive_template(self):
        """Test comprehensive evaluation template."""
        text = LLMJudgePrompts.COMPREHENSIVE_TEMPLATE.format(
            video_id="vid_001",
            annotation_text="Test annotation",
        )
        assert "OVERALL QUALITY" in text
        assert "factuality" in text
        assert "temporal_consistency" in text
