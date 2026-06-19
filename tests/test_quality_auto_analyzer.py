"""Tests for automatic quality analyzer."""

import pytest
from datetime import datetime

from dvas.data.schemas import (
    Action,
    Annotation,
    Hand,
    Segment,
    VideoMetadata,
)
from dvas.quality.auto_analyzer import AutomaticQualityAnalyzer
from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityScores,
    QualityThresholds,
)


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


@pytest.fixture
def analyzer():
    """Create an analyzer instance."""
    return AutomaticQualityAnalyzer()


class TestAutomaticQualityAnalyzer:
    """Test AutomaticQualityAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_basic(self, analyzer, sample_annotation):
        """Test basic analysis."""
        scores = await analyzer.analyze(sample_annotation)
        assert scores.annotation_id == "ann_001"
        assert scores.video_id == "vid_001"
        assert scores.overall_score >= 0.0
        assert scores.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_all_dimensions(self, analyzer, sample_annotation):
        """Test that all dimensions are analyzed."""
        scores = await analyzer.analyze(sample_annotation)

        # Check all dimensions have scores
        assert scores.factuality_score.score >= 0.0
        assert scores.temporal_consistency_score.score >= 0.0
        assert scores.object_grounding_score.score >= 0.0
        assert scores.action_grounding_score.score >= 0.0
        assert scores.affordance_score.score >= 0.0
        assert scores.robotic_usefulness_score.score >= 0.0
        assert scores.language_clarity_score.score >= 0.0
        assert scores.parse_confidence_score.score >= 0.0
        assert scores.reviewer_confidence_score.score >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        annotations = [
            Annotation(
                id=f"ann_{i:03d}",
                video_id=f"vid_{i:03d}",
                video_path=f"/path/to/video_{i}.mp4",
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
                        caption=f"Action {i}",
                        actions=[Action(verb="pick", noun="cup")],
                        objects=[],
                    ),
                ],
            )
            for i in range(3)
        ]

        results = await analyzer.analyze_batch(annotations)
        assert len(results) == 3
        assert "ann_000" in results
        assert "ann_001" in results
        assert "ann_002" in results


class TestFactualityAnalysis:
    """Test factuality dimension analysis."""

    @pytest.mark.asyncio
    async def test_no_actions_detected(self, analyzer):
        """Test factuality when no actions present."""
        annotation = Annotation(
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
                    caption="A person is standing",
                    actions=[],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        assert "no_actions_detected" in scores.factuality_score.issues

    @pytest.mark.asyncio
    async def test_caption_action_consistency(self, analyzer):
        """Test consistency between captions and actions."""
        annotation = Annotation(
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
                    caption="Person picks up a cup",
                    actions=[Action(verb="pick", noun="cup")],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        # Action verb "pick" should be found in caption
        assert scores.factuality_score.score > 0.5


class TestTemporalConsistencyAnalysis:
    """Test temporal consistency dimension analysis."""

    @pytest.mark.asyncio
    async def test_single_segment_consistency(self, analyzer):
        """Test consistency with single segment."""
        annotation = Annotation(
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
                    caption="Action",
                    actions=[],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        # Single segment should have neutral score
        assert scores.temporal_consistency_score.score > 0.5

    @pytest.mark.asyncio
    async def test_overlapping_segments(self, analyzer):
        """Test detection of overlapping segments."""
        annotation = Annotation(
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
                    caption="Action 1",
                    actions=[],
                    objects=[],
                ),
                Segment(
                    start_time=3.0,  # Overlaps with first
                    end_time=8.0,
                    caption="Action 2",
                    actions=[],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        # Should detect overlap
        assert len(scores.temporal_consistency_score.details.get("overlaps", [])) > 0


class TestObjectGroundingAnalysis:
    """Test object grounding dimension analysis."""

    @pytest.mark.asyncio
    async def test_no_objects(self, analyzer):
        """Test when no objects present."""
        annotation = Annotation(
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
                    caption="Action",
                    actions=[],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        assert "no_objects_detected" in scores.object_grounding_score.issues
        assert scores.object_grounding_score.score < 0.5


class TestActionGroundingAnalysis:
    """Test action grounding dimension analysis."""

    @pytest.mark.asyncio
    async def test_no_temporal_info(self, analyzer):
        """Test when actions lack temporal info."""
        annotation = Annotation(
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
                    caption="Action",
                    actions=[Action(verb="pick", noun="cup")],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        # Action without start/end times
        assert scores.action_grounding_score.details.get("actions_with_times", 0) == 0


class TestLanguageClarityAnalysis:
    """Test language clarity dimension analysis."""

    @pytest.mark.asyncio
    async def test_good_caption(self, analyzer):
        """Test with good quality caption."""
        annotation = Annotation(
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
                    actions=[Action(verb="pick", noun="cup")],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        # Good caption should have decent score
        assert scores.language_clarity_score.score > 0.5

    @pytest.mark.asyncio
    async def test_empty_caption(self, analyzer):
        """Test with empty caption."""
        annotation = Annotation(
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
                    caption="",
                    actions=[],
                    objects=[],
                ),
            ],
        )

        scores = await analyzer.analyze(annotation)
        assert "no_valid_captions" in scores.language_clarity_score.issues
