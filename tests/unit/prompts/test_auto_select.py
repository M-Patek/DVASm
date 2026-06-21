"""Tests for auto-selection logic."""

from dvas.prompts.auto_select import (
    AutoSelector,
    DomainDetector,
    VideoCharacteristics,
)
from dvas.prompts.registry import PromptDomain, PromptRegistry


class TestDomainDetector:
    """Test suite for DomainDetector."""

    def test_detect_kitchen_from_filename(self):
        """Test detecting kitchen domain from filename."""
        detector = DomainDetector()
        from pathlib import Path

        domain = detector.detect_from_filename(Path("cooking_video.mp4"))
        assert domain == PromptDomain.KITCHEN

    def test_detect_robot_from_filename(self):
        """Test detecting robot domain from filename."""
        detector = DomainDetector()
        from pathlib import Path

        domain = detector.detect_from_filename(Path("robot_grasping.mp4"))
        assert domain == PromptDomain.ROBOT

    def test_detect_general_from_filename(self):
        """Test fallback to general domain."""
        detector = DomainDetector()
        from pathlib import Path

        domain = detector.detect_from_filename(Path("random_video.mp4"))
        assert domain == PromptDomain.GENERAL

    def test_detect_from_metadata(self):
        """Test detecting domain from metadata."""
        detector = DomainDetector()
        metadata = {
            "description": "A cooking video in the kitchen",
            "title": "Kitchen tutorial",
            "tags": ["food", "recipe"],
        }
        domain = detector.detect_from_metadata(metadata)
        assert domain == PromptDomain.KITCHEN


class TestVideoCharacteristics:
    """Test suite for VideoCharacteristics."""

    def test_complexity_score(self):
        """Test complexity score computation."""
        chars = VideoCharacteristics(
            duration_seconds=120,
            scene_count=8,
            motion_score=0.8,
            object_count=15,
        )
        score = chars.complexity_score
        assert 0 <= score <= 1
        assert score > 0.5  # Should be complex

    def test_simple_video(self):
        """Test simple video characteristics."""
        chars = VideoCharacteristics(
            duration_seconds=5,
            scene_count=1,
            motion_score=0.1,
            object_count=2,
        )
        assert chars.is_simple is True
        assert chars.is_complex is False

    def test_complex_video(self):
        """Test complex video characteristics."""
        chars = VideoCharacteristics(
            duration_seconds=120,
            scene_count=10,
            motion_score=0.9,
            object_count=25,
        )
        assert chars.is_complex is True
        assert chars.is_simple is False


class TestAutoSelector:
    """Test suite for AutoSelector."""

    def test_select_returns_prompt(self):
        """Test that select returns a prompt."""
        registry = PromptRegistry()
        registry.create(
            name="kitchen_prompt",
            template="Describe kitchen video",
            domain=PromptDomain.KITCHEN,
            tags=["caption"],
        )

        selector = AutoSelector(registry=registry)
        from pathlib import Path

        result = selector.select(
            video_path=Path("cooking.mp4"),
            task_type="caption",
        )
        assert result is not None

    def test_select_prefers_domain(self):
        """Test that selection prefers matching domain."""
        registry = PromptRegistry()
        registry.create(
            name="kitchen",
            template="Kitchen template",
            domain=PromptDomain.KITCHEN,
        )
        registry.create(
            name="general",
            template="General template",
            domain=PromptDomain.GENERAL,
        )

        selector = AutoSelector(registry=registry)
        from pathlib import Path

        result = selector.select(
            video_path=Path("cooking.mp4"),
            preferred_domain=PromptDomain.KITCHEN,
        )
        assert result is not None
        assert result.metadata.domain == PromptDomain.KITCHEN

    def test_rank_prompts(self):
        """Test ranking prompts by suitability."""
        registry = PromptRegistry()
        registry.create(
            name="high_quality",
            template="High quality",
            domain=PromptDomain.KITCHEN,
        )
        registry.create(
            name="low_quality",
            template="Low quality",
            domain=PromptDomain.KITCHEN,
        )

        # Update quality scores
        for p in registry.list_all():
            if p.metadata.name == "high_quality":
                p.avg_quality_score = 0.9
            else:
                p.avg_quality_score = 0.3

        selector = AutoSelector(registry=registry)
        from pathlib import Path

        ranked = selector.rank_prompts(Path("cooking.mp4"), domain=PromptDomain.KITCHEN)
        assert len(ranked) > 0
        assert ranked[0][0].metadata.name == "high_quality"

    def test_select_for_characteristics(self):
        """Test selecting based on video characteristics."""
        registry = PromptRegistry()
        registry.create(
            name="complex_prompt",
            template="Complex template",
            domain=PromptDomain.GENERAL,
        )

        selector = AutoSelector(registry=registry)
        chars = VideoCharacteristics(
            duration_seconds=120,
            scene_count=10,
            motion_score=0.9,
            object_count=20,
        )
        result = selector.select_for_characteristics(chars, domain=PromptDomain.GENERAL)
        assert result is not None

    def test_exploration_rate(self):
        """Test setting exploration rate."""
        selector = AutoSelector()
        selector.set_exploration_rate(0.3)
        assert selector._exploration_rate == 0.3

    def test_exploration_rate_clamping(self):
        """Test that exploration rate is clamped to [0, 1]."""
        selector = AutoSelector()
        selector.set_exploration_rate(1.5)
        assert selector._exploration_rate == 1.0

        selector.set_exploration_rate(-0.5)
        assert selector._exploration_rate == 0.0
