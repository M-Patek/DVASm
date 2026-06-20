"""Tests for prompt quality attribution tracking."""

import pytest
from datetime import datetime, timezone

from dvas.data.schemas import (
    Annotation,
    VideoMetadata,
)
from dvas.prompts.attribution import (
    PromptAttributionRecord,
    PromptAttributionTracker,
    PromptPerformanceSummary,
)
from dvas.quality.schema import QualityDimension, QualityScores


class TestPromptAttributionRecord:
    """Test suite for PromptAttributionRecord."""

    def test_creation(self):
        """Test creating an attribution record."""
        record = PromptAttributionRecord(
            annotation_id="ann_1",
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            video_id="vid_1",
            quality_score=0.85,
            latency_ms=120.0,
            cost=0.02,
        )
        assert record.annotation_id == "ann_1"
        assert record.quality_score == 0.85
        assert record.latency_ms == 120.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        record = PromptAttributionRecord(
            annotation_id="ann_1",
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            video_id="vid_1",
            quality_score=0.85,
        )
        d = record.to_dict()
        assert d["annotation_id"] == "ann_1"
        assert d["quality_score"] == 0.85
        assert "timestamp" in d


class TestPromptPerformanceSummary:
    """Test suite for PromptPerformanceSummary."""

    def test_creation(self):
        """Test creating a performance summary."""
        summary = PromptPerformanceSummary(
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            total_annotations=10,
            avg_quality_score=0.85,
        )
        assert summary.prompt_id == "prompt_1"
        assert summary.total_annotations == 10
        assert summary.avg_quality_score == 0.85

    def test_to_dict(self):
        """Test converting to dictionary."""
        summary = PromptPerformanceSummary(
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            total_annotations=5,
            avg_quality_score=0.8,
        )
        d = summary.to_dict()
        assert d["prompt_id"] == "prompt_1"
        assert d["total_annotations"] == 5


class TestPromptAttributionTracker:
    """Test suite for PromptAttributionTracker."""

    def _create_annotation(self, ann_id="ann_1", video_id="vid_1"):
        """Helper to create a minimal annotation."""
        return Annotation(
            id=ann_id,
            video_id=video_id,
            video_path="/tmp/test.mp4",
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

    def test_record_attribution(self):
        """Test recording an attribution."""
        tracker = PromptAttributionTracker()
        ann = self._create_annotation()

        record = tracker.record_attribution(
            annotation=ann,
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            quality_scores=None,
            latency_ms=100.0,
            cost=0.01,
        )

        assert record.annotation_id == "ann_1"
        assert record.prompt_id == "prompt_1"
        assert record.prompt_version == "1.0.0"

    def test_record_attribution_with_quality(self):
        """Test recording attribution with quality scores."""
        tracker = PromptAttributionTracker()
        ann = self._create_annotation()
        quality = QualityScores(
            annotation_id="ann_1",
            video_id="vid_1",
            overall_score=0.85,
        )

        record = tracker.record_attribution(
            annotation=ann,
            prompt_id="prompt_1",
            prompt_version="1.0.0",
            quality_scores=quality,
        )

        assert record.quality_score == 0.85

    def test_get_records_for_prompt(self):
        """Test retrieving records for a prompt."""
        tracker = PromptAttributionTracker()
        ann1 = self._create_annotation("ann_1")
        ann2 = self._create_annotation("ann_2")

        tracker.record_attribution(ann1, "prompt_1", "1.0.0")
        tracker.record_attribution(ann2, "prompt_1", "1.0.0")

        records = tracker.get_records_for_prompt("prompt_1", "1.0.0")
        assert len(records) == 2

    def test_get_records_all_versions(self):
        """Test retrieving records across all versions."""
        tracker = PromptAttributionTracker()
        ann1 = self._create_annotation("ann_1")
        ann2 = self._create_annotation("ann_2")

        tracker.record_attribution(ann1, "prompt_1", "1.0.0")
        tracker.record_attribution(ann2, "prompt_1", "2.0.0")

        records = tracker.get_records_for_prompt("prompt_1")
        assert len(records) == 2

    def test_get_prompt_for_annotation(self):
        """Test finding which prompt produced an annotation."""
        tracker = PromptAttributionTracker()
        ann = self._create_annotation("ann_1")

        tracker.record_attribution(ann, "prompt_1", "1.0.0")

        prompt_key = tracker.get_prompt_for_annotation("ann_1")
        assert prompt_key == "prompt_1:1.0.0"

    def test_get_prompt_for_unknown_annotation(self):
        """Test finding prompt for unknown annotation."""
        tracker = PromptAttributionTracker()
        assert tracker.get_prompt_for_annotation("unknown") is None

    def test_compute_performance_summary(self):
        """Test computing performance summary."""
        tracker = PromptAttributionTracker()

        for i in range(5):
            ann = self._create_annotation(f"ann_{i}")
            tracker.record_attribution(
                ann, "prompt_1", "1.0.0",
                quality_scores=QualityScores(
                    annotation_id=f"ann_{i}",
                    video_id="vid_1",
                    overall_score=0.7 + i * 0.05,
                ),
                latency_ms=100.0 + i * 10,
                cost=0.01,
            )

        summary = tracker.compute_performance_summary("prompt_1", "1.0.0")
        assert summary.total_annotations == 5
        assert summary.avg_quality_score > 0
        assert summary.avg_latency_ms > 0
        assert summary.best_annotation_id is not None
        assert summary.worst_annotation_id is not None

    def test_compute_empty_summary(self):
        """Test computing summary with no records."""
        tracker = PromptAttributionTracker()
        summary = tracker.compute_performance_summary("prompt_1", "1.0.0")
        assert summary.total_annotations == 0
        assert summary.avg_quality_score == 0.0

    def test_compute_rollup(self):
        """Test computing rollup across prompts."""
        tracker = PromptAttributionTracker()

        for i in range(3):
            ann = self._create_annotation(f"ann_{i}")
            tracker.record_attribution(
                ann, "prompt_1", "1.0.0",
                quality_scores=QualityScores(
                    annotation_id=f"ann_{i}",
                    video_id="vid_1",
                    overall_score=0.8,
                ),
            )

        rollup = tracker.compute_rollup()
        assert len(rollup) > 0
        assert "prompt_1:1.0.0" in rollup

    def test_compare_prompts(self):
        """Test comparing two prompts."""
        tracker = PromptAttributionTracker()

        for i in range(3):
            ann = self._create_annotation(f"ann_a_{i}")
            tracker.record_attribution(
                ann, "prompt_A", "1.0.0",
                quality_scores=QualityScores(
                    annotation_id=f"ann_a_{i}",
                    video_id="vid_1",
                    overall_score=0.7,
                ),
                latency_ms=100.0,
                cost=0.01,
            )

        for i in range(3):
            ann = self._create_annotation(f"ann_b_{i}")
            tracker.record_attribution(
                ann, "prompt_B", "1.0.0",
                quality_scores=QualityScores(
                    annotation_id=f"ann_b_{i}",
                    video_id="vid_1",
                    overall_score=0.9,
                ),
                latency_ms=120.0,
                cost=0.015,
            )

        comparison = tracker.compare_prompts("prompt_A", "1.0.0", "prompt_B", "1.0.0")
        assert "prompt_a" in comparison
        assert "prompt_b" in comparison
        assert "difference" in comparison
        assert comparison["difference"]["quality"] > 0  # B is better
