"""Tests for review queue management."""

import pytest
from datetime import datetime, timedelta

from dvas.data.schemas import Annotation, VideoMetadata
from dvas.quality.review_queue import (
    DisagreementCase,
    DisagreementQueue,
    HumanReviewQueue,
    LowQualityQuarantine,
    QuarantineItem,
    ReviewItem,
    ReviewPriority,
    ReviewStatus,
)
from dvas.quality.schema import DimensionScore, QualityDimension, QualityScores


@pytest.fixture
def sample_annotation():
    """Create a sample annotation."""
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
        segments=[],
    )


@pytest.fixture
def sample_quality_scores():
    """Create sample quality scores."""
    return QualityScores(
        annotation_id="ann_001",
        video_id="vid_001",
        factuality_score=DimensionScore(dimension=QualityDimension.FACTUALITY, score=0.4),
        language_clarity_score=DimensionScore(
            dimension=QualityDimension.LANGUAGE_CLARITY, score=0.8
        ),
    )


class TestReviewPriority:
    """Test ReviewPriority enum."""

    def test_priority_values(self):
        """Test priority values."""
        assert ReviewPriority.CRITICAL.value == "critical"
        assert ReviewPriority.HIGH.value == "high"
        assert ReviewPriority.MEDIUM.value == "medium"
        assert ReviewPriority.LOW.value == "low"


class TestReviewStatus:
    """Test ReviewStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.ASSIGNED.value == "assigned"
        assert ReviewStatus.COMPLETED.value == "completed"


class TestReviewItem:
    """Test ReviewItem dataclass."""

    def test_basic_creation(self):
        """Test creating a review item."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.HIGH,
        )
        assert item.annotation_id == "ann_001"
        assert item.priority == ReviewPriority.HIGH
        assert item.status == ReviewStatus.PENDING

    def test_due_date_calculation(self):
        """Test due date is calculated from priority."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.HIGH,
        )
        assert item.due_by is not None
        # HIGH priority should be due in 1 day
        expected_due = item.created_at + timedelta(days=1)
        assert item.due_by == expected_due

    def test_is_overdue(self):
        """Test overdue detection."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.CRITICAL,
            due_by=datetime.utcnow() - timedelta(hours=1),
        )
        assert item.is_overdue() is True

    def test_assign(self):
        """Test assignment."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.HIGH,
        )
        item.assign("reviewer_001")
        assert item.assigned_to == "reviewer_001"
        assert item.status == ReviewStatus.ASSIGNED
        assert item.assigned_at is not None

    def test_complete(self):
        """Test completion."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.HIGH,
        )
        result = {"decision": "approved", "score": 0.9}
        item.complete(result, "Looks good")
        assert item.status == ReviewStatus.COMPLETED
        assert item.review_result == result
        assert item.reviewer_notes == "Looks good"

    def test_to_dict(self):
        """Test serialization."""
        item = ReviewItem(
            annotation_id="ann_001",
            video_id="vid_001",
            priority=ReviewPriority.HIGH,
        )
        data = item.to_dict()
        assert data["annotation_id"] == "ann_001"
        assert data["priority"] == "high"


class TestHumanReviewQueue:
    """Test HumanReviewQueue."""

    @pytest.mark.asyncio
    async def test_add_item(self, sample_annotation, sample_quality_scores):
        """Test adding item to queue."""
        queue = HumanReviewQueue()
        item = await queue.add(
            sample_annotation,
            sample_quality_scores,
            priority=ReviewPriority.HIGH,
        )
        assert item.annotation_id == "ann_001"

    @pytest.mark.asyncio
    async def test_get_next(self, sample_annotation):
        """Test getting next item for review."""
        queue = HumanReviewQueue()
        await queue.add(sample_annotation, priority=ReviewPriority.HIGH)

        item = await queue.get_next("reviewer_001")
        assert item is not None
        assert item.assigned_to == "reviewer_001"

    @pytest.mark.asyncio
    async def test_complete_review(self, sample_annotation):
        """Test completing a review."""
        queue = HumanReviewQueue()
        await queue.add(sample_annotation, priority=ReviewPriority.HIGH)
        await queue.get_next("reviewer_001")

        completed = await queue.complete_review(
            "ann_001",
            {"decision": "approved"},
            "Good quality",
        )
        assert completed is not None
        assert completed.status == ReviewStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_queue_stats(self, sample_annotation):
        """Test getting queue stats."""
        queue = HumanReviewQueue()
        await queue.add(sample_annotation, priority=ReviewPriority.HIGH)

        # Create a second annotation with different ID
        from copy import deepcopy

        ann2 = deepcopy(sample_annotation)
        ann2.id = "ann_002"
        await queue.add(ann2, priority=ReviewPriority.LOW)

        stats = await queue.get_queue_stats()
        assert stats["total_items"] == 2
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["low"] == 1


class TestDisagreementCase:
    """Test DisagreementCase dataclass."""

    def test_basic_creation(self):
        """Test creating a disagreement case."""
        case = DisagreementCase(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Person picks up a cup",
            student_output="Person picks up a glass",
            teacher_model="gpt-5.5",
            student_model="student-v1",
            disagreement_type="object",
            disagreement_score=0.8,
        )
        assert case.annotation_id == "ann_001"
        assert case.status == "pending"

    def test_to_dict(self):
        """Test serialization."""
        case = DisagreementCase(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Output 1",
            student_output="Output 2",
            teacher_model="gpt-5.5",
            student_model="student-v1",
            disagreement_type="caption",
            disagreement_score=0.7,
        )
        data = case.to_dict()
        assert data["annotation_id"] == "ann_001"
        assert data["disagreement_score"] == 0.7


class TestDisagreementQueue:
    """Test DisagreementQueue."""

    @pytest.mark.asyncio
    async def test_add_case(self):
        """Test adding a case."""
        queue = DisagreementQueue()
        case = await queue.add_case(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Output 1",
            student_output="Output 2",
            disagreement_type="caption",
            disagreement_score=0.8,
        )
        assert case.annotation_id == "ann_001"

    @pytest.mark.asyncio
    async def test_resolve_case(self):
        """Test resolving a case."""
        queue = DisagreementQueue()
        await queue.add_case(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Output 1",
            student_output="Output 2",
            disagreement_type="caption",
            disagreement_score=0.8,
        )

        resolved = await queue.resolve_case(
            "ann_001",
            correct_version="teacher",
            resolution_notes="Teacher was correct",
        )
        assert resolved is not None
        assert resolved.status == "resolved"
        assert resolved.correct_version == "teacher"

    @pytest.mark.asyncio
    async def test_get_pending_cases(self):
        """Test getting pending cases."""
        queue = DisagreementQueue()
        await queue.add_case(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Output 1",
            student_output="Output 2",
            disagreement_score=0.8,
        )
        await queue.add_case(
            annotation_id="ann_002",
            video_id="vid_002",
            teacher_output="Output 1",
            student_output="Output 2",
            disagreement_score=0.3,
        )

        # Get only high disagreement cases
        pending = await queue.get_pending_cases(min_disagreement=0.5)
        assert len(pending) == 1
        assert pending[0].annotation_id == "ann_001"

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test getting statistics."""
        queue = DisagreementQueue()
        await queue.add_case(
            annotation_id="ann_001",
            video_id="vid_001",
            teacher_output="Output 1",
            student_output="Output 2",
            disagreement_type="caption",
            disagreement_score=0.8,
        )

        stats = await queue.get_stats()
        assert stats["total_cases"] == 1
        assert stats["pending"] == 1


class TestQuarantineItem:
    """Test QuarantineItem dataclass."""

    def test_basic_creation(self):
        """Test creating a quarantine item."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        item = QuarantineItem(
            annotation_id="ann_001",
            video_id="vid_001",
            quality_scores=scores,
            quarantine_reason="Low quality",
        )
        assert item.annotation_id == "ann_001"
        assert item.status == "quarantined"

    def test_to_dict(self):
        """Test serialization."""
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )
        item = QuarantineItem(
            annotation_id="ann_001",
            video_id="vid_001",
            quality_scores=scores,
            quarantine_reason="Low quality",
        )
        data = item.to_dict()
        assert data["annotation_id"] == "ann_001"
        assert data["quarantine_reason"] == "Low quality"


class TestLowQualityQuarantine:
    """Test LowQualityQuarantine."""

    @pytest.mark.asyncio
    async def test_quarantine(self, sample_annotation):
        """Test adding to quarantine."""
        quarantine = LowQualityQuarantine()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )

        item = await quarantine.quarantine(
            sample_annotation,
            scores,
            reason="Overall score too low",
        )
        assert item.annotation_id == "ann_001"
        assert item.status == "quarantined"

    @pytest.mark.asyncio
    async def test_approve(self, sample_annotation):
        """Test manually approving quarantined item."""
        quarantine = LowQualityQuarantine()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )

        await quarantine.quarantine(sample_annotation, scores, reason="Low quality")
        approved = await quarantine.approve(
            "ann_001",
            approved_by="admin_001",
            notes="Approved after manual review",
        )

        assert approved is not None
        assert approved.status == "approved"
        assert approved.metadata["approved_by"] == "admin_001"

    @pytest.mark.asyncio
    async def test_get_active_items(self, sample_annotation):
        """Test getting active quarantined items."""
        quarantine = LowQualityQuarantine()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )

        await quarantine.quarantine(sample_annotation, scores, reason="Low quality")
        active = await quarantine.get_active_items()

        assert len(active) == 1
        assert active[0].annotation_id == "ann_001"

    @pytest.mark.asyncio
    async def test_stats(self, sample_annotation):
        """Test getting quarantine stats."""
        quarantine = LowQualityQuarantine()
        scores = QualityScores(
            annotation_id="ann_001",
            video_id="vid_001",
        )

        await quarantine.quarantine(sample_annotation, scores, reason="Low quality")
        stats = await quarantine.get_stats()

        assert stats["total_items"] == 1
        assert stats["quarantined"] == 1
