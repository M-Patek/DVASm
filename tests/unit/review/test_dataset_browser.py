"""Tests for dataset browser filtering and statistics."""

from datetime import datetime, timezone

import pytest

from dvas.data.schemas import Action, Annotation, Hand, Object, Segment, VideoMetadata
from dvas.quality.schema import DimensionScore, QualityDimension, QualityScores
from dvas.review.dataset_browser import (
    DatasetBrowser,
    DatasetFilter,
    DatasetStatistics,
    SortField,
    SortOrder,
)


class TestDatasetBrowser:
    """Test suite for DatasetBrowser."""

    def _make_annotation(
        self,
        ann_id: str = "ann1",
        video_id: str = "vid1",
        source: str = "teacher",
        quality_score: float = 0.8,
        tags: list = None,
        created_at: datetime = None,
    ) -> Annotation:
        """Create a test annotation."""
        metadata = VideoMetadata(
            fps=30.0,
            resolution=[1920, 1080],
            duration=10.0,
            total_frames=300,
        )
        segment = Segment(
            start_time=0.0,
            end_time=5.0,
            caption="Test caption",
            actions=[Action(verb="take", noun="cup", hand=Hand.RIGHT)],
            objects=[Object(name="cup")],
        )
        return Annotation(
            id=ann_id,
            video_id=video_id,
            video_path="test.mp4",
            metadata=metadata,
            source=source,
            quality_score=quality_score,
            tags=tags or [],
            created_at=created_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
            segments=[segment],
        )

    def test_filter_by_quality_score(self):
        """Test filtering by quality score range."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", quality_score=0.9))
        browser.add_annotation(self._make_annotation("a2", quality_score=0.4))
        browser.add_annotation(self._make_annotation("a3", quality_score=0.6))

        filter_criteria = DatasetFilter(min_quality_score=0.5)
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 2
        assert all((r.quality_score or 0) >= 0.5 for r in results)

    def test_filter_by_max_quality_score(self):
        """Test filtering by max quality score."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", quality_score=0.9))
        browser.add_annotation(self._make_annotation("a2", quality_score=0.4))

        filter_criteria = DatasetFilter(max_quality_score=0.5)
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 1
        assert results[0].id == "a2"

    def test_filter_by_source(self):
        """Test filtering by source."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", source="teacher"))
        browser.add_annotation(self._make_annotation("a2", source="student"))

        filter_criteria = DatasetFilter(source="teacher")
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 1
        assert results[0].source == "teacher"

    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", created_at=datetime(2024, 1, 15, tzinfo=timezone.utc)))
        browser.add_annotation(self._make_annotation("a2", created_at=datetime(2024, 2, 15, tzinfo=timezone.utc)))
        browser.add_annotation(self._make_annotation("a3", created_at=datetime(2024, 3, 15, tzinfo=timezone.utc)))

        filter_criteria = DatasetFilter(
            date_from=datetime(2024, 1, 20, tzinfo=timezone.utc),
            date_to=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 1
        assert results[0].id == "a2"

    def test_filter_by_tags(self):
        """Test filtering by tags."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", tags=["batch1"]))
        browser.add_annotation(self._make_annotation("a2", tags=["batch2"]))

        filter_criteria = DatasetFilter(tags=["batch1"])
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 1
        assert results[0].id == "a1"

    def test_filter_by_video_ids(self):
        """Test filtering by video IDs."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", video_id="vid1"))
        browser.add_annotation(self._make_annotation("a2", video_id="vid2"))

        filter_criteria = DatasetFilter(video_ids=["vid1"])
        results = browser.filter_annotations(filter_criteria)

        assert len(results) == 1
        assert results[0].video_id == "vid1"

    def test_pagination(self):
        """Test pagination."""
        browser = DatasetBrowser()
        for i in range(25):
            browser.add_annotation(self._make_annotation(f"a{i}", quality_score=0.8))

        result = browser.paginate(browser.filter_annotations(), page=1, page_size=10)
        assert len(result.items) == 10
        assert result.total == 25
        assert result.total_pages == 3

        result = browser.paginate(browser.filter_annotations(), page=2, page_size=10)
        assert len(result.items) == 10

        result = browser.paginate(browser.filter_annotations(), page=3, page_size=10)
        assert len(result.items) == 5

    def test_sort_by_quality_score(self):
        """Test sorting by quality score."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", quality_score=0.3))
        browser.add_annotation(self._make_annotation("a2", quality_score=0.9))
        browser.add_annotation(self._make_annotation("a3", quality_score=0.5))

        sorted_anns = browser.sort_annotations(
            browser.filter_annotations(),
            sort_field=SortField.QUALITY_SCORE,
            sort_order=SortOrder.DESC,
        )

        scores = [a.quality_score for a in sorted_anns]
        assert scores == [0.9, 0.5, 0.3]

    def test_sort_by_created_at(self):
        """Test sorting by created_at."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
        browser.add_annotation(self._make_annotation("a2", created_at=datetime(2024, 3, 1, tzinfo=timezone.utc)))
        browser.add_annotation(self._make_annotation("a3", created_at=datetime(2024, 2, 1, tzinfo=timezone.utc)))

        sorted_anns = browser.sort_annotations(
            browser.filter_annotations(),
            sort_field=SortField.CREATED_AT,
            sort_order=SortOrder.ASC,
        )

        assert sorted_anns[0].id == "a1"
        assert sorted_anns[1].id == "a3"
        assert sorted_anns[2].id == "a2"

    def test_statistics(self):
        """Test statistics aggregation."""
        browser = DatasetBrowser()
        browser.add_annotation(self._make_annotation("a1", source="teacher", quality_score=0.9))
        browser.add_annotation(self._make_annotation("a2", source="student", quality_score=0.5))

        stats = browser.get_statistics()

        assert stats.total_annotations == 2
        assert stats.avg_quality_score == pytest.approx(0.7, abs=0.01)
        assert stats.min_quality_score == 0.5
        assert stats.max_quality_score == 0.9
        assert stats.by_source["teacher"] == 1
        assert stats.by_source["student"] == 1
        assert stats.total_segments == 2
        assert stats.total_actions == 2
        assert stats.total_objects == 2

    def test_browse_combined(self):
        """Test combined browse (filter + sort + paginate)."""
        browser = DatasetBrowser()
        for i in range(10):
            browser.add_annotation(self._make_annotation(f"a{i}", quality_score=0.5 + i * 0.04))

        result = browser.browse(
            dataset_filter=DatasetFilter(min_quality_score=0.6),
            sort_field=SortField.QUALITY_SCORE,
            sort_order=SortOrder.DESC,
            page=1,
            page_size=5,
        )

        assert len(result.items) == 5
        assert result.total == 7  # Items with score >= 0.6 (a3-a9)
        scores = [a.quality_score for a in result.items]
        assert scores == sorted(scores, reverse=True)
