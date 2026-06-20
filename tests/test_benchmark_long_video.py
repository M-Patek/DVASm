"""Tests for long video benchmark."""

import tempfile

import pytest

from dvas.benchmarks.long_video import LongVideoSegment, LongVideoBenchmark


class TestLongVideoSegment:
    """Test LongVideoSegment dataclass."""

    def test_creation(self):
        """Test basic creation."""
        segment = LongVideoSegment(
            video_id="lv_001",
            start_time=0.0,
            end_time=10.0,
            segment_index=0,
            total_segments=5,
            caption="A person cutting vegetables",
            key_events=["pick_up_knife", "cut_tomato"],
        )
        assert segment.video_id == "lv_001"
        assert segment.start_time == 0.0
        assert segment.end_time == 10.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        segment = LongVideoSegment(
            video_id="lv_002",
            start_time=10.0,
            end_time=20.0,
            segment_index=1,
            total_segments=3,
            caption="Washing hands",
            key_events=["turn_on_tap"],
        )
        data = segment.to_dict()
        assert data["video_id"] == "lv_002"
        assert data["caption"] == "Washing hands"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "video_id": "lv_003",
            "start_time": 20.0,
            "end_time": 30.0,
            "segment_index": 2,
            "total_segments": 4,
            "caption": "Stirring soup",
            "key_events": ["pick_up_spoon"],
        }
        segment = LongVideoSegment.from_dict(data)
        assert segment.video_id == "lv_003"
        assert segment.caption == "Stirring soup"


class TestLongVideoBenchmark:
    """Test LongVideoBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LongVideoBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "long_video"
        assert temp_benchmark.results_dir.exists()

    def test_evaluate_temporal_coherence(self, temp_benchmark):
        """Test temporal coherence evaluation."""
        segments = [
            LongVideoSegment("v1", 0, 10, 0, 3, "cutting", ["e1"]),
            LongVideoSegment("v1", 10, 20, 1, 3, "washing", ["e2"]),
            LongVideoSegment("v1", 20, 30, 2, 3, "stirring", ["e3"]),
        ]
        predictions = ["cutting vegetables", "washing hands", "stirring soup"]
        ground_truth = ["cutting vegetables", "washing hands", "stirring soup"]
        result = temp_benchmark.evaluate_temporal_coherence(predictions, ground_truth, segments)
        assert "temporal_coherence" in result

    def test_evaluate_event_detection(self, temp_benchmark):
        """Test event detection evaluation."""
        predicted_events = [["event1", "event2"], ["event3"]]
        ground_truth_events = [["event1", "event2"], ["event3", "event4"]]
        result = temp_benchmark.evaluate_event_detection(predicted_events, ground_truth_events)
        assert "event_precision" in result
        assert "event_recall" in result

    def test_evaluate_by_duration(self, temp_benchmark):
        """Test duration-based evaluation."""
        segments = [
            LongVideoSegment("v1", 0, 5, 0, 3, "short", []),
            LongVideoSegment("v1", 5, 25, 1, 3, "medium", []),
            LongVideoSegment("v1", 25, 65, 2, 3, "long", []),
        ]
        predictions = ["short action", "medium action", "long action"]
        ground_truth = ["short action", "medium action", "long action"]
        result = temp_benchmark.evaluate_by_duration(predictions, ground_truth, segments)
        assert isinstance(result, dict)

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        segments = [
            LongVideoSegment("v1", 0, 10, 0, 2, "cutting", ["e1"]),
            LongVideoSegment("v1", 10, 20, 1, 2, "washing", ["e2"]),
        ]
        predictions = ["cutting", "washing"]
        ground_truth = ["cutting", "washing"]
        result = temp_benchmark.run_benchmark("test_model", predictions, ground_truth, segments)
        assert result.benchmark_name == "long_video"
        assert result.model_id == "test_model"

    def test_empty_segments(self, temp_benchmark):
        """Test with empty segments."""
        result = temp_benchmark.run_benchmark("test_model", [], [], [])
        assert result.benchmark_name == "long_video"
