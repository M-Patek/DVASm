"""Tests for EPIC-KITCHENS benchmark."""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.benchmarks.epic_kitchens import EPIKitchensBenchmark, EPICAction


class TestEPICAction:
    """Test EPICAction dataclass."""

    def test_creation(self):
        """Test basic creation."""
        action = EPICAction(
            verb="cut",
            noun="tomato",
            start_frame=100,
            end_frame=200,
            participant_id="P01",
            video_id="P01_01",
        )
        assert action.verb == "cut"
        assert action.noun == "tomato"
        assert action.start_frame == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        action = EPICAction(
            verb="wash",
            noun="hands",
            start_frame=50,
            end_frame=100,
            participant_id="P02",
            video_id="P02_01",
        )
        data = action.to_dict()
        assert data["verb"] == "wash"
        assert data["noun"] == "hands"
        assert data["start_frame"] == 50

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "verb": "stir",
            "noun": "soup",
            "start_frame": 300,
            "end_frame": 400,
            "participant_id": "P03",
            "video_id": "P03_01",
        }
        action = EPICAction.from_dict(data)
        assert action.verb == "stir"
        assert action.noun == "soup"
        assert action.start_frame == 300


class TestEPIKitchensBenchmark:
    """Test EPIKitchensBenchmark."""

    @pytest.fixture
    def temp_benchmark(self):
        """Create temporary benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EPIKitchensBenchmark(tmpdir)

    def test_init(self, temp_benchmark):
        """Test initialization."""
        assert temp_benchmark.name == "epic_kitchens"
        assert temp_benchmark.results_dir.exists()

    def test_evaluate_verb_accuracy(self, temp_benchmark):
        """Test verb accuracy evaluation."""
        predictions = ["cut", "wash", "stir", "open"]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
            EPICAction("open", "fridge", 0, 100, "P01", "V01"),
        ]

        accuracy = temp_benchmark.evaluate_verb_accuracy(predictions, ground_truth)
        assert accuracy == 1.0

    def test_evaluate_verb_accuracy_partial(self, temp_benchmark):
        """Test verb accuracy with partial matches."""
        predictions = ["cut", "wash", "stir", "open"]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("pour", "water", 0, 100, "P01", "V01"),  # mismatch
            EPICAction("close", "fridge", 0, 100, "P01", "V01"),  # mismatch
        ]

        accuracy = temp_benchmark.evaluate_verb_accuracy(predictions, ground_truth)
        assert accuracy == 0.5

    def test_evaluate_noun_accuracy(self, temp_benchmark):
        """Test noun accuracy evaluation."""
        predictions = ["tomato", "hands", "soup", "fridge"]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
            EPICAction("open", "fridge", 0, 100, "P01", "V01"),
        ]

        accuracy = temp_benchmark.evaluate_noun_accuracy(predictions, ground_truth)
        assert accuracy == 1.0

    def test_evaluate_action_retrieval(self, temp_benchmark):
        """Test action retrieval evaluation."""
        predictions = [("cut", "tomato"), ("wash", "hands"), ("stir", "soup")]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
        ]

        result = temp_benchmark.evaluate_action_retrieval(predictions, ground_truth)
        assert result["top1_accuracy"] == 1.0

    def test_evaluate_action_retrieval_partial(self, temp_benchmark):
        """Test action retrieval with partial matches."""
        predictions = [("cut", "tomato"), ("wash", "hands"), ("stir", "water")]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
        ]

        result = temp_benchmark.evaluate_action_retrieval(predictions, ground_truth)
        assert result["top1_accuracy"] == pytest.approx(2/3, abs=0.01)

    def test_run_benchmark(self, temp_benchmark):
        """Test full benchmark run."""
        verb_predictions = ["cut", "wash", "stir"]
        noun_predictions = ["tomato", "hands", "soup"]
        ground_truth = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
        ]

        result = temp_benchmark.run_benchmark(
            "test_model",
            verb_predictions,
            noun_predictions,
            ground_truth,
        )

        assert result.benchmark_name == "epic_kitchens"
        assert result.model_id == "test_model"
        assert result.metrics["verb_accuracy"] == 1.0
        assert result.metrics["noun_accuracy"] == 1.0

    def test_run_benchmark_mismatch_length(self, temp_benchmark):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            temp_benchmark.run_benchmark(
                "test_model",
                ["cut", "wash"],
                ["tomato", "hands", "soup"],
                [
                    EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
                    EPICAction("wash", "hands", 0, 100, "P01", "V01"),
                    EPICAction("stir", "soup", 0, 100, "P01", "V01"),
                ],
            )

    def test_get_top_verbs(self, temp_benchmark):
        """Test getting top verbs."""
        annotations = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("cut", "onion", 0, 100, "P01", "V01"),
            EPICAction("wash", "hands", 0, 100, "P01", "V01"),
            EPICAction("wash", "dishes", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
        ]

        top_verbs = temp_benchmark.get_top_verbs(annotations, n=2)
        assert len(top_verbs) == 2
        assert top_verbs[0][0] == "cut" or top_verbs[0][0] == "wash"

    def test_get_top_nouns(self, temp_benchmark):
        """Test getting top nouns."""
        annotations = [
            EPICAction("cut", "tomato", 0, 100, "P01", "V01"),
            EPICAction("wash", "tomato", 0, 100, "P01", "V01"),
            EPICAction("stir", "soup", 0, 100, "P01", "V01"),
            EPICAction("open", "fridge", 0, 100, "P01", "V01"),
        ]

        top_nouns = temp_benchmark.get_top_nouns(annotations, n=2)
        assert len(top_nouns) <= 2

    def test_empty_predictions(self, temp_benchmark):
        """Test with empty predictions."""
        accuracy = temp_benchmark.evaluate_verb_accuracy([], [])
        assert accuracy == 0.0
