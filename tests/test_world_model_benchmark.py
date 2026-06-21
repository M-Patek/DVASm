"""Tests for world_model benchmark module."""

import json
import tempfile
from pathlib import Path

import pytest

from dvas.data.schemas import Action, Segment
from dvas.world_model.benchmark import (
    BenchmarkResult,
    BenchmarkSuiteResult,
    CausalRelationBenchmark,
    CounterfactualBenchmark,
    StatePredictionBenchmark,
    WorldModelAnnotationBenchmark,
    load_benchmark_results,
    run_benchmarks,
)
from dvas.world_model.state_repr import WorldState


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_create(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            benchmark_name="test",
            metric_name="accuracy",
            value=0.95,
            unit="ratio",
        )
        assert result.benchmark_name == "test"
        assert result.value == 0.95

    def test_to_dict(self):
        """Test serialization."""
        result = BenchmarkResult(
            benchmark_name="test",
            metric_name="accuracy",
            value=0.95,
            unit="ratio",
            metadata={"test_id": 1},
        )
        data = result.to_dict()
        assert data["benchmark_name"] == "test"
        assert data["value"] == 0.95
        assert data["metadata"]["test_id"] == 1


class TestBenchmarkSuiteResult:
    """Tests for BenchmarkSuiteResult class."""

    def test_create(self):
        """Test creating suite result."""
        suite = BenchmarkSuiteResult(suite_name="test_suite")
        assert suite.suite_name == "test_suite"
        assert len(suite.results) == 0

    def test_to_dict(self):
        """Test serialization."""
        result = BenchmarkResult("test", "accuracy", 0.95)
        suite = BenchmarkSuiteResult(
            suite_name="test_suite",
            results=[result],
            summary={"mean": 0.95},
        )
        data = suite.to_dict()
        assert data["suite_name"] == "test_suite"
        assert len(data["results"]) == 1

    def test_to_json(self):
        """Test JSON serialization."""
        suite = BenchmarkSuiteResult(suite_name="test_suite")
        json_str = suite.to_json()
        assert "test_suite" in json_str
        # Verify valid JSON
        data = json.loads(json_str)
        assert data["suite_name"] == "test_suite"


class TestStatePredictionBenchmark:
    """Tests for StatePredictionBenchmark class."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic benchmark run."""
        benchmark = StatePredictionBenchmark()

        test_cases = [
            {
                "initial_state": WorldState(timestamp=0.0),
                "action": Action(verb="push", noun="block"),
                "expected_state": WorldState(timestamp=1.0),
            }
        ]

        results = await benchmark.run(test_cases)

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_run_empty(self):
        """Test with empty test cases."""
        benchmark = StatePredictionBenchmark()
        results = await benchmark.run([])
        assert results == []


class TestCausalRelationBenchmark:
    """Tests for CausalRelationBenchmark class."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic benchmark run."""
        benchmark = CausalRelationBenchmark()

        test_cases = [
            {
                "segment": Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Push the block",
                    actions=[Action(verb="push", noun="block")],
                ),
                "expected_causes": ["push causes movement"],
            }
        ]

        results = await benchmark.run(test_cases)

        assert isinstance(results, list)
        # Should have precision, recall, f1
        assert len(results) >= 3


class TestCounterfactualBenchmark:
    """Tests for CounterfactualBenchmark class."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic benchmark run."""
        benchmark = CounterfactualBenchmark()

        test_cases = [
            {
                "segment": Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Push the block",
                ),
                "actual_action": Action(verb="push", noun="block"),
                "alternative_actions": [Action(verb="pull", noun="block")],
            }
        ]

        results = await benchmark.run(test_cases)

        assert isinstance(results, list)
        assert len(results) > 0


class TestWorldModelAnnotationBenchmark:
    """Tests for WorldModelAnnotationBenchmark class."""

    @pytest.mark.asyncio
    async def test_run_quick_test(self):
        """Test quick test run."""
        benchmark = WorldModelAnnotationBenchmark()

        results = await benchmark.run_quick_test()

        assert isinstance(results, BenchmarkSuiteResult)
        assert results.suite_name == "world_model_annotation"
        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_run_all(self):
        """Test running all benchmarks."""
        benchmark = WorldModelAnnotationBenchmark()

        state_cases = [
            {
                "initial_state": WorldState(timestamp=0.0),
                "action": Action(verb="push", noun="block"),
                "expected_state": WorldState(timestamp=1.0),
            }
        ]

        causal_cases = [
            {
                "segment": Segment(
                    start_time=0.0,
                    end_time=1.0,
                    caption="Push",
                    actions=[Action(verb="push", noun="block")],
                ),
                "expected_causes": ["push causes movement"],
            }
        ]

        cf_cases = [
            {
                "segment": Segment(start_time=0.0, end_time=1.0, caption="Test"),
                "actual_action": Action(verb="push", noun="block"),
                "alternative_actions": [Action(verb="pull", noun="block")],
            }
        ]

        results = await benchmark.run_all(
            state_test_cases=state_cases,
            causal_test_cases=causal_cases,
            counterfactual_test_cases=cf_cases,
        )

        assert isinstance(results, BenchmarkSuiteResult)
        assert results.duration_seconds > 0

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WorldModelAnnotationBenchmark(output_dir=Path(tmpdir))

            result = BenchmarkSuiteResult(
                suite_name="test",
                results=[BenchmarkResult("test", "accuracy", 0.95)],
            )

            path = benchmark.save_results(result, "test_results.json")

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
                assert data["suite_name"] == "test"

    def test_generate_report(self):
        """Test report generation."""
        benchmark = WorldModelAnnotationBenchmark()

        result = BenchmarkSuiteResult(
            suite_name="test",
            results=[
                BenchmarkResult("state", "mae", 0.1, "meters"),
                BenchmarkResult("state", "rmse", 0.15, "meters"),
            ],
            summary={
                "state": {"mean": 0.125, "count": 2},
                "overall_score": 0.85,
            },
        )

        report = benchmark.generate_report(result)

        assert "World Model Annotation Benchmark Report" in report
        assert "0.1250" in report or "0.125" in report
        assert "0.8500" in report or "0.85" in report

    def test_compute_summary(self):
        """Test summary computation."""
        benchmark = WorldModelAnnotationBenchmark()

        results = [
            BenchmarkResult("state", "mae", 0.1),
            BenchmarkResult("state", "mae", 0.2),
            BenchmarkResult("causal", "f1", 0.8),
            BenchmarkResult("causal", "f1", 0.9),
        ]

        summary = benchmark._compute_summary(results)

        assert "benchmarks_by_type" in summary
        assert "state" in summary["benchmarks_by_type"]
        assert "causal" in summary["benchmarks_by_type"]


class TestLoadBenchmarkResults:
    """Tests for load_benchmark_results function."""

    def test_load_valid_file(self):
        """Test loading valid results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "suite_name": "test_suite",
                "results": [
                    {
                        "benchmark_name": "test",
                        "metric_name": "accuracy",
                        "value": 0.95,
                        "unit": "ratio",
                        "metadata": {},
                        "timestamp": 1234567890,
                    }
                ],
                "summary": {"mean": 0.95},
                "duration_seconds": 10.0,
                "timestamp": 1234567890,
            }

            path = Path(tmpdir) / "results.json"
            with open(path, "w") as f:
                json.dump(data, f)

            loaded = load_benchmark_results(path)

            assert isinstance(loaded, BenchmarkSuiteResult)
            assert loaded.suite_name == "test_suite"
            assert len(loaded.results) == 1


class TestRunBenchmarks:
    """Tests for run_benchmarks convenience function."""

    @pytest.mark.asyncio
    async def test_run_quick_test(self):
        """Test running quick test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = await run_benchmarks(
                output_dir=Path(tmpdir),
                quick_test=True,
            )

            assert isinstance(results, BenchmarkSuiteResult)
            assert len(results.results) > 0

            # Check that results file was created
            result_files = list(Path(tmpdir).glob("*.json"))
            assert len(result_files) > 0
