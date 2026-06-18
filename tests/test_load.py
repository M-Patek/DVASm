"""Load and performance tests for DVAS.

Tests that verify system performance under load.
"""

import asyncio
import time

import pytest

from dvas.testing import LoadTester, benchmark


class TestLoadTesting:
    """Test load testing utilities."""

    @pytest.mark.asyncio
    async def test_load_tester_basic(self):
        """Test basic load test execution."""
        tester = LoadTester()

        async def dummy_request():
            await asyncio.sleep(0.001)

        result = await tester.run(
            target=dummy_request,
            concurrent_users=2,
            requests_per_user=5,
        )

        assert result.total_requests == 10
        assert result.successful_requests == 10
        assert result.failed_requests == 0
        assert result.requests_per_second > 0

    @pytest.mark.asyncio
    async def test_load_tester_with_failures(self):
        """Test load test with some failures."""
        tester = LoadTester()
        counter = [0]

        async def failing_request():
            counter[0] += 1
            if counter[0] % 3 == 0:
                raise ValueError("Simulated failure")
            await asyncio.sleep(0.001)

        result = await tester.run(
            target=failing_request,
            concurrent_users=2,
            requests_per_user=5,
        )

        assert result.total_requests == 10
        assert result.failed_requests > 0
        assert result.error_rate > 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_load_tester_ramp_up(self):
        """Test load test with ramp up."""
        tester = LoadTester()
        start_times = []

        async def timed_request():
            start_times.append(time.time())
            await asyncio.sleep(0.001)

        result = await tester.run(
            target=timed_request,
            concurrent_users=3,
            requests_per_user=2,
            ramp_up=0.1,
        )

        assert result.total_requests == 6
        # With ramp up, users should start at different times
        assert len(start_times) == 6

    @pytest.mark.asyncio
    async def test_load_tester_empty(self):
        """Test load test with zero requests."""
        tester = LoadTester()

        async def dummy_request():
            pass

        result = await tester.run(
            target=dummy_request,
            concurrent_users=0,
            requests_per_user=0,
        )

        assert result.total_requests == 0
        assert result.requests_per_second == 0.0
        assert "No requests completed" in result.errors[0]


class TestBenchmarking:
    """Test benchmarking utilities."""

    def test_benchmark_function(self):
        """Test benchmarking a simple function."""
        def slow_add(a, b):
            time.sleep(0.001)
            return a + b

        result = benchmark(slow_add, 1, 2, iterations=10)

        assert result.name == "slow_add"
        assert result.iterations == 10
        assert result.total_time > 0
        assert result.avg_time > 0
        assert result.min_time > 0
        assert result.max_time >= result.min_time
        assert result.avg_time >= result.min_time
        assert result.avg_time <= result.max_time

    def test_benchmark_result_to_dict(self):
        """Test benchmark result serialization."""
        def identity(x):
            return x

        result = benchmark(identity, 42, iterations=5)
        d = result.to_dict()

        assert d["name"] == "identity"
        assert d["iterations"] == 5
        assert "avg_time" in d
        assert "min_time" in d
        assert "max_time" in d


class TestPerformanceAnnotations:
    """Performance tests for annotation operations."""

    def test_annotation_creation_performance(self):
        """Benchmark annotation creation."""
        from dvas.data.schemas import Annotation, Segment, VideoMetadata

        def create_annotations():
            for i in range(100):
                Annotation(
                    id=f"test_{i:03d}",
                    video_id=f"vid_{i:03d}",
                    video_path=f"/path/to/video_{i}.mp4",
                    segments=[
                        Segment(
                            start_time=0.0,
                            end_time=5.0,
                            caption=f"Segment {i}",
                        ),
                    ],
                    metadata=VideoMetadata(
                        fps=30.0,
                        resolution=[1920, 1080],
                        duration=10.0,
                        total_frames=300,
                    ),
                )

        result = benchmark(create_annotations, iterations=10)
        assert result.avg_time < 1.0  # Should be fast

    def test_llava_conversion_performance(self):
        """Benchmark LLaVA format conversion."""
        from dvas.data.schemas import Annotation, Segment, VideoMetadata, Action, Hand

        annotation = Annotation(
            id="perf_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=[
                Segment(
                    start_time=i * 2.0,
                    end_time=(i + 1) * 2.0,
                    caption=f"Segment {i}",
                    actions=[
                        Action(verb="cut", noun="vegetables", hand=Hand.RIGHT),
                        Action(verb="pick", noun="knife", hand=Hand.LEFT),
                    ],
                )
                for i in range(10)
            ],
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=20.0,
                total_frames=600,
            ),
        )

        def convert():
            annotation.to_llava_format()

        result = benchmark(convert, iterations=100)
        assert result.avg_time < 0.01  # Should be very fast

    def test_unique_verbs_performance(self):
        """Benchmark extracting unique verbs."""
        from dvas.data.schemas import Annotation, Segment, VideoMetadata, Action

        segments = [
            Segment(
                start_time=i * 1.0,
                end_time=(i + 1) * 1.0,
                caption=f"Segment {i}",
                actions=[
                    Action(verb=f"verb_{i % 20}", noun=f"noun_{i}")
                    for i in range(5)
                ],
            )
            for i in range(50)
        ]

        annotation = Annotation(
            id="perf_test",
            video_id="vid_001",
            video_path="/path/to/video.mp4",
            segments=segments,
            metadata=VideoMetadata(
                fps=30.0,
                resolution=[1920, 1080],
                duration=50.0,
                total_frames=1500,
            ),
        )

        def extract_verbs():
            annotation.get_action_verbs()

        result = benchmark(extract_verbs, iterations=100)
        assert result.avg_time < 0.01


class TestLoadTestMetrics:
    """Test load test metrics calculations."""

    def test_load_test_result_success_rate(self):
        """Test success rate calculation."""
        from dvas.testing import LoadTestResult

        result = LoadTestResult(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration=10.0,
            min_latency=0.01,
            max_latency=0.5,
            avg_latency=0.1,
            p50_latency=0.08,
            p95_latency=0.3,
            p99_latency=0.45,
            requests_per_second=10.0,
        )

        assert result.success_rate == 0.95
        assert abs(result.error_rate - 0.05) < 1e-10

    def test_load_test_result_zero_requests(self):
        """Test metrics with zero requests."""
        from dvas.testing import LoadTestResult

        result = LoadTestResult(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_duration=0.0,
            min_latency=0.0,
            max_latency=0.0,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
            requests_per_second=0.0,
        )

        assert result.success_rate == 0.0
        assert result.error_rate == 0.0

    def test_load_test_result_to_dict(self):
        """Test load test result serialization."""
        from dvas.testing import LoadTestResult

        result = LoadTestResult(
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            total_duration=10.0,
            min_latency=0.01,
            max_latency=0.5,
            avg_latency=0.1,
            p50_latency=0.08,
            p95_latency=0.3,
            p99_latency=0.45,
            requests_per_second=10.0,
            errors=["Error 1", "Error 2"],
        )

        d = result.to_dict()
        assert d["total_requests"] == 100
        assert abs(d["success_rate"] - 0.9) < 1e-10
        assert abs(d["error_rate"] - 0.1) < 1e-10
        assert d["requests_per_second"] == 10.0
        assert len(d["errors"]) == 2
