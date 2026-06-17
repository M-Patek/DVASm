"""Performance benchmark for DVAS pipeline optimizations.

Usage:
    python benchmarks/perf_benchmark.py

Measures:
- Video reading throughput (frames/sec)
- Frame sampling speed
- Scene detection speed
- Batch processing throughput
"""

import time
from pathlib import Path
from typing import Callable, List

import numpy as np


def benchmark(name: str, func: Callable, iterations: int = 3) -> dict:
    """Run a benchmark and return timing statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "name": name,
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "result": result,
    }


def create_mock_video(path: Path, num_frames: int = 300, fps: int = 30) -> None:
    """Create a small mock video for benchmarking."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))

    for i in range(num_frames):
        # Create frames with varying content
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some structure to make scene detection meaningful
        frame[:100, :100] = (i * 2) % 255
        writer.write(frame)

    writer.release()


def benchmark_video_reader(video_path: Path) -> List[dict]:
    """Benchmark video reading performance."""
    from dvas.data.video_reader import VideoReader

    def read_all_frames():
        with VideoReader(video_path) as reader:
            return list(reader.read_frames())

    def read_with_step():
        with VideoReader(video_path) as reader:
            return list(reader.read_frames(step=5))

    return [
        benchmark("video_reader_sequential", read_all_frames),
        benchmark("video_reader_step5", read_with_step),
    ]


def benchmark_frame_sampler(video_path: Path) -> List[dict]:
    """Benchmark frame sampling strategies."""
    from dvas.data.frame_sampler import SamplerConfig, UniformSampler
    from dvas.data.video_reader import VideoReader

    def sample_uniform():
        with VideoReader(video_path) as reader:
            sampler = UniformSampler(SamplerConfig(num_frames=16))
            return list(sampler.sample(reader))

    return [
        benchmark("uniform_sampler_16_frames", sample_uniform),
    ]


def benchmark_scene_detection(video_path: Path) -> List[dict]:
    """Benchmark scene detection performance."""
    from dvas.data.scene_detector import HistogramSceneDetector
    from dvas.data.video_reader import VideoReader

    def detect_scenes():
        with VideoReader(video_path) as reader:
            detector = HistogramSceneDetector()
            return detector.detect(reader)

    return [
        benchmark("scene_detection_histogram", detect_scenes, iterations=1),
    ]


def run_benchmarks():
    """Run all benchmarks and print results."""
    import tempfile

    print("=" * 60)
    print("DVAS Performance Benchmarks")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        print(f"\nCreating mock video: {video_path}")
        create_mock_video(video_path, num_frames=300, fps=30)
        print(f"Video created: {video_path.stat().st_size / 1024:.1f} KB")

        all_results = []

        print("\n--- Video Reading ---")
        for result in benchmark_video_reader(video_path):
            all_results.append(result)
            print(f"  {result['name']}: {result['mean']:.3f}s")

        print("\n--- Frame Sampling ---")
        for result in benchmark_frame_sampler(video_path):
            all_results.append(result)
            print(f"  {result['name']}: {result['mean']:.3f}s")

        print("\n--- Scene Detection ---")
        for result in benchmark_scene_detection(video_path):
            all_results.append(result)
            print(f"  {result['name']}: {result['mean']:.3f}s")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
