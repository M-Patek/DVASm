"""Performance benchmark for aiter_frames async streaming.

Compares memory usage and latency of the new asyncio.Queue-based
implementation vs the old ThreadPoolExecutor + list() approach.
"""

import asyncio
import time
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np

from dvas.data.video_loader import VideoLoader


def benchmark_old_approach(loader, num_frames=100):
    """Simulate the old approach: load all frames into memory first."""
    tracemalloc.start()
    start = time.perf_counter()

    # Old approach: read all frames synchronously into a list
    frames = list(loader.read_frames(num_frames=num_frames))

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "frames": len(frames),
        "elapsed_ms": elapsed * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
    }


async def benchmark_new_approach(loader, num_frames=100):
    """Benchmark the new async streaming approach."""
    tracemalloc.start()
    start = time.perf_counter()

    # New approach: stream frames asynchronously
    frames = []
    async for frame in loader.aiter_frames(num_frames=num_frames):
        frames.append(frame)

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "frames": len(frames),
        "elapsed_ms": elapsed * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def run_benchmark(num_frames=100):
    """Run benchmark comparing old vs new approaches."""
    import tempfile
    from pathlib import Path

    # Create a mock video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake_video_data")
        video_path = Path(f.name)

    with patch("dvas.data.video_loader.cv2.VideoCapture") as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True

        frame_count = [0]

        def mock_read():
            if frame_count[0] < num_frames:
                frame_count[0] += 1
                return (True, np.zeros((1920, 1080, 3), dtype=np.uint8))
            return (False, None)

        mock_instance.read.side_effect = mock_read
        mock_cap.return_value = mock_instance

        with VideoLoader(video_path) as loader:
            loader._metadata = MagicMock()
            loader._metadata.fps = 30.0
            loader._metadata.duration = 10.0
            loader._metadata.total_frames = 300

            print(f"\n{'=' * 60}")
            print(f"Benchmark: {num_frames} frames")
            print(f"{'=' * 60}")

            # Benchmark old approach
            old_result = benchmark_old_approach(loader, num_frames)
            print("\nOld approach (sync list):")
            print(f"  Frames: {old_result['frames']}")
            print(f"  Time: {old_result['elapsed_ms']:.2f} ms")
            print(f"  Peak memory: {old_result['peak_memory_mb']:.2f} MB")

            # Reset frame counter
            frame_count[0] = 0

            # Benchmark new approach
            new_result = asyncio.run(benchmark_new_approach(loader, num_frames))
            print("\nNew approach (async streaming):")
            print(f"  Frames: {new_result['frames']}")
            print(f"  Time: {new_result['elapsed_ms']:.2f} ms")
            print(f"  Peak memory: {new_result['peak_memory_mb']:.2f} MB")

            # Summary
            if old_result["peak_memory_mb"] > 0:
                memory_reduction = (
                    (old_result["peak_memory_mb"] - new_result["peak_memory_mb"])
                    / old_result["peak_memory_mb"]
                    * 100
                )
                print(f"\nMemory reduction: {memory_reduction:.1f}%")

    # Cleanup
    video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_benchmark(num_frames=50)
