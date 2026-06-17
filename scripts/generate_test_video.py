"""Generate a synthetic test video for end-to-end pipeline testing."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_test_video(
    output_path: Path,
    duration_seconds: float = 3.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Create a synthetic MP4 video with moving shapes.

    The video contains a red circle moving across a white background,
    simulating simple motion that scene detection and motion scoring
    can pick up.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(duration_seconds * fps)

    for i in range(total_frames):
        # White background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Moving red circle
        cx = int((width / total_frames) * i) + 50
        cy = height // 2
        radius = 40
        color = (0, 0, 255)  # Red in BGR
        cv2.circle(frame, (cx, cy), radius, color, -1)

        # Add timestamp text
        cv2.putText(
            frame,
            f"Frame {i}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"Created synthetic video: {output_path}")
    print(f"  Duration: {duration_seconds}s, Frames: {total_frames}, Resolution: {width}x{height}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic test video")
    parser.add_argument("--output", "-o", default="tmp/test_video.mp4", help="Output path")
    parser.add_argument("--duration", "-d", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    create_test_video(
        output_path=Path(args.output),
        duration_seconds=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
