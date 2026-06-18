"""Create mini test dataset for quick end-to-end validation.

This script creates a small synthetic dataset (or downloads a few sample videos)
for testing the full DVAS pipeline without downloading the full 120GB EPIC-KITCHENS.

Usage:
    # Create 5 synthetic test videos (random noise, for code testing)
    python examples/create_mini_dataset.py --num-videos 5 --synthetic

    # Download 5 real EPIC videos (recommended for actual testing)
    python examples/create_mini_dataset.py --num-videos 5 --real

    # Create LLaVA format training data from existing annotations
    python examples/create_mini_dataset.py --from-annotations data/annotations/gold

The mini dataset will be created at: data/mini_epic/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from dvas.data.schemas import (
    Action,
    Annotation,
    Object,
    QAPair,
    Segment,
    VideoMetadata,
)
from dvas.data.storage import AnnotationStore
from dvas.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# 示例 EPIC-KITCHENS 视频（前几个，体积小）
SAMPLE_EPIC_VIDEOS = [
    ("P01_01", "P01", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P01/P01_01.MP4"),
    ("P01_02", "P01", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P01/P01_02.MP4"),
    ("P01_03", "P01", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P01/P01_03.MP4"),
    ("P02_01", "P02", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P02/P02_01.MP4"),
    ("P02_02", "P02", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P02/P02_02.MP4"),
    ("P03_01", "P03", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P03/P03_01.MP4"),
    ("P03_02", "P03", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P03/P03_02.MP4"),
    ("P04_01", "P04", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P04/P04_01.MP4"),
    ("P04_02", "P04", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P04/P04_02.MP4"),
    ("P05_01", "P05", "https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/P05/P05_01.MP4"),
]


def create_synthetic_video(output_path: Path, duration_sec: int = 10, fps: int = 30) -> None:
    """Create a synthetic video using FFmpeg (color bars + random noise)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use FFmpeg to generate test video
    # testsrc=颜色条, noise=随机噪声, 合成一个模拟厨房场景
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=640x480:rate={fps}",
        "-f", "lavfi",
        "-i", f"noise=alls=20:allf=t+u",  # Add some noise"-filter_complex", "[0:v][1:v]blend=all_mode='addition':all_opacity=0.1[out]",
        "-map", "[out]",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        logger.info(f"Created synthetic video: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e}")
        # Fallback: create a simple MP4 with OpenCV if FFmpeg not available
        create_synthetic_video_opencv(output_path, duration_sec, fps)
    except FileNotFoundError:
        logger.warning("FFmpeg not found, using OpenCV fallback")
        create_synthetic_video_opencv(output_path, duration_sec, fps)


def create_synthetic_video_opencv(output_path: Path, duration_sec: int, fps: int) -> None:
    """Create synthetic video using OpenCV (no FFmpeg required)."""
    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

        total_frames = duration_sec * fps

        for i in range(total_frames):
            # Create a frame with changing colors (simulating movement)
            hue = (i * 2) % 180  # Changing hue
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = hue  # B
            frame[:, :, 1] = 128 + int(127 * np.sin(i * 0.1))  # G
            frame[:, :, 2] = 200  # R

            # Add some random noise
            noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

            out.write(frame)

        out.release()
        logger.info(f"Created synthetic video (OpenCV): {output_path}")

    except ImportError:
        logger.error("OpenCV not available. Cannot create synthetic video.")
        raise


def create_sample_annotation(video_id: str, video_path: Path) -> Annotation:
    """Create a sample annotation for testing."""

    # Generate deterministic but varied content based on video_id
    hash_val = int(hashlib.md5(video_id.encode()).hexdigest(), 16)
    random.seed(hash_val)

    actions_pool = [
        ("pick", "knife"), ("cut", "vegetable"), ("wash", "plate"),
        ("open", "fridge"), ("take", "bottle"), ("pour", "water"),
        ("stir", "soup"), ("taste", "food"), ("put", "lid"),
        ("close", "drawer"), ("hold", "cup"), ("move", "pan"),
    ]

    # Create 3-5 segments
    num_segments = random.randint(3, 5)
    segments = []

    for i in range(num_segments):
        start_time = float(i * 15)
        end_time = start_time + random.uniform(8, 15)

        verb, noun = random.choice(actions_pool)

        segment = Segment(
            start_time=start_time,
            end_time=end_time,
            caption=f"Person is {verb}ing a {noun}",
            actions=[Action(verb=verb, noun=noun)],
            objects=[Object(name=noun, confidence=0.9)],
        )
        segments.append(segment)

    return Annotation(
        id=f"ann_{video_id}",
        video_id=video_id,
        video_path=str(video_path),
        segments=segments,
        metadata=VideoMetadata(
            video_id=video_id,
            fps=30.0,
            resolution=[640, 480],
            duration=float(num_segments * 15),
            total_frames=num_segments * 15 * 30,
        ),
        source="teacher",
        model_version="gpt-5.5",
        quality_score=random.uniform(0.8, 0.95),
    )


def download_sample_video(video_id: str, participant: str, url: str, output_dir: Path) -> Path | None:
    """Download a single sample video."""
    import urllib.request

    video_dir = output_dir / "videos" / participant
    video_dir.mkdir(parents=True, exist_ok=True)

    output_path = video_dir / f"{video_id}.MP4"

    if output_path.exists():
        logger.info(f"Video already exists: {output_path}")
        return output_path

    try:
        logger.info(f"Downloading {video_id}...")
        # 设置超时和User-Agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=120) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())

        logger.info(f"Downloaded: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download {video_id}: {e}")
        return None


def create_mini_dataset(
    output_dir: Path,
    num_videos: int = 5,
    use_real: bool = False,
    create_annotations: bool = True,
) -> None:
    """Create mini dataset for testing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating mini dataset at: {output_dir}")

    video_ids = []

    if use_real:
        # Download real EPIC videos
        logger.info(f"Downloading {num_videos} real EPIC videos...")

        for video_id, participant, url in SAMPLE_EPIC_VIDEOS[:num_videos]:
            video_path = download_sample_video(video_id, participant, url, output_dir)
            if video_path:
                video_ids.append((video_id, video_path))

    else:
        # Create synthetic videos
        logger.info(f"Creating {num_videos} synthetic videos...")

        for i in range(num_videos):
            video_id = f"test_{i+1:03d}"
            video_dir = output_dir / "videos" / "test"
            video_dir.mkdir(parents=True, exist_ok=True)

            video_path = video_dir / f"{video_id}.mp4"
            create_synthetic_video(video_path, duration_sec=10, fps=30)
            video_ids.append((video_id, video_path))

    logger.info(f"Created/downloaded {len(video_ids)} videos")

    # Create sample annotations
    if create_annotations:
        logger.info("Creating sample annotations...")
        store = AnnotationStore()

        for video_id, video_path in video_ids:
            annotation = create_sample_annotation(video_id, video_path)
            store.save(annotation, source="gold", overwrite=True)
            logger.info(f"  Created annotation: {annotation.id}")

        logger.info(f"Saved {len(video_ids)} annotations to storage")

    # Create info file
    info = {
        "dataset": "EPIC-KITCHENS-MINI",
        "created_at": datetime.now().isoformat(),
        "num_videos": len(video_ids),
        "video_type": "real" if use_real else "synthetic",
        "video_ids": [vid for vid, _ in video_ids],
    }

    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Mini dataset creation complete!")
    logger.info(f"  Location: {output_dir}")
    logger.info(f"  Videos: {output_dir / 'videos'}")
    logger.info(f"  Annotations: [AnnotationStore default location]")


def export_to_llava(
    output_path: Path,
    num_samples: int = 10,
) -> None:
    """Export existing annotations to LLaVA format for training."""
    from dvas.export.adapters import LLaVAAdapter, export_annotations

    store = AnnotationStore()
    annotations = list(store.load_all(source="gold"))

    if not annotations:
        logger.error("No annotations found. Create mini dataset first.")
        return

    adapter = LLaVAAdapter()
    data = adapter.export(annotations[:num_samples])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create mini test dataset for DVAS"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mini_epic"),
        help="Output directory (default: data/mini_epic)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=5,
        help="Number of videos to create/download (default: 5)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Download real EPIC videos (default: create synthetic)",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip creating annotations",
    )
    parser.add_argument(
        "--export-llava",
        type=Path,
        help="Export annotations to LLaVA format file",
    )

    args = parser.parse_args()
    setup_logging()

    if args.export_llava:
        export_to_llava(args.export_llava, args.num_videos)
    else:
        create_mini_dataset(
            output_dir=args.output_dir,
            num_videos=args.num_videos,
            use_real=args.real,
            create_annotations=not args.no_annotations,
        )


if __name__ == "__main__":
    main()
