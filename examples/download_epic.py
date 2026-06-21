"""Download EPIC-KITCHENS dataset.

EPIC-KITCHENS 是一个公开的第一人称厨房视频数据集。
下载地址（国内可访问）：
- 官网：https://epic-kitchens.github.io/
- 直接下载：http://epic-kitchens.github.io/static/misc/download_scripts/epic-kitchens-100-download-videos.sh

注意：
1. EPIC-KITCHENS-100 完整视频约 120GB
2. 需要同意使用协议（Academic/Research Use）
3. 建议使用 wget 或 aria2c 多线程下载

Usage:
    # 下载标注文件（小文件，先下这个）
    python examples/download_epic.py --annotations-only

    # 下载训练集视频（约 90GB）
    python examples/download_epic.py --split train --output-dir data/epic-kitchens

    # 下载测试集视频（约 30GB）
    python examples/download_epic.py --split test --output-dir data/epic-kitchens

    # 下载前 10 个视频用于测试
    python examples/download_epic.py --split train --max-videos 10
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

from dvas.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# EPIC-KITCHENS-100 下载链接模板
# 格式：P{participant:02d}/P{participant:02d}_{video:02d}.MP4
EPIC_100_URL_TEMPLATE = (
    "https://data.bris.ac.uk/datasets/tar/1mv94ej3w96gj9dfhglwg4v0y/{participant}/{video}"
)

# 标注文件下载链接
ANNOTATIONS_URLS = {
    "train": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv",
    "test": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_test.csv",
    "validation": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_validation.csv",
    "narration": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_narration.csv",
}


def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download a single file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return True

    try:
        logger.info(f"Downloading: {url}")
        logger.info(f"  -> {output_path}")

        # 使用 urllib 下载
        urllib.request.urlretrieve(url, output_path)
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_annotations(output_dir: Path) -> None:
    """Download annotation CSV files."""
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading annotation files...")

    for name, url in ANNOTATIONS_URLS.items():
        output_path = annotations_dir / f"EPIC_100_{name}.csv"
        if download_file(url, output_path):
            logger.info(f"  [OK] {name}")
        else:
            logger.warning(f"  [FAIL] {name}")


def parse_video_list(csv_path: Path, split: str) -> list[tuple[str, str, str]]:
    """Parse CSV to get video list."""
    videos = []
    seen = set()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            if video_id not in seen:
                seen.add(video_id)
                # 解析 participant 和 video number
                # video_id format: P01_01
                parts = video_id.split("_")
                if len(parts) == 2:
                    participant = parts[0]  # P01
                    video_num = parts[1]  # 01
                    videos.append((video_id, participant, video_num))

    return videos


def download_epic_videos(
    output_dir: Path,
    split: str = "train",
    max_videos: Optional[int] = None,
    use_aria2: bool = False,
) -> None:
    """Download EPIC-KITCHENS videos."""

    # 首先确保标注文件已下载
    annotations_dir = output_dir / "annotations"
    csv_path = annotations_dir / f"EPIC_100_{split}.csv"

    if not csv_path.exists():
        logger.info("Annotation file not found, downloading first...")
        download_annotations(output_dir)

    if not csv_path.exists():
        logger.error(f"Cannot find annotation file: {csv_path}")
        return

    # 解析视频列表
    videos = parse_video_list(csv_path, split)
    logger.info(f"Found {len(videos)} unique videos in {split} split")

    if max_videos:
        videos = videos[:max_videos]
        logger.info(f"Limited to first {max_videos} videos")

    # 创建视频目录
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 下载视频
    success_count = 0

    for video_id, participant, video_num in videos:
        # EPIC-KITCHENS 视频在 university data repository
        # 实际的视频下载链接格式
        video_url = f"https://data.bris.ac.uk/datasets/3h81ucoatsrb2v7bw2edvk1aa2/{participant}/{participant}_{video_num}.MP4"

        participant_dir = videos_dir / participant
        participant_dir.mkdir(exist_ok=True)

        output_path = participant_dir / f"{video_id}.MP4"

        if output_path.exists():
            logger.info(f"Skip existing: {video_id}")
            success_count += 1
            continue

        # 尝试下载
        logger.info(f"Downloading {video_id}...")

        try:
            if (
                use_aria2
                and subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
            ):
                # 使用 aria2c 加速（如果可用）
                cmd = [
                    "aria2c",
                    "-x",
                    "4",  # 4 connections
                    "-s",
                    "4",  # 4 splits
                    "-o",
                    str(output_path),
                    video_url,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=600)
                if result.returncode == 0:
                    success_count += 1
            else:
                # 使用 urllib
                if download_file(video_url, output_path, timeout=600):
                    success_count += 1

        except Exception as e:
            logger.error(f"Failed to download {video_id}: {e}")

    logger.info(f"Download complete: {success_count}/{len(videos)} videos")


def main():
    parser = argparse.ArgumentParser(description="Download EPIC-KITCHENS dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/epic-kitchens"),
        help="Output directory (default: data/epic-kitchens)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "validation", "all"],
        default="train",
        help="Which split to download (default: train)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Limit number of videos (for testing)",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Only download annotation CSVs (small files)",
    )
    parser.add_argument(
        "--use-aria2",
        action="store_true",
        help="Use aria2c for faster downloads (if installed)",
    )

    args = parser.parse_args()
    setup_logging()

    if args.annotations_only:
        download_annotations(args.output_dir)
    elif args.split == "all":
        for split in ["train", "test", "validation"]:
            download_epic_videos(
                args.output_dir,
                split=split,
                max_videos=args.max_videos,
                use_aria2=args.use_aria2,
            )
    else:
        download_epic_videos(
            args.output_dir,
            split=args.split,
            max_videos=args.max_videos,
            use_aria2=args.use_aria2,
        )


if __name__ == "__main__":
    main()
