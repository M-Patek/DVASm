"""Annotate EPIC-KITCHENS videos using teacher models.

Example usage:
    python examples/annotate_epic.py --split train --num 10 --model gpt-5.5
    python examples/annotate_epic.py --split test --num 100 --model claude-opus-4-8 --output data/annotations/claude

Environment variables (set in .env file or environment):
    OPENAI_API_KEY: Required for GPT-5.5 teacher
    ANTHROPIC_API_KEY: Required for Claude teacher
    TOGETHER_API_KEY: Required for Together AI teacher
    EPIC_KITCHENS_ROOT: Path to EPIC-KITCHENS dataset (default: data/epic-kitchens)
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from dvas.data.video_loader import EPICKitchensLoader
from dvas.models.teacher import TeacherModel
from dvas.pipeline.core import AnnotationPipeline
from dvas.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# Model configurations
TEACHER_MODELS = {
    "gpt-5.5": "gpt-5.5",
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "together-llama": "meta-llama/Llama-3.2-90B-Vision-Instruct",
}


async def annotate_videos(
    split: str,
    num_videos: int,
    model_name: str,
    output_dir: Path,
    epic_root: Path,
) -> None:
    """Run annotation on EPIC-KITCHENS videos."""
    setup_logging()

    # Validate API key
    if "gpt-" in model_name and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable required for GPT models")
    if "claude" in model_name and not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable required for Claude models")
    if "together" in model_name and not os.getenv("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY environment variable required for Together models")

    # Initialize teacher model
    logger.info(f"Initializing {model_name} teacher model")
    teacher_model_name = TEACHER_MODELS.get(model_name, model_name)
    teacher = TeacherModel(model_name=teacher_model_name)

    # Initialize EPIC-KITCHENS loader
    logger.info(f"Loading EPIC-KITCHENS from {epic_root}")
    epic_loader = EPICKitchensLoader(epic_root)

    # Get videos for split
    # Note: EPIC has train/test splits in separate CSV files
    split_file = epic_root / f"EPIC_100_{split}.csv"

    video_paths = []
    if split_file.exists():
        import pandas as pd
        df = pd.read_csv(split_file)
        video_ids = df["video_id"].unique().tolist()[:num_videos]

        for video_id in video_ids:
            path = epic_loader.get_video_path(video_id)
            if path:
                video_paths.append((video_id, path))
    else:
        # Direct video discovery (for mini test datasets)
        logger.info("Using video directory directly")
        videos_dir = epic_root / "videos"
        if videos_dir.exists():
            for video_file in videos_dir.rglob("*.mp4"):
                video_id = video_file.stem
                video_paths.append((video_id, video_file))
                if len(video_paths) >= num_videos:
                    break
            # Also check for uppercase .MP4 (EPIC format)
            if len(video_paths) < num_videos:
                for video_file in videos_dir.rglob("*.MP4"):
                    video_id = video_file.stem
                    if (video_id, video_file) not in video_paths:
                        video_paths.append((video_id, video_file))
                    if len(video_paths) >= num_videos:
                        break

    logger.info(f"Found {len(video_paths)} videos to annotate")

    # Create annotation pipeline
    pipeline = AnnotationPipeline(
        teacher_model=teacher,
        checkpoint_path=output_dir / "checkpoint.json",
    )

    # Process videos
    results = []
    for video_id, video_path in video_paths:
        try:
            logger.info(f"Annotating {video_id}")
            result = await pipeline.annotate_video(
                video_path=video_path,
                video_id=video_id,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to annotate {video_id}", error=str(e))

    # Summary
    successful = sum(1 for r in results if r.segments)  # Check if annotation has segments
    logger.info(
        "Annotation complete",
        total=len(results),
        successful=successful,
        failed=len(results) - successful,
        output_dir=str(output_dir),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Annotate EPIC-KITCHENS videos with teacher models"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="EPIC-KITCHENS split to annotate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of videos to annotate",
    )
    parser.add_argument(
        "--model",
        choices=list(TEACHER_MODELS.keys()) + ["custom"],
        default="gpt-5.5",
        help="Teacher model to use",
    )
    parser.add_argument(
        "--custom-model",
        type=str,
        default=None,
        help="Custom model name (use with --model custom)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations"),
        help="Output directory for annotations",
    )
    parser.add_argument(
        "--epic-root",
        type=Path,
        default=Path(os.getenv("EPIC_KITCHENS_ROOT", "data/epic-kitchens")),
        help="Path to EPIC-KITCHENS dataset root",
    )

    args = parser.parse_args()

    # Use custom model if specified
    model_name = args.custom_model if args.model == "custom" else args.model

    asyncio.run(annotate_videos(
        split=args.split,
        num_videos=args.num,
        model_name=model_name,
        output_dir=args.output,
        epic_root=args.epic_root,
    ))


if __name__ == "__main__":
    main()
