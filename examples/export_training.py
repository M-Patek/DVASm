"""Export gold annotations as training data for student model.

Converts annotations to various formats (LLaVA, OpenAI, ShareGPT) for SFT/DPO training.

Example usage:
    # Export as LLaVA format (default)
    python examples/export_training.py --output data/training/sft.jsonl

    # Export specific videos
    python examples/export_training.py --videos vid_001 vid_002 --output data/training/subset.jsonl

    # Export for DPO (preference pairs)
    python examples/export_training.py --format dpo --output data/training/dpo.jsonl

    # Export as ShareGPT format
    python examples/export_training.py --format sharegpt --output data/training/sharegpt.jsonl

    # Split into train/val
    python examples/export_training.py --split 0.8 --output-dir data/training/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dvas.data.schemas import Annotation
from dvas.data.storage import AnnotationStore
from dvas.export.adapters import (
    LLaVAAdapter,
    OpenAIAdapter,
    ShareGPTAdapter,
)
from dvas.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


FORMAT_ADAPTERS = {
    "llava": LLaVAAdapter(),
    "openai": OpenAIAdapter(),
    "sharegpt": ShareGPTAdapter(),
}


def load_annotations(
    store: AnnotationStore,
    video_ids: list[str] | None = None,
    source: str = "gold",
) -> list[Annotation]:
    """Load annotations from store."""
    if video_ids:
        annotations = []
        for vid in video_ids:
            ann = store.load(vid, source=source)
            if ann:
                annotations.append(ann)
            else:
                logger.warning(f"Annotation not found: {vid}")
        return annotations
    else:
        return list(store.load_all(source=source))


def split_train_val(
    items: list[dict],
    train_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split items into train/validation sets."""
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    return train, val


def export_training_data(
    output_path: Path,
    format_name: str = "llava",
    video_ids: list[str] | None = None,
    source: str = "gold",
    split_ratio: float | None = None,
    output_dir: Path | None = None,
) -> None:
    """Export annotations as training data."""
    setup_logging()

    # Validate format
    if format_name not in FORMAT_ADAPTERS:
        raise ValueError(
            f"Unknown format: {format_name}. Choose from: {list(FORMAT_ADAPTERS.keys())}"
        )

    # Load annotations
    logger.info(f"Loading annotations from source: {source}")
    store = AnnotationStore()
    annotations = load_annotations(store, video_ids, source)

    if not annotations:
        logger.error("No annotations found")
        return

    logger.info(f"Loaded {len(annotations)} annotations")

    # Convert to training format
    adapter = FORMAT_ADAPTERS[format_name]
    logger.info(f"Converting to {format_name} format")

    # Use adapter export method
    training_items = adapter.export(annotations)

    logger.info(f"Created {len(training_items)} training examples")

    # Split if requested
    if split_ratio is not None and output_dir:
        train_items, val_items = split_train_val(training_items, split_ratio)

        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / f"train_{format_name}.jsonl"
        val_path = output_dir / f"val_{format_name}.jsonl"

        # Write train split
        with open(train_path, "w") as f:
            for item in train_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Write val split
        with open(val_path, "w") as f:
            for item in val_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            "Export complete",
            train_count=len(train_items),
            val_count=len(val_items),
            train_path=str(train_path),
            val_path=str(val_path),
        )

    else:
        # Single file output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for item in training_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            "Export complete",
            count=len(training_items),
            output_path=str(output_path),
        )


def main():
    parser = argparse.ArgumentParser(description="Export gold annotations as training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/sft.jsonl"),
        help="Output file path (default: data/training/sft.jsonl)",
    )
    parser.add_argument(
        "--format",
        choices=list(FORMAT_ADAPTERS.keys()),
        default="llava",
        help="Output format (default: llava)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Specific video IDs to export (default: all)",
    )
    parser.add_argument(
        "--source",
        choices=["gold", "model", "reviewed"],
        default="gold",
        help="Annotation source to export (default: gold)",
    )
    parser.add_argument(
        "--split",
        type=float,
        help="Train/val split ratio (e.g., 0.8 for 80%% train)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for split files (required with --split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )

    args = parser.parse_args()

    if args.split is not None and args.output_dir is None:
        parser.error("--output-dir is required when using --split")

    export_training_data(
        output_path=args.output,
        format_name=args.format,
        video_ids=args.videos,
        source=args.source,
        split_ratio=args.split,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
