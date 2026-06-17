"""Dataset utilities for video-language training."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import Dataset

logger = logging.getLogger(__name__)


class VideoAnnotationDataset:
    """Dataset for video annotation training."""

    def __init__(
        self,
        data_path: Union[str, Path],
        max_frames: int = 16,
        image_resolution: int = 448,
    ):
        self.data_path = Path(data_path)
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.data: List[Dict] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load dataset from JSONL file."""
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        return Dataset.from_list(self.data)

    @staticmethod
    def from_annotations(
        annotations: List[Any],
        output_path: Path,
        format_type: str = "conversational",
    ) -> "VideoAnnotationDataset":
        """Create dataset from Annotation objects."""
        data = []

        for ann in annotations:
            if format_type == "conversational":
                # Convert to conversational format
                for segment in ann.segments:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a video understanding AI that provides detailed descriptions of video content, focusing on hand actions and object interactions.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": ann.video_path},
                                {
                                    "type": "text",
                                    "text": "Describe the actions in this video segment.",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": segment.caption_dense or segment.caption,
                        },
                    ]
                    data.append({"messages": messages})

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return VideoAnnotationDataset(output_path)


class DPOPairDataset:
    """Dataset for DPO (Direct Preference Optimization) training."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.pairs: List[Dict] = []

        self._load_pairs()

    def _load_pairs(self) -> None:
        """Load preference pairs."""
        if not self.data_path.exists():
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    self.pairs.append(pair)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def create_pairs_from_rankings(
        video_annotations: Dict[str, List[Dict]],
        output_path: Path,
    ) -> "DPOPairDataset":
        """Create DPO pairs from ranked annotations.

        Args:
            video_annotations: Dict of video_id -> list of annotations with scores
            output_path: Where to save the pairs
        """
        pairs = []

        for video_id, annotations in video_annotations.items():
            # Sort by score
            ranked = sorted(annotations, key=lambda x: x.get("score", 0), reverse=True)

            # Generate pairs (winner vs loser)
            for i, winner in enumerate(ranked):
                for loser in ranked[i + 1 :]:
                    pair = {
                        "prompt": winner.get("prompt", ""),
                        "chosen": winner.get("response", ""),
                        "rejected": loser.get("response", ""),
                        "video_id": video_id,
                    }
                    pairs.append(pair)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        return DPOPairDataset(output_path)


def collate_fn(batch: List[Dict], tokenizer: Any) -> Dict[str, Any]:
    """Collate function for batching."""
    texts = []
    for item in batch:
        if "messages" in item:
            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                # Simple concatenation
                text = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in item["messages"]
                )
            texts.append(text)
        elif "text" in item:
            texts.append(item["text"])
        else:
            texts.append(str(item))

    # Tokenize
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone(),
    }
