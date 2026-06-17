"""Checkpoint management for resumable pipeline processing."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages checkpoint state for resumable pipeline processing.

    Separated from pipeline to allow independent testing and reuse.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_ids: Set[str] = set()
        self.failed_items: List[Dict] = []
        self._loaded = False

    def load(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded, False if not."""
        if self._loaded:
            return True

        if not self.checkpoint_path.exists():
            return False

        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            self.processed_ids = set(data.get("processed", []))
            self.failed_items = data.get("failed", [])
            self._loaded = True
            logger.info(
                "checkpoint_loaded",
                processed=len(self.processed_ids),
                failed=len(self.failed_items),
            )
            return True
        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            return False

    def save(self) -> None:
        """Save current checkpoint."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed": list(self.processed_ids),
                    "failed": self.failed_items,
                },
                f,
            )

    def mark_processed(self, video_id: str) -> None:
        """Mark video as processed."""
        self.processed_ids.add(video_id)

    def mark_failed(self, video_id: str, error: str) -> None:
        """Mark video as failed."""
        self.failed_items.append({"video_id": video_id, "error": error})

    def is_processed(self, video_id: str) -> bool:
        """Check if video has been processed."""
        return video_id in self.processed_ids

    def get_failed_count(self) -> int:
        """Get number of failed items."""
        return len(self.failed_items)

    def get_processed_count(self) -> int:
        """Get number of processed items."""
        return len(self.processed_ids)
