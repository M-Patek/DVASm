"""Retry utilities with checkpoint support for DVAS."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from dvas.exceptions import (
    RetryExhaustedError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries function calls with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt, delay)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt >= max_attempts:
                        raise RetryExhaustedError(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}",
                            attempts=max_attempts,
                            last_error=e,
                        ) from e

                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if on_retry:
                        on_retry(e, attempt, delay)
                    logger.warning(
                        "Retry attempt %d/%d for %s: %s (delay=%.1fs)",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            raise RetryExhaustedError(
                f"Unexpected exit from retry loop for {func.__name__}",
                attempts=max_attempts,
                last_error=last_exception,
            )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt >= max_attempts:
                        raise RetryExhaustedError(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}",
                            attempts=max_attempts,
                            last_error=e,
                        ) from e

                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if on_retry:
                        on_retry(e, attempt, delay)
                    logger.warning(
                        "Async retry attempt %d/%d for %s: %s (delay=%.1fs)",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(e),
                        delay,
                    )
                    await asyncio.sleep(delay)

            raise RetryExhaustedError(
                f"Unexpected exit from retry loop for {func.__name__}",
                attempts=max_attempts,
                last_error=last_exception,
            )

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@dataclass
class BatchCheckpoint:
    """Checkpoint data for batch processing."""

    processed_count: int = 0
    failed_items: List[Dict[str, Any]] = field(default_factory=list)
    last_processed_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_count": self.processed_count,
            "failed_items": self.failed_items,
            "last_processed_id": self.last_processed_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchCheckpoint":
        return cls(
            processed_count=data.get("processed_count", 0),
            failed_items=data.get("failed_items", []),
            last_processed_id=data.get("last_processed_id"),
            timestamp=data.get("timestamp", time.time()),
        )


class BatchProcessor:
    """Process items in batches with checkpoint support."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        batch_size: int = 10,
        save_interval: int = 5,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.checkpoint = BatchCheckpoint()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk if exists."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.checkpoint = BatchCheckpoint.from_dict(data)
                logger.info(
                    "checkpoint_loaded",
                    processed=self.checkpoint.processed_count,
                    failed=len(self.checkpoint.failed_items),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Checkpoint load failed: %s", str(e))
                self.checkpoint = BatchCheckpoint()

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        if not self.checkpoint_path:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.checkpoint.to_dict(), f, indent=2)

    def mark_processed(self, item_id: str) -> None:
        """Mark an item as successfully processed."""
        self.checkpoint.processed_count += 1
        self.checkpoint.last_processed_id = item_id

        if self.checkpoint.processed_count % self.save_interval == 0:
            self._save_checkpoint()

    def mark_failed(self, item_id: str, error: str, details: Optional[Dict] = None) -> None:
        """Mark an item as failed."""
        self.checkpoint.failed_items.append(
            {
                "item_id": item_id,
                "error": error,
                "details": details or {},
                "timestamp": time.time(),
            }
        )
        self.checkpoint.processed_count += 1

    def is_processed(self, item_id: str) -> bool:
        """Check if an item has already been processed."""
        # Check if item is in failed items
        for failed in self.checkpoint.failed_items:
            if failed.get("item_id") == item_id:
                return True
        return False

    def get_failed_items(self) -> List[Dict[str, Any]]:
        """Get list of failed items for retry."""
        return self.checkpoint.failed_items.copy()

    def clear_failed(self) -> None:
        """Clear failed items list."""
        self.checkpoint.failed_items = []
        self._save_checkpoint()

    def __enter__(self) -> "BatchProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure checkpoint is saved on exit."""
        self._save_checkpoint()

    def process_batch(
        self,
        items: List[Dict[str, Any]],
        processor: Callable[[Dict[str, Any]], T],
        skip_processed: bool = True,
    ) -> Tuple[List[T], List[Dict[str, Any]]]:
        """Process a batch of items with checkpoint support.

        Args:
            items: List of items to process
            processor: Function to process each item
            skip_processed: Whether to skip already processed items

        Returns:
            Tuple of (successful results, failed items)
        """
        results: List[T] = []
        failed: List[Dict[str, Any]] = []

        for item in items:
            item_id = item.get("id", str(item))

            if skip_processed and self.is_processed(item_id):
                continue

            try:
                result = processor(item)
                results.append(result)
                self.mark_processed(item_id)
            except Exception as e:
                self.mark_failed(item_id, str(e))
                failed.append({"item": item, "error": str(e)})

        return results, failed
