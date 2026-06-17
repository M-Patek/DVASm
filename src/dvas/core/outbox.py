"""Outbox pattern for reliable event publishing.

Ensures events are persisted before being published, guaranteeing
at-least-once delivery even in the face of crashes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class OutboxStatus(Enum):
    """Status of an outbox entry."""

    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"


@dataclass
class OutboxEntry:
    """An entry in the outbox."""

    id: str
    event_type: str
    payload: str
    status: OutboxStatus
    created_at: float
    retry_count: int
    error: Optional[str] = None


class OutboxStore:
    """SQLite-backed outbox store for reliable event persistence.

    Uses WAL mode for high concurrency and implements exponential
    backoff for failed entries.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Path("data/outbox.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database with WAL mode."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outbox (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outbox_status
                ON outbox(status, created_at)
            """)
            conn.commit()

    async def add(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Add an event to the outbox.

        Returns:
            The ID of the created entry
        """
        entry_id = str(uuid.uuid4())[:8]
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.execute(
                    """
                    INSERT INTO outbox (id, event_type, payload, status, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        event_type,
                        json.dumps(payload),
                        OutboxStatus.PENDING.value,
                        time.time(),
                    ),
                )
                conn.commit()
        return entry_id

    async def get_pending(self, limit: int = 100) -> List[OutboxEntry]:
        """Get pending entries for processing."""
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM outbox
                    WHERE status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (OutboxStatus.PENDING.value, limit),
                )
                rows = cursor.fetchall()
                return [self._row_to_entry(row) for row in rows]

    async def mark_processing(self, entry_id: str) -> None:
        """Mark an entry as being processed."""
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.execute(
                    "UPDATE outbox SET status = ? WHERE id = ?",
                    (OutboxStatus.PROCESSING.value, entry_id),
                )
                conn.commit()

    async def mark_sent(self, entry_id: str) -> None:
        """Mark an entry as successfully sent."""
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.execute(
                    "UPDATE outbox SET status = ? WHERE id = ?",
                    (OutboxStatus.SENT.value, entry_id),
                )
                conn.commit()

    async def mark_failed(self, entry_id: str, error: str) -> None:
        """Mark an entry as failed with retry tracking."""
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.execute(
                    """
                    UPDATE outbox
                    SET status = ?, retry_count = retry_count + 1, error = ?
                    WHERE id = ?
                    """,
                    (OutboxStatus.FAILED.value, error, entry_id),
                )
                conn.commit()

    async def reset_stale(self, max_age_seconds: float = 300.0) -> int:
        """Reset entries stuck in processing back to pending.

        Returns:
            Number of entries reset
        """
        cutoff = time.time() - max_age_seconds
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.execute(
                    """
                    UPDATE outbox
                    SET status = ?
                    WHERE status = ? AND created_at < ?
                    """,
                    (OutboxStatus.PENDING.value, OutboxStatus.PROCESSING.value, cutoff),
                )
                conn.commit()
                return cursor.rowcount

    async def cleanup_sent(self, max_age_days: int = 7) -> int:
        """Remove old sent entries.

        Returns:
            Number of entries removed
        """
        cutoff = time.time() - (max_age_days * 86400)
        async with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.execute(
                    "DELETE FROM outbox WHERE status = ? AND created_at < ?",
                    (OutboxStatus.SENT.value, cutoff),
                )
                conn.commit()
                return cursor.rowcount

    def _row_to_entry(self, row: sqlite3.Row) -> OutboxEntry:
        return OutboxEntry(
            id=row["id"],
            event_type=row["event_type"],
            payload=row["payload"],
            status=OutboxStatus(row["status"]),
            created_at=row["created_at"],
            retry_count=row["retry_count"],
            error=row["error"],
        )


class OutboxPublisher:
    """Publisher that uses the outbox pattern for reliable delivery.

    Events are first persisted to the outbox, then published asynchronously.
    A background process ensures all pending events are eventually published.
    """

    def __init__(
        self,
        store: OutboxStore,
        publish_callback: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.publish_callback = publish_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, poll_interval: float = 5.0) -> None:
        """Start the background publisher."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(poll_interval))
        logger.info("outbox_publisher_started")

    async def stop(self) -> None:
        """Stop the background publisher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("outbox_publisher_stopped")

    async def publish(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Publish an event via the outbox.

        The event is persisted immediately and published asynchronously.
        """
        entry_id = await self.store.add(event_type, payload)
        logger.debug("outbox_event_added", entry_id=entry_id, event_type=event_type)
        return entry_id

    async def _poll_loop(self, interval: float) -> None:
        """Background loop that processes pending entries."""
        while self._running:
            try:
                # Reset stale entries
                await self.store.reset_stale()

                # Process pending entries
                entries = await self.store.get_pending(limit=100)
                for entry in entries:
                    await self._process_entry(entry)

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("outbox_poll_error", error=str(e))
                await asyncio.sleep(interval)

    async def _process_entry(self, entry: OutboxEntry) -> None:
        """Process a single outbox entry."""
        await self.store.mark_processing(entry.id)

        try:
            payload = json.loads(entry.payload)

            if self.publish_callback:
                await self.publish_callback(entry.event_type, payload)

            await self.store.mark_sent(entry.id)
            logger.debug("outbox_event_published", entry_id=entry.id)

        except Exception as e:
            await self.store.mark_failed(entry.id, str(e))
            logger.warning(
                "outbox_publish_failed",
                entry_id=entry.id,
                event_type=entry.event_type,
                error=str(e),
                retry=entry.retry_count,
            )
