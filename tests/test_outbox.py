"""Tests for outbox pattern implementation."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from dvas.core.outbox import (
    OutboxEntry,
    OutboxPublisher,
    OutboxStatus,
    OutboxStore,
)


class TestOutboxEntry:
    def test_create_entry(self):
        entry = OutboxEntry(
            id="abc123",
            event_type="test_event",
            payload='{"key": "value"}',
            status=OutboxStatus.PENDING,
            created_at=1234567890.0,
            retry_count=0,
            error=None,
        )
        assert entry.id == "abc123"
        assert entry.status == OutboxStatus.PENDING


class TestOutboxStore:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_outbox.db"

    @pytest.fixture
    def store(self, temp_db):
        store = OutboxStore(db_path=temp_db)
        yield store
        # Cleanup: close any open connections
        import gc
        gc.collect()

    @pytest.mark.asyncio
    async def test_add_entry(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    @pytest.mark.asyncio
    async def test_get_pending(self, store):
        # Add entries
        await store.add("event1", {"a": 1})
        await store.add("event2", {"b": 2})

        pending = await store.get_pending(limit=10)
        assert len(pending) == 2
        assert all(e.status == OutboxStatus.PENDING for e in pending)

    @pytest.mark.asyncio
    async def test_mark_processing(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_processing(entry_id)

        pending = await store.get_pending()
        assert len(pending) == 0  # No longer pending

    @pytest.mark.asyncio
    async def test_mark_sent(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_processing(entry_id)
        await store.mark_sent(entry_id)

        # Should not appear in pending
        pending = await store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_failed(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_failed(entry_id, "connection error")

        # Failed entries should not be in pending
        pending = await store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_retry_count_incremented(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_failed(entry_id, "error 1")

        # Get all entries by checking the database directly
        import sqlite3

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT retry_count FROM outbox WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_reset_stale(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_processing(entry_id)

        # Reset stale entries (0 seconds age - should reset immediately)
        reset_count = await store.reset_stale(max_age_seconds=0)
        assert reset_count == 1

        # Should be back in pending
        pending = await store.get_pending()
        assert len(pending) == 1
        assert pending[0].id == entry_id

    @pytest.mark.asyncio
    async def test_cleanup_sent(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        await store.mark_processing(entry_id)
        await store.mark_sent(entry_id)

        # Clean up with 0 days (should remove all sent)
        cleaned = await store.cleanup_sent(max_age_days=0)
        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_concurrent_adds(self, store):
        # Test thread safety of add operations
        tasks = [store.add(f"event_{i}", {"i": i}) for i in range(10)]
        entry_ids = await asyncio.gather(*tasks)

        assert len(entry_ids) == 10
        assert len(set(entry_ids)) == 10  # All unique

    @pytest.mark.asyncio
    async def test_get_pending_limit(self, store):
        for i in range(5):
            await store.add(f"event_{i}", {"i": i})

        pending = await store.get_pending(limit=2)
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_row_to_entry(self, store):
        entry_id = await store.add("test_event", {"data": "value"})
        pending = await store.get_pending()

        assert len(pending) == 1
        entry = pending[0]
        assert isinstance(entry, OutboxEntry)
        assert entry.id == entry_id
        assert entry.event_type == "test_event"
        assert entry.status == OutboxStatus.PENDING
        assert entry.retry_count == 0


class TestOutboxPublisher:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_outbox.db"

    @pytest.fixture
    def store(self, temp_db):
        store = OutboxStore(db_path=temp_db)
        yield store
        import gc
        gc.collect()

    @pytest.mark.asyncio
    async def test_publish_adds_entry(self, store):
        publisher = OutboxPublisher(store=store)
        entry_id = await publisher.publish("my_event", {"key": "value"})

        assert isinstance(entry_id, str)

        pending = await store.get_pending()
        assert len(pending) == 1
        assert pending[0].id == entry_id

    @pytest.mark.asyncio
    async def test_start_stop(self, store):
        publisher = OutboxPublisher(store=store)
        await publisher.start(poll_interval=0.1)
        assert publisher._running is True
        assert publisher._task is not None

        await publisher.stop()
        assert publisher._running is False
        assert publisher._task is None

    @pytest.mark.asyncio
    async def test_background_polling(self, store):
        published_events = []

        async def mock_publish(event_type, payload):
            published_events.append((event_type, payload))

        publisher = OutboxPublisher(store=store, publish_callback=mock_publish)
        await publisher.publish("test_event", {"data": "value"})

        await publisher.start(poll_interval=0.05)
        await asyncio.sleep(0.2)  # Wait for polling
        await publisher.stop()

        assert len(published_events) == 1
        assert published_events[0][0] == "test_event"
        assert published_events[0][1] == {"data": "value"}

    @pytest.mark.asyncio
    async def test_publish_callback_error(self, store):
        async def failing_publish(event_type, payload):
            raise ValueError("publish failed")

        publisher = OutboxPublisher(store=store, publish_callback=failing_publish)
        await publisher.publish("test_event", {"data": "value"})

        await publisher.start(poll_interval=0.05)
        await asyncio.sleep(0.2)
        await publisher.stop()

        # Entry should be marked as failed
        pending = await store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_idempotent_start(self, store):
        publisher = OutboxPublisher(store=store)
        await publisher.start(poll_interval=0.1)
        await publisher.start(poll_interval=0.1)  # Should be idempotent
        assert publisher._running is True
        await publisher.stop()
