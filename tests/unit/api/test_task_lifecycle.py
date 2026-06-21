"""Tests for task lifecycle management."""

import pytest

from dvas.api.task_lifecycle import (
    Checkpoint,
    RetryConfig,
    RetryPolicy,
    TaskLifecycleManager,
)
from dvas.api.task_store import InMemoryTaskStore, TaskStatus, TaskType


class TestTaskLifecycleManager:
    """Test TaskLifecycleManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh lifecycle manager."""
        store = InMemoryTaskStore()
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            max_retries=3,
            base_delay_seconds=0.1,  # Fast for tests
            max_delay_seconds=1.0,
        )
        return TaskLifecycleManager(store, config)

    @pytest.mark.asyncio
    async def test_create_task(self, manager):
        """Test creating a task."""
        task = await manager.create_task(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
            priority=5,
        )
        assert task.id is not None
        assert task.status == TaskStatus.PENDING
        assert task.type == TaskType.ANNOTATION
        assert task.payload["video_id"] == "vid_123"

    @pytest.mark.asyncio
    async def test_start_processing(self, manager):
        """Test starting task processing."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        updated = await manager.start_processing(task.id)
        assert updated is not None
        assert updated.status == TaskStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_start_processing_not_found(self, manager):
        """Test starting processing for non-existent task."""
        result = await manager.start_processing("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_start_processing_cancelled(self, manager):
        """Test starting processing for cancelled task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.cancel_task(task.id)

        result = await manager.start_processing(task.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_progress(self, manager):
        """Test updating task progress."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.start_processing(task.id)

        await manager.update_progress(task.id, 50.0)
        status = await manager.get_task_status(task.id)
        assert status["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_update_progress_bounds(self, manager):
        """Test progress bounds (0-100)."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})

        await manager.update_progress(task.id, 150.0)
        status = await manager.get_task_status(task.id)
        assert status["progress"] == 100.0

        await manager.update_progress(task.id, -10.0)
        status = await manager.get_task_status(task.id)
        assert status["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, manager):
        """Test saving a checkpoint."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        checkpoint = await manager.save_checkpoint(
            task.id,
            state={"segment": 3, "frame": 120},
            progress=50.0,
            segment_index=3,
        )
        assert checkpoint.task_id == task.id
        assert checkpoint.progress == 50.0
        assert checkpoint.segment_index == 3
        assert checkpoint.state["segment"] == 3

    @pytest.mark.asyncio
    async def test_get_checkpoint(self, manager):
        """Test retrieving a checkpoint."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.save_checkpoint(task.id, state={"segment": 3}, progress=50.0)

        checkpoint = await manager.get_checkpoint(task.id)
        assert checkpoint is not None
        assert checkpoint.progress == 50.0

    @pytest.mark.asyncio
    async def test_get_checkpoint_not_found(self, manager):
        """Test retrieving non-existent checkpoint."""
        checkpoint = await manager.get_checkpoint("nonexistent")
        assert checkpoint is None

    @pytest.mark.asyncio
    async def test_complete_task(self, manager):
        """Test completing a task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        completed = await manager.complete_task(
            task.id,
            result={"annotation_id": "ann_456"},
        )
        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED
        assert completed.progress == 100.0
        assert completed.result["annotation_id"] == "ann_456"

        # Checkpoint should be cleaned up
        checkpoint = await manager.get_checkpoint(task.id)
        assert checkpoint is None

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self, manager):
        """Test completing non-existent task."""
        result = await manager.complete_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_fail_task(self, manager):
        """Test failing a task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        failed = await manager.fail_task(task.id, error="Timeout")
        assert failed is not None
        assert failed.status == TaskStatus.FAILED
        assert failed.error == "Timeout"

    @pytest.mark.asyncio
    async def test_retry_task(self, manager):
        """Test retrying a failed task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.fail_task(task.id, error="Timeout")

        retried = await manager.retry_task(task.id)
        assert retried is not None
        assert retried.retry_count == 1
        assert retried.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_retry_task_not_found(self, manager):
        """Test retrying non-existent task."""
        result = await manager.retry_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, manager):
        """Test retry exhaustion."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.fail_task(task.id, error="Timeout")

        # Retry up to max_retries
        for i in range(5):
            retried = await manager.retry_task(task.id)
            if retried is None:
                break

        status = await manager.get_task_status(task.id)
        # retry_count should be at least 1 (first retry), but could be more
        assert status["retry_count"] >= 1
        assert status["can_retry"] is False

    @pytest.mark.asyncio
    async def test_retry_no_retry_policy(self, manager):
        """Test retry with NO_RETRY policy."""
        store = InMemoryTaskStore()
        no_retry_config = RetryConfig(policy=RetryPolicy.NO_RETRY)
        mgr = TaskLifecycleManager(store, no_retry_config)

        task = await mgr.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await mgr.fail_task(task.id, error="Timeout")

        retried = await mgr.retry_task(task.id)
        assert retried is None

    @pytest.mark.asyncio
    async def test_cancel_task(self, manager):
        """Test cancelling a task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        cancelled = await manager.cancel_task(task.id)
        assert cancelled is True

        status = await manager.get_task_status(task.id)
        assert status["status"] == "CANCELLED"
        assert status["is_terminal"] is True

    @pytest.mark.asyncio
    async def test_cancel_terminal_task(self, manager):
        """Test cancelling already terminal task."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.complete_task(task.id)

        cancelled = await manager.cancel_task(task.id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, manager):
        """Test cancelling non-existent task."""
        result = await manager.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_task_with_checkpoint(self, manager):
        """Test resuming task from checkpoint."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.save_checkpoint(task.id, state={"segment": 3}, progress=50.0)
        await manager.fail_task(task.id, error="Timeout")

        resumed = await manager.resume_task(task.id)
        assert resumed is not None
        assert resumed.status == TaskStatus.PENDING
        assert resumed.progress == 50.0

    @pytest.mark.asyncio
    async def test_resume_task_without_checkpoint(self, manager):
        """Test resuming task without checkpoint."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.fail_task(task.id, error="Timeout")

        resumed = await manager.resume_task(task.id)
        assert resumed is not None
        assert resumed.status == TaskStatus.PENDING
        assert resumed.progress == 0.0

    @pytest.mark.asyncio
    async def test_resume_task_not_found(self, manager):
        """Test resuming non-existent task."""
        result = await manager.resume_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_status(self, manager):
        """Test getting full task status."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        status = await manager.get_task_status(task.id)
        assert status is not None
        assert status["task_id"] == task.id
        assert status["status"] == "PENDING"
        assert status["can_retry"] is False  # Not failed yet
        assert status["is_terminal"] is False
        assert status["checkpoint"] is None

    @pytest.mark.asyncio
    async def test_get_task_status_with_checkpoint(self, manager):
        """Test status with checkpoint."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.save_checkpoint(task.id, state={"segment": 3}, progress=50.0)

        status = await manager.get_task_status(task.id)
        assert status["checkpoint"] is not None
        assert status["checkpoint"]["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_progress_callback(self, manager):
        """Test progress callback registration."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})

        callbacks_received = []

        def callback(task_id, progress):
            callbacks_received.append((task_id, progress))

        manager.register_progress_callback(task.id, callback)
        await manager.update_progress(task.id, 50.0)

        assert len(callbacks_received) == 1
        assert callbacks_received[0] == (task.id, 50.0)

    @pytest.mark.asyncio
    async def test_unregister_progress_callback(self, manager):
        """Test unregistering progress callback."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})

        callbacks_received = []

        def callback(task_id, progress):
            callbacks_received.append((task_id, progress))

        manager.register_progress_callback(task.id, callback)
        manager.unregister_progress_callback(task.id, callback)
        await manager.update_progress(task.id, 50.0)

        assert len(callbacks_received) == 0

    @pytest.mark.asyncio
    async def test_stream_progress(self, manager):
        """Test progress streaming."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.start_processing(task.id)

        # Collect stream events
        events = []
        async for event in manager.stream_progress(task.id):
            events.append(event)
            if len(events) >= 1:
                break

        assert len(events) >= 1
        assert events[0]["task_id"] == task.id
        assert "status" in events[0]

    @pytest.mark.asyncio
    async def test_stream_progress_terminal(self, manager):
        """Test progress streaming stops at terminal state."""
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await manager.complete_task(task.id)

        events = []
        async for event in manager.stream_progress(task.id):
            events.append(event)
            if len(events) >= 1:
                break

        assert len(events) >= 1
        assert events[0]["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_stream_progress_not_found(self, manager):
        """Test streaming for non-existent task."""
        events = []
        async for event in manager.stream_progress("nonexistent"):
            events.append(event)

        assert len(events) == 1
        assert "error" in events[0]

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting lifecycle statistics."""
        # Create some tasks
        await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_1"})
        await manager.create_task(TaskType.EXPORT, payload={"format": "llava"})
        task = await manager.create_task(TaskType.ANNOTATION, payload={"video_id": "vid_2"})
        await manager.complete_task(task.id)

        stats = await manager.get_stats()
        assert stats["total_tasks"] == 3
        assert stats["pending"] == 2
        assert stats["completed"] == 1
        assert stats["checkpoints_stored"] == 0


class TestRetryConfig:
    """Test RetryConfig."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=2.0,
            backoff_multiplier=2.0,
            max_delay_seconds=60.0,
        )
        # Retry 0: 2.0 * 2^0 = 2.0
        delay0 = config.calculate_delay(0)
        assert 1.5 <= delay0 <= 2.5  # With jitter

        # Retry 1: 2.0 * 2^1 = 4.0
        delay1 = config.calculate_delay(1)
        assert 3.0 <= delay1 <= 5.0  # With jitter

    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        config = RetryConfig(
            policy=RetryPolicy.FIXED_DELAY,
            base_delay_seconds=5.0,
        )
        delay = config.calculate_delay(2)
        assert 3.75 <= delay <= 6.25  # With jitter

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = RetryConfig(
            policy=RetryPolicy.LINEAR_BACKOFF,
            base_delay_seconds=2.0,
        )
        delay = config.calculate_delay(3)
        assert 4.5 <= delay <= 7.5  # 2.0 * 3 = 6.0 with jitter

    def test_no_retry(self):
        """Test no retry policy."""
        config = RetryConfig(policy=RetryPolicy.NO_RETRY)
        assert config.calculate_delay(0) == 0.0

    def test_max_delay_cap(self):
        """Test max delay cap."""
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=10.0,
            backoff_multiplier=10.0,
            max_delay_seconds=50.0,
        )
        delay = config.calculate_delay(5)
        assert delay <= 50.0 * 1.25  # Allow for jitter


class TestCheckpoint:
    """Test Checkpoint."""

    def test_to_dict(self):
        """Test checkpoint serialization."""
        from datetime import datetime, timezone

        checkpoint = Checkpoint(
            task_id="task_123",
            state={"segment": 3},
            progress=50.0,
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            segment_index=3,
        )
        data = checkpoint.to_dict()
        assert data["task_id"] == "task_123"
        assert data["progress"] == 50.0
        assert data["segment_index"] == 3

    def test_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "task_id": "task_123",
            "state": {"segment": 3},
            "progress": 50.0,
            "timestamp": "2024-01-01T00:00:00+00:00",
            "segment_index": 3,
        }
        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.task_id == "task_123"
        assert checkpoint.progress == 50.0
        assert checkpoint.segment_index == 3
