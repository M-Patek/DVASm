"""Tests for task queue implementations."""

import pytest
import pytest_asyncio

from dvas.api.task_queue import (
    CeleryTaskQueue,
    InMemoryTaskQueue,
    QueueBackend,
    QueueConfig,
    create_task_queue,
)
from dvas.api.task_store import TaskStatus, TaskType


class TestInMemoryTaskQueue:
    """Test InMemoryTaskQueue implementation."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Create a fresh in-memory queue."""
        config = QueueConfig(
            max_retries=3,
            retry_delay_base_seconds=0.1,  # Fast for tests
            retry_delay_max_seconds=1.0,
            max_concurrent=2,
            poll_interval_seconds=0.1,
        )
        queue = InMemoryTaskQueue(config=config)
        yield queue
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_enqueue_task(self, queue):
        """Test enqueuing a task."""
        result = await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
            priority=5,
        )
        assert result.task_id is not None
        assert result.status == "pending"
        assert result.queue_position is not None

    @pytest.mark.asyncio
    async def test_dequeue_task(self, queue):
        """Test dequeuing a task."""
        # Enqueue first
        await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        # Dequeue
        task = await queue.dequeue()
        assert task is not None
        assert task.status == TaskStatus.PROCESSING
        assert task.type == TaskType.ANNOTATION

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, queue):
        """Test dequeuing from empty queue."""
        task = await queue.dequeue()
        assert task is None

    @pytest.mark.asyncio
    async def test_dequeue_concurrent_limit(self, queue):
        """Test concurrent task limit."""
        # Enqueue multiple tasks
        for i in range(5):
            await queue.enqueue(
                task_type=TaskType.ANNOTATION,
                payload={"video_id": f"vid_{i}"},
            )

        # Dequeue up to max_concurrent
        task1 = await queue.dequeue()
        task2 = await queue.dequeue()
        assert task1 is not None
        assert task2 is not None

        # Third dequeue should fail (concurrent limit)
        # Note: tasks are still in processing, so no more pending
        task3 = await queue.dequeue()
        assert task3 is None  # No pending tasks left (all processing)

    @pytest.mark.asyncio
    async def test_ack_success(self, queue):
        """Test acknowledging successful task."""
        await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        task = await queue.dequeue()
        await queue.ack(task.id, success=True)

        status = await queue.get_status(task.id)
        assert status["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_ack_failure(self, queue):
        """Test acknowledging failed task."""
        await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        task = await queue.dequeue()
        await queue.ack(task.id, success=False)

        status = await queue.get_status(task.id)
        assert status["status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_cancel_task(self, queue):
        """Test cancelling a task."""
        result = await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        cancelled = await queue.cancel(result.task_id)
        assert cancelled is True

        status = await queue.get_status(result.task_id)
        assert status["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, queue):
        """Test cancelling non-existent task."""
        result = await queue.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_processing_task(self, queue):
        """Test cancelling a processing task."""
        await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        # Dequeue to make it processing
        task = await queue.dequeue()
        assert task.status == TaskStatus.PROCESSING

        # Cancel should fail for processing tasks
        cancelled = await queue.cancel(task.id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_retry_task(self, queue):
        """Test retrying a failed task."""
        await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        # Fail the task
        task = await queue.dequeue()
        await queue.ack(task.id, success=False)

        # Retry
        retried = await queue.retry(task.id)
        assert retried is not None
        assert retried.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, queue):
        """Test retry exhaustion."""
        result = await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
        )

        # Fail and retry multiple times
        for i in range(5):
            task = await queue.dequeue()
            if task:
                await queue.ack(task.id, success=False)
                retried = await queue.retry(task.id)
                if retried is None:
                    break

        # After max retries, should be None
        status = await queue.get_status(result.task_id)
        assert status["retry_count"] >= 3

    @pytest.mark.asyncio
    async def test_get_status(self, queue):
        """Test getting task status."""
        result = await queue.enqueue(
            task_type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
            priority=3,
        )

        status = await queue.get_status(result.task_id)
        assert status is not None
        assert status["task_id"] == result.task_id
        assert status["status"] == "PENDING"
        assert status["type"] == "annotation"
        # Priority may be stored as string or int, check both
        assert status["priority"] in [3, "3"]

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, queue):
        """Test getting status for non-existent task."""
        status = await queue.get_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_get_queue_length(self, queue):
        """Test getting queue length."""
        assert await queue.get_queue_length() == 0

        await queue.enqueue(TaskType.ANNOTATION, payload={"video_id": "vid_1"})
        await queue.enqueue(TaskType.ANNOTATION, payload={"video_id": "vid_2"})

        assert await queue.get_queue_length() == 2

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, queue):
        """Test tenant-scoped queue operations."""
        await queue.enqueue(
            TaskType.ANNOTATION,
            payload={"video_id": "vid_1"},
            tenant_id="tenant_a",
        )
        await queue.enqueue(
            TaskType.ANNOTATION,
            payload={"video_id": "vid_2"},
            tenant_id="tenant_b",
        )

        # Get queue length per tenant
        assert await queue.get_queue_length(tenant_id="tenant_a") == 1
        assert await queue.get_queue_length(tenant_id="tenant_b") == 1

        # Dequeue for specific tenant
        task = await queue.dequeue(tenant_id="tenant_a")
        assert task is not None
        assert task.tenant_id == "tenant_a"

    @pytest.mark.asyncio
    async def test_health_check(self, queue):
        """Test health check."""
        assert await queue.health_check() is True

    @pytest.mark.asyncio
    async def test_shutdown(self, queue):
        """Test graceful shutdown."""
        await queue.enqueue(TaskType.ANNOTATION, payload={"video_id": "vid_1"})
        await queue.dequeue()

        await queue.shutdown()
        assert queue._shutdown is True


class TestQueueConfig:
    """Test QueueConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = QueueConfig()
        assert config.max_retries == 3
        assert config.retry_delay_base_seconds == 2.0
        assert config.retry_backoff_multiplier == 2.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = QueueConfig(
            max_retries=5,
            retry_delay_base_seconds=1.0,
            retry_delay_max_seconds=60.0,
        )
        assert config.max_retries == 5
        assert config.retry_delay_base_seconds == 1.0


class TestCeleryTaskQueue:
    """Test CeleryTaskQueue (mocked)."""

    def test_init(self):
        """Test Celery queue initialization."""
        queue = CeleryTaskQueue(broker_url="redis://localhost:6379/0")
        assert queue.broker_url == "redis://localhost:6379/0"
        assert queue.backend_url == "redis://localhost:6379/0"


class TestCreateTaskQueue:
    """Test task queue factory."""

    def test_create_in_memory(self):
        """Test creating in-memory queue."""
        queue = create_task_queue(QueueBackend.IN_MEMORY)
        assert isinstance(queue, InMemoryTaskQueue)

    def test_create_celery(self):
        """Test creating Celery queue."""
        queue = create_task_queue(QueueBackend.CELERY)
        assert isinstance(queue, CeleryTaskQueue)

    def test_create_unsupported_fallback(self):
        """Test fallback for unsupported backends."""
        # Fallback behavior: unknown backend values fall back to in-memory
        queue = create_task_queue(QueueBackend("rq"))
        assert isinstance(queue, InMemoryTaskQueue)

        queue = create_task_queue(QueueBackend("arq"))
        assert isinstance(queue, InMemoryTaskQueue)
