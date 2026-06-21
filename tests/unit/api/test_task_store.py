"""Tests for task store implementations."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from dvas.api.task_store import (
    InMemoryTaskStore,
    PostgresTaskStore,
    RedisTaskStore,
    Task,
    TaskStatus,
    TaskType,
)


class TestInMemoryTaskStore:
    """Test InMemoryTaskStore implementation."""

    @pytest_asyncio.fixture
    async def store(self):
        """Create a fresh in-memory store."""
        store = InMemoryTaskStore(max_tasks=100, finished_task_ttl_seconds=60.0)
        yield store

    @pytest.mark.asyncio
    async def test_create_task(self, store):
        """Test creating a task."""
        task = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        created = await store.create(task)
        assert created.id == task.id
        assert created.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_task(self, store):
        """Test retrieving a task by ID."""
        task = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await store.create(task)

        retrieved = await store.get(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.payload["video_id"] == "vid_123"

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, store):
        """Test retrieving non-existent task."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_task(self, store):
        """Test updating a task."""
        task = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await store.create(task)

        task.status = TaskStatus.PROCESSING
        updated = await store.update(task)
        assert updated.status == TaskStatus.PROCESSING

        retrieved = await store.get(task.id)
        assert retrieved.status == TaskStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_update_task_not_found(self, store):
        """Test updating non-existent task raises KeyError."""
        task = Task(id="nonexistent", type=TaskType.ANNOTATION)
        with pytest.raises(KeyError):
            await store.update(task)

    @pytest.mark.asyncio
    async def test_delete_task(self, store):
        """Test deleting a task."""
        task = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_123"})
        await store.create(task)

        deleted = await store.delete(task.id)
        assert deleted is True

        retrieved = await store.get(task.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_task_not_found(self, store):
        """Test deleting non-existent task."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_tasks(self, store):
        """Test listing tasks with filtering."""
        task1 = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_1"})
        task2 = Task(type=TaskType.EXPORT, payload={"format": "llava"})
        task3 = Task(type=TaskType.ANNOTATION, payload={"video_id": "vid_2"})

        await store.create(task1)
        await store.create(task2)
        await store.create(task3)

        # List all
        all_tasks = await store.list()
        assert len(all_tasks) == 3

        # Filter by type
        annotation_tasks = await store.list(task_type=TaskType.ANNOTATION)
        assert len(annotation_tasks) == 2

        # Filter by status
        pending_tasks = await store.list(status=TaskStatus.PENDING)
        assert len(pending_tasks) == 3

    @pytest.mark.asyncio
    async def test_list_tasks_with_pagination(self, store):
        """Test listing with pagination."""
        for i in range(5):
            task = Task(type=TaskType.ANNOTATION, payload={"idx": i})
            await store.create(task)

        tasks = await store.list(limit=2, offset=0)
        assert len(tasks) == 2

        tasks = await store.list(limit=2, offset=2)
        assert len(tasks) == 2

        tasks = await store.list(limit=2, offset=4)
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_count_tasks(self, store):
        """Test counting tasks."""
        task1 = Task(type=TaskType.ANNOTATION)
        task2 = Task(type=TaskType.EXPORT)
        task3 = Task(type=TaskType.ANNOTATION)

        await store.create(task1)
        await store.create(task2)
        await store.create(task3)

        assert await store.count() == 3
        assert await store.count(task_type=TaskType.ANNOTATION) == 2
        assert await store.count(task_type=TaskType.EXPORT) == 1

    @pytest.mark.asyncio
    async def test_get_next_pending(self, store):
        """Test getting next pending task by priority."""
        task1 = Task(type=TaskType.ANNOTATION, priority=5)
        task2 = Task(type=TaskType.ANNOTATION, priority=2)
        task3 = Task(type=TaskType.EXPORT, priority=1)

        await store.create(task1)
        await store.create(task2)
        await store.create(task3)

        # Should return highest priority (lowest number) pending task
        next_task = await store.get_next_pending(task_type=TaskType.ANNOTATION)
        assert next_task is not None
        assert next_task.priority == 2  # task2 has priority 2

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, store):
        """Test tenant-scoped access."""
        task1 = Task(type=TaskType.ANNOTATION, tenant_id="tenant_a")
        task2 = Task(type=TaskType.ANNOTATION, tenant_id="tenant_b")

        await store.create(task1)
        await store.create(task2)

        # Get with tenant filter
        retrieved_a = await store.get(task1.id, tenant_id="tenant_a")
        assert retrieved_a is not None
        assert retrieved_a.tenant_id == "tenant_a"

        # Wrong tenant should return None
        retrieved_wrong = await store.get(task1.id, tenant_id="tenant_b")
        assert retrieved_wrong is None

        # List with tenant filter
        tenant_a_tasks = await store.list(tenant_id="tenant_a")
        assert len(tenant_a_tasks) == 1

    @pytest.mark.asyncio
    async def test_health_check(self, store):
        """Test health check."""
        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_task_to_dict(self):
        """Test task serialization."""
        task = Task(
            id="task_abc123",
            type=TaskType.ANNOTATION,
            payload={"video_id": "vid_123"},
            status=TaskStatus.PENDING,
            tenant_id="tenant_1",
            priority=3,
        )
        data = task.to_dict()
        assert data["id"] == "task_abc123"
        assert data["type"] == "annotation"
        assert data["status"] == "PENDING"
        assert data["tenant_id"] == "tenant_1"
        assert data["priority"] == 3

    @pytest.mark.asyncio
    async def test_task_from_dict(self):
        """Test task deserialization."""
        data = {
            "id": "task_abc123",
            "status": "PROCESSING",
            "type": "export",
            "payload": {"format": "llava"},
            "result": None,
            "error": None,
            "progress": 50.0,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "tenant_id": "tenant_1",
            "priority": 2,
            "retry_count": 1,
            "max_retries": 3,
        }
        task = Task.from_dict(data)
        assert task.id == "task_abc123"
        assert task.status == TaskStatus.PROCESSING
        assert task.type == TaskType.EXPORT
        assert task.progress == 50.0

    @pytest.mark.asyncio
    async def test_task_is_terminal(self):
        """Test terminal state detection."""
        assert Task(status=TaskStatus.COMPLETED).is_terminal() is True
        assert Task(status=TaskStatus.FAILED).is_terminal() is True
        assert Task(status=TaskStatus.CANCELLED).is_terminal() is True
        assert Task(status=TaskStatus.PENDING).is_terminal() is False
        assert Task(status=TaskStatus.PROCESSING).is_terminal() is False

    @pytest.mark.asyncio
    async def test_task_can_retry(self):
        """Test retry eligibility."""
        task = Task(status=TaskStatus.FAILED, retry_count=0, max_retries=3)
        assert task.can_retry() is True

        task.retry_count = 3
        assert task.can_retry() is False

        task = Task(status=TaskStatus.COMPLETED, retry_count=0, max_retries=3)
        assert task.can_retry() is False


class TestRedisTaskStore:
    """Test RedisTaskStore (requires mocking)."""

    @pytest.fixture
    def store(self):
        """Create Redis store (mocked)."""
        # Note: Redis is mocked in tests, no real connection needed
        store = RedisTaskStore(redis_url="redis://localhost:6379/0")
        return store

    def test_init(self, store):
        """Test Redis store initialization."""
        assert store._redis_url == "redis://localhost:6379/0"
        assert store._key_prefix == "dvas:task:"

    def test_task_key(self, store):
        """Test task key generation."""
        assert store._task_key("task_123") == "dvas:task:task_123"

    def test_tenant_key(self, store):
        """Test tenant key generation."""
        assert store._tenant_key("tenant_1") == "dvas:task:tenant:tenant_1"


class TestPostgresTaskStore:
    """Test PostgresTaskStore (requires mocking)."""

    @pytest.fixture
    def store(self):
        """Create Postgres store (mocked)."""
        store = PostgresTaskStore(dsn="postgresql://localhost/dvas")
        return store

    def test_init(self, store):
        """Test Postgres store initialization."""
        assert store._dsn == "postgresql://localhost/dvas"
        assert store._table_name == "tasks"

    def test_row_to_task(self, store):
        """Test row to task conversion."""

        class MockRow:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data.get(key)

        row = MockRow(
            {
                "id": "task_123",
                "status": "PENDING",
                "type": "annotation",
                "payload": '{"video_id": "vid_1"}',
                "result": None,
                "error": None,
                "progress": 0.0,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "tenant_id": "tenant_1",
                "priority": 5,
                "retry_count": 0,
                "max_retries": 3,
            }
        )

        task = store._row_to_task(row)
        assert task.id == "task_123"
        assert task.status == TaskStatus.PENDING
        assert task.type == TaskType.ANNOTATION
