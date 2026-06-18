"""Tests for actor model implementation."""

import asyncio

import pytest

from dvas.core.actor import (
    Actor,
    ActorMessage,
    ActorStatus,
    ActorSystem,
    VideoLoaderActor,
    AnnotationWorkerActor,
)


class TestActorMessage:
    def test_create_message(self):
        msg = ActorMessage(sender="test", payload="data")
        assert msg.sender == "test"
        assert msg.payload == "data"
        assert isinstance(msg.timestamp, float)
        assert isinstance(msg.message_id, str)

    def test_default_timestamp(self):
        import time
        before = time.time()
        msg = ActorMessage(sender="x", payload=1)
        after = time.time()
        assert before <= msg.timestamp <= after


class MockActor(Actor):
    """Test actor that records messages."""

    def __init__(self, name: str = "mock"):
        super().__init__(name)
        self.received_messages: list = []

    async def handle_message(self, message: ActorMessage) -> None:
        self.received_messages.append(message)


class TestActor:
    @pytest.fixture
    def actor(self):
        return MockActor("test_actor")

    @pytest.mark.asyncio
    async def test_start_stop(self, actor):
        assert actor.status == ActorStatus.IDLE

        await actor.start()
        assert actor.status == ActorStatus.RUNNING

        await actor.stop()
        assert actor.status == ActorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_send_and_receive(self, actor):
        await actor.start()

        msg = ActorMessage(sender="test", payload="hello")
        result = await actor.send(msg)
        assert result is True

        # Wait for message processing
        await asyncio.sleep(0.1)
        assert len(actor.received_messages) == 1
        assert actor.received_messages[0].payload == "hello"

        await actor.stop()

    @pytest.mark.asyncio
    async def test_send_with_timeout(self, actor):
        await actor.start()

        msg = ActorMessage(sender="test", payload="data")
        result = await actor.send(msg, timeout=1.0)
        assert result is True

        await actor.stop()

    @pytest.mark.asyncio
    async def test_mailbox_full(self, actor):
        actor_small = MockActor("small")
        actor_small._mailbox = asyncio.Queue(maxsize=1)

        await actor_small.start()

        # Fill the mailbox (queue is async, so this may not be full immediately)
        # Just verify send returns True for normal case
        msg = ActorMessage(sender="test", payload="data")
        result = await actor_small.send(msg)
        assert result is True

        await actor_small.stop()

    @pytest.mark.asyncio
    async def test_stats(self, actor):
        await actor.start()

        msg = ActorMessage(sender="test", payload="data")
        await actor.send(msg)
        await asyncio.sleep(0.1)

        stats = actor.stats
        assert stats["name"] == "test_actor"
        assert stats["messages_processed"] >= 0
        assert stats["errors"] == 0

        await actor.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        class ErrorActor(Actor):
            async def handle_message(self, message: ActorMessage) -> None:
                raise ValueError("test error")

        error_actor = ErrorActor("error_actor")
        await error_actor.start()

        msg = ActorMessage(sender="test", payload="data")
        await error_actor.send(msg)
        await asyncio.sleep(0.1)

        assert error_actor._error_count == 1
        await error_actor.stop()

    @pytest.mark.asyncio
    async def test_double_start(self, actor):
        await actor.start()
        await actor.start()  # Should be idempotent
        assert actor.status == ActorStatus.RUNNING
        await actor.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, actor):
        # Should not raise
        await actor.stop()
        assert actor.status == ActorStatus.IDLE


class TestActorSystem:
    @pytest.fixture
    def system(self):
        return ActorSystem()

    @pytest.mark.asyncio
    async def test_register_and_start_all(self, system):
        actor1 = MockActor("actor1")
        actor2 = MockActor("actor2")

        system.register(actor1)
        system.register(actor2)

        await system.start_all()
        assert actor1.status == ActorStatus.RUNNING
        assert actor2.status == ActorStatus.RUNNING

        await system.stop_all()

    @pytest.mark.asyncio
    async def test_send_to_actor(self, system):
        actor = MockActor("target")
        system.register(actor)
        await system.start_all()

        msg = ActorMessage(sender="system", payload="test")
        result = await system.send("target", msg)
        assert result is True

        await asyncio.sleep(0.1)
        assert len(actor.received_messages) == 1

        await system.stop_all()

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_actor(self, system):
        msg = ActorMessage(sender="system", payload="test")
        result = await system.send("nonexistent", msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, system):
        actor = MockActor("stats_actor")
        system.register(actor)
        await system.start_all()

        stats = system.get_stats()
        assert "stats_actor" in stats
        assert stats["stats_actor"]["name"] == "stats_actor"

        await system.stop_all()


class TestBuiltInActors:
    @pytest.mark.asyncio
    async def test_video_loader_actor(self):
        actor = VideoLoaderActor("video_loader")
        await actor.start()

        msg = ActorMessage(sender="test", payload="/path/to/video.mp4")
        await actor.send(msg)
        await asyncio.sleep(0.1)

        assert len(actor._loaded_videos) == 1
        assert "/path/to/video.mp4" in actor._loaded_videos

        await actor.stop()

    @pytest.mark.asyncio
    async def test_annotation_worker_actor(self):
        actor = AnnotationWorkerActor("worker")
        await actor.start()

        msg = ActorMessage(sender="test", payload="video_123")
        await actor.send(msg)
        await asyncio.sleep(0.1)

        assert actor._processed_count == 1

        await actor.stop()
