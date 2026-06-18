"""Tests for saga pattern implementation."""

import asyncio

import pytest

from dvas.core.saga import (
    AnnotationSagaBuilder,
    FunctionSagaStep,
    Saga,
    SagaContext,
    SagaExecutionError,
    SagaOrchestrator,
    SagaStepResult,
    SagaStatus,
    SagaStepStatus,
)


class TestSagaContext:
    def test_set_and_get(self):
        ctx = SagaContext(saga_id="test-123")
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = SagaContext(saga_id="test-123")
        assert ctx.get("missing", "default") == "default"

    def test_results_list(self):
        ctx = SagaContext(saga_id="test-123")
        result = SagaStepResult(success=True, data="test")
        ctx.results.append(result)
        assert len(ctx.results) == 1


class TestSagaStepResult:
    def test_success_result(self):
        result = SagaStepResult(success=True, data="data", latency_ms=100.0)
        assert result.success is True
        assert result.data == "data"
        assert result.latency_ms == 100.0
        assert result.error is None

    def test_failure_result(self):
        result = SagaStepResult(success=False, error="something failed")
        assert result.success is False
        assert result.error == "something failed"


class TestFunctionSagaStep:
    @pytest.mark.asyncio
    async def test_execute_action(self):
        async def action(ctx: SagaContext) -> SagaStepResult:
            return SagaStepResult(success=True, data="action_result")

        async def compensation(ctx: SagaContext) -> SagaStepResult:
            return SagaStepResult(success=True)

        step = FunctionSagaStep("test_step", action, compensation)
        ctx = SagaContext(saga_id="test")

        result = await step.execute(ctx)
        assert result.success is True
        assert result.data == "action_result"

    @pytest.mark.asyncio
    async def test_compensate(self):
        async def action(ctx: SagaContext) -> SagaStepResult:
            return SagaStepResult(success=True)

        async def compensation(ctx: SagaContext) -> SagaStepResult:
            return SagaStepResult(success=True, data="compensated")

        step = FunctionSagaStep("test_step", action, compensation)
        ctx = SagaContext(saga_id="test")

        result = await step.compensate(ctx)
        assert result.success is True
        assert result.data == "compensated"


class TestSaga:
    def test_create_saga(self):
        saga = Saga("test_saga")
        assert saga.name == "test_saga"
        assert len(saga) == 0

    def test_add_step(self):
        saga = Saga("test")

        async def action(ctx):
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        step = FunctionSagaStep("step1", action, comp)
        saga.add_step(step)

        assert len(saga) == 1
        assert saga.steps[0].name == "step1"

    def test_add_step_chaining(self):
        saga = Saga("test")

        async def action(ctx):
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        saga.add_step(FunctionSagaStep("s1", action, comp)).add_step(
            FunctionSagaStep("s2", action, comp)
        )

        assert len(saga) == 2


class TestSagaOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return SagaOrchestrator()

    @pytest.mark.asyncio
    async def test_successful_saga(self, orchestrator):
        async def action(ctx):
            ctx.set("result", "ok")
            return SagaStepResult(success=True, data="ok")

        async def comp(ctx):
            return SagaStepResult(success=True)

        saga = Saga("success_saga")
        saga.add_step(FunctionSagaStep("step1", action, comp))
        saga.add_step(FunctionSagaStep("step2", action, comp))

        ctx = await orchestrator.execute(saga)

        assert ctx.get("result") == "ok"
        assert orchestrator.get_status(ctx.saga_id) == SagaStatus.COMPLETED
        assert len(ctx.results) == 2

    @pytest.mark.asyncio
    async def test_failed_saga_with_compensation(self, orchestrator):
        async def success_action(ctx):
            return SagaStepResult(success=True)

        async def fail_action(ctx):
            return SagaStepResult(success=False, error="step failed")

        async def comp(ctx):
            ctx.set("compensated", True)
            return SagaStepResult(success=True)

        saga = Saga("fail_saga")
        saga.add_step(FunctionSagaStep("step1", success_action, comp))
        saga.add_step(FunctionSagaStep("step2", fail_action, comp))

        with pytest.raises(SagaExecutionError) as exc_info:
            await orchestrator.execute(saga)

        assert "step2" in str(exc_info.value)
        assert exc_info.value.failed_step == "step2"

    @pytest.mark.asyncio
    async def test_saga_with_timeout(self, orchestrator):
        async def slow_action(ctx):
            await asyncio.sleep(1.0)
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        saga = Saga("timeout_saga")
        saga.add_step(FunctionSagaStep("slow", slow_action, comp))

        with pytest.raises(SagaExecutionError) as exc_info:
            await orchestrator.execute(saga, timeout=0.1)

        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_saga_step_status_tracking(self, orchestrator):
        async def action(ctx):
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        step = FunctionSagaStep("track_step", action, comp)
        assert step.status == SagaStepStatus.PENDING

        saga = Saga("track_saga")
        saga.add_step(step)

        await orchestrator.execute(saga)
        assert step.status == SagaStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_saga_context_passed_between_steps(self, orchestrator):
        async def step1_action(ctx):
            ctx.set("step1_data", "from_step1")
            return SagaStepResult(success=True)

        async def step2_action(ctx):
            data = ctx.get("step1_data")
            ctx.set("step2_data", f"got_{data}")
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        saga = Saga("context_saga")
        saga.add_step(FunctionSagaStep("s1", step1_action, comp))
        saga.add_step(FunctionSagaStep("s2", step2_action, comp))

        ctx = await orchestrator.execute(saga)
        assert ctx.get("step2_data") == "got_from_step1"

    @pytest.mark.asyncio
    async def test_compensation_failure_handled(self, orchestrator):
        async def success_action(ctx):
            return SagaStepResult(success=True)

        async def fail_action(ctx):
            return SagaStepResult(success=False, error="fail")

        async def bad_comp(ctx):
            raise ValueError("compensation error")

        saga = Saga("bad_comp_saga")
        saga.add_step(FunctionSagaStep("s1", success_action, bad_comp))
        saga.add_step(FunctionSagaStep("s2", fail_action, bad_comp))

        # Should not raise from compensation error
        with pytest.raises(SagaExecutionError):
            await orchestrator.execute(saga)

    @pytest.mark.asyncio
    async def test_multiple_sagas(self, orchestrator):
        async def action(ctx):
            return SagaStepResult(success=True)

        async def comp(ctx):
            return SagaStepResult(success=True)

        saga1 = Saga("saga1")
        saga1.add_step(FunctionSagaStep("s1", action, comp))

        saga2 = Saga("saga2")
        saga2.add_step(FunctionSagaStep("s1", action, comp))

        ctx1 = await orchestrator.execute(saga1)
        ctx2 = await orchestrator.execute(saga2)

        assert ctx1.saga_id != ctx2.saga_id
        assert orchestrator.get_status(ctx1.saga_id) == SagaStatus.COMPLETED
        assert orchestrator.get_status(ctx2.saga_id) == SagaStatus.COMPLETED


class TestAnnotationSagaBuilder:
    def test_create_annotation_saga(self):
        saga = AnnotationSagaBuilder.create_annotation_saga(
            video_path="/path/to/video.mp4",
            video_id="vid_123",
            teacher_model=None,
            store=None,
        )

        assert saga.name == "annotate_vid_123"
        assert len(saga) == 4
        assert saga.steps[0].name == "load_video"
        assert saga.steps[1].name == "detect_scenes"
        assert saga.steps[2].name == "annotate_segments"
        assert saga.steps[3].name == "save_annotation"

    @pytest.mark.asyncio
    async def test_load_video_action(self):
        action = AnnotationSagaBuilder._load_video_action("/path/to/video.mp4", "vid_123")
        ctx = SagaContext(saga_id="test")

        # Will fail because video doesn't exist, but tests the structure
        result = await action(ctx)
        # Video doesn't exist, so this should fail
        assert result.success is False

    @pytest.mark.asyncio
    async def test_noop_compensation(self):
        comp = AnnotationSagaBuilder._noop_compensation()
        ctx = SagaContext(saga_id="test")
        result = await comp(ctx)
        assert result.success is True
