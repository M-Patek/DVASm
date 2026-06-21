"""Tests for WorldModelAnnotator — pytest function style.

Covers placeholder state prediction, dynamics generation, and
counterfactual generation. All methods are async.
"""

import pytest

from dvas.data.schemas import Action, Hand, Segment
from dvas.world_model import WorldModelAnnotator


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def annotator() -> WorldModelAnnotator:
    return WorldModelAnnotator()


@pytest.fixture
def sample_segment() -> Segment:
    return Segment(
        start_time=0.0,
        end_time=2.0,
        caption="pick cup from table",
        actions=[
            Action(
                verb="pick",
                noun="cup",
                hand=Hand.RIGHT,
            )
        ],
    )


@pytest.fixture
def sample_action() -> Action:
    return Action(
        verb="pick",
        noun="cup",
        hand=Hand.RIGHT,
    )


@pytest.fixture
def alternative_action() -> Action:
    return Action(
        verb="place",
        noun="cup",
        hand=Hand.RIGHT,
    )


# ── State prediction tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_state_prediction(annotator: WorldModelAnnotator, sample_segment: Segment) -> None:
    prediction = await annotator.generate_state_prediction(sample_segment)
    assert prediction.predicted_next_frame_desc is not None
    assert prediction.expected_state_change is not None
    assert prediction.preconditions == []
    assert prediction.effects == []


@pytest.mark.asyncio
async def test_generate_state_prediction_with_action(
    annotator: WorldModelAnnotator,
    sample_segment: Segment,
    sample_action: Action,
) -> None:
    prediction = await annotator.generate_state_prediction(sample_segment, sample_action)
    assert prediction is not None
    assert prediction.predicted_next_frame_desc is not None


# ── Dynamics tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_dynamics(annotator: WorldModelAnnotator, sample_segment: Segment) -> None:
    dynamics = await annotator.generate_dynamics(sample_segment)
    assert dynamics.physical_constraints == []
    assert dynamics.causal_links == []
    assert dynamics.counterfactuals == []


# ── Counterfactual tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_counterfactual(
    annotator: WorldModelAnnotator,
    sample_segment: Segment,
    sample_action: Action,
    alternative_action: Action,
) -> None:
    dynamics = await annotator.generate_counterfactual(
        sample_segment, sample_action, alternative_action
    )
    assert len(dynamics.counterfactuals) == 1
    cf = dynamics.counterfactuals[0]
    assert cf["if"] == "place cup"
    assert cf["instead_of"] == "pick cup"
    assert "set down" in cf["then"]


@pytest.mark.asyncio
async def test_generate_counterfactual_same_action(
    annotator: WorldModelAnnotator,
    sample_segment: Segment,
    sample_action: Action,
) -> None:
    dynamics = await annotator.generate_counterfactual(
        sample_segment, sample_action, sample_action
    )
    assert len(dynamics.counterfactuals) == 1
    cf = dynamics.counterfactuals[0]
    assert cf["if"] == cf["instead_of"]
