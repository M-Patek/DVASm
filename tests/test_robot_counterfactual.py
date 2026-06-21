"""Tests for Counterfactual schema.

Tests the counterfactual.py module including what-if scenarios,
alternative actions, and counterfactual outcomes.
"""

from dvas.data.robot_schemas.counterfactual import (
    AlternativeAction,
    CounterfactualAnnotation,
    CounterfactualOutcome,
    CounterfactualType,
    OutcomeLikelihood,
    ScenarioValidity,
    SingleCounterfactual,
    StateChange,
    WhatIfQuery,
)


class TestStateChange:
    """Test state change representation."""

    def test_state_change_initialization(self):
        """Test state change initialization."""
        change = StateChange(
            property_name="object_position",
            original_value=(0.0, 0.0, 0.0),
            counterfactual_value=(0.5, 0.5, 0.0),
            description="Object moved to different location",
        )
        assert change.property_name == "object_position"
        assert change.is_reversible is True

    def test_state_change_serialization(self):
        """Test state change round-trip."""
        original = StateChange(
            property_name="gripper_state",
            original_value="open",
            counterfactual_value="closed",
            is_reversible=True,
        )
        data = original.to_dict()
        restored = StateChange.from_dict(data)

        assert restored.property_name == original.property_name
        assert restored.original_value == original.original_value


class TestAlternativeAction:
    """Test alternative action representation."""

    def test_alternative_action_initialization(self):
        """Test alternative action initialization."""
        alt = AlternativeAction(
            original_action_id="action_001",
            alternative_verb="push",
            alternative_noun="button",
            description="Push instead of press",
        )
        assert alt.original_action_id == "action_001"
        assert alt.alternative_verb == "push"

    def test_alternative_action_with_rationale(self):
        """Test alternative action with rationale."""
        alt = AlternativeAction(
            original_action_id="action_002",
            alternative_verb="slide",
            alternative_noun="drawer",
            rationale="Sliding is more controlled than pulling",
            outcome_difference="Less risk of spilling contents",
        )
        assert alt.rationale is not None
        assert alt.outcome_difference is not None

    def test_alternative_action_serialization(self):
        """Test alternative action round-trip."""
        original = AlternativeAction(
            original_action_id="action_003",
            alternative_verb="grasp",
            alternative_noun="handle",
            alternative_hand="left",
            description="Use left hand instead",
        )
        data = original.to_dict()
        restored = AlternativeAction.from_dict(data)

        assert restored.original_action_id == original.original_action_id
        assert restored.alternative_hand == original.alternative_hand


class TestCounterfactualOutcome:
    """Test counterfactual outcome representation."""

    def test_outcome_initialization(self):
        """Test outcome initialization."""
        outcome = CounterfactualOutcome(
            description="Object falls and breaks",
            likelihood=OutcomeLikelihood.UNLIKELY,
            probability=0.15,
        )
        assert outcome.description == "Object falls and breaks"
        assert outcome.likelihood == OutcomeLikelihood.UNLIKELY

    def test_outcome_with_state_changes(self):
        """Test outcome with state changes."""
        changes = [
            StateChange(
                property_name="object_state",
                original_value="intact",
                counterfactual_value="broken",
            ),
        ]
        outcome = CounterfactualOutcome(
            description="Object breaks",
            state_changes=changes,
            task_success=False,
        )
        assert len(outcome.state_changes) == 1
        assert outcome.task_success is False

    def test_outcome_safety_notes(self):
        """Test outcome with safety notes."""
        outcome = CounterfactualOutcome(
            description="Collision occurs",
            safety_notes="May damage equipment - avoid in real execution",
        )
        assert outcome.safety_notes is not None

    def test_outcome_serialization(self):
        """Test outcome round-trip."""
        original = CounterfactualOutcome(
            description="Task completed successfully",
            likelihood=OutcomeLikelihood.LIKELY,
            probability=0.85,
            efficiency_comparison="better",
        )
        data = original.to_dict()
        restored = CounterfactualOutcome.from_dict(data)

        assert restored.description == original.description
        assert restored.efficiency_comparison == original.efficiency_comparison


class TestSingleCounterfactual:
    """Test single counterfactual scenario."""

    def test_counterfactual_initialization(self):
        """Test counterfactual initialization."""
        cf = SingleCounterfactual(
            counterfactual_id="cf_001",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="What if we used the other hand?",
        )
        assert cf.counterfactual_id == "cf_001"
        assert cf.cf_type == CounterfactualType.ALTERNATIVE_ACTION

    def test_counterfactual_types(self):
        """Test different counterfactual types."""
        types = [
            CounterfactualType.ALTERNATIVE_ACTION,
            CounterfactualType.DIFFERENT_OBJECT,
            CounterfactualType.TEMPORAL_VARIATION,
            CounterfactualType.FAILURE_OUTCOME,
        ]
        for cf_type in types:
            cf = SingleCounterfactual(
                counterfactual_id=f"cf_{cf_type.value}",
                cf_type=cf_type,
                description=f"Test {cf_type.value}",
            )
            assert cf.cf_type == cf_type

    def test_counterfactual_with_alternatives(self):
        """Test counterfactual with alternative actions."""
        alt = AlternativeAction(
            original_action_id="action_001",
            alternative_verb="slide",
            description="Slide instead of push",
        )
        cf = SingleCounterfactual(
            counterfactual_id="cf_002",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Alternative manipulation",
            alternative_actions=[alt],
        )
        assert len(cf.alternative_actions) == 1

    def test_counterfactual_with_outcomes(self):
        """Test counterfactual with outcomes."""
        primary = CounterfactualOutcome(
            description="Success with less force",
            likelihood=OutcomeLikelihood.LIKELY,
        )
        secondary = CounterfactualOutcome(
            description="Object moves too far",
            likelihood=OutcomeLikelihood.POSSIBLE,
        )
        cf = SingleCounterfactual(
            counterfactual_id="cf_003",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Try different force",
            primary_outcome=primary,
            expected_outcomes=[primary, secondary],
        )
        assert cf.primary_outcome is not None
        assert len(cf.expected_outcomes) == 2

    def test_counterfactual_validity(self):
        """Test counterfactual validity states."""
        valid_cf = SingleCounterfactual(
            counterfactual_id="cf_valid",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Valid alternative",
            validity=ScenarioValidity.VALID,
        )
        assert valid_cf.is_valid_for_training() is True

        invalid_cf = SingleCounterfactual(
            counterfactual_id="cf_invalid",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Invalid scenario",
            validity=ScenarioValidity.INVALID,
        )
        assert invalid_cf.is_valid_for_training() is False

    def test_counterfactual_validation_success(self):
        """Test validation of valid counterfactual."""
        cf = SingleCounterfactual(
            counterfactual_id="cf_004",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test",
            alternative_actions=[
                AlternativeAction(
                    original_action_id="a1",
                    alternative_verb="test",
                    description="Test",
                )
            ],
            confidence=0.8,
        )
        is_valid, errors = cf.validate()
        assert is_valid is True

    def test_counterfactual_validation_failures(self):
        """Test validation catches errors."""
        # Missing ID
        cf_no_id = SingleCounterfactual(
            counterfactual_id="",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test",
        )
        is_valid, errors = cf_no_id.validate()
        assert is_valid is False
        assert any("counterfactual_id" in e for e in errors)

        # Missing description
        cf_no_desc = SingleCounterfactual(
            counterfactual_id="cf_005",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="",
        )
        is_valid, errors = cf_no_desc.validate()
        assert is_valid is False
        assert any("description" in e for e in errors)

        # Wrong type for alternative action
        cf_wrong_type = SingleCounterfactual(
            counterfactual_id="cf_006",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test",
            alternative_actions=[],  # Missing required alternatives
        )
        is_valid, errors = cf_wrong_type.validate()
        assert is_valid is False

    def test_get_primary_outcome_description(self):
        """Test getting primary outcome description."""
        primary = CounterfactualOutcome(
            description="Primary outcome",
            probability=0.8,
        )
        cf = SingleCounterfactual(
            counterfactual_id="cf_007",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test",
            primary_outcome=primary,
        )
        desc = cf.get_primary_outcome_description()
        assert desc == "Primary outcome"

    def test_get_primary_outcome_from_list(self):
        """Test getting primary outcome from list when no primary set."""
        outcomes = [
            CounterfactualOutcome(description="Low prob", probability=0.2),
            CounterfactualOutcome(description="High prob", probability=0.8),
        ]
        cf = SingleCounterfactual(
            counterfactual_id="cf_008",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test",
            expected_outcomes=outcomes,
        )
        desc = cf.get_primary_outcome_description()
        assert desc == "High prob"

    def test_counterfactual_serialization(self):
        """Test counterfactual round-trip."""
        original = SingleCounterfactual(
            counterfactual_id="cf_full",
            cf_type=CounterfactualType.TEMPORAL_VARIATION,
            description="What if we did this faster?",
            original_action_id="action_010",
            generation_rationale="Testing speed variations",
            training_value=["speed_adaptation", "timing_robustness"],
            confidence=0.9,
            annotated_by="human",
        )
        data = original.to_dict()
        restored = SingleCounterfactual.from_dict(data)

        assert restored.counterfactual_id == original.counterfactual_id
        assert restored.training_value == original.training_value


class TestWhatIfQuery:
    """Test what-if query representation."""

    def test_query_initialization(self):
        """Test query initialization."""
        query = WhatIfQuery(
            query_id="q_001",
            query_text="What if I used a different grip?",
        )
        assert query.query_id == "q_001"
        assert query.query_text == "What if I used a different grip?"

    def test_query_structured(self):
        """Test structured query components."""
        query = WhatIfQuery(
            query_id="q_002",
            query_text="What if I grasp from the top instead of the side?",
            condition="grasp from the top",
            action="grasping",
            outcome_question="will it be more stable?",
            answer_type="comparison",
        )
        assert query.condition == "grasp from the top"
        assert query.answer_type == "comparison"

    def test_query_with_references(self):
        """Test query with action references."""
        query = WhatIfQuery(
            query_id="q_003",
            query_text="What if I did this earlier?",
            reference_segment_id="seg_001",
            reference_action_id="action_001",
        )
        assert query.reference_segment_id == "seg_001"

    def test_query_serialization(self):
        """Test query round-trip."""
        original = WhatIfQuery(
            query_id="q_004",
            query_text="Test query",
            generated_counterfactuals=["cf_001", "cf_002"],
        )
        data = original.to_dict()
        restored = WhatIfQuery.from_dict(data)

        assert restored.query_id == original.query_id
        assert restored.generated_counterfactuals == original.generated_counterfactuals


class TestCounterfactualAnnotation:
    """Test counterfactual annotation container."""

    def test_annotation_initialization(self):
        """Test annotation initialization."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_001",
            video_id="video_001",
        )
        assert ann.annotation_id == "ann_001"
        assert ann.video_id == "video_001"

    def test_add_counterfactual(self):
        """Test adding counterfactual to annotation."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_002",
            video_id="video_002",
        )

        valid_cf = SingleCounterfactual(
            counterfactual_id="cf_valid",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Valid CF",
            validity=ScenarioValidity.VALID,
        )
        ann.add_counterfactual(valid_cf)

        assert len(ann.counterfactuals) == 1
        assert ann.num_valid_cfs == 1

    def test_add_invalid_counterfactual(self):
        """Test adding invalid counterfactual."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_003",
            video_id="video_003",
        )

        invalid_cf = SingleCounterfactual(
            counterfactual_id="cf_invalid",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Invalid CF",
            validity=ScenarioValidity.INVALID,
        )
        ann.add_counterfactual(invalid_cf)

        assert ann.num_invalid_cfs == 1

    def test_add_what_if_query(self):
        """Test adding what-if query."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_004",
            video_id="video_004",
        )

        query = WhatIfQuery(
            query_id="q_001",
            query_text="What if?",
        )
        ann.add_what_if_query(query)

        assert len(ann.what_if_queries) == 1

    def test_get_valid_counterfactuals(self):
        """Test getting only valid counterfactuals."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_005",
            video_id="video_005",
        )

        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf1",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Valid",
                validity=ScenarioValidity.VALID,
            )
        )
        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf2",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Invalid",
                validity=ScenarioValidity.INVALID,
            )
        )

        valid = ann.get_valid_counterfactuals()
        assert len(valid) == 1
        assert valid[0].counterfactual_id == "cf1"

    def test_get_counterfactuals_by_type(self):
        """Test getting counterfactuals by type."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_006",
            video_id="video_006",
        )

        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf_alt",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Alt action",
            )
        )
        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf_obj",
                cf_type=CounterfactualType.DIFFERENT_OBJECT,
                description="Diff object",
            )
        )

        alts = ann.get_counterfactuals_by_type(CounterfactualType.ALTERNATIVE_ACTION)
        assert len(alts) == 1
        assert alts[0].counterfactual_id == "cf_alt"

    def test_get_counterfactuals_for_action(self):
        """Test getting counterfactuals for specific action."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_007",
            video_id="video_007",
        )

        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf_for_a1",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="For action 1",
                original_action_id="action_001",
            )
        )
        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf_for_a2",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="For action 2",
                original_action_id="action_002",
            )
        )

        cfs = ann.get_counterfactuals_for_action("action_001")
        assert len(cfs) == 1
        assert cfs[0].counterfactual_id == "cf_for_a1"

    def test_find_alternative_actions(self):
        """Test finding alternative actions."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_008",
            video_id="video_008",
        )

        cf = SingleCounterfactual(
            counterfactual_id="cf_alt",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Alt actions",
            original_action_id="action_001",
            alternative_actions=[
                AlternativeAction(
                    original_action_id="action_001",
                    alternative_verb="slide",
                    description="Slide instead",
                ),
            ],
        )
        ann.add_counterfactual(cf)

        alts = ann.find_alternative_actions("action_001")
        assert len(alts) == 1
        assert alts[0].alternative_verb == "slide"

    def test_generate_summary(self):
        """Test summary generation."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_009",
            video_id="video_009",
        )

        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf1",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Alt 1",
                validity=ScenarioValidity.VALID,
            )
        )
        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf2",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Alt 2",
                validity=ScenarioValidity.VALID,
            )
        )
        ann.add_counterfactual(
            SingleCounterfactual(
                counterfactual_id="cf3",
                cf_type=CounterfactualType.DIFFERENT_OBJECT,
                description="Diff obj",
                validity=ScenarioValidity.VALID,
            )
        )

        summary = ann.generate_summary()
        assert summary["total_counterfactuals"] == 3
        assert summary["valid_for_training"] == 3
        assert summary["by_type"]["alternative_action"] == 2

    def test_annotation_serialization(self):
        """Test annotation round-trip."""
        ann = CounterfactualAnnotation(
            annotation_id="ann_full",
            video_id="video_full",
            segment_id="seg_001",
            created_at="2024-01-15T10:00:00",
            annotated_by="model",
        )

        cf = SingleCounterfactual(
            counterfactual_id="cf_test",
            cf_type=CounterfactualType.ALTERNATIVE_ACTION,
            description="Test CF",
            validity=ScenarioValidity.VALID,
        )
        ann.add_counterfactual(cf)

        query = WhatIfQuery(query_id="q_test", query_text="Test?")
        ann.add_what_if_query(query)

        data = ann.to_dict()
        restored = CounterfactualAnnotation.from_dict(data)

        assert restored.annotation_id == ann.annotation_id
        assert len(restored.counterfactuals) == 1
        assert len(restored.what_if_queries) == 1


class TestOutcomeLikelihood:
    """Test outcome likelihood enumeration."""

    def test_likelihood_values(self):
        """Test all likelihood values."""
        values = [
            OutcomeLikelihood.IMPOSSIBLE,
            OutcomeLikelihood.UNLIKELY,
            OutcomeLikelihood.POSSIBLE,
            OutcomeLikelihood.LIKELY,
            OutcomeLikelihood.CERTAIN,
        ]
        for val in values:
            outcome = CounterfactualOutcome(
                description="Test",
                likelihood=val,
            )
            assert outcome.likelihood == val


class TestScenarioValidity:
    """Test scenario validity enumeration."""

    def test_validity_for_training(self):
        """Test validity check for training."""
        valid_values = [ScenarioValidity.VALID, ScenarioValidity.REDUNDANT]
        for val in valid_values:
            cf = SingleCounterfactual(
                counterfactual_id=f"cf_{val.value}",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Test",
                validity=val,
            )
            assert cf.is_valid_for_training() is True

        invalid_values = [
            ScenarioValidity.INVALID,
            ScenarioValidity.DANGEROUS,
            ScenarioValidity.UNINFORMATIVE,
        ]
        for val in invalid_values:
            cf = SingleCounterfactual(
                counterfactual_id=f"cf_{val.value}",
                cf_type=CounterfactualType.ALTERNATIVE_ACTION,
                description="Test",
                validity=val,
            )
            assert cf.is_valid_for_training() is False
