"""Counterfactual annotations for robot learning.

Provides "what-if" scenarios and alternative action sequences for
robust policy training through counterfactual reasoning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CounterfactualType(str, Enum):
    """Types of counterfactual scenarios."""

    # Alternative action at a decision point
    ALTERNATIVE_ACTION = "alternative_action"

    # Different object interaction
    DIFFERENT_OBJECT = "different_object"

    # Different timing/sequencing
    TEMPORAL_VARIATION = "temporal_variation"

    # Different location/spatial arrangement
    SPATIAL_VARIATION = "spatial_variation"

    # Different tool/instrument
    TOOL_SUBSTITUTION = "tool_substitution"

    # Missing step (ablation)
    STEP_REMOVAL = "step_removal"

    # Additional step
    STEP_INSERTION = "step_insertion"

    # Different initial conditions
    INITIAL_CONDITION = "initial_condition"

    # Different environmental parameters
    ENVIRONMENT_CHANGE = "environment_change"

    # Failure scenario
    FAILURE_OUTCOME = "failure_outcome"

    # Success condition variation
    SUCCESS_VARIATION = "success_variation"


class OutcomeLikelihood(str, Enum):
    """Likelihood of counterfactual outcome."""

    IMPOSSIBLE = "impossible"  # Cannot happen
    UNLIKELY = "unlikely"  # < 30% chance
    POSSIBLE = "possible"  # 30-70% chance
    LIKELY = "likely"  # > 70% chance
    CERTAIN = "certain"  # Will happen


class ScenarioValidity(str, Enum):
    """Validity of counterfactual scenario for training."""

    VALID = "valid"  # Useful for training
    INVALID = "invalid"  # Physically/logically impossible
    DANGEROUS = "dangerous"  # Safety concern
    REDUNDANT = "redundant"  # Too similar to original
    UNINFORMATIVE = "uninformative"  # No learning value


@dataclass
class StateChange:
    """Description of a state change in counterfactual."""

    # What changed
    property_name: str

    # Original value
    original_value: Optional[Any] = None

    # Counterfactual value
    counterfactual_value: Optional[Any] = None

    # Human-readable description
    description: Optional[str] = None

    # Whether change is reversible
    is_reversible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "property_name": self.property_name,
            "original_value": self.original_value,
            "counterfactual_value": self.counterfactual_value,
            "description": self.description,
            "is_reversible": self.is_reversible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateChange":
        """Create from dictionary."""
        return cls(
            property_name=data["property_name"],
            original_value=data.get("original_value"),
            counterfactual_value=data.get("counterfactual_value"),
            description=data.get("description"),
            is_reversible=data.get("is_reversible", True),
        )


@dataclass
class AlternativeAction:
    """Alternative action for a counterfactual scenario."""

    # Reference to original action
    original_action_id: str

    # Alternative verb
    alternative_verb: str

    # Alternative noun (object)
    alternative_noun: Optional[str] = None

    # Alternative hand
    alternative_hand: Optional[str] = None

    # Description of the alternative
    description: str = ""

    # Why this is an alternative
    rationale: Optional[str] = None

    # Expected outcome difference
    outcome_difference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_action_id": self.original_action_id,
            "alternative_verb": self.alternative_verb,
            "alternative_noun": self.alternative_noun,
            "alternative_hand": self.alternative_hand,
            "description": self.description,
            "rationale": self.rationale,
            "outcome_difference": self.outcome_difference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlternativeAction":
        """Create from dictionary."""
        return cls(
            original_action_id=data["original_action_id"],
            alternative_verb=data["alternative_verb"],
            alternative_noun=data.get("alternative_noun"),
            alternative_hand=data.get("alternative_hand"),
            description=data["description"],
            rationale=data.get("rationale"),
            outcome_difference=data.get("outcome_difference"),
        )


@dataclass
class CounterfactualOutcome:
    """Expected outcome of a counterfactual scenario."""

    # Description of outcome
    description: str

    # Likelihood of this outcome
    likelihood: OutcomeLikelihood = OutcomeLikelihood.POSSIBLE

    # Numerical probability if known (0.0 to 1.0)
    probability: Optional[float] = None

    # State changes from original
    state_changes: List[StateChange] = field(default_factory=list)

    # Success/failure of task
    task_success: Optional[bool] = None

    # Efficiency comparison to original
    efficiency_comparison: str = "equivalent"  # "better", "worse", "equivalent"

    # Safety considerations
    safety_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "likelihood": self.likelihood.value,
            "probability": self.probability,
            "state_changes": [s.to_dict() for s in self.state_changes],
            "task_success": self.task_success,
            "efficiency_comparison": self.efficiency_comparison,
            "safety_notes": self.safety_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterfactualOutcome":
        """Create from dictionary."""
        return cls(
            description=data["description"],
            likelihood=OutcomeLikelihood(data.get("likelihood", "possible")),
            probability=data.get("probability"),
            state_changes=[StateChange.from_dict(s) for s in data.get("state_changes", [])],
            task_success=data.get("task_success"),
            efficiency_comparison=data.get("efficiency_comparison", "equivalent"),
            safety_notes=data.get("safety_notes"),
        )


@dataclass
class SingleCounterfactual:
    """Single counterfactual scenario.

    Represents one alternative scenario derived from an original action sequence.
    """

    # Unique identifier
    counterfactual_id: str

    # Type of counterfactual
    cf_type: CounterfactualType

    # Human-readable description
    description: str

    # Reference to original segment/action
    original_segment_id: Optional[str] = None
    original_action_id: Optional[str] = None

    # Alternative actions (if applicable)
    alternative_actions: List[AlternativeAction] = field(default_factory=list)

    # State modifications
    state_modifications: List[StateChange] = field(default_factory=list)

    # Expected outcomes
    expected_outcomes: List[CounterfactualOutcome] = field(default_factory=list)

    # Primary outcome (most likely)
    primary_outcome: Optional[CounterfactualOutcome] = None

    # Validity for training
    validity: ScenarioValidity = ScenarioValidity.VALID

    # Why this scenario was generated
    generation_rationale: Optional[str] = None

    # Use cases for training
    training_value: List[str] = field(default_factory=list)

    # Metadata
    confidence: float = 1.0
    annotated_by: str = "auto"  # "auto", "human", "model"
    annotation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "counterfactual_id": self.counterfactual_id,
            "cf_type": self.cf_type.value,
            "description": self.description,
            "original_segment_id": self.original_segment_id,
            "original_action_id": self.original_action_id,
            "alternative_actions": [a.to_dict() for a in self.alternative_actions],
            "state_modifications": [s.to_dict() for s in self.state_modifications],
            "expected_outcomes": [o.to_dict() for o in self.expected_outcomes],
            "primary_outcome": self.primary_outcome.to_dict() if self.primary_outcome else None,
            "validity": self.validity.value,
            "generation_rationale": self.generation_rationale,
            "training_value": self.training_value,
            "confidence": self.confidence,
            "annotated_by": self.annotated_by,
            "annotation_metadata": self.annotation_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleCounterfactual":
        """Create from dictionary."""
        primary_outcome = None
        if data.get("primary_outcome"):
            primary_outcome = CounterfactualOutcome.from_dict(data["primary_outcome"])

        return cls(
            counterfactual_id=data["counterfactual_id"],
            cf_type=CounterfactualType(data.get("cf_type", "alternative_action")),
            description=data["description"],
            original_segment_id=data.get("original_segment_id"),
            original_action_id=data.get("original_action_id"),
            alternative_actions=[
                AlternativeAction.from_dict(a) for a in data.get("alternative_actions", [])
            ],
            state_modifications=[
                StateChange.from_dict(s) for s in data.get("state_modifications", [])
            ],
            expected_outcomes=[
                CounterfactualOutcome.from_dict(o) for o in data.get("expected_outcomes", [])
            ],
            primary_outcome=primary_outcome,
            validity=ScenarioValidity(data.get("validity", "valid")),
            generation_rationale=data.get("generation_rationale"),
            training_value=data.get("training_value", []),
            confidence=data.get("confidence", 1.0),
            annotated_by=data.get("annotated_by", "auto"),
            annotation_metadata=data.get("annotation_metadata", {}),
        )

    def validate(self) -> tuple[bool, List[str]]:
        """Validate this counterfactual.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.counterfactual_id:
            errors.append("counterfactual_id is required")

        if not self.description:
            errors.append("description is required")

        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0.0 and 1.0")

        # Type-specific validation
        if self.cf_type == CounterfactualType.ALTERNATIVE_ACTION and not self.alternative_actions:
            errors.append("ALTERNATIVE_ACTION type must have alternative_actions")

        if self.cf_type == CounterfactualType.INITIAL_CONDITION and not self.state_modifications:
            errors.append("INITIAL_CONDITION type must have state_modifications")

        return len(errors) == 0, errors

    def is_valid_for_training(self) -> bool:
        """Check if this counterfactual is valid for training."""
        return self.validity in (ScenarioValidity.VALID, ScenarioValidity.REDUNDANT)

    def get_primary_outcome_description(self) -> str:
        """Get the description of the primary/most likely outcome."""
        if self.primary_outcome:
            return self.primary_outcome.description
        if self.expected_outcomes:
            # Return outcome with highest probability
            sorted_outcomes = sorted(
                self.expected_outcomes,
                key=lambda o: o.probability or 0.0,
                reverse=True
            )
            return sorted_outcomes[0].description
        return "No outcome specified"


@dataclass
class WhatIfQuery:
    """A "what-if" query for counterfactual reasoning.

    Represents a natural language or structured query about alternative scenarios.
    """

    # Query ID
    query_id: str

    # Query text
    query_text: str

    # Structured query components
    condition: Optional[str] = None  # "What if..."
    action: Optional[str] = None  # "...I did X..."
    outcome_question: Optional[str] = None  # "...would Y happen?"

    # Reference to original scenario
    reference_segment_id: Optional[str] = None
    reference_action_id: Optional[str] = None

    # Expected answer type
    answer_type: str = "outcome"  # "outcome", "comparison", "feasibility"

    # Generated counterfactuals answering this query
    generated_counterfactuals: List[str] = field(default_factory=list)  # IDs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "condition": self.condition,
            "action": self.action,
            "outcome_question": self.outcome_question,
            "reference_segment_id": self.reference_segment_id,
            "reference_action_id": self.reference_action_id,
            "answer_type": self.answer_type,
            "generated_counterfactuals": self.generated_counterfactuals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WhatIfQuery":
        """Create from dictionary."""
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            condition=data.get("condition"),
            action=data.get("action"),
            outcome_question=data.get("outcome_question"),
            reference_segment_id=data.get("reference_segment_id"),
            reference_action_id=data.get("reference_action_id"),
            answer_type=data.get("answer_type", "outcome"),
            generated_counterfactuals=data.get("generated_counterfactuals", []),
        )


@dataclass
class CounterfactualAnnotation:
    """Container for all counterfactual annotations of a video/segment.

    Collects counterfactual scenarios and what-if queries for training data.
    """

    # Annotation ID
    annotation_id: str

    # Video/segment reference
    video_id: str
    segment_id: Optional[str] = None

    # Counterfactual scenarios
    counterfactuals: List[SingleCounterfactual] = field(default_factory=list)

    # What-if queries
    what_if_queries: List[WhatIfQuery] = field(default_factory=list)

    # Counterfactual relationships (hierarchical or sequential)
    cf_relations: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    num_valid_cfs: int = 0
    num_invalid_cfs: int = 0

    # Metadata
    created_at: Optional[str] = None
    annotated_by: str = "auto"
    annotation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "segment_id": self.segment_id,
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "what_if_queries": [q.to_dict() for q in self.what_if_queries],
            "cf_relations": self.cf_relations,
            "num_valid_cfs": self.num_valid_cfs,
            "num_invalid_cfs": self.num_invalid_cfs,
            "created_at": self.created_at,
            "annotated_by": self.annotated_by,
            "annotation_metadata": self.annotation_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterfactualAnnotation":
        """Create from dictionary."""
        return cls(
            annotation_id=data["annotation_id"],
            video_id=data["video_id"],
            segment_id=data.get("segment_id"),
            counterfactuals=[
                SingleCounterfactual.from_dict(cf) for cf in data.get("counterfactuals", [])
            ],
            what_if_queries=[
                WhatIfQuery.from_dict(q) for q in data.get("what_if_queries", [])
            ],
            cf_relations=data.get("cf_relations", []),
            num_valid_cfs=data.get("num_valid_cfs", 0),
            num_invalid_cfs=data.get("num_invalid_cfs", 0),
            created_at=data.get("created_at"),
            annotated_by=data.get("annotated_by", "auto"),
            annotation_metadata=data.get("annotation_metadata", {}),
        )

    def add_counterfactual(self, cf: SingleCounterfactual) -> None:
        """Add a counterfactual scenario."""
        self.counterfactuals.append(cf)
        if cf.is_valid_for_training():
            self.num_valid_cfs += 1
        else:
            self.num_invalid_cfs += 1

    def add_what_if_query(self, query: WhatIfQuery) -> None:
        """Add a what-if query."""
        self.what_if_queries.append(query)

    def get_valid_counterfactuals(self) -> List[SingleCounterfactual]:
        """Get all counterfactuals valid for training."""
        return [cf for cf in self.counterfactuals if cf.is_valid_for_training()]

    def get_counterfactuals_by_type(self, cf_type: CounterfactualType) -> List[SingleCounterfactual]:
        """Get counterfactuals of a specific type."""
        return [cf for cf in self.counterfactuals if cf.cf_type == cf_type]

    def get_counterfactuals_for_action(self, action_id: str) -> List[SingleCounterfactual]:
        """Get all counterfactuals related to a specific action."""
        return [
            cf for cf in self.counterfactuals
            if cf.original_action_id == action_id
        ]

    def find_alternative_actions(self, action_id: str) -> List[AlternativeAction]:
        """Find all alternative actions for a given action."""
        alternatives = []
        for cf in self.counterfactuals:
            if cf.original_action_id == action_id:
                alternatives.extend(cf.alternative_actions)
        return alternatives

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of counterfactuals."""
        type_counts = {}
        for cf in self.counterfactuals:
            cf_type = cf.cf_type.value
            type_counts[cf_type] = type_counts.get(cf_type, 0) + 1

        return {
            "total_counterfactuals": len(self.counterfactuals),
            "valid_for_training": len(self.get_valid_counterfactuals()),
            "num_queries": len(self.what_if_queries),
            "by_type": type_counts,
            "validity_breakdown": {
                "valid": self.num_valid_cfs,
                "invalid": self.num_invalid_cfs,
            },
        }
