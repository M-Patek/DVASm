"""World Model module for state prediction and dynamics annotation.

Provides comprehensive world model capabilities for training data generation:

- **State Representation** (`state_repr`): WorldState, ObjectState, SceneGraph
- **Dynamics** (`dynamics`): PhysicalDynamics, ContactDynamics, MotionPrediction
- **Temporal Graphs** (`temporal_graph`): TemporalEventGraph, ObjectStateTransitionGraph
- **Annotation** (`annotator`): WorldModelAnnotator for generating WM training data
- **Training Export** (`training_export`): Export to Sarlo, SAPIEN formats
- **Quality Evaluation** (`quality_evaluator`): Evaluate state prediction accuracy
- **Benchmarks** (`benchmark`): Standardized benchmarks for WM capabilities

Quick start:
    from dvas.world_model import WorldModelAnnotator, WorldState

    annotator = WorldModelAnnotator()
    state_before = await annotator.generate_state_before(segment)
    state_after = await annotator.generate_state_after(segment)
    prediction = await annotator.predict_next_state(state_before, action)

Teacher-based annotation:
    from dvas.models.teacher.base import TeacherModel
    from dvas.world_model import WorldModelAnnotator

    teacher = TeacherModel(model_name="gpt-5.5")
    annotator = WorldModelAnnotator(teacher_model=teacher, use_teacher=True)
"""

from dvas.world_model.annotator import WorldModelAnnotator
from dvas.world_model.benchmark import (
    BenchmarkResult,
    BenchmarkSuiteResult,
    CausalRelationBenchmark,
    CounterfactualBenchmark,
    StatePredictionBenchmark,
    WorldModelAnnotationBenchmark,
    load_benchmark_results,
    run_benchmarks,
)
from dvas.world_model.dynamics import (
    ContactDynamics,
    ContactEvent,
    ContactType,
    DynamicsType,
    ForceVector,
    MotionPrediction,
    PhysicalDynamics,
    PhysicalProperties,
    Trajectory,
)
from dvas.world_model.quality_evaluator import (
    CounterfactualMetrics,
    DynamicsQualityMetrics,
    StatePredictionMetrics,
    WorldModelQualityEvaluator,
    WorldModelQualityReport,
)
from dvas.world_model.state_repr import (
    AffordanceState,
    ContactState,
    ObjectRole,
    ObjectState,
    Relationship,
    SceneGraph,
    WorldState,
)
from dvas.world_model.temporal_graph import (
    EventType,
    MultiObjectTransitionGraph,
    ObjectStateTransitionGraph,
    StateTransition,
    TemporalEvent,
    TemporalEventGraph,
    TemporalRelation,
    TemporalRelationType,
)
from dvas.world_model.training_export import (
    GenericExporter,
    SarloExporter,
    SapienExporter,
    TrajectorySample,
    export_trajectories,
)

__all__ = [
    # Core annotator
    "WorldModelAnnotator",
    # State representation
    "WorldState",
    "ObjectState",
    "SceneGraph",
    "Relationship",
    "AffordanceState",
    "ContactState",
    "ObjectRole",
    # Dynamics
    "PhysicalDynamics",
    "ContactDynamics",
    "MotionPrediction",
    "PhysicalProperties",
    "ContactEvent",
    "ForceVector",
    "Trajectory",
    "DynamicsType",
    "ContactType",
    # Temporal graphs
    "TemporalEventGraph",
    "ObjectStateTransitionGraph",
    "MultiObjectTransitionGraph",
    "TemporalEvent",
    "TemporalRelation",
    "StateTransition",
    "EventType",
    "TemporalRelationType",
    # Training export
    "TrajectorySample",
    "SarloExporter",
    "SapienExporter",
    "GenericExporter",
    "export_trajectories",
    # Quality evaluation
    "WorldModelQualityEvaluator",
    "WorldModelQualityReport",
    "StatePredictionMetrics",
    "CounterfactualMetrics",
    "DynamicsQualityMetrics",
    # Benchmarks
    "WorldModelAnnotationBenchmark",
    "StatePredictionBenchmark",
    "CausalRelationBenchmark",
    "CounterfactualBenchmark",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "run_benchmarks",
    "load_benchmark_results",
]
