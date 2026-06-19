"""Robot-specific annotation schemas for DVAS.

This package provides schemas for VLA (Vision-Language-Action) and
robot learning applications, extending the base annotation format
with robot-specific capabilities.
"""

from dvas.data.robot_schemas.robot_action import (
    ActionCondition,
    ActionPrimitive,
    CausalLink,
    ContactEvent,
    EnhancedRobotAction,
    FailureAnnotation,
    FailureMode,
    GripperData,
    GripperState,
    HandPose,
    Pose6D,
    RobotPolicyHint,
)

from dvas.data.robot_schemas.affordance import (
    AffordanceAnnotation,
    AffordanceType,
    ForceRequirements,
    GraspConstraints,
    Handedness,
    ObjectAffordance,
    SingleAffordance,
    SpatialRegion,
)

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

__all__ = [
    # Robot action
    "ActionCondition",
    "ActionPrimitive",
    "CausalLink",
    "ContactEvent",
    "EnhancedRobotAction",
    "FailureAnnotation",
    "FailureMode",
    "GripperData",
    "GripperState",
    "HandPose",
    "Pose6D",
    "RobotPolicyHint",
    # Affordance
    "AffordanceAnnotation",
    "AffordanceType",
    "ForceRequirements",
    "GraspConstraints",
    "Handedness",
    "ObjectAffordance",
    "SingleAffordance",
    "SpatialRegion",
    # Counterfactual
    "AlternativeAction",
    "CounterfactualAnnotation",
    "CounterfactualOutcome",
    "CounterfactualType",
    "OutcomeLikelihood",
    "ScenarioValidity",
    "SingleCounterfactual",
    "StateChange",
    "WhatIfQuery",
]
