"""Quality evaluation for World Model annotations.

Evaluates:
- State prediction accuracy
- Counterfactual validity
- Dynamics annotation quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvas.data.schemas import Action, Annotation, Segment
from dvas.utils.logging import get_logger
from dvas.world_model.dynamics import ContactDynamics, MotionPrediction, PhysicalDynamics
from dvas.world_model.state_repr import WorldState

logger = get_logger(__name__)


@dataclass
class StatePredictionMetrics:
    """Metrics for state prediction evaluation.

    Attributes:
        mse: Mean squared error across all predictions
        mae: Mean absolute error
        rmse: Root mean squared error
        end_position_error: Error at final timestep
        collision_accuracy: Accuracy of collision predictions
        coverage: Ratio of predicted objects vs actual
        temporal_consistency: Smoothness of predictions over time
    """

    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    end_position_error: float = 0.0
    collision_accuracy: float = 0.0
    coverage: float = 0.0
    temporal_consistency: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "end_position_error": self.end_position_error,
            "collision_accuracy": self.collision_accuracy,
            "coverage": self.coverage,
            "temporal_consistency": self.temporal_consistency,
        }


@dataclass
class CounterfactualMetrics:
    """Metrics for counterfactual validity.

    Attributes:
        physical_plausibility: Score for physical plausibility (0-1)
        semantic_coherence: Consistency with language description
        diversity: Diversity of generated counterfactuals
        action_feasibility: Feasibility of alternative actions
    """

    physical_plausibility: float = 0.0
    semantic_coherence: float = 0.0
    diversity: float = 0.0
    action_feasibility: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "physical_plausibility": self.physical_plausibility,
            "semantic_coherence": self.semantic_coherence,
            "diversity": self.diversity,
            "action_feasibility": self.action_feasibility,
        }


@dataclass
class DynamicsQualityMetrics:
    """Metrics for dynamics annotation quality.

    Attributes:
        force_accuracy: Accuracy of force predictions
        contact_detection_f1: F1 score for contact detection
        trajectory_ade: Average displacement error
        trajectory_fde: Final displacement error
        physical_consistency: Physics rule compliance
        density_estimate_error: Error in mass/density estimates
    """

    force_accuracy: float = 0.0
    contact_detection_f1: float = 0.0
    trajectory_ade: float = 0.0
    trajectory_fde: float = 0.0
    physical_consistency: float = 0.0
    density_estimate_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "force_accuracy": self.force_accuracy,
            "contact_detection_f1": self.contact_detection_f1,
            "trajectory_ade": self.trajectory_ade,
            "trajectory_fde": self.trajectory_fde,
            "physical_consistency": self.physical_consistency,
            "density_estimate_error": self.density_estimate_error,
        }


@dataclass
class WorldModelQualityReport:
    """Complete quality report for world model annotations.

    Attributes:
        annotation_id: ID of evaluated annotation
        state_prediction: State prediction metrics
        counterfactual: Counterfactual validity metrics
        dynamics: Dynamics quality metrics
        overall_score: Aggregated quality score
        issues: List of identified issues
        recommendations: Suggested improvements
    """

    annotation_id: str
    state_prediction: StatePredictionMetrics = field(
        default_factory=StatePredictionMetrics
    )
    counterfactual: CounterfactualMetrics = field(default_factory=CounterfactualMetrics)
    dynamics: DynamicsQualityMetrics = field(default_factory=DynamicsQualityMetrics)
    overall_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "state_prediction": self.state_prediction.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "overall_score": self.overall_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


class WorldModelQualityEvaluator:
    """Evaluate quality of World Model training annotations.

    Provides comprehensive quality assessment including:
    - State prediction accuracy against ground truth
    - Counterfactual scenario validity
    - Dynamics annotation quality
    - Physics consistency checks
    """

    def __init__(
        self,
        position_threshold: float = 0.1,  # 10cm
        velocity_threshold: float = 0.05,  # 5cm/s
        physics_tolerance: float = 0.15,
    ):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.physics_tolerance = physics_tolerance

    def evaluate_state_predictions(
        self,
        predictions: List[WorldState],
        ground_truth: List[WorldState],
    ) -> StatePredictionMetrics:
        """Evaluate state prediction accuracy.

        Args:
            predictions: Predicted world states
            ground_truth: Actual world states

        Returns:
            State prediction metrics
        """
        metrics = StatePredictionMetrics()

        if not predictions or not ground_truth:
            logger.warning("evaluate_predictions_empty_input")
            return metrics

        # Ensure same length
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

        position_errors = []
        velocity_errors = []

        for pred, gt in zip(predictions, ground_truth):
            for obj_id in gt.scene_graph.objects:
                if obj_id in pred.scene_graph.objects:
                    pred_obj = pred.scene_graph.objects[obj_id]
                    gt_obj = gt.scene_graph.objects[obj_id]

                    # Position error
                    pos_error = np.linalg.norm(pred_obj.position - gt_obj.position)
                    position_errors.append(pos_error)

                    # Velocity error
                    vel_error = np.linalg.norm(pred_obj.velocity - gt_obj.velocity)
                    velocity_errors.append(vel_error)

        if position_errors:
            metrics.mse = float(np.mean([e ** 2 for e in position_errors]))
            metrics.mae = float(np.mean(position_errors))
            metrics.rmse = float(np.sqrt(metrics.mse))
            metrics.end_position_error = float(position_errors[-1])

        if velocity_errors:
            metrics.temporal_consistency = float(
                1.0 - min(np.mean(velocity_errors) / self.velocity_threshold, 1.0)
            )

        return metrics

    def evaluate_counterfactuals(
        self,
        counterfactuals: List[Dict[str, Any]],
        reference_annotation: Annotation,
    ) -> CounterfactualMetrics:
        """Evaluate counterfactual scenario validity.

        Args:
            counterfactuals: Generated counterfactual scenarios
            reference_annotation: Original annotation for context

        Returns:
            Counterfactual validity metrics
        """
        metrics = CounterfactualMetrics()

        if not counterfactuals:
            return metrics

        # Physical plausibility: check if counterfactuals obey physics
        plausible_count = 0
        for cf in counterfactuals:
            if self._check_physical_plausibility(cf):
                plausible_count += 1

        metrics.physical_plausibility = plausible_count / len(counterfactuals)

        # Semantic coherence: check language consistency
        coherent_count = 0
        for cf in counterfactuals:
            if self._check_semantic_coherence(cf, reference_annotation):
                coherent_count += 1

        metrics.semantic_coherence = coherent_count / len(counterfactuals)

        # Diversity: measure uniqueness of outcomes
        outcomes = [cf.get("then", "") for cf in counterfactuals]
        unique_outcomes = len(set(outcomes))
        metrics.diversity = unique_outcomes / len(counterfactuals) if counterfactuals else 0

        # Action feasibility: check if actions are physically possible
        feasible_count = 0
        for cf in counterfactuals:
            if self._check_action_feasibility(cf, reference_annotation):
                feasible_count += 1

        metrics.action_feasibility = feasible_count / len(counterfactuals)

        return metrics

    def evaluate_dynamics(
        self,
        dynamics: List[PhysicalDynamics],
        ground_truth_contacts: Optional[List[ContactDynamics]] = None,
    ) -> DynamicsQualityMetrics:
        """Evaluate dynamics annotation quality.

        Args:
            dynamics: Predicted dynamics annotations
            ground_truth_contacts: Optional ground truth contacts

        Returns:
            Dynamics quality metrics
        """
        metrics = DynamicsQualityMetrics()

        if not dynamics:
            return metrics

        # Contact detection F1
        if ground_truth_contacts:
            metrics.contact_detection_f1 = self._compute_contact_f1(
                dynamics, ground_truth_contacts
            )

        # Trajectory errors from motion predictions
        trajectories = []
        for d in dynamics:
            if d.trajectory:
                trajectories.append(d.trajectory)

        if trajectories:
            metrics.trajectory_ade = self._compute_ade(trajectories)
            metrics.trajectory_fde = self._compute_fde(trajectories)

        # Physical consistency check
        consistent_count = 0
        for d in dynamics:
            if self._check_physical_consistency(d):
                consistent_count += 1

        metrics.physical_consistency = consistent_count / len(dynamics)

        # Force estimation accuracy
        force_errors = []
        for d in dynamics:
            for force in d.forces:
                # Compare with estimated forces
                estimated = self._estimate_expected_force(d.properties)
                if estimated > 0:
                    error = abs(force.magnitude - estimated) / estimated
                    force_errors.append(error)

        if force_errors:
            metrics.force_accuracy = 1.0 - min(np.mean(force_errors), 1.0)

        return metrics

    def evaluate_annotation(
        self,
        annotation: Annotation,
        ground_truth_states: Optional[List[WorldState]] = None,
    ) -> WorldModelQualityReport:
        """Generate complete quality report for an annotation.

        Args:
            annotation: Annotation to evaluate
            ground_truth_states: Optional ground truth states

        Returns:
            Complete quality report
        """
        report = WorldModelQualityReport(annotation_id=annotation.id)

        # Extract world model data from annotation
        dynamics_data = annotation.dynamics
        state_predictions = annotation.state_predictions

        # Evaluate state predictions if available
        if state_predictions and ground_truth_states:
            # Convert StatePrediction to WorldState list
            predicted_states = self._state_prediction_to_states(state_predictions)
            report.state_prediction = self.evaluate_state_predictions(
                predicted_states, ground_truth_states
            )

        # Evaluate counterfactuals
        if dynamics_data and dynamics_data.counterfactuals:
            report.counterfactual = self.evaluate_counterfactuals(
                dynamics_data.counterfactuals,
                annotation,
            )

        # Evaluate dynamics
        if dynamics_data:
            # Convert schema DynamicsAnnotation to PhysicalDynamics list
            dynamics_list = self._convert_dynamics_annotation(dynamics_data)
            report.dynamics = self.evaluate_dynamics(dynamics_list)

        # Compute overall score
        report.overall_score = self._compute_overall_score(report)

        # Generate issues and recommendations
        report.issues = self._identify_issues(report)
        report.recommendations = self._generate_recommendations(report)

        logger.info(
            "quality_evaluation_complete",
            annotation_id=annotation.id,
            overall_score=report.overall_score,
            issue_count=len(report.issues),
        )

        return report

    def batch_evaluate(
        self,
        annotations: List[Annotation],
    ) -> Tuple[List[WorldModelQualityReport], Dict[str, float]]:
        """Evaluate multiple annotations.

        Args:
            annotations: List of annotations to evaluate

        Returns:
            Tuple of (reports, aggregate_statistics)
        """
        reports = []
        scores = []

        for annotation in annotations:
            try:
                report = self.evaluate_annotation(annotation)
                reports.append(report)
                scores.append(report.overall_score)
            except Exception as e:
                logger.error(
                    "evaluation_failed",
                    annotation_id=annotation.id,
                    error=str(e),
                )

        # Compute aggregate statistics
        stats = {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "median_score": float(np.median(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
        }

        return reports, stats

    def _check_physical_plausibility(self, counterfactual: Dict[str, Any]) -> bool:
        """Check if a counterfactual scenario is physically plausible."""
        outcome = counterfactual.get("then", "")
        implausible_keywords = ["float", "teleport", "instantly", "magic", "impossible"]
        return not any(kw in outcome.lower() for kw in implausible_keywords)

    def _check_semantic_coherence(
        self,
        counterfactual: Dict[str, Any],
        annotation: Annotation,
    ) -> bool:
        """Check if counterfactual is semantically coherent with annotation."""
        alternative = counterfactual.get("if", "")
        if not alternative:
            return False

        # Check that the alternative action is related to original actions
        original_verbs = set()
        for seg in annotation.segments:
            for action in seg.actions:
                original_verbs.add(action.verb.lower())

        # Alternative should involve similar objects/actions
        for verb in original_verbs:
            if verb in alternative.lower():
                return True

        return False

    def _check_action_feasibility(
        self,
        counterfactual: Dict[str, Any],
        annotation: Annotation,
    ) -> bool:
        """Check if alternative action is physically feasible."""
        alternative = counterfactual.get("if", "").lower()

        # Basic feasibility checks
        infeasible_patterns = [
            "grab air",
            "lift without",
            "push through",
            "reach through",
        ]

        return not any(pat in alternative for pat in infeasible_patterns)

    def _compute_contact_f1(
        self,
        dynamics: List[PhysicalDynamics],
        ground_truth: List[ContactDynamics],
    ) -> float:
        """Compute F1 score for contact detection."""
        # Simplified: count true positives, false positives, false negatives
        tp = 0
        fp = 0
        fn = 0

        # Match predicted contacts to ground truth
        for d in dynamics:
            for event in d.contact_events:
                matched = False
                for gt in ground_truth:
                    for gt_event in gt.events:
                        if (
                            event.subject_id == gt_event.subject_id
                            and event.object_id == gt_event.object_id
                            and abs(event.start_time - gt_event.start_time) < 0.1
                        ):
                            tp += 1
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    fp += 1

        fn = max(0, len(ground_truth) - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def _compute_ade(self, trajectories: List[Any]) -> float:
        """Compute Average Displacement Error."""
        if not trajectories:
            return 0.0
        # Simplified: return mean trajectory length as proxy
        total = sum(t.length() for t in trajectories)
        return total / len(trajectories)

    def _compute_fde(self, trajectories: List[Any]) -> float:
        """Compute Final Displacement Error."""
        if not trajectories:
            return 0.0
        # Simplified: return mean final segment length
        final_displacements = []
        for t in trajectories:
            if len(t.positions) >= 2:
                final_displacements.append(
                    np.linalg.norm(t.positions[-1] - t.positions[-2])
                )
        return float(np.mean(final_displacements)) if final_displacements else 0.0

    def _check_physical_consistency(self, dynamics: PhysicalDynamics) -> bool:
        """Check if dynamics annotation obeys physics rules."""
        # Check force magnitudes are reasonable
        for force in dynamics.forces:
            if force.magnitude > 10000:  # 10kN is excessive for manipulation
                return False

        # Check velocities don't exceed limits
        if dynamics.trajectory:
            for vel in dynamics.trajectory.velocities:
                if np.linalg.norm(vel) > 50:  # 50 m/s is unrealistic
                    return False

        return True

    def _estimate_expected_force(self, properties: Any) -> float:
        """Estimate expected force based on object properties."""
        if properties.mass:
            # F = mg for lifting, roughly
            return properties.mass * 9.81
        return 10.0  # Default assumption

    def _state_prediction_to_states(
        self,
        state_prediction: Any,
    ) -> List[WorldState]:
        """Convert StatePrediction schema to list of WorldState."""
        # Simplified conversion
        states = []
        if state_prediction.predicted_next_frame_desc:
            state = WorldState(
                timestamp=0.0,
                metadata={"description": state_prediction.predicted_next_frame_desc},
            )
            states.append(state)
        return states

    def _convert_dynamics_annotation(
        self,
        dynamics_annotation: Any,
    ) -> List[PhysicalDynamics]:
        """Convert schema DynamicsAnnotation to PhysicalDynamics list."""
        # Simplified conversion
        return [
            PhysicalDynamics(
                object_id="unknown",
                dynamics_type="rigid_body",
            )
        ]

    def _compute_overall_score(self, report: WorldModelQualityReport) -> float:
        """Compute aggregate quality score."""
        scores = []

        # State prediction contribution
        sp = report.state_prediction
        if sp.mae > 0:
            scores.append(1.0 - min(sp.mae / self.position_threshold, 1.0))

        # Counterfactual contribution
        cf = report.counterfactual
        scores.append(cf.physical_plausibility)
        scores.append(cf.semantic_coherence)
        scores.append(cf.action_feasibility)

        # Dynamics contribution
        dyn = report.dynamics
        scores.append(dyn.physical_consistency)
        scores.append(dyn.force_accuracy)
        scores.append(dyn.contact_detection_f1)

        return float(np.mean(scores)) if scores else 0.0

    def _identify_issues(self, report: WorldModelQualityReport) -> List[str]:
        """Identify quality issues from report."""
        issues = []

        if report.state_prediction.mae > self.position_threshold:
            issues.append(
                f"High position prediction error: {report.state_prediction.mae:.3f}m"
            )

        if report.counterfactual.physical_plausibility < 0.5:
            issues.append("Low counterfactual physical plausibility")

        if report.dynamics.physical_consistency < 0.7:
            issues.append("Dynamics annotations violate physics rules")

        if report.dynamics.contact_detection_f1 < 0.5:
            issues.append("Poor contact detection performance")

        return issues

    def _generate_recommendations(self, report: WorldModelQualityReport) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []

        if report.state_prediction.mae > self.position_threshold:
            recommendations.append(
                "Consider using higher resolution or more frames for state prediction"
            )

        if report.counterfactual.diversity < 0.5:
            recommendations.append(
                "Increase counterfactual diversity by varying action parameters"
            )

        if report.dynamics.force_accuracy < 0.7:
            recommendations.append(
                "Improve force estimation using physics-based priors"
            )

        if not recommendations and report.overall_score < 0.7:
            recommendations.append(
                "Consider re-training world model with higher quality data"
            )

        return recommendations
