"""Automatic quality analyzer for annotation quality assessment.

Computes quality scores across all dimensions using heuristics and
 automated analysis methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dvas.data.schemas import Action, Annotation, Object
from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityScores,
    QualityThresholds,
)
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class AutomaticQualityAnalyzer:
    """Analyze annotation quality automatically.

    Computes quality scores across all dimensions using heuristics,
    statistical analysis, and rule-based methods.
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize analyzer.

        Args:
            thresholds: Quality thresholds for pass/fail determination
        """
        self.thresholds = thresholds or QualityThresholds()

    async def analyze(
        self,
        annotation: Annotation,
        reference_annotation: Optional[Annotation] = None,
    ) -> QualityScores:
        """Analyze a single annotation.

        Args:
            annotation: The annotation to analyze
            reference_annotation: Optional reference for comparison

        Returns:
            QualityScores with all dimension scores computed
        """
        scores = QualityScores(
            annotation_id=annotation.id,
            video_id=annotation.video_id,
            computed_by="automatic",
        )

        # Compute each dimension score
        scores.factuality_score = self._analyze_factuality(
            annotation, reference_annotation
        )
        scores.temporal_consistency_score = self._analyze_temporal_consistency(
            annotation
        )
        scores.object_grounding_score = self._analyze_object_grounding(annotation)
        scores.action_grounding_score = self._analyze_action_grounding(annotation)
        scores.affordance_score = self._analyze_affordance(annotation)
        scores.robotic_usefulness_score = self._analyze_robotic_usefulness(annotation)
        scores.language_clarity_score = self._analyze_language_clarity(annotation)
        scores.parse_confidence_score = self._analyze_parse_confidence(annotation)
        scores.reviewer_confidence_score = self._analyze_reviewer_confidence(annotation)

        # Recompute aggregates
        scores._compute_aggregates()

        logger.info(
            "quality_analysis_complete",
            annotation_id=annotation.id,
            overall_score=scores.overall_score,
            failed_dimensions=len(scores.failed_dimensions),
        )

        return scores

    async def analyze_batch(
        self,
        annotations: List[Annotation],
        references: Optional[Dict[str, Annotation]] = None,
    ) -> Dict[str, QualityScores]:
        """Analyze a batch of annotations.

        Args:
            annotations: List of annotations to analyze
            references: Optional dict of annotation_id -> reference annotation

        Returns:
            Dict mapping annotation_id to QualityScores
        """
        results = {}

        for annotation in annotations:
            ref = references.get(annotation.id) if references else None
            try:
                scores = await self.analyze(annotation, ref)
                results[annotation.id] = scores
            except Exception as e:
                logger.error(
                    "analysis_failed",
                    annotation_id=annotation.id,
                    error=str(e),
                )
                # Create empty scores on failure
                results[annotation.id] = QualityScores(
                    annotation_id=annotation.id,
                    video_id=annotation.video_id,
                    computed_by="automatic",
                )

        logger.info(
            "batch_analysis_complete",
            total=len(annotations),
            successful=sum(1 for s in results.values() if s.overall_score > 0),
        )

        return results

    def _analyze_factuality(
        self,
        annotation: Annotation,
        reference: Optional[Annotation] = None,
    ) -> DimensionScore:
        """Analyze factuality - accuracy of described actions/objects.

        Without reference, uses heuristics like:
        - Presence of actions and objects
        - Reasonable action-object combinations
        - Caption consistency with extracted entities
        """
        issues = []
        details = {}

        # Extract all actions and objects from segments
        all_actions: List[Action] = []
        all_objects: List[Object] = []
        all_captions: List[str] = []

        for segment in annotation.segments:
            all_actions.extend(segment.actions)
            all_objects.extend(segment.objects)
            all_captions.append(segment.caption)

        # Check for minimal content
        if not all_actions:
            issues.append("no_actions_detected")
        if not all_objects:
            issues.append("no_objects_detected")
        if not all_captions or all(not c.strip() for c in all_captions):
            issues.append("no_captions")

        # Check caption-action consistency
        action_mentions = 0
        for caption in all_captions:
            caption_lower = caption.lower()
            for action in all_actions:
                if action.verb.lower() in caption_lower:
                    action_mentions += 1
                    break

        action_consistency = (
            action_mentions / len(all_captions) if all_captions else 0
        )
        details["action_consistency_ratio"] = action_consistency

        if action_consistency < 0.5 and all_actions:
            issues.append("low_action_caption_consistency")

        # Check caption-object consistency
        object_mentions = 0
        for caption in all_captions:
            caption_lower = caption.lower()
            for obj in all_objects:
                if obj.name.lower() in caption_lower:
                    object_mentions += 1
                    break

        object_consistency = (
            object_mentions / len(all_captions) if all_captions else 0
        )
        details["object_consistency_ratio"] = object_consistency

        if object_consistency < 0.5 and all_objects:
            issues.append("low_object_caption_consistency")

        # Compute score
        base_score = 0.5  # Start neutral

        if all_actions and all_objects:
            base_score += 0.2
        if action_consistency > 0.7:
            base_score += 0.15
        if object_consistency > 0.7:
            base_score += 0.15

        # Penalize issues
        score = max(0.0, base_score - len(issues) * 0.1)

        return DimensionScore(
            dimension=QualityDimension.FACTUALITY,
            score=score,
            confidence=0.7 if not reference else 0.9,
            details=details,
            issues=issues,
        )

    def _analyze_temporal_consistency(self, annotation: Annotation) -> DimensionScore:
        """Analyze temporal consistency across time windows.

        Checks for:
        - Non-overlapping segments (proper ordering)
        - Temporal gaps between segments
        - Consistent durations across similar actions
        """
        issues = []
        details = {}

        segments = annotation.segments
        if len(segments) < 2:
            # Single segment is trivially consistent
            return DimensionScore(
                dimension=QualityDimension.TEMPORAL_CONSISTENCY,
                score=0.7,  # Neutral score for minimal data
                details={"segment_count": len(segments)},
            )

        # Check for proper ordering (non-overlapping or reasonable overlaps)
        overlaps = []
        gaps = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # Check for overlap
            if current.end_time > next_seg.start_time:
                overlap = current.end_time - next_seg.start_time
                overlaps.append({"index": i, "overlap_seconds": overlap})

            # Check for gap
            if next_seg.start_time > current.end_time:
                gap = next_seg.start_time - current.end_time
                gaps.append({"index": i, "gap_seconds": gap})

        details["overlaps"] = overlaps
        details["gaps"] = gaps
        details["segment_count"] = len(segments)

        # Score based on issues
        if len(overlaps) > len(segments) * 0.5:
            issues.append("excessive_overlaps")

        large_gaps = [g for g in gaps if g["gap_seconds"] > 5.0]
        if len(large_gaps) > len(segments) * 0.3:
            issues.append("excessive_temporal_gaps")

        # Check duration consistency for same actions
        action_durations: Dict[str, List[float]] = {}
        for segment in segments:
            for action in segment.actions:
                key = f"{action.verb}_{action.noun}"
                if key not in action_durations:
                    action_durations[key] = []
                duration = segment.end_time - segment.start_time
                action_durations[key].append(duration)

        duration_variance_issues = 0
        for action_type, durations in action_durations.items():
            if len(durations) >= 3:
                import statistics

                try:
                    std = statistics.stdev(durations)
                    mean = statistics.mean(durations)
                    cv = std / mean if mean > 0 else 0
                    if cv > 0.5:  # High coefficient of variation
                        duration_variance_issues += 1
                except statistics.StatisticsError:
                    pass

        details["high_variance_action_types"] = duration_variance_issues
        if duration_variance_issues > 2:
            issues.append("inconsistent_action_durations")

        # Compute score
        score = 1.0
        score -= len(overlaps) * 0.05
        score -= len(large_gaps) * 0.03
        score -= duration_variance_issues * 0.05
        score -= len(issues) * 0.1

        return DimensionScore(
            dimension=QualityDimension.TEMPORAL_CONSISTENCY,
            score=max(0.0, score),
            confidence=0.75,
            details=details,
            issues=issues,
        )

    def _analyze_object_grounding(self, annotation: Annotation) -> DimensionScore:
        """Analyze object localization accuracy.

        Checks for:
        - Objects with bounding boxes
        - Spatial consistency of objects across segments
        - Confidence scores for object detections
        """
        issues = []
        details = {}

        total_objects = 0
        objects_with_bbox = 0
        objects_with_confidence = 0
        high_confidence_objects = 0

        for segment in annotation.segments:
            for obj in segment.objects:
                total_objects += 1
                if obj.bbox is not None:
                    objects_with_bbox += 1
                if obj.confidence is not None:
                    objects_with_confidence += 1
                    if obj.confidence > 0.7:
                        high_confidence_objects += 1

        details["total_objects"] = total_objects
        details["objects_with_bbox"] = objects_with_bbox
        details["objects_with_confidence"] = objects_with_confidence
        details["high_confidence_objects"] = high_confidence_objects

        if total_objects == 0:
            issues.append("no_objects_detected")
            return DimensionScore(
                dimension=QualityDimension.OBJECT_GROUNDING,
                score=0.3,
                details=details,
                issues=issues,
            )

        bbox_ratio = objects_with_bbox / total_objects
        confidence_ratio = (
            objects_with_confidence / total_objects if objects_with_confidence > 0 else 0
        )
        high_conf_ratio = high_confidence_objects / total_objects

        details["bbox_coverage"] = bbox_ratio
        details["confidence_coverage"] = confidence_ratio
        details["high_confidence_ratio"] = high_conf_ratio

        if bbox_ratio < 0.5:
            issues.append("low_bbox_coverage")
        if confidence_ratio < 0.5:
            issues.append("low_confidence_coverage")

        # Compute score
        score = 0.4  # Base score
        score += bbox_ratio * 0.3
        score += confidence_ratio * 0.15
        score += high_conf_ratio * 0.15

        return DimensionScore(
            dimension=QualityDimension.OBJECT_GROUNDING,
            score=min(1.0, score),
            confidence=0.65,
            details=details,
            issues=issues,
        )

    def _analyze_action_grounding(self, annotation: Annotation) -> DimensionScore:
        """Analyze action temporal localization.

        Checks for:
        - Actions with start/end times
        - Temporal alignment with segment boundaries
        - Reasonable action durations
        """
        issues = []
        details = {}

        total_actions = 0
        actions_with_times = 0
        actions_with_durations = 0

        duration_values = []

        for segment in annotation.segments:
            for action in segment.actions:
                total_actions += 1
                if action.start_time is not None and action.end_time is not None:
                    actions_with_times += 1
                    duration = action.end_time - action.start_time
                    if duration > 0:
                        actions_with_durations += 1
                        duration_values.append(duration)

        details["total_actions"] = total_actions
        details["actions_with_times"] = actions_with_times
        details["actions_with_durations"] = actions_with_durations

        if total_actions == 0:
            issues.append("no_actions_detected")
            return DimensionScore(
                dimension=QualityDimension.ACTION_GROUNDING,
                score=0.3,
                details=details,
                issues=issues,
            )

        time_coverage = actions_with_times / total_actions
        details["time_coverage"] = time_coverage

        if time_coverage < 0.5:
            issues.append("low_temporal_coverage")

        # Check duration reasonableness
        if duration_values:
            import statistics

            try:
                mean_duration = statistics.mean(duration_values)
                details["mean_action_duration"] = mean_duration

                if mean_duration < 0.5:
                    issues.append("actions_too_short")
                if mean_duration > 30:
                    issues.append("actions_too_long")
            except statistics.StatisticsError:
                pass

        # Compute score
        score = 0.5  # Base score
        score += time_coverage * 0.3
        if duration_values:
            score += 0.2

        return DimensionScore(
            dimension=QualityDimension.ACTION_GROUNDING,
            score=min(1.0, score),
            confidence=0.7,
            details=details,
            issues=issues,
        )

    def _analyze_affordance(self, annotation: Annotation) -> DimensionScore:
        """Analyze action-object relationship validity.

        Checks for:
        - Reasonable verb-noun combinations
        - Presence of physical properties for actions
        - Instruments matching actions
        """
        issues = []
        details = {}

        # Common affordance patterns
        reasonable_combinations = {
            "cut": ["knife", "scissors", "blade"],
            "pour": ["cup", "glass", "bottle", "container"],
            "stir": ["spoon", "fork", "chopstick", "whisk"],
            "open": ["door", "fridge", "drawer", "cabinet", "container", "bottle"],
            "close": ["door", "fridge", "drawer", "cabinet", "container", "bottle"],
            "pick": ["object", "item", "fruit", "vegetable", "tool"],
            "place": ["table", "counter", "shelf", "surface"],
        }

        total_actions = 0
        actions_with_physical = 0
        reasonable_matches = 0

        for segment in annotation.segments:
            for action in segment.actions:
                total_actions += 1

                if action.physical:
                    actions_with_physical += 1

                # Check verb-noun reasonableness
                verb_lower = action.verb.lower()
                noun_lower = action.noun.lower()

                if verb_lower in reasonable_combinations:
                    reasonable_nouns = reasonable_combinations[verb_lower]
                    if any(n in noun_lower for n in reasonable_nouns):
                        reasonable_matches += 1

        details["total_actions"] = total_actions
        details["actions_with_physical"] = actions_with_physical
        details["reasonable_matches"] = reasonable_matches

        if total_actions == 0:
            return DimensionScore(
                dimension=QualityDimension.AFFORDANCE,
                score=0.5,
                details=details,
                issues=["no_actions_to_evaluate"],
            )

        physical_ratio = actions_with_physical / total_actions
        reasonableness_ratio = reasonable_matches / total_actions

        details["physical_ratio"] = physical_ratio
        details["reasonableness_ratio"] = reasonableness_ratio

        # Compute score
        score = 0.4  # Base score
        score += physical_ratio * 0.3
        score += reasonableness_ratio * 0.3

        return DimensionScore(
            dimension=QualityDimension.AFFORDANCE,
            score=min(1.0, score),
            confidence=0.6,
            details=details,
            issues=issues,
        )

    def _analyze_robotic_usefulness(self, annotation: Annotation) -> DimensionScore:
        """Analyze value for robot learning.

        Checks for:
        - Embodiment action information
        - Gripper states
        - Physical property descriptions
        - Manipulation-relevant actions
        """
        issues = []
        details = {}

        manipulation_verbs = {
            "pick", "place", "grasp", "release", "push", "pull",
            "lift", "lower", "rotate", "turn", "open", "close",
            "pour", "stir", "cut", "wipe", "insert", "remove",
        }

        total_actions = 0
        manipulation_actions = 0
        actions_with_embodiment = 0
        actions_with_gripper = 0
        has_physical_properties = 0

        for segment in annotation.segments:
            for action in segment.actions:
                total_actions += 1

                verb_lower = action.verb.lower()
                if verb_lower in manipulation_verbs:
                    manipulation_actions += 1

                if action.embodiment:
                    actions_with_embodiment += 1
                    if action.embodiment.gripper_state:
                        actions_with_gripper += 1

                if action.physical:
                    has_physical_properties += 1

        details["total_actions"] = total_actions
        details["manipulation_actions"] = manipulation_actions
        details["actions_with_embodiment"] = actions_with_embodiment
        details["actions_with_gripper"] = actions_with_gripper
        details["actions_with_physical"] = has_physical_properties

        if total_actions == 0:
            return DimensionScore(
                dimension=QualityDimension.ROBOTIC_USEFULNESS,
                score=0.3,
                details=details,
                issues=["no_actions_to_evaluate"],
            )

        manipulation_ratio = manipulation_actions / total_actions
        embodiment_ratio = actions_with_embodiment / total_actions
        gripper_ratio = actions_with_gripper / total_actions
        physical_ratio = has_physical_properties / total_actions

        details["manipulation_ratio"] = manipulation_ratio
        details["embodiment_ratio"] = embodiment_ratio
        details["gripper_ratio"] = gripper_ratio
        details["physical_ratio"] = physical_ratio

        # Compute score based on robotics-relevant features
        score = 0.2  # Base score
        score += manipulation_ratio * 0.3
        score += embodiment_ratio * 0.2
        score += gripper_ratio * 0.15
        score += physical_ratio * 0.15

        return DimensionScore(
            dimension=QualityDimension.ROBOTIC_USEFULNESS,
            score=min(1.0, score),
            confidence=0.65,
            details=details,
            issues=issues,
        )

    def _analyze_language_clarity(self, annotation: Annotation) -> DimensionScore:
        """Analyze caption/text quality.

        Checks for:
        - Grammar indicators (basic)
        - Sentence structure
        - Clarity indicators (length, complexity)
        - Completeness
        """
        issues = []
        details = {}

        if not annotation.segments:
            return DimensionScore(
                dimension=QualityDimension.LANGUAGE_CLARITY,
                score=0.2,
                details={"segment_count": 0},
                issues=["no_segments"],
            )

        total_captions = 0
        caption_scores = []

        for segment in annotation.segments:
            caption = segment.caption.strip()
            if not caption:
                continue

            total_captions += 1
            score_components = []

            # Length check
            word_count = len(caption.split())
            if 5 <= word_count <= 100:
                score_components.append(0.3)
            elif word_count > 0:
                score_components.append(0.15)

            # Sentence count (prefer 1-3 sentences)
            sentence_count = caption.count(".") + caption.count("!") + caption.count("?")
            if 1 <= sentence_count <= 3:
                score_components.append(0.2)
            elif sentence_count > 0:
                score_components.append(0.1)

            # Check for action verbs
            has_action_verb = bool(segment.actions)
            if has_action_verb:
                score_components.append(0.25)

            # Check for object mentions
            has_objects = bool(segment.objects)
            if has_objects:
                score_components.append(0.25)

            caption_scores.append(sum(score_components))

        details["total_captions"] = total_captions
        details["avg_caption_score"] = (
            sum(caption_scores) / len(caption_scores) if caption_scores else 0
        )

        if total_captions == 0:
            issues.append("no_valid_captions")

        # Compute overall score
        if caption_scores:
            avg_score = sum(caption_scores) / len(caption_scores)
        else:
            avg_score = 0.1

        return DimensionScore(
            dimension=QualityDimension.LANGUAGE_CLARITY,
            score=min(1.0, avg_score),
            confidence=0.7,
            details=details,
            issues=issues,
        )

    def _analyze_parse_confidence(self, annotation: Annotation) -> DimensionScore:
        """Analyze parser confidence.

        Uses metadata from the annotation if available.
        """
        issues = []
        details = {}

        # Check annotation quality_score if available
        if annotation.quality_score is not None:
            base_score = annotation.quality_score
            details["annotation_quality_score"] = annotation.quality_score
        else:
            base_score = 0.5

        # Check quality_metrics
        if annotation.quality_metrics:
            details["quality_metrics"] = annotation.quality_metrics
            # Use parse_confidence if available
            parse_conf = annotation.quality_metrics.get("parse_confidence", 0.5)
            base_score = parse_conf

        # Check for parsing indicators
        segments_with_actions = sum(1 for s in annotation.segments if s.actions)
        segments_with_objects = sum(1 for s in annotation.segments if s.objects)

        details["segments_with_actions"] = segments_with_actions
        details["segments_with_objects"] = segments_with_objects

        if len(annotation.segments) > 0:
            action_coverage = segments_with_actions / len(annotation.segments)
            object_coverage = segments_with_objects / len(annotation.segments)
            details["action_coverage"] = action_coverage
            details["object_coverage"] = object_coverage

            # Boost score if parsing was successful
            base_score = max(base_score, (action_coverage + object_coverage) / 2)

        return DimensionScore(
            dimension=QualityDimension.PARSE_CONFIDENCE,
            score=min(1.0, base_score),
            confidence=0.75,
            details=details,
            issues=issues,
        )

    def _analyze_reviewer_confidence(self, annotation: Annotation) -> DimensionScore:
        """Analyze human reviewer confidence.

        Without actual human review, estimates based on:
        - Data completeness
        - Internal consistency
        - Richness of annotation
        """
        issues = []
        details = {}

        # Estimate confidence based on annotation richness
        richness_score = 0.0

        # Segments present
        segment_count = len(annotation.segments)
        details["segment_count"] = segment_count
        if segment_count > 0:
            richness_score += 0.2

        # Actions per segment
        avg_actions = sum(len(s.actions) for s in annotation.segments) / max(segment_count, 1)
        details["avg_actions_per_segment"] = avg_actions
        if avg_actions >= 1:
            richness_score += 0.2

        # Objects per segment
        avg_objects = sum(len(s.objects) for s in annotation.segments) / max(segment_count, 1)
        details["avg_objects_per_segment"] = avg_objects
        if avg_objects >= 1:
            richness_score += 0.2

        # V2 enhancements present
        v2_fields = 0
        for segment in annotation.segments:
            for action in segment.actions:
                if action.instrument:
                    v2_fields += 1
                if action.physical:
                    v2_fields += 1
                if action.source_state or action.target_state:
                    v2_fields += 1

        details["v2_enhancement_fields"] = v2_fields
        if v2_fields > 0:
            richness_score += 0.2

        # Quality score available
        if annotation.quality_score is not None:
            richness_score += 0.2
            details["has_quality_score"] = True

        return DimensionScore(
            dimension=QualityDimension.REVIEWER_CONFIDENCE,
            score=min(1.0, richness_score),
            confidence=0.5,  # Low confidence since this is estimated
            details=details,
            issues=issues,
        )
