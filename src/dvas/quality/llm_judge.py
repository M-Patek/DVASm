"""LLM-as-Judge pipeline for quality evaluation.

Uses LLM to evaluate annotation quality across different dimensions
with structured prompts and confidence scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Annotation, Segment
from dvas.models.base import GenerationResult
from dvas.quality.schema import DimensionScore, QualityDimension, QualityScores
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMJudgeConfig:
    """Configuration for LLM judge."""

    model_name: str = "gpt-5.5"
    temperature: float = 0.2
    max_tokens: int = 2048
    use_structured_output: bool = True
    confidence_threshold: float = 0.7


class LLMJudgePrompts:
    """Prompt templates for LLM quality judging."""

    SYSTEM_PROMPT = """You are an expert annotation quality evaluator for video understanding datasets.
Your task is to evaluate the quality of video annotations across multiple dimensions.

Evaluate objectively and provide specific, actionable feedback.
Respond in the requested JSON format only."""

    FACTUALITY_TEMPLATE = """Evaluate the FACTUALITY of this video annotation.

Video ID: {video_id}

Annotation Content:
{annotation_text}

Rate the factuality (0.0 to 1.0) based on:
- Are the described actions plausible?
- Are the mentioned objects consistent?
- Is the description internally coherent?

Respond in JSON format:
{{
    "score": float,  // 0.0 to 1.0
    "confidence": float,  // 0.0 to 1.0
    "reasoning": string,
    "issues": [string]
}}"""

    TEMPORAL_CONSISTENCY_TEMPLATE = """Evaluate the TEMPORAL CONSISTENCY of this video annotation.

Video ID: {video_id}

Segments:
{segment_times}

Rate temporal consistency (0.0 to 1.0) based on:
- Are timestamps in logical order?
- Are durations reasonable for the actions?
- Are there gaps or overlaps that make sense?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    OBJECT_GROUNDING_TEMPLATE = """Evaluate the OBJECT GROUNDING quality.

Video ID: {video_id}

Objects mentioned:
{objects_text}

Rate object grounding (0.0 to 1.0) based on:
- Are objects clearly identified?
- Are spatial references (if any) clear?
- Is there sufficient detail about objects?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    ACTION_GROUNDING_TEMPLATE = """Evaluate the ACTION GROUNDING quality.

Video ID: {video_id}

Actions described:
{actions_text}

Rate action grounding (0.0 to 1.0) based on:
- Are actions clearly described?
- Are temporal boundaries specified?
- Are action sequences logical?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    AFFORDANCE_TEMPLATE = """Evaluate the AFFORDANCE validity.

Video ID: {video_id}

Actions and objects:
{affordance_text}

Rate affordance (0.0 to 1.0) based on:
- Are action-object combinations physically plausible?
- Are the described interactions reasonable?
- Are tools/instruments appropriate for the actions?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    ROBOTIC_USEFULNESS_TEMPLATE = """Evaluate the ROBOTIC USEFULNESS for robot learning.

Video ID: {video_id}

Annotation:
{annotation_text}

Rate robotic usefulness (0.0 to 1.0) based on:
- Does it describe manipulable actions?
- Are there clear action-object relationships?
- Would this help a robot learn manipulation skills?
- Are physical properties (force, trajectory) described?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    LANGUAGE_CLARITY_TEMPLATE = """Evaluate the LANGUAGE CLARITY.

Video ID: {video_id}

Captions:
{captions_text}

Rate language clarity (0.0 to 1.0) based on:
- Is the language clear and grammatical?
- Are descriptions concise yet complete?
- Is the vocabulary appropriate?
- Is the style consistent?

Respond in JSON format:
{{
    "score": float,
    "confidence": float,
    "reasoning": string,
    "issues": [string]
}}"""

    COMPREHENSIVE_TEMPLATE = """Evaluate the OVERALL QUALITY of this video annotation.

Video ID: {video_id}

Complete Annotation:
{annotation_text}

Provide scores for ALL dimensions (0.0 to 1.0):
{{
    "factuality": {{"score": float, "confidence": float, "issues": [string]}},
    "temporal_consistency": {{"score": float, "confidence": float, "issues": [string]}},
    "object_grounding": {{"score": float, "confidence": float, "issues": [string]}},
    "action_grounding": {{"score": float, "confidence": float, "issues": [string]}},
    "affordance": {{"score": float, "confidence": float, "issues": [string]}},
    "robotic_usefulness": {{"score": float, "confidence": float, "issues": [string]}},
    "language_clarity": {{"score": float, "confidence": float, "issues": [string]}},
    "overall": {{"score": float, "confidence": float}}
}}"""

    @classmethod
    def format_annotation(cls, annotation: Annotation) -> str:
        """Format annotation for prompt."""
        lines = [f"Annotation ID: {annotation.id}", f"Source: {annotation.source}"]

        for i, segment in enumerate(annotation.segments):
            lines.append(
                f"\nSegment {i + 1} ({segment.start_time:.1f}s - {segment.end_time:.1f}s):"
            )
            lines.append(f"  Caption: {segment.caption}")

            if segment.actions:
                lines.append("  Actions:")
                for action in segment.actions:
                    action_str = f"    - {action.verb} {action.noun}"
                    if action.hand != "unknown":
                        action_str += f" ({action.hand} hand)"
                    if action.instrument:
                        action_str += f" using {action.instrument}"
                    lines.append(action_str)

            if segment.objects:
                lines.append("  Objects:")
                for obj in segment.objects:
                    obj_str = f"    - {obj.name}"
                    if obj.state:
                        obj_str += f" (state: {obj.state})"
                    lines.append(obj_str)

        return "\n".join(lines)

    @classmethod
    def format_segments(cls, segments: List[Segment]) -> str:
        """Format segment times for prompt."""
        lines = []
        for i, seg in enumerate(segments):
            duration = seg.end_time - seg.start_time
            lines.append(
                f"Segment {i + 1}: {seg.start_time:.1f}s - {seg.end_time:.1f}s (duration: {duration:.1f}s)"
            )
        return "\n".join(lines)

    @classmethod
    def format_objects(cls, segments: List[Segment]) -> str:
        """Format objects for prompt."""
        objects_by_segment = []
        for i, seg in enumerate(segments):
            if seg.objects:
                obj_strs = []
                for obj in seg.objects:
                    s = obj.name
                    if obj.state:
                        s += f" [{obj.state}]"
                    if obj.material:
                        s += f" ({obj.material})"
                    obj_strs.append(s)
                objects_by_segment.append(f"Segment {i + 1}: {', '.join(obj_strs)}")
        return "\n".join(objects_by_segment) or "No objects specified"

    @classmethod
    def format_actions(cls, segments: List[Segment]) -> str:
        """Format actions for prompt."""
        actions_by_segment = []
        for i, seg in enumerate(segments):
            if seg.actions:
                action_strs = []
                for action in seg.actions:
                    s = f"{action.verb} {action.noun}"
                    if action.start_time is not None and action.end_time is not None:
                        s += f" [{action.start_time:.1f}s - {action.end_time:.1f}s]"
                    action_strs.append(s)
                actions_by_segment.append(f"Segment {i + 1}: {', '.join(action_strs)}")
        return "\n".join(actions_by_segment) or "No actions specified"

    @classmethod
    def format_affordances(cls, segments: List[Segment]) -> str:
        """Format action-object pairs for affordance evaluation."""
        pairs = []
        for i, seg in enumerate(segments):
            if seg.actions:
                for action in seg.actions:
                    obj_list = (
                        [obj.name for obj in seg.objects] if seg.objects else ["unknown object"]
                    )
                    s = f"Segment {i + 1}: {action.verb} {action.noun}"
                    if action.instrument:
                        s += f" with {action.instrument}"
                    s += f" (objects present: {', '.join(obj_list)})"
                    pairs.append(s)
        return "\n".join(pairs) or "No action-object pairs"

    @classmethod
    def format_captions(cls, segments: List[Segment]) -> str:
        """Format captions for prompt."""
        return "\n".join(f"Segment {i + 1}: {seg.caption}" for i, seg in enumerate(segments))


class LLMJudgePipeline:
    """Pipeline for using LLM as a judge for quality evaluation."""

    def __init__(
        self,
        config: Optional[LLMJudgeConfig] = None,
        teacher_model=None,
    ):
        """Initialize LLM judge pipeline.

        Args:
            config: Judge configuration
            teacher_model: Optional teacher model instance to use
        """
        self.config = config or LLMJudgeConfig()
        self._teacher_model = teacher_model
        self._prompts = LLMJudgePrompts()

    async def evaluate(
        self,
        annotation: Annotation,
        dimensions: Optional[List[QualityDimension]] = None,
    ) -> QualityScores:
        """Evaluate annotation quality using LLM.

        Args:
            annotation: Annotation to evaluate
            dimensions: Specific dimensions to evaluate (None = all)

        Returns:
            QualityScores from LLM evaluation
        """
        if dimensions is None:
            dimensions = [
                QualityDimension.FACTUALITY,
                QualityDimension.TEMPORAL_CONSISTENCY,
                QualityDimension.OBJECT_GROUNDING,
                QualityDimension.ACTION_GROUNDING,
                QualityDimension.AFFORDANCE,
                QualityDimension.ROBOTIC_USEFULNESS,
                QualityDimension.LANGUAGE_CLARITY,
            ]

        scores = QualityScores(
            annotation_id=annotation.id,
            video_id=annotation.video_id,
            computed_by="llm_judge",
        )

        # Use comprehensive evaluation for efficiency
        if len(dimensions) > 3:
            comprehensive_result = await self._evaluate_comprehensive(annotation)
            if comprehensive_result:
                scores = comprehensive_result
            else:
                # Fall back to individual evaluation
                for dimension in dimensions:
                    dim_score = await self._evaluate_dimension(annotation, dimension)
                    self._set_dimension_score(scores, dimension, dim_score)
        else:
            # Evaluate specific dimensions
            for dimension in dimensions:
                dim_score = await self._evaluate_dimension(annotation, dimension)
                self._set_dimension_score(scores, dimension, dim_score)

        # Recompute aggregates
        scores._compute_aggregates()

        logger.info(
            "llm_judge_evaluation_complete",
            annotation_id=annotation.id,
            overall_score=scores.overall_score,
            dimensions_evaluated=len(dimensions),
        )

        return scores

    async def evaluate_batch(
        self,
        annotations: List[Annotation],
    ) -> Dict[str, QualityScores]:
        """Evaluate a batch of annotations.

        Args:
            annotations: List of annotations to evaluate

        Returns:
            Dict mapping annotation_id to QualityScores
        """
        results = {}

        for annotation in annotations:
            try:
                scores = await self.evaluate(annotation)
                results[annotation.id] = scores
            except Exception as e:
                logger.error(
                    "llm_judge_evaluation_failed",
                    annotation_id=annotation.id,
                    error=str(e),
                )
                # Create empty scores on failure
                results[annotation.id] = QualityScores(
                    annotation_id=annotation.id,
                    video_id=annotation.video_id,
                    computed_by="llm_judge",
                )

        return results

    async def _evaluate_dimension(
        self,
        annotation: Annotation,
        dimension: QualityDimension,
    ) -> DimensionScore:
        """Evaluate a single dimension."""
        # Get appropriate prompt
        prompt = self._get_prompt_for_dimension(annotation, dimension)

        # Call LLM
        result = await self._call_llm(prompt)

        # Parse result
        return self._parse_dimension_result(result, dimension)

    async def _evaluate_comprehensive(
        self,
        annotation: Annotation,
    ) -> Optional[QualityScores]:
        """Evaluate all dimensions in one call."""
        annotation_text = self._prompts.format_annotation(annotation)

        prompt = self._prompts.COMPREHENSIVE_TEMPLATE.format(
            video_id=annotation.video_id,
            annotation_text=annotation_text,
        )

        result = await self._call_llm(prompt)

        if not result or not result.text:
            return None

        try:
            parsed = self._extract_json(result.text)
            if not parsed:
                return None

            scores = QualityScores(
                annotation_id=annotation.id,
                video_id=annotation.video_id,
                computed_by="llm_judge",
            )

            # Map parsed results to dimensions
            dimension_map = {
                "factuality": QualityDimension.FACTUALITY,
                "temporal_consistency": QualityDimension.TEMPORAL_CONSISTENCY,
                "object_grounding": QualityDimension.OBJECT_GROUNDING,
                "action_grounding": QualityDimension.ACTION_GROUNDING,
                "affordance": QualityDimension.AFFORDANCE,
                "robotic_usefulness": QualityDimension.ROBOTIC_USEFULNESS,
                "language_clarity": QualityDimension.LANGUAGE_CLARITY,
            }

            for key, dim in dimension_map.items():
                if key in parsed:
                    dim_data = parsed[key]
                    score = DimensionScore(
                        dimension=dim,
                        score=float(dim_data.get("score", 0.5)),
                        confidence=float(dim_data.get("confidence", 0.7)),
                        issues=dim_data.get("issues", []),
                    )
                    self._set_dimension_score(scores, dim, score)

            # Set overall if provided
            if "overall" in parsed:
                scores.overall_score = float(parsed["overall"].get("score", 0.5))

            return scores

        except Exception as e:
            logger.error("comprehensive_evaluation_parse_failed", error=str(e))
            return None

    def _get_prompt_for_dimension(
        self,
        annotation: Annotation,
        dimension: QualityDimension,
    ) -> str:
        """Get the appropriate prompt for a dimension."""
        video_id = annotation.video_id

        if dimension == QualityDimension.FACTUALITY:
            return self._prompts.FACTUALITY_TEMPLATE.format(
                video_id=video_id,
                annotation_text=self._prompts.format_annotation(annotation),
            )
        elif dimension == QualityDimension.TEMPORAL_CONSISTENCY:
            return self._prompts.TEMPORAL_CONSISTENCY_TEMPLATE.format(
                video_id=video_id,
                segment_times=self._prompts.format_segments(annotation.segments),
            )
        elif dimension == QualityDimension.OBJECT_GROUNDING:
            return self._prompts.OBJECT_GROUNDING_TEMPLATE.format(
                video_id=video_id,
                objects_text=self._prompts.format_objects(annotation.segments),
            )
        elif dimension == QualityDimension.ACTION_GROUNDING:
            return self._prompts.ACTION_GROUNDING_TEMPLATE.format(
                video_id=video_id,
                actions_text=self._prompts.format_actions(annotation.segments),
            )
        elif dimension == QualityDimension.AFFORDANCE:
            return self._prompts.AFFORDANCE_TEMPLATE.format(
                video_id=video_id,
                affordance_text=self._prompts.format_affordances(annotation.segments),
            )
        elif dimension == QualityDimension.ROBOTIC_USEFULNESS:
            return self._prompts.ROBOTIC_USEFULNESS_TEMPLATE.format(
                video_id=video_id,
                annotation_text=self._prompts.format_annotation(annotation),
            )
        elif dimension == QualityDimension.LANGUAGE_CLARITY:
            return self._prompts.LANGUAGE_CLARITY_TEMPLATE.format(
                video_id=video_id,
                captions_text=self._prompts.format_captions(annotation.segments),
            )
        else:
            return self._prompts.FACTUALITY_TEMPLATE.format(
                video_id=video_id,
                annotation_text=self._prompts.format_annotation(annotation),
            )

    async def _call_llm(self, prompt: str) -> Optional[GenerationResult]:
        """Call the LLM with the prompt."""
        if self._teacher_model is None:
            # Lazy import to avoid circular dependency
            from dvas.models.teacher.base import TeacherModel

            self._teacher_model = TeacherModel(model_name=self.config.model_name)

        try:
            result = await self._teacher_model.generate(
                prompt=prompt,
                task="quality_evaluation",
                temperature=self.config.temperature,
            )
            return result
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            return None

    def _parse_dimension_result(
        self,
        result: Optional[GenerationResult],
        dimension: QualityDimension,
    ) -> DimensionScore:
        """Parse LLM result into DimensionScore."""
        if result is None or not result.text:
            return DimensionScore(
                dimension=dimension,
                score=0.5,
                confidence=0.0,
                issues=["llm_evaluation_failed"],
            )

        try:
            parsed = self._extract_json(result.text)
            if not parsed:
                return DimensionScore(
                    dimension=dimension,
                    score=0.5,
                    confidence=0.3,
                    issues=["failed_to_parse_llm_response"],
                )

            return DimensionScore(
                dimension=dimension,
                score=float(parsed.get("score", 0.5)),
                confidence=float(parsed.get("confidence", 0.7)),
                details={"reasoning": parsed.get("reasoning", "")},
                issues=parsed.get("issues", []),
            )
        except Exception as e:
            logger.error("parse_dimension_result_failed", error=str(e))
            return DimensionScore(
                dimension=dimension,
                score=0.5,
                confidence=0.3,
                issues=["parse_error"],
            )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text."""
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("json_decode_failed", text=text[:200])
            return None

    def _set_dimension_score(
        self,
        scores: QualityScores,
        dimension: QualityDimension,
        score: DimensionScore,
    ) -> None:
        """Set a dimension score on QualityScores."""
        if dimension == QualityDimension.FACTUALITY:
            scores.factuality_score = score
        elif dimension == QualityDimension.TEMPORAL_CONSISTENCY:
            scores.temporal_consistency_score = score
        elif dimension == QualityDimension.OBJECT_GROUNDING:
            scores.object_grounding_score = score
        elif dimension == QualityDimension.ACTION_GROUNDING:
            scores.action_grounding_score = score
        elif dimension == QualityDimension.AFFORDANCE:
            scores.affordance_score = score
        elif dimension == QualityDimension.ROBOTIC_USEFULNESS:
            scores.robotic_usefulness_score = score
        elif dimension == QualityDimension.LANGUAGE_CLARITY:
            scores.language_clarity_score = score
        elif dimension == QualityDimension.PARSE_CONFIDENCE:
            scores.parse_confidence_score = score
        elif dimension == QualityDimension.REVIEWER_CONFIDENCE:
            scores.reviewer_confidence_score = score
