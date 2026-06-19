"""LLM-as-a-Judge for annotation quality assessment."""

import asyncio
from typing import Any, Dict, List, Optional

from dvas.models.base import GenerationResult
from dvas.models.teacher import TeacherModel


class LLMJudge:
    """
    Use a language model to judge annotation quality.
    This provides semantic evaluation beyond n-gram metrics.
    """

    # Quality dimensions to evaluate
    DIMENSIONS = [
        "accuracy",  # Factual correctness
        "completeness",  # Coverage of important aspects
        "clarity",  # Clear and understandable
        "relevance",  # Relevant to the video content
        "structure",  # Well-organized and logical
    ]

    @staticmethod
    def _result_text(result: GenerationResult) -> str:
        """Return text from the standardized teacher response."""
        return result.text

    def __init__(
        self,
        judge_model: Optional[TeacherModel] = None,
    ):
        self.judge = judge_model or TeacherModel(model_name="gpt-5.5")

    async def evaluate_segment(
        self,
        annotation_text: str,
        reference_text: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single annotation segment.

        Args:
            annotation_text: The generated annotation
            reference_text: Optional reference (gold) annotation
            dimensions: Quality dimensions to evaluate

        Returns:
            Dict with scores and feedback
        """
        dims = dimensions or self.DIMENSIONS

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            annotation_text=annotation_text,
            reference_text=reference_text,
            dimensions=dims,
        )

        # Call judge model
        result = await self.judge.annotate(
            frames=[],  # No video needed for text-only evaluation
            prompt=prompt,
            temperature=0.0,  # Deterministic for consistency
        )

        # Parse result. TeacherModel.annotate() returns GenerationResult, not a dict.
        return self._parse_evaluation(self._result_text(result))

    async def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple annotations concurrently.

        Args:
            items: List of dicts with 'annotation' and optional 'reference'
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_one(item: Dict) -> Dict[str, Any]:
            async with semaphore:
                result = await self.evaluate_segment(
                    annotation_text=item["annotation"],
                    reference_text=item.get("reference"),
                )
                result["id"] = item.get("id", "unknown")
                return result

        tasks = [evaluate_one(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _build_evaluation_prompt(
        self,
        annotation_text: str,
        reference_text: Optional[str],
        dimensions: List[str],
    ) -> str:
        """Build the evaluation prompt."""
        prompt = """You are an expert evaluator of video annotation quality.

Evaluate the following annotation based on these dimensions:
"""

        for dim in dimensions:
            desc = self._get_dimension_description(dim)
            prompt += f"\n- **{dim.capitalize()}**: {desc}"

        prompt += f"""

---

**Annotation to Evaluate**:
{annotation_text}
"""

        if reference_text:
            prompt += f"""

**Reference (Gold Standard)**:
{reference_text}

Compare the annotation to the reference when scoring.
"""

        prompt += """

---

**Output Format** (respond only in this format):

```
Overall Score: X/10

Dimension Scores:
- accuracy: X/10
- completeness: X/10
- clarity: X/10
- relevance: X/10
- structure: X/10

Justification:
[2-3 sentences explaining the main strengths and weaknesses]

Suggestions for Improvement:
- [specific suggestion 1]
- [specific suggestion 2]
```
"""
        return prompt

    def _get_dimension_description(self, dimension: str) -> str:
        """Get description for a quality dimension."""
        descriptions = {
            "accuracy": "Factual correctness and alignment with visual content",
            "completeness": "Coverage of all important actions, objects, and events",
            "clarity": "Clear, understandable language without ambiguity",
            "relevance": "Information directly related to the video content",
            "structure": "Logical organization and flow of information",
        }
        return descriptions.get(dimension, "Quality of this aspect")

    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """Parse the evaluation response."""
        result = {
            "overall_score": None,
            "dimension_scores": {},
            "justification": "",
            "suggestions": [],
            "raw_response": text,
        }

        # Extract overall score
        overall_match = __import__("re").search(r"Overall Score:\s*(\d+(?:\.\d+)?)/10", text)
        if overall_match:
            result["overall_score"] = float(overall_match.group(1))

        # Extract dimension scores
        for dim in self.DIMENSIONS:
            pattern = rf"- {dim}:\s*(\d+(?:\.\d+)?)/10"
            match = __import__("re").search(pattern, text, __import__("re").IGNORECASE)
            if match:
                result["dimension_scores"][dim] = float(match.group(1))

        # Extract justification
        just_match = __import__("re").search(
            r"Justification:\s*(.+?)(?=Suggestions|$)", text, __import__("re").DOTALL
        )
        if just_match:
            result["justification"] = just_match.group(1).strip()

        # Extract suggestions
        sugg_section = __import__("re").search(
            r"Suggestions for Improvement:(.+?)(?=```|$)", text, __import__("re").DOTALL
        )
        if sugg_section:
            suggestions = __import__("re").findall(r"-\s*(.+)", sugg_section.group(1))
            result["suggestions"] = [s.strip() for s in suggestions]

        return result


class ConsistencyChecker:
    """Check temporal and semantic consistency across segments."""

    def __init__(self):
        self.metrics = __import__(
            "dvas.models.evaluator.metrics", fromlist=["MetricsCalculator"]
        ).MetricsCalculator()

    def check_temporal_consistency(
        self,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if adjacent segments have consistent narratives.

        Args:
            segments: List of segment dicts with 'caption' and timestamps

        Returns:
            Consistency report
        """
        if len(segments) < 2:
            return {"consistent": True, "score": 1.0, "issues": []}

        issues = []
        similarities = []

        for i in range(len(segments) - 1):
            curr = segments[i]
            next_seg = segments[i + 1]

            # Check for temporal overlap
            if curr.get("end_time", 0) > next_seg.get("start_time", float("inf")):
                issues.append(f"Temporal overlap between segments {i} and {i + 1}")

            # Check semantic continuity
            curr_caption = curr.get("caption", "")
            next_caption = next_seg.get("caption", "")

            if curr_caption and next_caption:
                rouge_scores = self.metrics.rouge(curr_caption, next_caption)
                similarities.append(rouge_scores["rougeL_f"])

        # Average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

        # Consistency score (lower similarity between segments is expected,
        # but very high similarity might indicate redundancy)
        consistency_score = 1.0 - abs(avg_similarity - 0.3)  # Optimal around 0.3

        return {
            "consistent": len(issues) == 0,
            "score": max(0.0, consistency_score),
            "avg_segment_similarity": avg_similarity,
            "issues": issues,
        }

    def check_action_consistency(
        self,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if actions across segments form a coherent sequence.

        Args:
            segments: List of segments with 'actions' field

        Returns:
            Action consistency report
        """
        all_actions = []
        for seg in segments:
            actions = seg.get("actions", [])
            all_actions.extend(actions)

        if not all_actions:
            return {"consistent": True, "actions": [], "issues": []}

        # Check for repeated actions (might indicate redundancy)
        action_counts = {}
        for action in all_actions:
            key = f"{action.get('verb', '')}_{action.get('noun', '')}"
            action_counts[key] = action_counts.get(key, 0) + 1

        issues = []
        for action, count in action_counts.items():
            if count > 3:  # Same action repeated many times
                issues.append(f"Action '{action}' repeated {count} times")

        return {
            "consistent": len(issues) == 0,
            "actions": list(action_counts.keys()),
            "action_counts": action_counts,
            "issues": issues,
        }
