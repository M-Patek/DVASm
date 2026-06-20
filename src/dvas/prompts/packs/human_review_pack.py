"""Human review and quality assessment prompt packs.

Provides specialized prompts for human reviewers to assess annotation
quality, provide feedback, and validate model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.prompts.registry import PromptDomain, PromptTemplate
from dvas.quality.schema import QualityDimension
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HumanReviewPromptPack:
    """Prompt pack for human review and quality assessment."""

    name: str = "human_review"
    domain: PromptDomain = PromptDomain.HUMAN_REVIEW

    TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "review_overall_quality": """Review the following video annotation for overall quality.

Video: {video_description}
Annotation: {annotation_text}

Please assess:
1. ACCURACY (1-5): Does the annotation accurately describe the video?
2. COMPLETENESS (1-5): Does it cover all important aspects?
3. CLARITY (1-5): Is the language clear and unambiguous?
4. TEMPORAL_CORRECTNESS (1-5): Are timestamps and sequences correct?
5. OBJECT_GROUNDING (1-5): Are objects correctly identified and described?

For each dimension, provide:
- Score (1-5)
- Brief justification
- Specific issues (if any)

Overall score: {overall_score}/5

Reviewer notes: """,

        "review_factuality": """Fact-check this annotation against the video.

Annotation: {annotation_text}

Identify any factual errors:
1. FALSE CLAIMS: Statements that contradict the video
2. HALLUCINATIONS: Details not present in the video
3. MISIDENTIFICATIONS: Wrong object/action labels
4. TEMPORAL_ERRORS: Events described in wrong order
5. QUANTITATIVE_ERRORS: Wrong counts, sizes, durations

For each error found:
- Quote the incorrect statement
- Explain why it's wrong
- Suggest correction

Fact-check result: """,

        "review_vla_specific": """Review this annotation for VLA/Robot training suitability.

Annotation: {annotation_text}

Assess for robot action learning:
1. ACTION_GRANULARITY: Are actions described at the right level of detail?
2. GRASP_DESCRIPTIONS: Are grasp types specified?
3. SPATIAL_PRECISION: Are positions and trajectories clear?
4. FORCE_MODULATION: Is force/pressure described?
5. EXECUTABILITY: Could a robot execute from this description?

Score each from 1-5 and provide actionable feedback.

VLA review: """,

        "review_comparison": """Compare two annotations of the same video.

Annotation A: {annotation_a}

Annotation B: {annotation_b}

Compare them on:
1. ACCURACY: Which is more accurate?
2. DETAIL: Which has more useful detail?
3. CLARITY: Which is clearer?
4. COMPLETENESS: Which is more complete?
5. ROBOTIC_USEFULNESS: Which is more useful for robot training?

Declare a winner (A, B, or tie) for each dimension with justification.

Comparison result: """,

        "review_correction": """Provide a corrected version of this annotation.

Original annotation: {annotation_text}

Known issues: {known_issues}

Please provide a corrected annotation that:
1. Fixes all identified errors
2. Maintains correct information from the original
3. Adds missing details where appropriate
4. Uses clear, precise language
5. Follows the annotation standard format

Corrected annotation: """,

        "review_consensus": """Evaluate consensus among multiple reviewers.

Review 1: {review_1}
Review 2: {review_2}
Review 3: {review_3}

Determine:
1. AGREEMENT_LEVEL: High/medium/low agreement on scores
2. CONSENSUS_SCORE: Average or median of dimension scores
3. DISSENT_AREAS: Where reviewers disagree significantly
4. CONFIDENCE: How confident should we be in the consensus?

Consensus evaluation: """,
    })

    def get_template(self, name: str) -> Optional[str]:
        """Get a template by name."""
        return self.TEMPLATES.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.TEMPLATES.keys())

    def create_prompt_template(
        self,
        template_name: str,
        custom_variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptTemplate]:
        """Create a PromptTemplate from a named template.

        Args:
            template_name: Name of the template to use.
            custom_variables: Optional custom variables to substitute.

        Returns:
            PromptTemplate or None if name not found.
        """
        template_text = self.get_template(template_name)
        if template_text is None:
            return None

        if custom_variables:
            for key, value in custom_variables.items():
                template_text = template_text.replace(f"{{{key}}}", value)

        from dvas.prompts.registry import PromptMetadata
        return PromptTemplate(
            id=f"hr_{template_name}",
            metadata=PromptMetadata(
                name=template_name,
                version="1.0.0",
                domain=self.domain,
                description=f"Human review prompt: {template_name}",
                tags=["human_review", "quality", template_name],
            ),
            template=template_text,
            variables=list(custom_variables.keys()) if custom_variables else [],
        )

    def get_quality_review_prompt(
        self,
        annotation_text: str,
        video_description: str = "",
        dimensions: Optional[List[QualityDimension]] = None,
    ) -> str:
        """Get a quality review prompt with specific dimensions.

        Args:
            annotation_text: The annotation to review.
            video_description: Optional video description.
            dimensions: Specific quality dimensions to assess.

        Returns:
            Formatted review prompt.
        """
        base = self.TEMPLATES["review_overall_quality"]
        base = base.replace("{annotation_text}", annotation_text)
        base = base.replace("{video_description}", video_description)

        if dimensions:
            dim_text = "\n".join(f"- {d.value}" for d in dimensions)
            base += f"\n\nFocus on these dimensions:\n{dim_text}"

        return base

    def get_comparison_prompt(
        self,
        annotation_a: str,
        annotation_b: str,
        criteria: Optional[List[str]] = None,
    ) -> str:
        """Get a comparison prompt for two annotations.

        Args:
            annotation_a: First annotation.
            annotation_b: Second annotation.
            criteria: Optional custom comparison criteria.

        Returns:
            Formatted comparison prompt.
        """
        base = self.TEMPLATES["review_comparison"]
        base = base.replace("{annotation_a}", annotation_a)
        base = base.replace("{annotation_b}", annotation_b)

        if criteria:
            criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
            base = base.replace(
                "1. ACCURACY: Which is more accurate?",
                criteria_text,
            )

        return base
