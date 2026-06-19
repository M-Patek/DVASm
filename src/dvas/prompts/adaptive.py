"""Adaptive prompt engineering based on video characteristics."""

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dvas.data.video_loader import VideoLoader
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class VideoCategory(str, Enum):
    """Video content categories."""

    KITCHEN_COOKING = "kitchen_cooking"
    ROBOT_MANIPULATION = "robot_manipulation"
    ASSEMBLY = "assembly"
    MEDICAL = "medical"
    SPORTS = "sports"
    GENERAL = "general"


class ComplexityLevel(str, Enum):
    """Video complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PromptTemplate:
    """Prompt template with metadata."""

    name: str
    template: str
    category: VideoCategory
    complexity: ComplexityLevel
    focus_areas: List[str]
    example_count: int = 0
    avg_quality_score: float = 0.0


class PromptLibrary:
    """Library of specialized prompts for different video types."""

    TEMPLATES = {
        # Kitchen cooking templates
        "kitchen_detailed": PromptTemplate(
            name="kitchen_detailed",
            category=VideoCategory.KITCHEN_COOKING,
            complexity=ComplexityLevel.COMPLEX,
            focus_areas=["hand_actions", "ingredients", "tools", "sequence"],
            template="""Analyze this first-person cooking video in detail.

Focus on:
1. HAND ACTIONS: Which hand (left/right/both) performs each action, grip type, movement trajectory
2. INGREDIENTS: Identify all ingredients, their states (raw/cooked/chopped), and quantities
3. TOOLS: List all kitchen tools used and how they are manipulated
4. SEQUENCE: Describe the exact temporal order of actions
5. OBJECT RELATIONS: How hands interact with objects, object-object interactions

Structure your response as:
- Scene overview (1-2 sentences)
- Step-by-step breakdown with timestamps
- List of key objects and their states
- Hand usage patterns

Caption: """,
        ),
        "kitchen_quick": PromptTemplate(
            name="kitchen_quick",
            category=VideoCategory.KITCHEN_COOKING,
            complexity=ComplexityLevel.SIMPLE,
            focus_areas=["main_action", "primary_objects"],
            template="""Briefly describe the main cooking activity in this video.

Mention:
- What is being prepared
- Main steps (3-5 bullet points)
- Primary ingredients and tools

Caption: """,
        ),
        # Robot manipulation templates
        "robot_fine_grained": PromptTemplate(
            name="robot_fine_grained",
            category=VideoCategory.ROBOT_MANIPULATION,
            complexity=ComplexityLevel.COMPLEX,
            focus_areas=["hand_trajectory", "grasp", "force", "precision"],
            template="""Provide fine-grained analysis of this robotic manipulation video.

Required details:
1. HAND TRAJECTORY: Precise path of hands/fingers in 3D space
2. GRASP TYPES: Pinch, power, tripod, hook, etc.
3. FORCE MODULATION: Gentle vs firm grasps, pressure changes
4. PRECISION: Exact finger positioning, contact points
5. FAILURES: Any slips, adjustments, or repositioning
6. OBJECT PHYSICS: How objects deform, move, or respond

Use this JSON structure:
{{
  "scene": "brief description",
  "actions": [{{"time": "X-Ys", "hand": "L/R/B", "action": "...", "grasp": "..."}}],
  "objects": [{{"name": "...", "state": "...", "interactions": []}}]
}}

Caption: """,
        ),
        # General templates
        "general_detailed": PromptTemplate(
            name="general_detailed",
            category=VideoCategory.GENERAL,
            complexity=ComplexityLevel.MODERATE,
            focus_areas=["description", "actions", "objects"],
            template="""Describe this video comprehensively.

Include:
- Overall scene and setting
- Temporal sequence: What happens first, then next
- Actions: Who/what is doing what, with which body parts
- Objects: Key objects involved and their relationships
- Spatial relationships: Positions, movements, distances

Caption: """,
        ),
        "qa_generation": PromptTemplate(
            name="qa_generation",
            category=VideoCategory.GENERAL,
            complexity=ComplexityLevel.MODERATE,
            focus_areas=["questions", "answers", "grounding"],
            template="""Based on this video, generate 3-5 question-answer pairs.

Each QA pair should:
- Be answerable from the video content
- Cover different aspects (what, how, why, when)
- Include temporal grounding where relevant
- Be suitable for training a video understanding model

Format: Q: [question]\nA: [answer]

Caption: """,
        ),
    }

    def get_template(self, name: str) -> PromptTemplate:
        """Get template by name."""
        if name not in self.TEMPLATES:
            return self.TEMPLATES["general_detailed"]
        return self.TEMPLATES[name]

    def get_by_category(
        self, category: VideoCategory, complexity: Optional[ComplexityLevel] = None
    ) -> List[PromptTemplate]:
        """Get templates for category."""
        templates = [t for t in self.TEMPLATES.values() if t.category == category]

        if complexity:
            templates = [t for t in templates if t.complexity == complexity]

        return templates


class VideoTypeClassifier:
    """Classify video type for prompt selection."""

    KEYWORDS = {
        VideoCategory.KITCHEN_COOKING: [
            "cook",
            "kitchen",
            "cut",
            "chop",
            "food",
            "ingredient",
            "recipe",
            "pan",
            "oven",
            "knife",
        ],
        VideoCategory.ROBOT_MANIPULATION: [
            "grasp",
            "pick",
            "place",
            "manipulate",
            "hand",
            "finger",
            "grip",
            "assembly",
        ],
        VideoCategory.ASSEMBLY: [
            "assemble",
            "build",
            "construct",
            "screw",
            "bolt",
            "part",
            "component",
        ],
        VideoCategory.MEDICAL: [
            "surgery",
            "medical",
            "patient",
            "instrument",
            "operation",
            "tissue",
        ],
    }

    def classify(self, video_path: Path) -> VideoCategory:
        """Classify video based on visual features and metadata."""
        # Analyze first few seconds
        with VideoLoader(video_path) as loader:
            frames = []
            for frame in loader.read_frames(end_time=2.0, num_frames=5):
                # Simple color analysis for scene type
                avg_color = frame.data.mean(axis=(0, 1))
                frames.append(avg_color)

            # Motion analysis
            motion = loader.compute_motion_score(end_time=2.0, sample_frames=5)

        # Simple heuristic classification
        # In production, use actual object detection and scene classification

        # Check filename for clues
        filename = video_path.name.lower()
        for category, keywords in self.KEYWORDS.items():
            if any(kw in filename for kw in keywords):
                return category

        # Default based on motion
        if motion > 0.5:
            return VideoCategory.GENERAL

        return VideoCategory.GENERAL


class AdaptivePromptEngine:
    """Generate adaptive prompts based on video characteristics."""

    def __init__(self):
        self.library = PromptLibrary()
        self.classifier = VideoTypeClassifier()
        self.prompt_history: Dict[str, List[Dict]] = {}

    def generate_prompt(
        self,
        video_path: Path,
        task_type: str = "caption",
        preferred_complexity: Optional[ComplexityLevel] = None,
    ) -> str:
        """Generate optimal prompt for video."""
        # Classify video
        category = self.classifier.classify(video_path)

        # Determine complexity if not specified
        if not preferred_complexity:
            preferred_complexity = self._estimate_complexity(video_path)

        # Select template
        templates = self.library.get_by_category(category, preferred_complexity)

        if not templates:
            templates = [self.library.get_template("general_detailed")]

        # Choose best performing template or random
        template = self._select_best_template(templates, video_path)

        # Add few-shot examples if available
        prompt = self._augment_with_examples(template.template, category)

        logger.info(
            "prompt_generated",
            video=str(video_path),
            category=category.value,
            complexity=preferred_complexity.value,
            template=template.name,
        )

        return prompt

    def _estimate_complexity(self, video_path: Path) -> ComplexityLevel:
        """Estimate video complexity."""
        with VideoLoader(video_path) as loader:
            metadata = loader.metadata
            scenes = loader.detect_scenes(max_scenes=10)

            scene_count = len(scenes)
            duration = metadata.duration

            # Simple complexity heuristic
            if scene_count > 5 or duration > 30:
                return ComplexityLevel.COMPLEX
            elif scene_count > 2 or duration > 10:
                return ComplexityLevel.MODERATE
            else:
                return ComplexityLevel.SIMPLE

    def _select_best_template(
        self, templates: List[PromptTemplate], video_path: str
    ) -> PromptTemplate:
        """Select best template based on historical performance."""
        if not templates:
            raise ValueError("No templates available")

        # Sort by average quality score
        scored_templates = [(t, t.avg_quality_score) for t in templates]
        scored_templates.sort(key=lambda x: x[1], reverse=True)

        # Use top performer 80% of time, explore 20%
        if random.random() < 0.8 or len(scored_templates) == 1:
            return scored_templates[0][0]
        else:
            # Random exploration
            return random.choice(templates)

    def _augment_with_examples(self, template: str, category: VideoCategory) -> str:
        """Add few-shot examples to prompt."""
        # Get examples for category
        examples = self._get_examples(category)

        if not examples:
            return template

        # Add examples before main prompt
        example_text = "\n\nExamples:\n"
        for i, ex in enumerate(examples[:2], 1):
            example_text += f"{i}. {ex}\n"

        # Insert before "Caption:"
        if "Caption:" in template:
            return template.replace("Caption:", example_text + "\nCaption:")

        return template + example_text

    def _get_examples(self, category: VideoCategory) -> List[str]:
        """Get example outputs for category."""
        # Placeholder: In production, retrieve from high-quality examples database
        examples = {
            VideoCategory.KITCHEN_COOKING: [
                "Person uses right hand to grip knife, left hand to steady onion on cutting board...",
                "Left hand holds tomato, right hand rotates it while knife makes precise cuts...",
            ],
            VideoCategory.ROBOT_MANIPULATION: [
                "Grasp: Tripod grasp on small screw. Approach from 45-degree angle...",
                "Hand trajectory: Arcing motion from left to right, maintaining constant height...",
            ],
        }
        return examples.get(category, [])

    def feedback_loop(
        self,
        template_name: str,
        video_path: str,
        quality_score: float,
    ) -> None:
        """Update template performance based on feedback."""
        template = self.library.get_template(template_name)

        # Update moving average
        n = template.example_count
        template.avg_quality_score = (template.avg_quality_score * n + quality_score) / (n + 1)
        template.example_count += 1

        logger.info(
            "prompt_feedback_recorded",
            template=template_name,
            score=quality_score,
            new_avg=template.avg_quality_score,
        )


class DynamicPromptOptimizer:
    """Optimize prompts based on evaluation metrics."""

    def __init__(self):
        self.performance_log: List[Dict] = []

    def record_performance(
        self,
        prompt: str,
        video_category: VideoCategory,
        metrics: Dict[str, float],
    ) -> None:
        """Record prompt performance."""
        self.performance_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_hash": hash(prompt) % 10000,
                "category": video_category.value,
                "metrics": metrics,
            }
        )

    def suggest_improvements(self) -> List[Dict]:
        """Suggest prompt improvements based on data."""
        if len(self.performance_log) < 10:
            return []

        # Analyze low-performing configurations
        suggestions = []

        # Group by category
        by_category = {}
        for entry in self.performance_log:
            cat = entry["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(entry["metrics"].get("quality", 0))

        # Find underperforming categories
        for cat, scores in by_category.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.7:
                suggestions.append(
                    {
                        "category": cat,
                        "current_avg_score": avg_score,
                        "suggestion": f"Consider adding more specific guidance for {cat} videos",
                        "priority": "high" if avg_score < 0.5 else "medium",
                    }
                )

        return suggestions
