"""Smart Router - Adaptive model selection based on video complexity."""

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dvas.data.video_loader import VideoLoader
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Available model types."""

    TEACHER_GPT4V = "gpt-4v"
    TEACHER_CLAUDE = "claude"
    TEACHER_TOGETHER = "together"
    STUDENT_LOCAL = "student-local"
    STUDENT_EDGE = "student-edge"


class RoutingStrategy(str, Enum):
    """Routing strategies."""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
    BALANCED = "balanced"  # Balance cost and quality
    ADAPTIVE = "adaptive"  # Dynamically choose based on complexity


@dataclass
class RoutingDecision:
    """Routing decision result."""

    model_type: ModelType
    confidence_threshold: float
    reasoning: str
    estimated_cost: float
    estimated_quality: float
    fallback_to_teacher: bool
    reasoning_metadata: Dict


@dataclass
class VideoComplexityProfile:
    """Complexity analysis of a video."""

    motion_score: float  # 0-1, higher = more motion
    scene_complexity: int  # number of scene changes
    object_density: float  # objects per frame
    temporal_consistency: float  # 0-1, higher = more consistent
    hand_interaction_density: float  # hand action frequency
    duration_seconds: float

    @property
    def overall_complexity(self) -> float:
        """Calculate overall complexity score (0-1)."""
        weights = {
            "motion": 0.25,
            "scene": 0.20,
            "object": 0.20,
            "temporal": 0.15,
            "hand": 0.20,
        }

        # Normalize scene count (assuming max 20 scenes is complex)
        scene_norm = min(self.scene_complexity / 20, 1.0)

        # Temporal consistency is inverse (consistent = simpler)
        temporal_inv = 1 - self.temporal_consistency

        score = (
            weights["motion"] * self.motion_score
            + weights["scene"] * scene_norm
            + weights["object"] * self.object_density
            + weights["temporal"] * temporal_inv
            + weights["hand"] * self.hand_interaction_density
        )

        return min(1.0, max(0.0, score))


class VideoComplexityAnalyzer:
    """Analyze video complexity for routing decisions."""

    def __init__(self, sample_duration: float = 10.0):
        self.sample_duration = sample_duration

    def analyze(self, video_path: Path) -> VideoComplexityProfile:
        """Analyze video complexity asynchronously."""
        start_time = time.time()

        with VideoLoader(video_path) as loader:
            metadata = loader.metadata

            # Sample frames for analysis (avoid full video scan)
            sample_positions = np.linspace(
                0, metadata.duration, min(5, int(metadata.duration / 10) + 1)
            )

            motion_scores = []
            object_densities = []
            hand_interactions = []

            for pos in sample_positions:
                frames = list(loader.read_frames(start_time=pos, num_frames=8))
                if len(frames) < 2:
                    continue

                # Compute motion
                motion = loader.compute_motion_score(
                    start_time=pos, end_time=pos + 2, sample_frames=5
                )
                motion_scores.append(motion)

                # Estimate object density (placeholder for actual detection)
                # In production, use YOLO or similar
                frame_variance = np.var(frames[0][1])
                object_densities.append(min(frame_variance / 10000, 1.0))

                # Estimate hand interaction (based on lower frame region activity)
                # Placeholder: check motion in bottom half where hands typically are
                hand_interactions.append(motion * 0.7)  # Simplified

            # Detect scenes
            scenes = loader.detect_scenes(max_scenes=20)

            # Calculate temporal consistency
            temporal_consistency = 1.0 - min(len(scenes) / 10, 1.0)

        profile = VideoComplexityProfile(
            motion_score=float(np.mean(motion_scores)) if motion_scores else 0.5,
            scene_complexity=len(scenes),
            object_density=float(np.mean(object_densities)) if object_densities else 0.5,
            temporal_consistency=temporal_consistency,
            hand_interaction_density=float(np.mean(hand_interactions))
            if hand_interactions
            else 0.3,
            duration_seconds=metadata.duration,
        )

        logger.info(
            "complexity_analysis_complete",
            video=str(video_path),
            complexity=profile.overall_complexity,
            duration=time.time() - start_time,
        )

        return profile


class SmartRouter:
    """Intelligent model routing based on complexity and strategy."""

    # Cost estimates (per video in USD)
    COST_TABLE = {
        ModelType.TEACHER_GPT4V: 0.05,  # ~5 cents per video
        ModelType.TEACHER_CLAUDE: 0.04,
        ModelType.TEACHER_TOGETHER: 0.02,
        ModelType.STUDENT_LOCAL: 0.001,  # Compute cost only
        ModelType.STUDENT_EDGE: 0.0005,
    }

    # Quality estimates (0-1 scale)
    QUALITY_TABLE = {
        ModelType.TEACHER_GPT4V: 0.95,
        ModelType.TEACHER_CLAUDE: 0.93,
        ModelType.TEACHER_TOGETHER: 0.88,
        ModelType.STUDENT_LOCAL: 0.80,
        ModelType.STUDENT_EDGE: 0.70,
    }

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        budget_limit: Optional[float] = None,
        quality_threshold: float = 0.75,
        student_confidence_threshold: float = 0.8,
    ):
        self.strategy = strategy
        self.budget_limit = budget_limit
        self.quality_threshold = quality_threshold
        self.student_confidence_threshold = student_confidence_threshold
        self.analyzer = VideoComplexityAnalyzer()

        # Performance tracking
        self.routing_history: List[Dict] = []

    async def route(self, video_path: Path) -> RoutingDecision:
        """Determine best model for video annotation."""
        # Analyze video complexity
        complexity = self.analyzer.analyze(video_path)
        complexity_score = complexity.overall_complexity

        # Make routing decision based on strategy
        if self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            decision = self._route_quality_first(complexity_score)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            decision = self._route_cost_first(complexity_score)
        elif self.strategy == RoutingStrategy.BALANCED:
            decision = self._route_balanced(complexity_score)
        else:  # ADAPTIVE
            decision = self._route_adaptive(complexity, video_path)

        # Log decision
        self.routing_history.append({
            "video": str(video_path),
            "complexity": complexity_score,
            "strategy": self.strategy.value,
            "decision": decision.model_type.value,
            "reasoning": decision.reasoning,
        })

        logger.info(
            "routing_decision",
            video=str(video_path),
            complexity=complexity_score,
            model=decision.model_type.value,
            reasoning=decision.reasoning,
        )

        return decision

    def _route_quality_first(self, complexity: float) -> RoutingDecision:
        """Always choose best quality model within budget."""
        if self.budget_limit and self.budget_limit < 0.03:
            model = ModelType.TEACHER_TOGETHER
            reasoning = "Budget constrained, using most affordable teacher"
        else:
            model = ModelType.TEACHER_GPT4V
            reasoning = "Quality-first strategy: using best available model"

        return RoutingDecision(
            model_type=model,
            confidence_threshold=0.9,
            reasoning=reasoning,
            estimated_cost=self.COST_TABLE[model],
            estimated_quality=self.QUALITY_TABLE[model],
            fallback_to_teacher=False,
            reasoning_metadata={"strategy": "quality_first", "complexity": complexity},
        )

    def _route_cost_first(self, complexity: float) -> RoutingDecision:
        """Always choose cheapest adequate model."""
        # Try student first
        if complexity < 0.4:  # Simple video
            model = ModelType.STUDENT_EDGE
            reasoning = "Cost-first: simple video, using edge student model"
        elif complexity < 0.6:
            model = ModelType.STUDENT_LOCAL
            reasoning = "Cost-first: moderate complexity, using local student"
        else:
            model = ModelType.TEACHER_TOGETHER
            reasoning = "Cost-first: complex video required teacher fallback"

        return RoutingDecision(
            model_type=model,
            confidence_threshold=0.6 if "STUDENT" in model.value else 0.9,
            reasoning=reasoning,
            estimated_cost=self.COST_TABLE[model],
            estimated_quality=self.QUALITY_TABLE[model],
            fallback_to_teacher="STUDENT" in model.value,
            reasoning_metadata={"strategy": "cost_first", "complexity": complexity},
        )

    def _route_balanced(self, complexity: float) -> RoutingDecision:
        """Balance cost and quality based on complexity."""
        if complexity < 0.3:
            model = ModelType.STUDENT_LOCAL
            reasoning = "Balanced: simple video handled by student"
        elif complexity < 0.6:
            model = ModelType.TEACHER_TOGETHER
            reasoning = "Balanced: moderate complexity uses affordable teacher"
        else:
            model = ModelType.TEACHER_GPT4V
            reasoning = "Balanced: complex video needs best quality"

        return RoutingDecision(
            model_type=model,
            confidence_threshold=0.75,
            reasoning=reasoning,
            estimated_cost=self.COST_TABLE[model],
            estimated_quality=self.QUALITY_TABLE[model],
            fallback_to_teacher=complexity > 0.5 and "STUDENT" in model.value,
            reasoning_metadata={"strategy": "balanced", "complexity": complexity},
        )

    def _route_adaptive(
        self, complexity: VideoComplexityProfile, video_path: Path
    ) -> RoutingDecision:
        """Adaptive routing with multiple factors."""
        score = complexity.overall_complexity

        # Factor 1: Motion complexity
        if complexity.motion_score > 0.8:
            score += 0.1

        # Factor 2: Hand interactions (important for robotics)
        if complexity.hand_interaction_density > 0.7:
            score += 0.1

        # Factor 3: Video hash for consistent routing
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
        consistency_offset = int(video_hash[:2], 16) / 256 * 0.1 - 0.05
        score = min(1.0, max(0.0, score + consistency_offset))

        # Make decision
        if score < 0.25:
            model = ModelType.STUDENT_EDGE
            reasoning = f"Adaptive: very simple video (score={score:.2f})"
        elif score < 0.45:
            model = ModelType.STUDENT_LOCAL
            reasoning = f"Adaptive: simple video (score={score:.2f})"
        elif score < 0.65:
            model = ModelType.TEACHER_TOGETHER
            reasoning = f"Adaptive: moderate complexity (score={score:.2f})"
        elif score < 0.8:
            model = ModelType.TEACHER_CLAUDE
            reasoning = f"Adaptive: high complexity (score={score:.2f})"
        else:
            model = ModelType.TEACHER_GPT4V
            reasoning = f"Adaptive: very complex video (score={score:.2f})"

        # Check budget constraint
        if self.budget_limit and self.COST_TABLE[model] > self.budget_limit:
            model = ModelType.TEACHER_TOGETHER
            reasoning += ", downgraded due to budget constraint"

        # Determine confidence threshold and fallback
        if "STUDENT" in model.value:
            confidence_threshold = self.student_confidence_threshold - (score * 0.2)
            fallback = True
        else:
            confidence_threshold = 0.9
            fallback = False

        return RoutingDecision(
            model_type=model,
            confidence_threshold=confidence_threshold,
            reasoning=reasoning,
            estimated_cost=self.COST_TABLE[model],
            estimated_quality=self.QUALITY_TABLE[model],
            fallback_to_teacher=fallback,
            reasoning_metadata={
                "strategy": "adaptive",
                "complexity": complexity.overall_complexity,
                "adjusted_score": score,
                "motion": complexity.motion_score,
                "hand_density": complexity.hand_interaction_density,
            },
        )

    def get_routing_statistics(self) -> Dict:
        """Get statistics on routing decisions."""
        if not self.routing_history:
            return {}

        model_counts = {}
        complexity_sum = 0

        for entry in self.routing_history:
            model = entry["decision"]
            model_counts[model] = model_counts.get(model, 0) + 1
            complexity_sum += entry["complexity"]

        total = len(self.routing_history)
        avg_complexity = complexity_sum / total

        # Calculate savings from student usage
        teacher_cost = self.COST_TABLE[ModelType.TEACHER_GPT4V] * total
        actual_cost = sum(
            self.COST_TABLE[ModelType(m)]
            for m in [e["decision"] for e in self.routing_history]
        )
        savings_pct = (1 - actual_cost / teacher_cost) * 100 if teacher_cost > 0 else 0

        return {
            "total_videos": total,
            "model_distribution": {
                k: {"count": v, "percentage": v / total * 100}
                for k, v in model_counts.items()
            },
            "average_complexity": avg_complexity,
            "estimated_savings_percent": savings_pct,
            "total_estimated_cost": actual_cost,
            "teacher_only_cost": teacher_cost,
        }


async def route_and_annotate(
    video_path: Path,
    router: SmartRouter,
    student_engine=None,
    teacher_pool: Optional[Dict] = None,
) -> Tuple[str, RoutingDecision]:
    """Full pipeline: route and annotate."""
    # Get routing decision
    decision = await router.route(video_path)

    # Execute annotation
    annotation_text = ""

    if decision.model_type in (ModelType.STUDENT_LOCAL, ModelType.STUDENT_EDGE):
        # Try student first
        if student_engine:
            try:
                annotation_text = student_engine.generate(
                    video_path=video_path,
                )

                # Check confidence
                confidence = student_engine._estimate_confidence(annotation_text)

                if confidence < decision.confidence_threshold and decision.fallback_to_teacher:
                    logger.info("student_confidence_low_fallback", confidence=confidence)
                    # Fallback to teacher
                    if teacher_pool:
                        teacher = teacher_pool.get(ModelType.TEACHER_GPT4V)
                        if teacher:
                            # Import here to avoid circular dependency

                            result = await teacher.annotate(
                                frames=[],  # Would need to load frames
                                task="fine_grained",
                            )
                            annotation_text = result.get("text", "")

            except Exception as e:
                logger.error("student_inference_failed", error=str(e))
                if decision.fallback_to_teacher:
                    annotation_text = "[Fallback to teacher needed due to error]"

    return annotation_text, decision
