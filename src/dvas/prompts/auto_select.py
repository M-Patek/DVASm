"""Prompt auto-selection based on video characteristics.

Automatically selects the best prompt based on video domain detection,
performance history, and video metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dvas.prompts.registry import PromptDomain, PromptRegistry, PromptTemplate
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class DomainDetector:
    """Detects video domain from metadata and content."""

    DOMAIN_KEYWORDS = {
        PromptDomain.KITCHEN: [
            "cook", "kitchen", "cut", "chop", "food", "ingredient",
            "recipe", "pan", "oven", "knife", "stove", "plate",
        ],
        PromptDomain.ROBOT: [
            "grasp", "pick", "place", "manipulate", "hand", "finger",
            "grip", "assembly", "robot", "arm", "gripper",
        ],
        PromptDomain.MEDICAL: [
            "surgery", "medical", "patient", "instrument", "operation",
            "tissue", "doctor", "hospital", "clinic",
        ],
        PromptDomain.SPORTS: [
            "run", "jump", "sport", "ball", "game", "match",
            "exercise", "fitness", "athlete", "court", "field",
        ],
        PromptDomain.ASSEMBLY: [
            "assemble", "build", "construct", "screw", "bolt",
            "part", "component", "wrench", "tool", "factory",
        ],
    }

    def detect_from_filename(self, video_path: Path) -> PromptDomain:
        """Detect domain from video filename."""
        filename = video_path.name.lower()
        scores: Dict[PromptDomain, int] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in filename)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore
        return PromptDomain.GENERAL

    def detect_from_metadata(self, metadata: Dict[str, any]) -> PromptDomain:
        """Detect domain from video metadata."""
        tags = metadata.get("tags", [])
        description = metadata.get("description", "").lower()
        title = metadata.get("title", "").lower()

        scores: Dict[PromptDomain, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in description or kw in title)
            for tag in tags:
                if any(kw in tag.lower() for kw in keywords):
                    score += 1
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore
        return PromptDomain.GENERAL


@dataclass
class VideoCharacteristics:
    """Characteristics of a video for prompt selection."""

    duration_seconds: float = 0.0
    scene_count: int = 0
    motion_score: float = 0.0
    object_count: int = 0
    has_audio: bool = False
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    tags: List[str] = field(default_factory=list)

    @property
    def complexity_score(self) -> float:
        """Compute a complexity score (0-1)."""
        duration_factor = min(self.duration_seconds / 60.0, 1.0)
        scene_factor = min(self.scene_count / 10.0, 1.0)
        motion_factor = min(self.motion_score, 1.0)
        object_factor = min(self.object_count / 20.0, 1.0)

        return (duration_factor + scene_factor + motion_factor + object_factor) / 4.0

    @property
    def is_complex(self) -> bool:
        return self.complexity_score > 0.6

    @property
    def is_simple(self) -> bool:
        return self.complexity_score < 0.3


class AutoSelector:
    """Automatically selects the best prompt for a video."""

    def __init__(self, registry: Optional[PromptRegistry] = None) -> None:
        self.registry = registry or PromptRegistry()
        self.domain_detector = DomainDetector()
        self._exploration_rate = 0.2

    def select(
        self,
        video_path: Path,
        video_characteristics: Optional[VideoCharacteristics] = None,
        task_type: str = "caption",
        preferred_domain: Optional[PromptDomain] = None,
    ) -> Optional[PromptTemplate]:
        """Select the best prompt for a video.

        Args:
            video_path: Path to the video file.
            video_characteristics: Optional pre-computed characteristics.
            task_type: Type of annotation task.
            preferred_domain: Optional preferred domain override.

        Returns:
            The selected prompt template, or None if no suitable prompt found.
        """
        # Detect domain
        domain = preferred_domain or self.domain_detector.detect_from_filename(video_path)

        # Get prompts for domain
        candidates = self.registry.list_by_domain(domain)
        if not candidates:
            candidates = self.registry.list_by_domain(PromptDomain.GENERAL)

        if not candidates:
            logger.warning("no_prompts_found", domain=domain.value)
            return None
        # Filter by task type if specified
        task_candidates = [
            p for p in candidates
            if task_type in p.metadata.tags or not p.metadata.tags
        ]
        if task_candidates:
            candidates = task_candidates

        # Rank by quality score
        ranked = sorted(
            candidates,
            key=lambda p: p.avg_quality_score,
            reverse=True,
        )

        # Exploration vs exploitation
        if random.random() < self._exploration_rate and len(ranked) > 1:
            # Explore: pick a random candidate from top 3
            selected = random.choice(ranked[:min(3, len(ranked))])
            logger.info(
                "prompt_selected_explore",
                prompt_id=selected.id,
                domain=domain.value,
            )
            return selected

        # Exploit: pick the best performer
        if ranked:
            selected = ranked[0]
            logger.info(
                "prompt_selected_exploit",
                prompt_id=selected.id,
                domain=domain.value,
                quality_score=selected.avg_quality_score,
            )
            return selected

        return None

    def select_for_characteristics(
        self,
        characteristics: VideoCharacteristics,
        domain: PromptDomain = PromptDomain.GENERAL,
    ) -> Optional[PromptTemplate]:
        """Select prompt based on video characteristics.

        Args:
            characteristics: Video characteristics.
            domain: Domain to select from.

        Returns:
            Selected prompt or None.
        """
        candidates = self.registry.list_by_domain(domain)
        if not candidates:
            candidates = self.registry.list_by_domain(PromptDomain.GENERAL)

        if not candidates:
            return None

        # For complex videos, prefer prompts with higher detail
        if characteristics.is_complex:
            # Filter for complex prompts (those with higher quality scores)
            candidates = sorted(candidates, key=lambda p: p.avg_quality_score, reverse=True)

        # For simple videos, prefer simpler prompts
        if characteristics.is_simple:
            # Use any available prompt
            pass

        if candidates:
            return candidates[0]
        return None

    def rank_prompts(
        self,
        video_path: Path,
        domain: Optional[PromptDomain] = None,
    ) -> List[Tuple[PromptTemplate, float]]:
        """Rank all prompts by suitability for a video.

        Returns:
            List of (prompt, score) tuples sorted by score descending.
        """
        detected_domain = domain or self.domain_detector.detect_from_filename(video_path)
        candidates = self.registry.list_by_domain(detected_domain)
        if not candidates:
            candidates = self.registry.list_by_domain(PromptDomain.GENERAL)

        scored: List[Tuple[PromptTemplate, float]] = []
        for prompt in candidates:
            score = prompt.avg_quality_score
            scored.append((prompt, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def set_exploration_rate(self, rate: float) -> None:
        """Set the exploration rate (0.0 to 1.0)."""
        self._exploration_rate = max(0.0, min(1.0, rate))
