"""Few-shot example retrieval with semantic indexing.

Provides vector-based example retrieval, domain-specific example packs,
and similarity-based selection for prompt augmentation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dvas.prompts.registry import PromptDomain
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Example:
    """A single few-shot example."""

    id: str
    input_text: str
    output_text: str
    domain: PromptDomain = PromptDomain.GENERAL
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "domain": self.domain.value,
            "tags": self.tags.copy(),
            "quality_score": self.quality_score,
        }


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _simple_embedding(text: str, dim: int = 64) -> List[float]:
    """Create a simple deterministic embedding from text.

    In production, replace with actual embedding model (e.g., sentence-transformers).
    """
    # Simple character n-gram hash-based embedding
    vec = [0.0] * dim
    text_lower = text.lower()

    for i in range(len(text_lower) - 2):
        ngram = text_lower[i : i + 3]
        h = hashlib.md5(ngram.encode()).hexdigest()
        idx = int(h, 16) % dim
        vec[idx] += 1.0

    # Normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


class SemanticExampleIndex:
    """Index for semantic retrieval of few-shot examples."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._examples: Dict[str, Example] = {}
        self._domain_index: Dict[PromptDomain, List[str]] = {}
        self._embedding_dim = embedding_dim

    def add_example(self, example: Example) -> None:
        """Add an example to the index."""
        if example.embedding is None:
            example.embedding = _simple_embedding(
                example.input_text, self._embedding_dim
            )

        self._examples[example.id] = example

        if example.domain not in self._domain_index:
            self._domain_index[example.domain] = []
        if example.id not in self._domain_index[example.domain]:
            self._domain_index[example.domain].append(example.id)

    def add_examples(self, examples: List[Example]) -> None:
        """Add multiple examples."""
        for example in examples:
            self.add_example(example)

    def search(
        self,
        query: str,
        domain: Optional[PromptDomain] = None,
        top_k: int = 3,
        min_quality: float = 0.0,
    ) -> List[Tuple[Example, float]]:
        """Search for similar examples.

        Args:
            query: Query text to search for.
            domain: Optional domain filter.
            top_k: Number of results to return.
            min_quality: Minimum quality score filter.

        Returns:
            List of (example, similarity) tuples sorted by similarity.
        """
        query_embedding = _simple_embedding(query, self._embedding_dim)

        candidates: List[Example] = []
        if domain and domain in self._domain_index:
            for ex_id in self._domain_index[domain]:
                if ex_id in self._examples:
                    candidates.append(self._examples[ex_id])
        else:
            candidates = list(self._examples.values())

        # Filter by quality
        candidates = [e for e in candidates if e.quality_score >= min_quality]

        # Compute similarities
        scored: List[Tuple[Example, float]] = []
        for example in candidates:
            if example.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, example.embedding)
            scored.append((example, sim))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def get_by_domain(self, domain: PromptDomain) -> List[Example]:
        """Get all examples for a domain."""
        ids = self._domain_index.get(domain, [])
        return [self._examples[i] for i in ids if i in self._examples]

    def get_by_tag(self, tag: str) -> List[Example]:
        """Get all examples with a tag."""
        return [e for e in self._examples.values() if tag in e.tags]

    def size(self) -> int:
        """Get total number of examples."""
        return len(self._examples)


class ExamplePack:
    """A pack of few-shot examples for a specific domain."""

    def __init__(self, name: str, domain: PromptDomain) -> None:
        self.name = name
        self.domain = domain
        self.examples: List[Example] = []

    def add(self, example: Example) -> None:
        """Add an example to the pack."""
        self.examples.append(example)

    def get_examples(
        self,
        query: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Example]:
        """Get examples, optionally filtered by query similarity.

        Args:
            query: Optional query for similarity search.
            top_k: Number of examples to return.

        Returns:
            List of examples.
        """
        if query is None:
            return self.examples[:top_k]

        # Simple keyword matching for fallback
        query_lower = query.lower()
        scored: List[Tuple[Example, float]] = []
        for ex in self.examples:
            score = 0.0
            if query_lower in ex.input_text.lower():
                score += 1.0
            for tag in ex.tags:
                if tag.lower() in query_lower:
                    score += 0.5
            scored.append((ex, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def to_prompt_text(self, examples: Optional[List[Example]] = None) -> str:
        """Convert examples to prompt text.

        Args:
            examples: Optional list of examples to format. Uses all if not provided.

        Returns:
            Formatted prompt text with examples.
        """
        if examples is None:
            examples = self.examples

        lines: List[str] = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {ex.input_text}")
            lines.append(f"Output: {ex.output_text}")
            lines.append("")

        return "\n".join(lines)


def create_domain_example_packs() -> Dict[PromptDomain, ExamplePack]:
    """Create default example packs for common domains.

    Returns:
        Dictionary mapping domains to example packs.
    """
    packs: Dict[PromptDomain, ExamplePack] = {}

    # Kitchen domain examples
    kitchen_pack = ExamplePack("kitchen_default", PromptDomain.KITCHEN)
    kitchen_pack.add(Example(
        id="kitchen_ex_1",
        input_text="Video of person chopping onions on a cutting board",
        output_text="The person uses their right hand to grip the knife and their left hand to steady the onion. They make several downward cutting motions to dice the onion into small pieces.",
        domain=PromptDomain.KITCHEN,
        tags=["chopping", "knife"],
        quality_score=0.9,
    ))
    kitchen_pack.add(Example(
        id="kitchen_ex_2",
        input_text="Video of person stirring soup in a pot",
        output_text="The person holds a wooden spoon in their right hand and stirs the soup in a circular motion inside the pot on the stove.",
        domain=PromptDomain.KITCHEN,
        tags=["stirring", "pot"],
        quality_score=0.85,
    ))
    packs[PromptDomain.KITCHEN] = kitchen_pack

    # Robot domain examples
    robot_pack = ExamplePack("robot_default", PromptDomain.ROBOT)
    robot_pack.add(Example(
        id="robot_ex_1",
        input_text="Video of robotic arm picking up a red block",
        output_text="The robotic arm moves to position above the red block, opens the gripper, descends, and closes the gripper to grasp the block firmly.",
        domain=PromptDomain.ROBOT,
        tags=["grasp", "block"],
        quality_score=0.92,
    ))
    robot_pack.add(Example(
        id="robot_ex_2",
        input_text="Video of robot placing object on shelf",
        output_text="The robot arm extends forward while holding the object, positions it above the shelf, and releases the gripper to place the object down.",
        domain=PromptDomain.ROBOT,
        tags=["place", "shelf"],
        quality_score=0.88,
    ))
    packs[PromptDomain.ROBOT] = robot_pack

    # General domain examples
    general_pack = ExamplePack("general_default", PromptDomain.GENERAL)
    general_pack.add(Example(
        id="general_ex_1",
        input_text="Video of a person walking in a park",
        output_text="A person is walking along a path in a park. Trees and grass are visible in the background. The person moves at a steady pace.",
        domain=PromptDomain.GENERAL,
        tags=["walking", "outdoor"],
        quality_score=0.8,
    ))
    packs[PromptDomain.GENERAL] = general_pack

    return packs
