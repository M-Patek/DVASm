"""Structured response parsing for model outputs.

Replaces fragile regex-based parsing with explicit strategies.
All parsing failures are tracked for later analysis.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.data.schemas import Action, Hand, Object, QAPair
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedSegment:
    """Result of parsing a model response into structured segment data."""

    scene_description: str = ""
    qa_pairs: List[QAPair] = field(default_factory=list)
    objects: List[Object] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    raw_text: str = ""
    parse_method: str = ""
    confidence: float = 0.0


@dataclass
class ParseFailure:
    """Record of a parsing failure for later analysis."""

    raw_text: str
    error_type: str
    error_message: str
    attempted_methods: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure record to dictionary for serialization."""
        return {
            "raw_text": self.raw_text[:1000],  # Truncate for storage
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempted_methods": self.attempted_methods,
            "timestamp": self.timestamp,
        }


class ParseStrategy(ABC):
    """Abstract base for response parsing strategies."""

    @abstractmethod
    def parse(self, text: str) -> Optional[ParsedSegment]:
        """Attempt to parse text. Return None if this strategy cannot handle it."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and debugging."""
        pass


class JsonBlockStrategy(ParseStrategy):
    """Parse JSON block from model response (most structured)."""

    @property
    def name(self) -> str:
        return "json_block"

    def parse(self, text: str) -> Optional[ParsedSegment]:
        # Find JSON block between triple backticks or standalone
        json_text = self._extract_json(text)
        if not json_text:
            return None

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        return self._build_segment(data, text)

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON string from text."""
        # Try markdown code block first
        import re

        code_block = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Try standalone JSON object
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            return brace_match.group().strip()

        return None

    def _build_segment(self, data: Dict, raw_text: str) -> ParsedSegment:
        """Build ParsedSegment from JSON data."""
        # Extract scene description from multiple possible field names
        description = (
            data.get("scene_description")
            or data.get("description")
            or data.get("caption")
            or raw_text[:500]
        )

        # Extract QA pairs from multiple possible structures
        qa_pairs = self._extract_qa_pairs(data)

        # Extract objects
        objects = self._extract_objects(data)

        # Extract actions
        actions = self._extract_actions(data)

        # Confidence: more structured fields = higher confidence
        confidence = self._calculate_confidence(data, qa_pairs, objects, actions)

        return ParsedSegment(
            scene_description=description,
            qa_pairs=qa_pairs,
            objects=objects,
            actions=actions,
            raw_text=raw_text[:500],
            parse_method=self.name,
            confidence=confidence,
        )

    def _extract_qa_pairs(self, data: Dict) -> List[QAPair]:
        """Extract QA pairs from various possible structures."""
        qa_pairs = []

        # Structure 1: steps array with action/details
        steps = data.get("steps", [])
        for i, step in enumerate(steps[:5]):
            if isinstance(step, dict):
                action = step.get("action", "")
                details = step.get("details", "")
                qa_pairs.append(
                    QAPair(
                        question=f"Step {i + 1}: What action is performed?",
                        answer=f"{action} {details}".strip(),
                    )
                )

        # Structure 2: explicit qa_pairs array
        explicit_qa = data.get("qa_pairs", data.get("qa", []))
        for qa in explicit_qa[:5]:
            if isinstance(qa, dict):
                qa_pairs.append(
                    QAPair(
                        question=qa.get("question", ""),
                        answer=qa.get("answer", ""),
                        question_type=qa.get("type", "other"),
                    )
                )

        return qa_pairs

    def _extract_objects(self, data: Dict) -> List[Object]:
        """Extract objects from various possible structures."""
        objects = []
        obj_list = data.get("objects", [])

        for obj in obj_list[:20]:  # Limit to prevent abuse
            if isinstance(obj, dict):
                objects.append(
                    Object(
                        name=obj.get("name", "unknown"),
                        confidence=None,
                        attributes={"state": obj.get("state", "")},
                    )
                )
            elif isinstance(obj, str):
                objects.append(Object(name=obj, confidence=None))

        return objects

    def _extract_actions(self, data: Dict) -> List[Action]:
        """Extract actions from various possible structures."""
        actions = []

        # Structure 1: hand_actions array
        hand_actions = data.get("hand_actions", [])
        for ha in hand_actions[:10]:
            if isinstance(ha, dict):
                action_text = ha.get("action", "")
                verb = action_text.split()[0] if action_text else "unknown"
                actions.append(
                    Action(
                        verb=verb,
                        noun=ha.get("target", ""),
                        confidence=None,
                        hand=self._parse_hand(ha.get("hand", "")),
                    )
                )

        # Structure 2: actions array with verb/noun
        action_list = data.get("actions", [])
        for a in action_list[:10]:
            if isinstance(a, dict):
                actions.append(
                    Action(
                        verb=a.get("verb", a.get("action", "unknown")),
                        noun=a.get("noun", a.get("target", "")),
                        confidence=None,
                        hand=self._parse_hand(a.get("hand", "")),
                    )
                )

        return actions

    def _parse_hand(self, hand_str: str) -> Hand:
        """Parse hand string to Hand enum."""
        hand_lower = str(hand_str).lower()
        if "left" in hand_lower:
            return Hand.LEFT
        if "right" in hand_lower:
            return Hand.RIGHT
        if "both" in hand_lower:
            return Hand.BOTH
        return Hand.UNKNOWN

    def _calculate_confidence(
        self, data: Dict, qa_pairs: List, objects: List, actions: List
    ) -> float:
        """Calculate parse confidence based on structure richness."""
        score = 0.3  # Base for having valid JSON

        if data.get("scene_description") or data.get("description"):
            score += 0.2
        if qa_pairs:
            score += 0.2
        if objects:
            score += 0.15
        if actions:
            score += 0.15

        return min(1.0, score)


class StructuredTextStrategy(ParseStrategy):
    """Parse structured text with clear section markers."""

    @property
    def name(self) -> str:
        return "structured_text"

    def parse(self, text: str) -> Optional[ParsedSegment]:
        import re

        # Look for structured sections like "Scene:", "Actions:", "Objects:"
        has_structure = any(
            marker in text.lower()
            for marker in ["scene:", "actions:", "objects:", "steps:", "description:"]
        )

        if not has_structure:
            return None

        # Extract scene description - match from marker to next section or end
        scene_match = re.search(
            r"(?:scene|description):\s*(.+?)(?=\n\s*(?:actions|objects|steps):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        scene_description = scene_match.group(1).strip() if scene_match else text[:500]

        # Extract actions - match from "Actions:" to next section or end
        actions = []
        action_section = re.search(
            r"actions?:\s*(.+?)(?=\n\s*(?:objects|scene|steps):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if action_section:
            for line in action_section.group(1).strip().split("\n")[:10]:
                line = line.strip("- *1234567890. ")
                if line:
                    words = line.split()
                    if len(words) >= 2:
                        actions.append(Action(verb=words[0], noun=words[-1], confidence=None))
                    elif len(words) == 1:
                        actions.append(Action(verb=words[0], noun="", confidence=None))

        # Extract objects - match from "Objects:" to next section or end
        objects = []
        obj_section = re.search(
            r"objects?:\s*(.+?)(?=\n\s*(?:actions|scene|steps):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if obj_section:
            for item in obj_section.group(1).strip().split(",")[:10]:
                item = item.strip("- *1234567890. \n")
                if item:
                    objects.append(Object(name=item, confidence=None))

        confidence = 0.5 if (actions or objects) else 0.3

        return ParsedSegment(
            scene_description=scene_description,
            actions=actions,
            objects=objects,
            raw_text=text[:500],
            parse_method=self.name,
            confidence=confidence,
        )


class PlainTextStrategy(ParseStrategy):
    """Fallback: treat entire text as scene description."""

    @property
    def name(self) -> str:
        return "plain_text"

    def parse(self, text: str) -> Optional[ParsedSegment]:
        # This is the ultimate fallback - always succeeds
        return ParsedSegment(
            scene_description=text[:2000],  # Allow longer for plain text
            raw_text=text[:500],
            parse_method=self.name,
            confidence=0.1,  # Low confidence since unstructured
        )


class StructuredParser:
    """Multi-strategy parser for model responses.

    Tries strategies in order of preference, tracks failures.
    """

    # Default strategy order: most structured first
    DEFAULT_STRATEGIES = [
        JsonBlockStrategy(),
        StructuredTextStrategy(),
        PlainTextStrategy(),  # Always succeeds
    ]

    def __init__(
        self,
        strategies: Optional[List[ParseStrategy]] = None,
        failure_log_path: Optional[Path] = None,
    ):
        self.strategies = strategies or self.DEFAULT_STRATEGIES.copy()
        self.failure_log_path = failure_log_path
        self.failures: List[ParseFailure] = []

    def parse(self, text: str) -> ParsedSegment:
        """Parse text using available strategies.

        Tries each strategy in order. If all structured strategies fail,
        falls back to plain text. Records failures for analysis.
        """
        if not text or not text.strip():
            return ParsedSegment(
                scene_description="",
                parse_method="empty_input",
                confidence=0.0,
            )

        attempted: List[str] = []

        for strategy in self.strategies:
            try:
                result = strategy.parse(text)
                if result is not None:
                    logger.debug(
                        "parse_success",
                        method=strategy.name,
                        confidence=result.confidence,
                    )
                    return result
            except Exception as e:
                logger.warning("parse_strategy_failed", method=strategy.name, error=str(e))

            attempted.append(strategy.name)

        # This should not happen (PlainTextStrategy always succeeds)
        # but record as failure if it does
        failure = ParseFailure(
            raw_text=text,
            error_type="all_strategies_failed",
            error_message="No strategy could parse the response",
            attempted_methods=attempted,
        )
        self._record_failure(failure)

        return ParsedSegment(
            scene_description=text[:2000],
            parse_method="emergency_fallback",
            confidence=0.0,
        )

    def _record_failure(self, failure: ParseFailure) -> None:
        """Record a parsing failure."""
        self.failures.append(failure)

        if self.failure_log_path:
            try:
                self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.failure_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(failure.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                pass  # Don't let logging failures crash parsing

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get statistics on parsing failures."""
        if not self.failures:
            return {"total_failures": 0}

        from collections import Counter

        error_types: Counter[str] = Counter(f.error_type for f in self.failures)
        attempted_counts: Counter[str] = Counter()
        for f in self.failures:
            for method in f.attempted_methods:
                attempted_counts[method] += 1

        return {
            "total_failures": len(self.failures),
            "error_type_distribution": dict(error_types),
            "attempted_strategies": dict(attempted_counts),
        }

    def to_legacy_dict(self, parsed: ParsedSegment) -> Dict[str, Any]:
        """Convert ParsedSegment to legacy dict format for backward compatibility."""
        return {
            "scene_description": parsed.scene_description,
            "qa_pairs": parsed.qa_pairs,
            "objects": parsed.objects,
            "actions": parsed.actions,
            "_parse_metadata": {
                "method": parsed.parse_method,
                "confidence": parsed.confidence,
            },
        }
