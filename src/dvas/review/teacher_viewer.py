"""Teacher output viewer for reviewing raw teacher responses.

Displays raw teacher model outputs, parse confidence, and
fallback chain visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a teacher output."""

    raw_text: str
    parsed_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    parse_method: str = "unknown"
    fallback_used: bool = False
    fallback_level: int = 0
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_data": self.parsed_data,
            "confidence": self.confidence,
            "parse_method": self.parse_method,
            "fallback_used": self.fallback_used,
            "fallback_level": self.fallback_level,
            "parse_errors": self.parse_errors,
        }


@dataclass
class FallbackStep:
    """A single step in the fallback chain."""

    step_index: int
    model_name: str
    status: str  # "success", "failed", "skipped"
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    output_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "model_name": self.model_name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "output_preview": self.output_preview,
        }


@dataclass
class TeacherOutput:
    """Complete teacher output with metadata."""

    annotation_id: str
    video_id: str
    teacher_model: str
    raw_response: str
    parse_result: Optional[ParseResult] = None
    fallback_chain: List[FallbackStep] = field(default_factory=list)
    generation_time_ms: Optional[float] = None
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "teacher_model": self.teacher_model,
            "raw_response": self.raw_response,
            "parse_result": self.parse_result.to_dict() if self.parse_result else None,
            "fallback_chain": [s.to_dict() for s in self.fallback_chain],
            "generation_time_ms": self.generation_time_ms,
            "token_count": self.token_count,
            "cost_usd": self.cost_usd,
        }


class TeacherOutputViewer:
    """Viewer for raw teacher model outputs.

    Displays raw responses, parse confidence, and fallback chain
    visualization for debugging and review.
    """

    def __init__(self):
        self._outputs: Dict[str, TeacherOutput] = {}

    def add_output(self, output: TeacherOutput) -> None:
        """Add a teacher output to the viewer.

        Args:
            output: TeacherOutput to add
        """
        self._outputs[output.annotation_id] = output
        logger.info(
            "teacher_output_added",
            annotation_id=output.annotation_id,
            model=output.teacher_model,
        )

    def get_output(self, annotation_id: str) -> Optional[TeacherOutput]:
        """Get a teacher output by annotation ID.

        Args:
            annotation_id: ID of the annotation

        Returns:
            TeacherOutput or None if not found
        """
        return self._outputs.get(annotation_id)

    def get_parse_confidence(self, annotation_id: str) -> float:
        """Get parse confidence for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            Parse confidence (0.0 to 1.0)
        """
        output = self._outputs.get(annotation_id)
        if output and output.parse_result:
            return output.parse_result.confidence
        return 0.0

    def get_fallback_chain(self, annotation_id: str) -> List[FallbackStep]:
        """Get the fallback chain for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            List of fallback steps
        """
        output = self._outputs.get(annotation_id)
        if output:
            return output.fallback_chain
        return []

    def get_fallback_summary(self, annotation_id: str) -> Dict[str, Any]:
        """Get a summary of the fallback chain.

        Args:
            annotation_id: ID of the annotation

        Returns:
            Dict with fallback summary statistics
        """
        chain = self.get_fallback_chain(annotation_id)
        if not chain:
            return {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "primary_model_used": False,
                "fallback_triggered": False,
            }

        successful = sum(1 for s in chain if s.status == "success")
        failed = sum(1 for s in chain if s.status == "failed")

        return {
            "total_steps": len(chain),
            "successful_steps": successful,
            "failed_steps": failed,
            "primary_model_used": len(chain) > 0 and chain[0].status == "success",
            "fallback_triggered": len(chain) > 1 or any(s.step_index > 0 for s in chain),
            "avg_latency_ms": sum(s.latency_ms or 0 for s in chain) / len(chain) if chain else 0.0,
        }

    def visualize_fallback_chain(self, annotation_id: str) -> List[Dict[str, Any]]:
        """Get a visualization-ready representation of the fallback chain.

        Args:
            annotation_id: ID of the annotation

        Returns:
            List of dicts with visualization data
        """
        chain = self.get_fallback_chain(annotation_id)
        result = []

        for step in chain:
            result.append(
                {
                    "step": step.step_index,
                    "model": step.model_name,
                    "status": step.status,
                    "latency_ms": step.latency_ms,
                    "has_error": step.error_message is not None,
                    "output_preview": step.output_preview,
                }
            )

        return result

    def get_all_outputs(self) -> List[TeacherOutput]:
        """Get all teacher outputs.

        Returns:
            List of all TeacherOutput
        """
        return list(self._outputs.values())

    def filter_by_model(self, model_name: str) -> List[TeacherOutput]:
        """Filter outputs by teacher model name.

        Args:
            model_name: Name of the model to filter by

        Returns:
            List of matching TeacherOutput
        """
        return [o for o in self._outputs.values() if o.teacher_model == model_name]

    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of parse confidence scores.

        Returns:
            Dict with confidence ranges and counts
        """
        ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for output in self._outputs.values():
            conf = output.parse_result.confidence if output.parse_result else 0.0
            if conf < 0.2:
                ranges["0.0-0.2"] += 1
            elif conf < 0.4:
                ranges["0.2-0.4"] += 1
            elif conf < 0.6:
                ranges["0.4-0.6"] += 1
            elif conf < 0.8:
                ranges["0.6-0.8"] += 1
            else:
                ranges["0.8-1.0"] += 1

        return ranges
