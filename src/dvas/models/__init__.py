"""DVAS models: teacher, student, and evaluation."""

from dvas.models.evaluator.llm_judge import ConsistencyChecker, LLMJudge
from dvas.models.evaluator.metrics import MetricsCalculator, compare_annotations
from dvas.models.teacher.base import TeacherModel

# Student models are optional (heavy training dependencies)
try:
    from dvas.models.student.inference import StudentInferenceEngine, StudentTeacherBridge
except ImportError:
    StudentInferenceEngine = None  # type: ignore
    StudentTeacherBridge = None  # type: ignore

__all__ = [
    "ConsistencyChecker",
    "LLMJudge",
    "MetricsCalculator",
    "StudentInferenceEngine",
    "StudentTeacherBridge",
    "TeacherModel",
    "compare_annotations",
]
