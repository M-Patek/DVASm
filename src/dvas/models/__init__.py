"""DVAS models: teacher, student, and evaluation."""

from dvas.models.evaluator.llm_judge import ConsistencyChecker, LLMJudge
from dvas.models.evaluator.metrics import MetricsCalculator, compare_annotations
from dvas.models.teacher.base import TeacherModel
from dvas.models.teacher.claude import ClaudeTeacher
from dvas.models.teacher.gpt4v import GPT4VTeacher, GPT4VisionTeacher
from dvas.models.teacher.together import TogetherTeacher

# Student models are optional (heavy training dependencies)
try:
    from dvas.models.student.inference import StudentInferenceEngine, StudentTeacherBridge
except ImportError:
    StudentInferenceEngine = None  # type: ignore
    StudentTeacherBridge = None  # type: ignore

__all__ = [
    "ClaudeTeacher",
    "ConsistencyChecker",
    "GPT4VTeacher",
    "GPT4VisionTeacher",
    "LLMJudge",
    "MetricsCalculator",
    "StudentInferenceEngine",
    "StudentTeacherBridge",
    "TeacherModel",
    "TogetherTeacher",
    "compare_annotations",
]
