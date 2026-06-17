"""Teacher models for generating gold-standard annotations."""

from dvas.models.teacher.base import TeacherModel
from dvas.models.teacher.claude import ClaudeTeacher
from dvas.models.teacher.gpt4v import GPT4VTeacher, GPT4VisionTeacher
from dvas.models.teacher.together import TogetherTeacher

__all__ = [
    "ClaudeTeacher",
    "GPT4VTeacher",
    "GPT4VisionTeacher",
    "TeacherModel",
    "TogetherTeacher",
]
