"""Governance module — annotation standard management.

Public API:
    get_adapter(standard) -> StandardAdapter
    list_standards() -> List[AnnotationStandard]
"""

from dvas.data.schemas import AnnotationStandard
from dvas.governance.adapters import (
    EPICAdapter,
    Ego4DAdapter,
    OpenXAdapter,
    StandardAdapter,
    get_adapter,
    list_standards,
)

__all__ = [
    "AnnotationStandard",
    "StandardAdapter",
    "EPICAdapter",
    "Ego4DAdapter",
    "OpenXAdapter",
    "get_adapter",
    "list_standards",
]
