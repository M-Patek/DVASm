"""Pipeline module for DVAS."""

from dvas.pipeline.core import (
    AnnotationPipeline,
    EPICAnnotationPipeline,
    create_training_data_from_gold,
)

__all__ = [
    "AnnotationPipeline",
    "EPICAnnotationPipeline",
    "create_training_data_from_gold",
]
