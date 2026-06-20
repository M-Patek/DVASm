"""Review workbench for DVAS video annotation platform.

Provides tools for human review of annotations including dataset browsing,
annotation editing, diff viewing, reviewer assignment, and approval workflows.
"""

from dvas.review.annotation_editor import AnnotationEditor, ChangeRecord
from dvas.review.dataset_browser import (
    DatasetBrowser,
    DatasetFilter,
    DatasetStatistics,
    PaginationResult,
    SortField,
    SortOrder,
)
from dvas.review.diff_viewer import (
    AnnotationDiff,
    AnnotationDiffResult,
    DiffType,
    FieldDiff,
    SegmentDiff,
)
from dvas.review.reviewer_assignment import (
    Assignment,
    Reviewer,
    ReviewerAssignment,
)
from dvas.review.review_queue import (
    QueueItem,
    QueueItemStatus,
    QueuePriority,
    ReviewQueue,
)
from dvas.review.workflow import (
    ApprovalWorkflow,
    RejectionReason,
    RejectionRecord,
    StageTransition,
    WorkflowAnnotation,
    WorkflowStage,
)
from dvas.review.reviewer_metrics import (
    ReviewerMetrics,
    ReviewerPerformance,
    ReviewSession,
)

__all__ = [
    # Dataset Browser
    "DatasetBrowser",
    "DatasetFilter",
    "DatasetStatistics",
    "PaginationResult",
    "SortField",
    "SortOrder",
    # Annotation Editor
    "AnnotationEditor",
    "ChangeRecord",
    # Diff Viewer
    "AnnotationDiff",
    "AnnotationDiffResult",
    "DiffType",
    "FieldDiff",
    "SegmentDiff",
    # Reviewer Assignment
    "ReviewerAssignment",
    "Reviewer",
    "Assignment",
    # Review Queue
    "ReviewQueue",
    "QueueItem",
    "QueuePriority",
    "QueueItemStatus",
    # Workflow
    "ApprovalWorkflow",
    "WorkflowStage",
    "WorkflowAnnotation",
    "StageTransition",
    "RejectionReason",
    "RejectionRecord",
    # Metrics
    "ReviewerMetrics",
    "ReviewerPerformance",
    "ReviewSession",
]
