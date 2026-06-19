"""Quality loop system for annotation quality control.

This module provides comprehensive quality analysis and feedback systems
for annotations, including automatic analysis, LLM-as-judge evaluation,
review queue management, and trend tracking.
"""

from dvas.quality.acceptance import (
    AcceptanceCriteria,
    AcceptanceCriteriaRegistry,
    AcceptanceGate,
    AcceptanceLevel,
)
from dvas.quality.analyzer import (
    AnomalyDetector,
    DataAugmenter,
    DataDistribution,
    DataQualityAnalyzer,
    DatasetQualityMetrics,
)
from dvas.quality.auto_analyzer import AutomaticQualityAnalyzer
from dvas.quality.llm_judge import LLMJudgeConfig, LLMJudgePipeline, LLMJudgePrompts
from dvas.quality.review_queue import (
    DisagreementCase,
    DisagreementQueue,
    HumanReviewQueue,
    LowQualityQuarantine,
    QuarantineItem,
    ReviewItem,
    ReviewPriority,
    ReviewStatus,
)
from dvas.quality.schema import (
    DimensionScore,
    QualityDimension,
    QualityProfile,
    QualityScores,
    QualityThresholds,
)
from dvas.quality.trend_dashboard import (
    DatasetQualityRollup,
    DimensionTrend,
    ModelQualityRollup,
    QualitySnapshot,
    QualityTrendDashboard,
    TimePeriod,
)

__all__ = [
    # Schema
    "QualityDimension",
    "DimensionScore",
    "QualityScores",
    "QualityThresholds",
    "QualityProfile",
    # Analyzer (existing)
    "AnomalyDetector",
    "DataQualityAnalyzer",
    "DatasetQualityMetrics",
    "DataDistribution",
    "DataAugmenter",
    # Auto Analyzer (new)
    "AutomaticQualityAnalyzer",
    # LLM Judge
    "LLMJudgePipeline",
    "LLMJudgeConfig",
    "LLMJudgePrompts",
    # Review Queue
    "HumanReviewQueue",
    "ReviewItem",
    "ReviewPriority",
    "ReviewStatus",
    "DisagreementQueue",
    "DisagreementCase",
    "LowQualityQuarantine",
    "QuarantineItem",
    # Trend Dashboard
    "QualityTrendDashboard",
    "QualitySnapshot",
    "DimensionTrend",
    "DatasetQualityRollup",
    "ModelQualityRollup",
    "TimePeriod",
    # Acceptance
    "AcceptanceCriteria",
    "AcceptanceGate",
    "AcceptanceLevel",
    "AcceptanceCriteriaRegistry",
]
