"""DVAS Prompt System — Phase 9 Upgrade.

Comprehensive prompt management with versioning, A/B testing,
auto-selection, few-shot retrieval, and regression testing.
"""

from dvas.prompts.ab_testing import (
    ABTestConfig,
    ABTestMetrics,
    ABTestResult,
    ABTestRunner,
    AssignmentMethod,
)
from dvas.prompts.adaptive import (
    AdaptivePromptEngine,
    ComplexityLevel,
    DynamicPromptOptimizer,
    PromptLibrary,
    PromptTemplate as AdaptivePromptTemplate,
    VideoCategory,
    VideoTypeClassifier,
)
from dvas.prompts.attribution import (
    PromptAttributionRecord,
    PromptAttributionTracker,
    PromptPerformanceSummary,
)
from dvas.prompts.auto_select import (
    AutoSelector,
    DomainDetector,
    VideoCharacteristics,
)
from dvas.prompts.few_shot import (
    Example,
    ExamplePack,
    SemanticExampleIndex,
    create_domain_example_packs,
)
from dvas.prompts.registry import (
    PromptDomain,
    PromptMetadata,
    PromptRegistry,
    PromptTemplate,
)
from dvas.prompts.regression import (
    GoldenAnnotation,
    PromptRegressionTest,
    RegressionResult,
    RegressionStatus,
)
from dvas.prompts.versioning import (
    PromptVersion,
    PromptVersionManager,
    VersionDiff,
    compute_diff,
    is_compatible,
    suggest_version_bump,
)

__all__ = [
    # Registry
    "PromptRegistry",
    "PromptTemplate",
    "PromptMetadata",
    "PromptDomain",
    # Versioning
    "PromptVersion",
    "PromptVersionManager",
    "VersionDiff",
    "compute_diff",
    "is_compatible",
    "suggest_version_bump",
    # A/B Testing
    "ABTestConfig",
    "ABTestRunner",
    "ABTestResult",
    "ABTestMetrics",
    "AssignmentMethod",
    # Attribution
    "PromptAttributionTracker",
    "PromptAttributionRecord",
    "PromptPerformanceSummary",
    # Auto-selection
    "AutoSelector",
    "DomainDetector",
    "VideoCharacteristics",
    # Few-shot
    "SemanticExampleIndex",
    "Example",
    "ExamplePack",
    "create_domain_example_packs",
    # Regression
    "PromptRegressionTest",
    "RegressionResult",
    "RegressionStatus",
    "GoldenAnnotation",
    # Adaptive (existing)
    "AdaptivePromptEngine",
    "AdaptivePromptTemplate",
    "PromptLibrary",
    "VideoTypeClassifier",
    "DynamicPromptOptimizer",
    "VideoCategory",
    "ComplexityLevel",
]
