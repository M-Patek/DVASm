"""Student model training and inference."""

from dvas.models.student.benchmark import (
    BenchmarkResult,
    RegressionReport,
    StudentRegressionBenchmark,
    quick_benchmark,
)
from dvas.models.student.calibration import (
    CalibrationMetrics,
    ConfidenceCalibrator,
    ConfidenceThresholdOptimizer,
    TemperatureScaler,
)
from dvas.models.student.config import DPOConfig, SFTConfig
from dvas.models.student.dpo_trainer import train_dpo
from dvas.models.student.evaluation import (
    ComparisonReport,
    CostComparisonResult,
    LatencyComparisonResult,
    QualityComparisonResult,
    TeacherStudentEvaluator,
)
from dvas.models.student.fallback import (
    AdaptiveFallback,
    FallbackStats,
    LowConfidenceFallback,
    create_fallback_router,
)
from dvas.models.student.inference import (
    StudentInferenceEngine,
    StudentTeacherBridge,
    batch_inference,
)
from dvas.models.student.registry import (
    AdapterMetadata,
    LoRAAdapterRegistry,
    ModelArtifactRegistry,
)
from dvas.models.student.selection import (
    ActiveLearningSampler,
    DiversitySampling,
    ExpectedModelChange,
    HybridSelection,
    QueryByCommittee,
    SampleScore,
    SelectionStrategy,
    UncertaintySampling,
    create_strategy,
)
from dvas.models.student.sft_trainer import train_sft

__all__ = [
    # Config
    "SFTConfig",
    "DPOConfig",
    # Training
    "train_sft",
    "train_dpo",
    # Inference
    "StudentInferenceEngine",
    "StudentTeacherBridge",
    "batch_inference",
    # Registry
    "LoRAAdapterRegistry",
    "ModelArtifactRegistry",
    "AdapterMetadata",
    # Calibration
    "ConfidenceCalibrator",
    "TemperatureScaler",
    "CalibrationMetrics",
    "ConfidenceThresholdOptimizer",
    # Selection
    "ActiveLearningSampler",
    "SelectionStrategy",
    "UncertaintySampling",
    "DiversitySampling",
    "ExpectedModelChange",
    "QueryByCommittee",
    "HybridSelection",
    "SampleScore",
    "create_strategy",
    # Evaluation
    "TeacherStudentEvaluator",
    "ComparisonReport",
    "QualityComparisonResult",
    "CostComparisonResult",
    "LatencyComparisonResult",
    # Fallback
    "LowConfidenceFallback",
    "AdaptiveFallback",
    "FallbackStats",
    "create_fallback_router",
    # Benchmark
    "StudentRegressionBenchmark",
    "BenchmarkResult",
    "RegressionReport",
    "quick_benchmark",
]
