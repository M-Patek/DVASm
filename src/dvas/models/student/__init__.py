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
from dvas.models.student.checkpoint_resume import (
    cleanup_old_checkpoints,
    find_checkpoints,
    get_latest_checkpoint,
    get_resume_kwargs,
    is_checkpoint_valid,
    load_checkpoint_state,
    resume_from_checkpoint,
    save_checkpoint_state,
)
from dvas.models.student.fsdp_utils import (
    DistributedConfig,
    FSDPConfig,
    cleanup_distributed,
    get_auto_wrap_policy,
    get_backward_prefetch,
    get_mixed_precision_policy,
    get_sharding_strategy,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    setup_distributed,
    wrap_model_with_fsdp,
)
from dvas.models.student.validation_gate import (
    ValidationGate,
    ValidationGateResult,
    run_validation_cli,
)
from dvas.models.student.wandb_tracker import WandBTracker, init_wandb_for_transformers

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
    # Checkpoint Resume
    "find_checkpoints",
    "get_latest_checkpoint",
    "is_checkpoint_valid",
    "load_checkpoint_state",
    "save_checkpoint_state",
    "resume_from_checkpoint",
    "get_resume_kwargs",
    "cleanup_old_checkpoints",
    # FSDP
    "FSDPConfig",
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_with_fsdp",
    "get_sharding_strategy",
    "get_backward_prefetch",
    "get_mixed_precision_policy",
    "get_auto_wrap_policy",
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    # Validation Gate
    "ValidationGate",
    "ValidationGateResult",
    "run_validation_cli",
    # W&B Tracking
    "WandBTracker",
    "init_wandb_for_transformers",
]
