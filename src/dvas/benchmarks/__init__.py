"""Benchmark suite for DVAS video annotation platform.

Provides standardized evaluation across multiple dimensions:
- Dataset-specific benchmarks (EPIC-KITCHENS, Ego4D, Open X-Embodiment)
- Synthetic and long-video benchmarks
- Model comparison leaderboards (teacher and student)
- Cost/quality and latency/quality Pareto analysis
- Annotation consistency and human agreement metrics
- Regression tracking and nightly reporting
"""

from dvas.benchmarks.base import BenchmarkResult, BenchmarkSuite
from dvas.benchmarks.epic_kitchens import EPIKitchensBenchmark
from dvas.benchmarks.ego4d import Ego4DBenchmark
from dvas.benchmarks.open_x_embodiment import OpenXEmbodimentBenchmark
from dvas.benchmarks.synthetic_video import SyntheticVideoBenchmark
from dvas.benchmarks.long_video import LongVideoBenchmark
from dvas.benchmarks.teacher_leaderboard import TeacherLeaderboard
from dvas.benchmarks.student_leaderboard import StudentLeaderboard
from dvas.benchmarks.cost_quality_pareto import CostQualityPareto
from dvas.benchmarks.latency_quality_pareto import LatencyQualityPareto
from dvas.benchmarks.annotation_consistency import AnnotationConsistencyBenchmark
from dvas.benchmarks.human_agreement import HumanAgreementBenchmark
from dvas.benchmarks.regression import RegressionBenchmark
from dvas.benchmarks.nightly_report import NightlyBenchmarkReport

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "EPIKitchensBenchmark",
    "Ego4DBenchmark",
    "OpenXEmbodimentBenchmark",
    "SyntheticVideoBenchmark",
    "LongVideoBenchmark",
    "TeacherLeaderboard",
    "StudentLeaderboard",
    "CostQualityPareto",
    "LatencyQualityPareto",
    "AnnotationConsistencyBenchmark",
    "HumanAgreementBenchmark",
    "RegressionBenchmark",
    "NightlyBenchmarkReport",
]
