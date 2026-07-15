"""Model execution and benchmark data structures."""

from edge_ai_demo.model.benchmark_result import (
    BenchmarkComparison,
    BenchmarkReport,
    BenchmarkResult,
    Runtime,
)
from edge_ai_demo.model.model_config import GenerationSettings, ModelConfig

__all__ = [
    "BenchmarkComparison",
    "BenchmarkReport",
    "BenchmarkResult",
    "GenerationSettings",
    "ModelConfig",
    "Runtime",
]
