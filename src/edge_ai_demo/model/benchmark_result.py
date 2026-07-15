"""Strongly typed results returned by runners and the controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from statistics import fmean


class Runtime(str, Enum):
    PYTORCH = "PyTorch"
    OPENVINO = "OpenVINO"


@dataclass(frozen=True, slots=True)
class LoadTimings:
    """One-time preparation timings for the current loaded runner."""

    loading_seconds: float
    conversion_seconds: float = 0.0
    compilation_seconds: float = 0.0
    converted_model_cache_hit: bool = False
    reused_loaded_model: bool = False

    def as_reused(self) -> LoadTimings:
        return replace(self, reused_loaded_model=True)


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Estimated process resources sampled during generation."""

    cpu_usage_percent: float | None
    peak_ram_delta_mb: float | None


@dataclass(frozen=True, slots=True)
class SingleRunResult:
    generated_text: str
    input_token_ids: tuple[int, ...]
    input_tokens: int
    generated_tokens: int
    tokenization_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_per_second: float
    resource_usage: ResourceUsage


def _mean_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return fmean(available) if available else None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Average metrics for measured runs plus their raw measurements."""

    runtime: Runtime
    device: str
    model_id: str
    model_name: str
    model_revision: str
    generated_text: str
    input_tokens: int
    generated_tokens: float
    loading_seconds: float
    conversion_seconds: float
    compilation_seconds: float
    tokenization_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_per_second: float
    cpu_usage_percent: float | None
    peak_ram_delta_mb: float | None
    measured_runs: int
    warmup_runs: int
    converted_model_cache_hit: bool
    reused_loaded_model: bool
    runs: tuple[SingleRunResult, ...]

    @classmethod
    def from_runs(
        cls,
        *,
        runtime: Runtime,
        device: str,
        model_id: str,
        model_name: str,
        model_revision: str,
        load_timings: LoadTimings,
        runs: tuple[SingleRunResult, ...],
        warmup_runs: int = 1,
    ) -> BenchmarkResult:
        if not runs:
            raise ValueError("At least one measured run is required.")

        input_tokens = {run.input_tokens for run in runs}
        input_token_ids = {run.input_token_ids for run in runs}
        if len(input_tokens) != 1 or len(input_token_ids) != 1:
            raise ValueError("Tokenized input changed between measured runs.")

        generation_seconds = sum(run.generation_seconds for run in runs)
        generated_tokens = sum(run.generated_tokens for run in runs)
        throughput = generated_tokens / generation_seconds if generation_seconds > 0 else 0.0

        return cls(
            runtime=runtime,
            device=device,
            model_id=model_id,
            model_name=model_name,
            model_revision=model_revision,
            generated_text=runs[-1].generated_text,
            input_tokens=input_tokens.pop(),
            generated_tokens=fmean(run.generated_tokens for run in runs),
            loading_seconds=load_timings.loading_seconds,
            conversion_seconds=load_timings.conversion_seconds,
            compilation_seconds=load_timings.compilation_seconds,
            tokenization_seconds=fmean(run.tokenization_seconds for run in runs),
            generation_seconds=fmean(run.generation_seconds for run in runs),
            total_seconds=fmean(run.total_seconds for run in runs),
            tokens_per_second=throughput,
            cpu_usage_percent=_mean_optional(
                [run.resource_usage.cpu_usage_percent for run in runs]
            ),
            peak_ram_delta_mb=_mean_optional(
                [run.resource_usage.peak_ram_delta_mb for run in runs]
            ),
            measured_runs=len(runs),
            warmup_runs=warmup_runs,
            converted_model_cache_hit=load_timings.converted_model_cache_hit,
            reused_loaded_model=load_timings.reused_loaded_model,
            runs=runs,
        )

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return the stable, display-oriented fields without raw run data."""

        return {
            "runtime": self.runtime.value,
            "device": self.device,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "generated_text": self.generated_text,
            "input_tokens": self.input_tokens,
            "generated_tokens": self.generated_tokens,
            "loading_seconds": self.loading_seconds,
            "conversion_seconds": self.conversion_seconds,
            "compilation_seconds": self.compilation_seconds,
            "tokenization_seconds": self.tokenization_seconds,
            "generation_seconds": self.generation_seconds,
            "total_seconds": self.total_seconds,
            "tokens_per_second": self.tokens_per_second,
            "cpu_usage_percent": self.cpu_usage_percent,
            "peak_ram_delta_mb": self.peak_ram_delta_mb,
            "measured_runs": self.measured_runs,
            "warmup_runs": self.warmup_runs,
            "converted_model_cache_hit": self.converted_model_cache_hit,
            "reused_loaded_model": self.reused_loaded_model,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    faster_runtime: Runtime | None
    absolute_time_difference_seconds: float
    openvino_performance_percent: float
    tokens_per_second_difference: float
    peak_ram_delta_difference_mb: float | None
    device_category_match: bool

    @property
    def performance_label(self) -> str:
        if self.openvino_performance_percent > 0:
            return "OpenVINO speedup"
        if self.openvino_performance_percent < 0:
            return "OpenVINO slowdown"
        return "No measurable difference"


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    runtime: Runtime
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    results: tuple[BenchmarkResult, ...]
    comparison: BenchmarkComparison | None
    failures: tuple[RuntimeFailure, ...] = ()
    warnings: tuple[str, ...] = ()

    def result_for(self, runtime: Runtime) -> BenchmarkResult | None:
        return next((result for result in self.results if result.runtime is runtime), None)

    def failure_for(self, runtime: Runtime) -> RuntimeFailure | None:
        return next((failure for failure in self.failures if failure.runtime is runtime), None)
