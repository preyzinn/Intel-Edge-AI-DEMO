"""Coordinate equivalent PyTorch and OpenVINO benchmark executions."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from edge_ai_demo.config import AppConfig
from edge_ai_demo.model.benchmark_result import (
    BenchmarkReport,
    BenchmarkResult,
    LoadTimings,
    Runtime,
    RuntimeFailure,
    SingleRunResult,
)
from edge_ai_demo.model.errors import BenchmarkError
from edge_ai_demo.model.metrics import compare_results
from edge_ai_demo.model.model_config import (
    SUPPORTED_MODELS,
    GenerationSettings,
    ModelConfig,
    get_model_config,
)
from edge_ai_demo.model.openvino_runner import OpenVINORunner
from edge_ai_demo.model.pytorch_runner import PyTorchRunner

LOGGER = logging.getLogger(__name__)


class Runner(Protocol):
    runtime: Runtime
    device: str
    model_config: ModelConfig

    def load(self) -> LoadTimings: ...

    def run_once(self, settings: GenerationSettings) -> SingleRunResult: ...

    def release(self) -> None: ...


RunnerFactory = Callable[[Runtime, ModelConfig, str], Runner]
ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    prompt: str
    model_id: str
    max_new_tokens: int
    temperature: float
    benchmark_runs: int
    pytorch_device: str
    openvino_device: str
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42


class BenchmarkController:
    """Validate, run, and compare the two benchmark runtimes."""

    def __init__(
        self,
        config: AppConfig | None = None,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self._runner_factory = runner_factory or self._create_runner
        self._runners: dict[tuple[Runtime, str, str], Runner] = {}

    @property
    def models(self) -> tuple[ModelConfig, ...]:
        return SUPPORTED_MODELS

    def available_devices(self) -> Mapping[Runtime, tuple[str, ...]]:
        return {
            Runtime.PYTORCH: PyTorchRunner.available_devices(),
            Runtime.OPENVINO: OpenVINORunner.available_devices(),
        }

    def default_request(self, prompt: str = "") -> BenchmarkRequest:
        return BenchmarkRequest(
            prompt=prompt,
            model_id=self.config.model_id,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
            benchmark_runs=self.config.benchmark_runs,
            pytorch_device=self.config.pytorch_device,
            openvino_device=self.config.openvino_device,
        )

    def compare(
        self,
        request: BenchmarkRequest,
        progress: ProgressCallback | None = None,
    ) -> BenchmarkReport:
        model_config, settings = self._validate_request(request)
        progress_callback = progress or (lambda _message, _fraction: None)
        runtime_order = (Runtime.PYTORCH, Runtime.OPENVINO)
        requested_devices = {
            Runtime.PYTORCH: request.pytorch_device.lower(),
            Runtime.OPENVINO: request.openvino_device.upper(),
        }
        measurements: dict[Runtime, list[SingleRunResult]] = {
            runtime: [] for runtime in runtime_order
        }
        runners: dict[Runtime, Runner] = {}
        load_timings: dict[Runtime, LoadTimings] = {}
        failures: dict[Runtime, RuntimeFailure] = {}
        completed_steps = 0
        total_steps = 4 + 2 * request.benchmark_runs

        for runtime in runtime_order:
            progress_callback(f"Loading {runtime.value}...", completed_steps / total_steps)
            try:
                runner = self._get_runner(runtime, model_config, requested_devices[runtime])
                load_timings[runtime] = runner.load()
                runners[runtime] = runner
            except Exception as exc:
                failures[runtime] = self._failure_from_exception(runtime, exc, "loading")
            completed_steps += 1

        for runtime in runtime_order:
            if runtime not in runners:
                completed_steps += 1
                continue
            progress_callback(
                f"Warming up {runtime.value} (unmeasured)...",
                completed_steps / total_steps,
            )
            try:
                runners[runtime].run_once(settings)
            except Exception as exc:
                failures[runtime] = self._failure_from_exception(runtime, exc, "warm-up")
                runners.pop(runtime)
            completed_steps += 1

        for run_index in range(request.benchmark_runs):
            measured_order = runtime_order if run_index % 2 == 0 else tuple(reversed(runtime_order))
            for runtime in measured_order:
                if runtime not in runners:
                    completed_steps += 1
                    continue
                progress_callback(
                    f"{runtime.value}: measured run {run_index + 1}/{request.benchmark_runs}",
                    completed_steps / total_steps,
                )
                try:
                    measurements[runtime].append(runners[runtime].run_once(settings))
                except Exception as exc:
                    failures[runtime] = self._failure_from_exception(runtime, exc, "inference")
                    runners.pop(runtime)
                completed_steps += 1

        results: list[BenchmarkResult] = []
        for runtime in runtime_order:
            runtime_runs = measurements[runtime]
            if len(runtime_runs) != request.benchmark_runs:
                if runtime not in failures and runtime in load_timings:
                    failures[runtime] = RuntimeFailure(
                        runtime,
                        "benchmark",
                        f"{runtime.value} did not complete every measured run.",
                    )
                continue
            results.append(
                BenchmarkResult.from_runs(
                    runtime=runtime,
                    device=requested_devices[runtime],
                    model_id=model_config.model_id,
                    model_name=model_config.display_name,
                    model_revision=model_config.revision,
                    load_timings=load_timings[runtime],
                    runs=tuple(runtime_runs),
                    warmup_runs=1,
                )
            )

        warnings: list[str] = []
        comparison = None
        pytorch_result = next(
            (result for result in results if result.runtime is Runtime.PYTORCH), None
        )
        openvino_result = next(
            (result for result in results if result.runtime is Runtime.OPENVINO), None
        )
        if pytorch_result is not None and openvino_result is not None:
            pytorch_ids = pytorch_result.runs[0].input_token_ids
            openvino_ids = openvino_result.runs[0].input_token_ids
            if pytorch_ids != openvino_ids:
                warnings.append(
                    "The runtimes produced different input token IDs, so no performance "
                    "comparison was calculated. Clear the converted-model cache and retry."
                )
            else:
                comparison = compare_results(pytorch_result, openvino_result)
                if not comparison.device_category_match:
                    warnings.append(
                        "PyTorch and OpenVINO used different device categories. The results "
                        "are shown, but this is a cross-device comparison."
                    )

        progress_callback("Benchmark complete.", 1.0)
        return BenchmarkReport(
            results=tuple(results),
            comparison=comparison,
            failures=tuple(failures[runtime] for runtime in runtime_order if runtime in failures),
            warnings=tuple(warnings),
        )

    def _validate_request(
        self, request: BenchmarkRequest
    ) -> tuple[ModelConfig, GenerationSettings]:
        prompt = request.prompt.strip()
        if not prompt:
            raise ValueError("Enter a prompt before starting the benchmark.")
        if len(prompt) > 20_000:
            raise ValueError("The prompt is too long. Use at most 20,000 characters.")

        model_config = get_model_config(request.model_id)
        if not 1 <= request.max_new_tokens <= 512:
            raise ValueError("Maximum new tokens must be between 1 and 512.")
        if not 1 <= request.benchmark_runs <= 10:
            raise ValueError("Benchmark runs must be between 1 and 10.")
        if not math.isfinite(request.temperature) or not 0 <= request.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not math.isfinite(request.top_p) or not 0 < request.top_p <= 1:
            raise ValueError("Top-p must be greater than 0 and at most 1.")
        if (
            not math.isfinite(request.repetition_penalty)
            or not 0.1 <= request.repetition_penalty <= 2
        ):
            raise ValueError("Repetition penalty must be between 0.1 and 2.")
        if not 0 <= request.seed <= 2_147_483_647:
            raise ValueError("Seed must be between 0 and 2,147,483,647.")
        if not request.pytorch_device.strip():
            raise ValueError("Select a PyTorch device.")
        if not request.openvino_device.strip():
            raise ValueError("Select an OpenVINO device.")

        return model_config, GenerationSettings(
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
        )

    def _get_runner(self, runtime: Runtime, model_config: ModelConfig, device: str) -> Runner:
        cache_key = (runtime, model_config.model_id, device)
        if cache_key not in self._runners:
            self._runners[cache_key] = self._runner_factory(runtime, model_config, device)
        return self._runners[cache_key]

    def _create_runner(self, runtime: Runtime, model_config: ModelConfig, device: str) -> Runner:
        if runtime is Runtime.PYTORCH:
            return PyTorchRunner(model_config, self.config.model_cache_dir, device)
        return OpenVINORunner(
            model_config,
            self.config.model_cache_dir,
            self.config.openvino_cache_dir,
            device,
        )

    @staticmethod
    def _failure_from_exception(
        runtime: Runtime, exc: Exception, default_stage: str
    ) -> RuntimeFailure:
        LOGGER.exception("%s benchmark failed during %s", runtime.value, default_stage)
        if isinstance(exc, BenchmarkError):
            return RuntimeFailure(runtime, exc.stage, exc.user_message)
        return RuntimeFailure(
            runtime,
            default_stage,
            f"{runtime.value} failed unexpectedly. Check the application log for details.",
        )

    def release(self) -> None:
        for runner in self._runners.values():
            runner.release()
        self._runners.clear()
