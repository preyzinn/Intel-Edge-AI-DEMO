"""Typed errors crossing the model/controller boundary."""

from __future__ import annotations

from edge_ai_demo.model.benchmark_result import Runtime


class BenchmarkError(RuntimeError):
    def __init__(self, runtime: Runtime, stage: str, user_message: str) -> None:
        super().__init__(user_message)
        self.runtime = runtime
        self.stage = stage
        self.user_message = user_message


class UnsupportedDeviceError(BenchmarkError):
    pass


class ModelLoadError(BenchmarkError):
    pass


class ConversionError(BenchmarkError):
    pass


class CompilationError(BenchmarkError):
    pass


class InferenceError(BenchmarkError):
    pass
