"""Timing, resource sampling, averaging, and comparison helpers."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import psutil

from edge_ai_demo.model.benchmark_result import (
    BenchmarkComparison,
    BenchmarkResult,
    ResourceUsage,
    Runtime,
)

LOGGER = logging.getLogger(__name__)
MEBIBYTE = 1024 * 1024


def tokens_per_second(generated_tokens: int, generation_seconds: float) -> float:
    if generation_seconds <= 0:
        return 0.0
    return generated_tokens / generation_seconds


def openvino_performance_percent(
    pytorch_generation_seconds: float, openvino_generation_seconds: float
) -> float:
    """Return positive OV speedup or negative OV slowdown versus PyTorch."""

    if pytorch_generation_seconds <= 0 or openvino_generation_seconds <= 0:
        return 0.0
    if openvino_generation_seconds < pytorch_generation_seconds:
        return (pytorch_generation_seconds / openvino_generation_seconds - 1.0) * 100.0
    if openvino_generation_seconds > pytorch_generation_seconds:
        return -(openvino_generation_seconds / pytorch_generation_seconds - 1.0) * 100.0
    return 0.0


def device_category(runtime: Runtime, device: str) -> str:
    normalized = device.upper().split(".", maxsplit=1)[0].split(":", maxsplit=1)[0]
    if runtime is Runtime.PYTORCH and normalized in {"CUDA", "XPU", "MPS"}:
        return "GPU"
    return normalized


def compare_results(pytorch: BenchmarkResult, openvino: BenchmarkResult) -> BenchmarkComparison:
    pytorch_time = pytorch.generation_seconds
    openvino_time = openvino.generation_seconds

    if pytorch_time == openvino_time:
        faster_runtime = None
    elif pytorch_time < openvino_time:
        faster_runtime = Runtime.PYTORCH
    else:
        faster_runtime = Runtime.OPENVINO

    memory_difference = None
    if pytorch.peak_ram_delta_mb is not None and openvino.peak_ram_delta_mb is not None:
        memory_difference = openvino.peak_ram_delta_mb - pytorch.peak_ram_delta_mb

    return BenchmarkComparison(
        faster_runtime=faster_runtime,
        absolute_time_difference_seconds=abs(pytorch_time - openvino_time),
        openvino_performance_percent=openvino_performance_percent(pytorch_time, openvino_time),
        tokens_per_second_difference=(openvino.tokens_per_second - pytorch.tokens_per_second),
        peak_ram_delta_difference_mb=memory_difference,
        device_category_match=(
            device_category(Runtime.PYTORCH, pytorch.device)
            == device_category(Runtime.OPENVINO, openvino.device)
        ),
    )


@dataclass(slots=True)
class _CpuSnapshot:
    user: float
    system: float


class ProcessResourceMonitor:
    """Sample process RSS and estimate normalized CPU use during one generation."""

    def __init__(self, sample_interval_seconds: float = 0.02) -> None:
        self._sample_interval = sample_interval_seconds
        self._process: psutil.Process | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_cpu: _CpuSnapshot | None = None
        self._start_time: float | None = None
        self._start_rss: int | None = None
        self._peak_rss: int | None = None

    def start(self) -> None:
        try:
            self._process = psutil.Process()
            cpu_times = self._process.cpu_times()
            self._start_cpu = _CpuSnapshot(cpu_times.user, cpu_times.system)
            self._start_time = time.perf_counter()
            self._start_rss = self._process.memory_info().rss
            self._peak_rss = self._start_rss
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._sample_until_stopped,
                name="benchmark-resource-monitor",
                daemon=True,
            )
            self._thread.start()
        except (OSError, psutil.Error) as exc:
            LOGGER.warning("Resource monitoring is unavailable: %s", exc)
            self._process = None

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sample_interval):
            self._sample_rss()

    def _sample_rss(self) -> None:
        if self._process is None:
            return
        try:
            rss = self._process.memory_info().rss
            if self._peak_rss is None or rss > self._peak_rss:
                self._peak_rss = rss
        except (OSError, psutil.Error) as exc:
            LOGGER.debug("Could not sample process memory: %s", exc)

    def stop(self) -> ResourceUsage:
        if self._process is None or self._start_time is None:
            return ResourceUsage(None, None)

        self._sample_rss()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self._sample_interval * 3))

        try:
            end_time = time.perf_counter()
            end_cpu = self._process.cpu_times()
            elapsed = max(end_time - self._start_time, 1e-9)
            start_cpu = self._start_cpu
            if start_cpu is None:
                cpu_percent = None
            else:
                cpu_seconds = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
                logical_cpus = psutil.cpu_count(logical=True) or 1
                cpu_percent = max(0.0, min(100.0, cpu_seconds / elapsed / logical_cpus * 100.0))

            if self._start_rss is None or self._peak_rss is None:
                peak_delta_mb = None
            else:
                peak_delta_mb = max(0, self._peak_rss - self._start_rss) / MEBIBYTE
            return ResourceUsage(cpu_percent, peak_delta_mb)
        except (OSError, psutil.Error) as exc:
            LOGGER.warning("Could not finalize resource metrics: %s", exc)
            return ResourceUsage(None, None)
