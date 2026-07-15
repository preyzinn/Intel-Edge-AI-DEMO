import pytest

from edge_ai_demo.model.benchmark_result import BenchmarkResult, Runtime
from edge_ai_demo.model.metrics import (
    compare_results,
    device_category,
    openvino_performance_percent,
    tokens_per_second,
)


def _result(
    runtime: Runtime,
    *,
    generation_seconds: float,
    throughput: float,
    device: str,
    peak_ram_mb: float | None,
) -> BenchmarkResult:
    return BenchmarkResult(
        runtime=runtime,
        device=device,
        model_id="example/model",
        model_name="Example",
        model_revision="revision",
        generated_text="text",
        input_tokens=3,
        generated_tokens=2.0,
        loading_seconds=0.1,
        conversion_seconds=0.0,
        compilation_seconds=0.0,
        tokenization_seconds=0.01,
        generation_seconds=generation_seconds,
        total_seconds=generation_seconds + 0.01,
        tokens_per_second=throughput,
        cpu_usage_percent=10.0,
        peak_ram_delta_mb=peak_ram_mb,
        measured_runs=1,
        warmup_runs=1,
        converted_model_cache_hit=False,
        reused_loaded_model=False,
        runs=(),
    )


@pytest.mark.parametrize(
    ("generated_tokens", "generation_seconds", "expected"),
    [(20, 2.0, 10.0), (1, 0.25, 4.0), (5, 0.0, 0.0), (5, -1.0, 0.0)],
)
def test_tokens_per_second(
    generated_tokens: int, generation_seconds: float, expected: float
) -> None:
    assert tokens_per_second(generated_tokens, generation_seconds) == expected


@pytest.mark.parametrize(
    ("pytorch_seconds", "openvino_seconds", "expected"),
    [
        (2.0, 1.0, 100.0),
        (1.0, 2.0, -100.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_openvino_performance_percent(
    pytorch_seconds: float, openvino_seconds: float, expected: float
) -> None:
    assert openvino_performance_percent(pytorch_seconds, openvino_seconds) == expected


@pytest.mark.parametrize(
    ("pytorch_seconds", "openvino_seconds", "winner", "percent"),
    [
        (2.0, 1.0, Runtime.OPENVINO, 100.0),
        (1.0, 2.0, Runtime.PYTORCH, -100.0),
        (1.5, 1.5, None, 0.0),
    ],
)
def test_compare_results_reports_winner_speed_and_differences(
    pytorch_seconds: float,
    openvino_seconds: float,
    winner: Runtime | None,
    percent: float,
) -> None:
    pytorch = _result(
        Runtime.PYTORCH,
        generation_seconds=pytorch_seconds,
        throughput=3.0,
        device="cpu",
        peak_ram_mb=20.0,
    )
    openvino = _result(
        Runtime.OPENVINO,
        generation_seconds=openvino_seconds,
        throughput=7.5,
        device="CPU",
        peak_ram_mb=14.0,
    )

    comparison = compare_results(pytorch, openvino)

    assert comparison.faster_runtime is winner
    assert comparison.absolute_time_difference_seconds == abs(pytorch_seconds - openvino_seconds)
    assert comparison.openvino_performance_percent == percent
    assert comparison.tokens_per_second_difference == 4.5
    assert comparison.peak_ram_delta_difference_mb == -6.0
    assert comparison.device_category_match is True


def test_compare_results_handles_unavailable_memory_and_cross_device_comparison() -> None:
    pytorch = _result(
        Runtime.PYTORCH,
        generation_seconds=1.0,
        throughput=2.0,
        device="cuda",
        peak_ram_mb=None,
    )
    openvino = _result(
        Runtime.OPENVINO,
        generation_seconds=1.0,
        throughput=2.0,
        device="CPU",
        peak_ram_mb=5.0,
    )

    comparison = compare_results(pytorch, openvino)

    assert comparison.peak_ram_delta_difference_mb is None
    assert comparison.device_category_match is False
    assert comparison.performance_label == "No measurable difference"


@pytest.mark.parametrize(
    ("runtime", "device", "expected"),
    [
        (Runtime.PYTORCH, "cpu", "CPU"),
        (Runtime.PYTORCH, "cuda", "GPU"),
        (Runtime.PYTORCH, "xpu", "GPU"),
        (Runtime.OPENVINO, "GPU.0", "GPU"),
        (Runtime.OPENVINO, "NPU", "NPU"),
    ],
)
def test_device_category(runtime: Runtime, device: str, expected: str) -> None:
    assert device_category(runtime, device) == expected
