import pytest

from edge_ai_demo.model.benchmark_result import (
    BenchmarkResult,
    LoadTimings,
    ResourceUsage,
    Runtime,
    SingleRunResult,
)


def _run(
    *,
    text: str = "response",
    token_ids: tuple[int, ...] = (10, 20),
    generated_tokens: int = 2,
    tokenization_seconds: float = 0.1,
    generation_seconds: float = 1.0,
    total_seconds: float = 1.2,
    cpu_percent: float | None = 20.0,
    ram_mb: float | None = 4.0,
) -> SingleRunResult:
    return SingleRunResult(
        generated_text=text,
        input_token_ids=token_ids,
        input_tokens=len(token_ids),
        generated_tokens=generated_tokens,
        tokenization_seconds=tokenization_seconds,
        generation_seconds=generation_seconds,
        total_seconds=total_seconds,
        tokens_per_second=(
            generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        ),
        resource_usage=ResourceUsage(cpu_percent, ram_mb),
    )


def test_benchmark_result_averages_runs_and_formats_stable_fields() -> None:
    first = _run(cpu_percent=None, ram_mb=2.0)
    second = _run(
        text="last response",
        generated_tokens=4,
        tokenization_seconds=0.3,
        generation_seconds=2.0,
        total_seconds=2.4,
        cpu_percent=40.0,
        ram_mb=6.0,
    )
    load_timings = LoadTimings(
        loading_seconds=3.0,
        conversion_seconds=4.0,
        compilation_seconds=5.0,
        converted_model_cache_hit=True,
    )

    result = BenchmarkResult.from_runs(
        runtime=Runtime.OPENVINO,
        device="CPU",
        model_id="example/model",
        model_name="Example",
        model_revision="abc123",
        load_timings=load_timings,
        runs=(first, second),
    )

    assert result.generated_text == "last response"
    assert result.input_tokens == 2
    assert result.generated_tokens == 3.0
    assert result.tokenization_seconds == pytest.approx(0.2)
    assert result.generation_seconds == 1.5
    assert result.total_seconds == pytest.approx(1.8)
    assert result.tokens_per_second == 2.0
    assert result.cpu_usage_percent == 40.0
    assert result.peak_ram_delta_mb == 4.0
    assert result.measured_runs == 2
    assert result.warmup_runs == 1
    assert result.converted_model_cache_hit is True

    formatted = result.to_dict()
    assert formatted["runtime"] == "OpenVINO"
    assert formatted["generated_text"] == "last response"
    assert formatted["measured_runs"] == 2
    assert "runs" not in formatted


def test_benchmark_result_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="At least one measured run"):
        BenchmarkResult.from_runs(
            runtime=Runtime.PYTORCH,
            device="cpu",
            model_id="example/model",
            model_name="Example",
            model_revision="abc123",
            load_timings=LoadTimings(1.0),
            runs=(),
        )


@pytest.mark.parametrize(
    "runs",
    [
        (_run(token_ids=(1, 2)), _run(token_ids=(1, 3))),
        (_run(token_ids=(1, 2)), _run(token_ids=(1, 2, 3))),
    ],
)
def test_benchmark_result_rejects_changed_tokenization(
    runs: tuple[SingleRunResult, SingleRunResult],
) -> None:
    with pytest.raises(ValueError, match="Tokenized input changed"):
        BenchmarkResult.from_runs(
            runtime=Runtime.PYTORCH,
            device="cpu",
            model_id="example/model",
            model_name="Example",
            model_revision="abc123",
            load_timings=LoadTimings(1.0),
            runs=runs,
        )


def test_load_timings_can_be_marked_reused_without_mutating_original() -> None:
    timings = LoadTimings(1.0, conversion_seconds=2.0)

    reused = timings.as_reused()

    assert timings.reused_loaded_model is False
    assert reused.reused_loaded_model is True
    assert reused.loading_seconds == timings.loading_seconds
    assert reused.conversion_seconds == timings.conversion_seconds
