"""Small Streamlit result components with no inference logic."""

from __future__ import annotations

import streamlit as st

from edge_ai_demo.model.benchmark_result import (
    BenchmarkComparison,
    BenchmarkReport,
    BenchmarkResult,
    Runtime,
)


def _seconds(value: float) -> str:
    return f"{value:.4f} s"


def _optional(value: float | None, suffix: str) -> str:
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def render_runtime_result(result: BenchmarkResult) -> None:
    st.subheader(result.runtime.value)
    st.caption(
        f"{result.model_name} - {result.device} - average of "
        f"{result.measured_runs} measured runs after {result.warmup_runs} warm-up"
    )

    st.markdown("**Generated response**")
    st.write(result.generated_text or "No text was generated.")

    first_row = st.columns(3)
    first_row[0].metric("Input tokens", str(result.input_tokens))
    first_row[1].metric("Generated tokens", f"{result.generated_tokens:.1f} avg")
    first_row[2].metric("Tokens/second", f"{result.tokens_per_second:.2f}")

    second_row = st.columns(2)
    second_row[0].metric("Model loading (one-time)", _seconds(result.loading_seconds))
    second_row[1].metric("Tokenization (avg/run)", _seconds(result.tokenization_seconds))

    third_row = st.columns(2)
    third_row[0].metric("Generation (avg/run)", _seconds(result.generation_seconds))
    third_row[1].metric("Total (avg/run)", _seconds(result.total_seconds))

    fourth_row = st.columns(2)
    fourth_row[0].metric(
        "Process CPU (avg estimate)",
        _optional(result.cpu_usage_percent, "%"),
    )
    fourth_row[1].metric(
        "Peak process RAM increase (avg estimate)",
        _optional(result.peak_ram_delta_mb, " MiB"),
    )

    if result.runtime is Runtime.OPENVINO:
        with st.expander("OpenVINO preparation timings"):
            st.metric("Conversion (one-time)", _seconds(result.conversion_seconds))
            st.metric("Compilation (one-time)", _seconds(result.compilation_seconds))
            cache_status = "hit" if result.converted_model_cache_hit else "created"
            st.caption(f"Converted-model cache: {cache_status}")

    if result.reused_loaded_model:
        st.caption("The already-loaded model was reused for this comparison.")


def render_runtime_failure(runtime: Runtime, report: BenchmarkReport) -> None:
    st.subheader(runtime.value)
    failure = report.failure_for(runtime)
    if failure is None:
        st.info("No result is available.")
        return
    st.error(failure.message)
    st.caption(f"Failed stage: {failure.stage}")


def _performance_text(comparison: BenchmarkComparison) -> str:
    percentage = comparison.openvino_performance_percent
    if percentage > 0:
        return f"OpenVINO is {percentage:.2f}% faster than PyTorch"
    if percentage < 0:
        return f"OpenVINO is {abs(percentage):.2f}% slower than PyTorch"
    return "No measurable speed difference"


def render_comparison(report: BenchmarkReport) -> None:
    comparison = report.comparison
    pytorch = report.result_for(Runtime.PYTORCH)
    openvino = report.result_for(Runtime.OPENVINO)
    if comparison is None or pytorch is None or openvino is None:
        return

    st.header("Comparison")
    faster = comparison.faster_runtime.value if comparison.faster_runtime is not None else "Tie"
    summary = st.columns(3)
    summary[0].metric("Faster runtime", faster)
    summary[1].metric(
        "Generation-time difference",
        _seconds(comparison.absolute_time_difference_seconds),
    )
    summary[2].metric(comparison.performance_label, _performance_text(comparison))

    differences = st.columns(2)
    differences[0].metric(
        "OpenVINO throughput difference",
        f"{comparison.tokens_per_second_difference:+.2f} tokens/s",
    )
    differences[1].metric(
        "OpenVINO peak RAM difference",
        _optional(comparison.peak_ram_delta_difference_mb, " MiB"),
    )
    st.caption(
        "Throughput and RAM differences are OpenVINO minus PyTorch; a positive RAM "
        "difference means a larger estimated process RSS increase for OpenVINO."
    )

    table_rows = []
    for result in (pytorch, openvino):
        table_rows.append(
            {
                "Runtime": result.runtime.value,
                "Device": result.device,
                "Generation (s)": round(result.generation_seconds, 4),
                "Total (s)": round(result.total_seconds, 4),
                "Tokens/s": round(result.tokens_per_second, 2),
                "CPU estimate (%)": (
                    None if result.cpu_usage_percent is None else round(result.cpu_usage_percent, 2)
                ),
                "Peak RAM increase (MiB)": (
                    None if result.peak_ram_delta_mb is None else round(result.peak_ram_delta_mb, 2)
                ),
            }
        )
    st.table(table_rows)

    chart_columns = st.columns(3)
    with chart_columns[0]:
        st.caption("Average generation time (lower is better)")
        st.bar_chart(
            {
                "Runtime": [Runtime.PYTORCH.value, Runtime.OPENVINO.value],
                "Seconds": [
                    pytorch.generation_seconds,
                    openvino.generation_seconds,
                ],
            },
            x="Runtime",
            y="Seconds",
        )
    with chart_columns[1]:
        st.caption("Throughput (higher is better)")
        st.bar_chart(
            {
                "Runtime": [Runtime.PYTORCH.value, Runtime.OPENVINO.value],
                "Tokens/s": [
                    pytorch.tokens_per_second,
                    openvino.tokens_per_second,
                ],
            },
            x="Runtime",
            y="Tokens/s",
        )
    with chart_columns[2]:
        st.caption("Average peak process RAM increase (estimate)")
        if pytorch.peak_ram_delta_mb is None or openvino.peak_ram_delta_mb is None:
            st.info("RAM sampling was unavailable for one or both runtimes.")
        else:
            st.bar_chart(
                {
                    "Runtime": [Runtime.PYTORCH.value, Runtime.OPENVINO.value],
                    "MiB": [
                        pytorch.peak_ram_delta_mb,
                        openvino.peak_ram_delta_mb,
                    ],
                },
                x="Runtime",
                y="MiB",
            )
