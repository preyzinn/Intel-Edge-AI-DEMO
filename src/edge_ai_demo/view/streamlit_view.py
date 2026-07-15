"""Single-page Streamlit view for the local benchmark."""

from __future__ import annotations

import logging

import streamlit as st

from edge_ai_demo.config import AppConfig
from edge_ai_demo.controller.benchmark_controller import (
    BenchmarkController,
    BenchmarkRequest,
)
from edge_ai_demo.model.benchmark_result import BenchmarkReport, Runtime
from edge_ai_demo.view.components import (
    render_comparison,
    render_runtime_failure,
    render_runtime_result,
)

LOGGER = logging.getLogger(__name__)


def _controller() -> BenchmarkController:
    if "benchmark_controller" not in st.session_state:
        st.session_state.benchmark_controller = BenchmarkController(AppConfig.from_env())
    return st.session_state.benchmark_controller


def _pick_default(options: tuple[str, ...], configured: str) -> str:
    if configured in options:
        return configured
    configured_lower = configured.lower()
    for option in options:
        if option.lower() == configured_lower:
            return option
    return options[0]


def _initialize_state() -> None:
    defaults = {
        "benchmark_running": False,
        "pending_request": None,
        "benchmark_report": None,
        "benchmark_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _execute_pending(controller: BenchmarkController) -> None:
    request = st.session_state.pending_request
    if request is None:
        st.session_state.benchmark_running = False
        return

    status = st.status(
        "Preparing the benchmark. The first run may download or convert the model...",
        expanded=True,
    )
    progress_bar = st.progress(0.0)

    def update_progress(message: str, fraction: float) -> None:
        status.update(label=message, state="running", expanded=True)
        progress_bar.progress(max(0.0, min(1.0, fraction)))

    try:
        report = controller.compare(request, progress=update_progress)
        st.session_state.benchmark_report = report
        st.session_state.benchmark_error = None
        status.update(label="Benchmark complete.", state="complete", expanded=False)
        progress_bar.progress(1.0)
    except ValueError as exc:
        st.session_state.benchmark_error = str(exc)
        status.update(label="Invalid benchmark settings.", state="error")
    except Exception:
        LOGGER.exception("Unexpected benchmark controller failure")
        st.session_state.benchmark_error = (
            "The benchmark failed unexpectedly. Check the Streamlit console for details."
        )
        status.update(label="Benchmark failed.", state="error")
    finally:
        st.session_state.pending_request = None
        st.session_state.benchmark_running = False

    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="PyTorch vs OpenVINO Benchmark",
        layout="wide",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _initialize_state()
    controller = _controller()
    config = controller.config
    running = bool(st.session_state.benchmark_running)

    st.title("PyTorch vs OpenVINO LLM Benchmark")
    st.write(
        "Run the same model and prompt through both runtimes under equivalent "
        "generation settings. Run metrics are averages; loading, conversion, and "
        "compilation are one-time timings."
    )

    devices = controller.available_devices()
    pytorch_devices = devices[Runtime.PYTORCH]
    openvino_devices = devices[Runtime.OPENVINO]

    with st.container(border=True):
        st.subheader("Runtime status")
        status_columns = st.columns(2)
        status_columns[0].write(
            "**PyTorch devices:** "
            + (", ".join(pytorch_devices) if pytorch_devices else "unavailable")
        )
        status_columns[1].write(
            "**OpenVINO devices:** "
            + (", ".join(openvino_devices) if openvino_devices else "unavailable")
        )
        previous_report = st.session_state.benchmark_report
        if previous_report is None:
            st.caption("Models are not loaded until the first comparison starts.")
        else:
            loaded_names = ", ".join(result.runtime.value for result in previous_report.results)
            st.caption(f"Last successful runtime results: {loaded_names or 'none'}.")

    if not pytorch_devices:
        st.error("PyTorch is unavailable. Run the setup script and restart the app.")
    if not openvino_devices:
        st.error(
            "OpenVINO did not report a CPU, GPU, or NPU device. Run the setup script "
            "or repair the OpenVINO installation."
        )

    selectable_pytorch = pytorch_devices or (config.pytorch_device.lower(),)
    selectable_openvino = openvino_devices or (config.openvino_device.upper(),)
    model_ids = tuple(model.model_id for model in controller.models)
    configured_model = config.model_id if config.model_id in model_ids else model_ids[0]

    with st.form("benchmark_configuration"):
        st.header("Configuration")
        model_id = st.selectbox(
            "Model",
            options=model_ids,
            index=model_ids.index(configured_model),
            format_func=lambda value: next(
                model.display_name for model in controller.models if model.model_id == value
            ),
            disabled=running,
        )
        prompt = st.text_area(
            "Prompt",
            value="Explain edge AI in three concise sentences.",
            height=130,
            disabled=running,
        )

        setting_columns = st.columns(3)
        max_new_tokens = setting_columns[0].number_input(
            "Maximum new tokens",
            min_value=1,
            max_value=512,
            value=config.max_new_tokens,
            step=1,
            disabled=running,
        )
        temperature = setting_columns[1].number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            disabled=running,
        )
        benchmark_runs = setting_columns[2].number_input(
            "Measured benchmark runs",
            min_value=1,
            max_value=10,
            value=config.benchmark_runs,
            step=1,
            disabled=running,
        )

        device_columns = st.columns(2)
        pytorch_device = device_columns[0].selectbox(
            "PyTorch device",
            options=selectable_pytorch,
            index=selectable_pytorch.index(
                _pick_default(selectable_pytorch, config.pytorch_device)
            ),
            disabled=running,
        )
        openvino_device = device_columns[1].selectbox(
            "OpenVINO device",
            options=selectable_openvino,
            index=selectable_openvino.index(
                _pick_default(selectable_openvino, config.openvino_device)
            ),
            disabled=running,
        )

        with st.expander("Advanced generation settings"):
            advanced_columns = st.columns(3)
            top_p = advanced_columns[0].number_input(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                step=0.05,
                disabled=running,
            )
            repetition_penalty = advanced_columns[1].number_input(
                "Repetition penalty",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                disabled=running,
            )
            seed = advanced_columns[2].number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
                disabled=running,
            )

        submitted = st.form_submit_button(
            "Run comparison",
            type="primary",
            use_container_width=True,
            disabled=running or not pytorch_devices or not openvino_devices,
        )

    if st.button(
        "Clear results",
        use_container_width=True,
        disabled=(
            running
            or (
                st.session_state.benchmark_report is None
                and st.session_state.benchmark_error is None
            )
        ),
    ):
        st.session_state.benchmark_report = None
        st.session_state.benchmark_error = None
        st.rerun()

    if submitted:
        st.session_state.benchmark_report = None
        st.session_state.benchmark_error = None
        st.session_state.pending_request = BenchmarkRequest(
            prompt=prompt,
            model_id=model_id,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            benchmark_runs=int(benchmark_runs),
            pytorch_device=pytorch_device,
            openvino_device=openvino_device,
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=int(seed),
        )
        st.session_state.benchmark_running = True
        st.rerun()

    if running:
        _execute_pending(controller)

    if st.session_state.benchmark_error:
        st.error(st.session_state.benchmark_error)

    report: BenchmarkReport | None = st.session_state.benchmark_report
    if report is None:
        return

    for warning in report.warnings:
        st.warning(warning)

    st.header("Results")
    result_columns = st.columns(2)
    for column, runtime in zip(result_columns, (Runtime.PYTORCH, Runtime.OPENVINO), strict=True):
        with column:
            result = report.result_for(runtime)
            if result is not None:
                render_runtime_result(result)
            else:
                render_runtime_failure(runtime, report)

    render_comparison(report)
