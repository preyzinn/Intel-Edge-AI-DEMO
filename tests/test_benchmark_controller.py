from __future__ import annotations

from collections.abc import Mapping

import pytest

import edge_ai_demo.controller.benchmark_controller as controller_module
from edge_ai_demo.config import AppConfig
from edge_ai_demo.controller.benchmark_controller import (
    BenchmarkController,
    BenchmarkRequest,
)
from edge_ai_demo.model.benchmark_result import (
    LoadTimings,
    ResourceUsage,
    Runtime,
    SingleRunResult,
)
from edge_ai_demo.model.errors import ModelLoadError
from edge_ai_demo.model.model_config import (
    DEFAULT_MODEL_ID,
    GenerationSettings,
    ModelConfig,
)


def _request(**changes: object) -> BenchmarkRequest:
    values: dict[str, object] = {
        "prompt": "  Explain edge AI.  ",
        "model_id": DEFAULT_MODEL_ID,
        "max_new_tokens": 24,
        "temperature": 0.0,
        "benchmark_runs": 3,
        "pytorch_device": "CPU",
        "openvino_device": "cpu",
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seed": 42,
    }
    values.update(changes)
    return BenchmarkRequest(**values)  # type: ignore[arg-type]


class FakeRunner:
    def __init__(
        self,
        runtime: Runtime,
        model_config: ModelConfig,
        device: str,
        events: list[tuple[str, Runtime]],
        token_ids: tuple[int, ...],
        fail_loading: bool = False,
    ) -> None:
        self.runtime = runtime
        self.model_config = model_config
        self.device = device
        self.events = events
        self.token_ids = token_ids
        self.fail_loading = fail_loading
        self.load_calls = 0
        self.release_calls = 0
        self.settings: list[GenerationSettings] = []

    def load(self) -> LoadTimings:
        self.events.append(("load", self.runtime))
        self.load_calls += 1
        if self.fail_loading:
            raise ModelLoadError(
                self.runtime,
                "loading",
                f"{self.runtime.value} fake load failure.",
            )
        timings = LoadTimings(loading_seconds=0.25)
        return timings if self.load_calls == 1 else timings.as_reused()

    def run_once(self, settings: GenerationSettings) -> SingleRunResult:
        self.events.append(("run", self.runtime))
        self.settings.append(settings)
        generation_seconds = 2.0 if self.runtime is Runtime.PYTORCH else 1.0
        generated_tokens = 4
        return SingleRunResult(
            generated_text=f"{self.runtime.value} response",
            input_token_ids=self.token_ids,
            input_tokens=len(self.token_ids),
            generated_tokens=generated_tokens,
            tokenization_seconds=0.1,
            generation_seconds=generation_seconds,
            total_seconds=generation_seconds + 0.2,
            tokens_per_second=generated_tokens / generation_seconds,
            resource_usage=ResourceUsage(
                cpu_usage_percent=20.0,
                peak_ram_delta_mb=(10.0 if self.runtime is Runtime.PYTORCH else 7.0),
            ),
        )

    def release(self) -> None:
        self.release_calls += 1


class RecordingFactory:
    def __init__(
        self,
        *,
        token_ids: Mapping[Runtime, tuple[int, ...]] | None = None,
        failing_runtime: Runtime | None = None,
    ) -> None:
        self.events: list[tuple[str, Runtime]] = []
        self.created: dict[Runtime, list[FakeRunner]] = {
            Runtime.PYTORCH: [],
            Runtime.OPENVINO: [],
        }
        self.token_ids = token_ids or {
            Runtime.PYTORCH: (10, 20, 30),
            Runtime.OPENVINO: (10, 20, 30),
        }
        self.failing_runtime = failing_runtime

    def __call__(self, runtime: Runtime, model_config: ModelConfig, device: str) -> FakeRunner:
        runner = FakeRunner(
            runtime,
            model_config,
            device,
            self.events,
            self.token_ids[runtime],
            fail_loading=runtime is self.failing_runtime,
        )
        self.created[runtime].append(runner)
        return runner


def test_compare_uses_equivalent_settings_warmup_runs_and_alternating_order(
    app_config: AppConfig,
) -> None:
    factory = RecordingFactory()
    controller = BenchmarkController(app_config, runner_factory=factory)
    progress: list[tuple[str, float]] = []

    report = controller.compare(
        _request(), progress=lambda text, value: progress.append((text, value))
    )

    assert factory.events == [
        ("load", Runtime.PYTORCH),
        ("load", Runtime.OPENVINO),
        ("run", Runtime.PYTORCH),
        ("run", Runtime.OPENVINO),
        ("run", Runtime.PYTORCH),
        ("run", Runtime.OPENVINO),
        ("run", Runtime.OPENVINO),
        ("run", Runtime.PYTORCH),
        ("run", Runtime.PYTORCH),
        ("run", Runtime.OPENVINO),
    ]
    pytorch_runner = factory.created[Runtime.PYTORCH][0]
    openvino_runner = factory.created[Runtime.OPENVINO][0]
    assert len(pytorch_runner.settings) == 4
    assert len(openvino_runner.settings) == 4
    all_settings = pytorch_runner.settings + openvino_runner.settings
    assert len({id(settings) for settings in all_settings}) == 1
    assert all_settings[0] == GenerationSettings(
        prompt="Explain edge AI.",
        max_new_tokens=24,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=42,
    )
    assert pytorch_runner.device == "cpu"
    assert openvino_runner.device == "CPU"
    assert [result.runtime for result in report.results] == [
        Runtime.PYTORCH,
        Runtime.OPENVINO,
    ]
    assert all(result.measured_runs == 3 for result in report.results)
    assert all(result.warmup_runs == 1 for result in report.results)
    assert report.comparison is not None
    assert report.comparison.faster_runtime is Runtime.OPENVINO
    assert progress[-1] == ("Benchmark complete.", 1.0)
    assert all(0.0 <= fraction <= 1.0 for _, fraction in progress)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt": "   "}, "Enter a prompt"),
        ({"prompt": "x" * 20_001}, "prompt is too long"),
        ({"model_id": "unsupported/model"}, "Unsupported model"),
        ({"max_new_tokens": 0}, "Maximum new tokens"),
        ({"max_new_tokens": 513}, "Maximum new tokens"),
        ({"benchmark_runs": 0}, "Benchmark runs"),
        ({"benchmark_runs": 11}, "Benchmark runs"),
        ({"temperature": -0.1}, "Temperature"),
        ({"temperature": float("inf")}, "Temperature"),
        ({"top_p": 0.0}, "Top-p"),
        ({"top_p": float("nan")}, "Top-p"),
        ({"repetition_penalty": 0.09}, "Repetition penalty"),
        ({"repetition_penalty": 2.01}, "Repetition penalty"),
        ({"seed": -1}, "Seed"),
        ({"seed": 2_147_483_648}, "Seed"),
        ({"pytorch_device": "  "}, "PyTorch device"),
        ({"openvino_device": "  "}, "OpenVINO device"),
    ],
)
def test_invalid_requests_are_rejected_before_runner_creation(
    app_config: AppConfig, changes: dict[str, object], message: str
) -> None:
    def unexpected_factory(
        _runtime: Runtime, _model_config: ModelConfig, _device: str
    ) -> FakeRunner:
        raise AssertionError("Runner factory must not be called for invalid input")

    controller = BenchmarkController(app_config, runner_factory=unexpected_factory)

    with pytest.raises(ValueError, match=message):
        controller.compare(_request(**changes))


def test_controller_caches_runners_until_release(app_config: AppConfig) -> None:
    factory = RecordingFactory()
    controller = BenchmarkController(app_config, runner_factory=factory)
    request = _request(benchmark_runs=1)

    controller.compare(request)
    second_report = controller.compare(request)

    assert len(factory.created[Runtime.PYTORCH]) == 1
    assert len(factory.created[Runtime.OPENVINO]) == 1
    assert factory.created[Runtime.PYTORCH][0].load_calls == 2
    assert factory.created[Runtime.OPENVINO][0].load_calls == 2
    assert all(result.reused_loaded_model for result in second_report.results)

    cached_runners = [
        factory.created[Runtime.PYTORCH][0],
        factory.created[Runtime.OPENVINO][0],
    ]
    controller.release()
    assert all(runner.release_calls == 1 for runner in cached_runners)

    controller.compare(request)
    assert len(factory.created[Runtime.PYTORCH]) == 2
    assert len(factory.created[Runtime.OPENVINO]) == 2


def test_one_runtime_failure_preserves_the_other_runtime_result(
    app_config: AppConfig,
) -> None:
    factory = RecordingFactory(failing_runtime=Runtime.OPENVINO)
    controller = BenchmarkController(app_config, runner_factory=factory)

    report = controller.compare(_request(benchmark_runs=2))

    assert report.result_for(Runtime.PYTORCH) is not None
    assert report.result_for(Runtime.OPENVINO) is None
    assert report.comparison is None
    failure = report.failure_for(Runtime.OPENVINO)
    assert failure is not None
    assert failure.stage == "loading"
    assert failure.message == "OpenVINO fake load failure."
    assert len(factory.created[Runtime.PYTORCH][0].settings) == 3
    assert factory.created[Runtime.OPENVINO][0].settings == []


def test_token_id_mismatch_returns_results_but_omits_comparison(
    app_config: AppConfig,
) -> None:
    factory = RecordingFactory(
        token_ids={
            Runtime.PYTORCH: (10, 20, 30),
            Runtime.OPENVINO: (10, 20, 99),
        }
    )
    controller = BenchmarkController(app_config, runner_factory=factory)

    report = controller.compare(_request(benchmark_runs=1))

    assert len(report.results) == 2
    assert report.comparison is None
    assert len(report.warnings) == 1
    assert "different input token IDs" in report.warnings[0]


def test_create_runner_selects_runtime_and_passes_cache_directories(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple[str, tuple[object, ...]]] = []

    class FakePyTorchRunner:
        def __init__(self, *args: object) -> None:
            created.append(("pytorch", args))

    class FakeOpenVINORunner:
        def __init__(self, *args: object) -> None:
            created.append(("openvino", args))

    monkeypatch.setattr(controller_module, "PyTorchRunner", FakePyTorchRunner)
    monkeypatch.setattr(controller_module, "OpenVINORunner", FakeOpenVINORunner)
    controller = BenchmarkController(app_config)
    model = controller.models[0]

    pytorch = controller._create_runner(Runtime.PYTORCH, model, "cpu")
    openvino = controller._create_runner(Runtime.OPENVINO, model, "CPU")

    assert isinstance(pytorch, FakePyTorchRunner)
    assert isinstance(openvino, FakeOpenVINORunner)
    assert created == [
        ("pytorch", (model, app_config.model_cache_dir, "cpu")),
        (
            "openvino",
            (
                model,
                app_config.model_cache_dir,
                app_config.openvino_cache_dir,
                "CPU",
            ),
        ),
    ]


def test_default_request_uses_central_configuration(app_config: AppConfig) -> None:
    request = BenchmarkController(app_config).default_request("hello")

    assert request == BenchmarkRequest(
        prompt="hello",
        model_id=app_config.model_id,
        max_new_tokens=app_config.max_new_tokens,
        temperature=0.0,
        benchmark_runs=app_config.benchmark_runs,
        pytorch_device=app_config.pytorch_device,
        openvino_device=app_config.openvino_device,
    )
