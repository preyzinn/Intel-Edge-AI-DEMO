from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import edge_ai_demo.model.runner as runner_module
from edge_ai_demo.model.benchmark_result import LoadTimings, ResourceUsage, Runtime
from edge_ai_demo.model.errors import InferenceError, UnsupportedDeviceError
from edge_ai_demo.model.model_config import GenerationSettings, ModelConfig
from edge_ai_demo.model.openvino_runner import OpenVINORunner
from edge_ai_demo.model.pytorch_runner import PyTorchRunner
from edge_ai_demo.model.runner import BaseRunner


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig("example/model", "Example", "1234567890abcdef")


class FakeTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.decoded_ids: tuple[int, ...] | None = None

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, list[list[int]]]:
        assert prompt == "prompt"
        assert return_tensors == "pt"
        return {
            "input_ids": [[101, 102, 103]],
            "attention_mask": [[1, 1, 1]],
        }

    def decode(self, values: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = tuple(values)
        return " generated text "


class FakeGenerateModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(sequences=[[101, 102, 103, 201, 202]])


class InspectableRunner(BaseRunner):
    runtime = Runtime.PYTORCH

    def __init__(self, model_config: ModelConfig, cache_dir: Path, *, loaded: bool) -> None:
        super().__init__(model_config, cache_dir, "cpu")
        self.prepared_inputs: dict[str, Any] | None = None
        self.seed: int | None = None
        self.synchronize_calls = 0
        if loaded:
            self._model = FakeGenerateModel()
            self._tokenizer = FakeTokenizer()
            self._load_timings = LoadTimings(0.1)

    def load(self) -> LoadTimings:
        raise AssertionError("This test runner is loaded explicitly")

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.prepared_inputs = inputs
        return inputs

    def _inference_context(self) -> Any:
        return nullcontext()

    def _set_seed(self, seed: int) -> None:
        self.seed = seed

    def _synchronize(self) -> None:
        self.synchronize_calls += 1

    def _release_runtime_resources(self) -> None:
        return None


class FakeMonitor:
    def start(self) -> None:
        return None

    def stop(self) -> ResourceUsage:
        return ResourceUsage(cpu_usage_percent=12.5, peak_ram_delta_mb=3.0)


def test_base_runner_counts_tokenizer_ids_and_slices_generated_tokens(
    offline_tmp_path: Path,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "ProcessResourceMonitor", FakeMonitor)
    runner = InspectableRunner(model_config, offline_tmp_path, loaded=True)
    settings = GenerationSettings(prompt="prompt", max_new_tokens=2, seed=99)

    result = runner.run_once(settings)

    assert result.input_token_ids == (101, 102, 103)
    assert result.input_tokens == 3
    assert result.generated_tokens == 2
    assert result.generated_text == "generated text"
    assert result.resource_usage == ResourceUsage(12.5, 3.0)
    assert result.tokens_per_second == pytest.approx(
        result.generated_tokens / result.generation_seconds
    )
    assert runner.seed == 99
    assert runner.synchronize_calls == 2
    assert isinstance(runner._tokenizer, FakeTokenizer)
    assert runner._tokenizer.decoded_ids == (201, 202)
    assert isinstance(runner._model, FakeGenerateModel)
    assert runner._model.kwargs == {
        "input_ids": [[101, 102, 103]],
        "attention_mask": [[1, 1, 1]],
        "max_new_tokens": 2,
        "repetition_penalty": 1.0,
        "do_sample": False,
        "use_cache": True,
    }


def test_base_runner_rejects_inference_before_load(
    offline_tmp_path: Path, model_config: ModelConfig
) -> None:
    runner = InspectableRunner(model_config, offline_tmp_path, loaded=False)

    with pytest.raises(InferenceError, match="model is not loaded"):
        runner.run_once(GenerationSettings(prompt="prompt"))


def test_pytorch_runner_loads_lazily_reuses_model_and_releases_resources(
    offline_tmp_path: Path,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
        "tokenizer": [],
        "model": [],
    }

    class LoadedModel:
        def __init__(self) -> None:
            self.device: object | None = None
            self.eval_called = False

        def to(self, device: object) -> None:
            self.device = device

        def eval(self) -> None:
            self.eval_called = True

    loaded_model = LoadedModel()
    loaded_tokenizer = object()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            calls["tokenizer"].append((args, kwargs))
            return loaded_tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> LoadedModel:
            calls["model"].append((args, kwargs))
            return loaded_model

    torch = ModuleType("torch")
    torch.device = lambda name: f"torch-device:{name}"  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(PyTorchRunner, "available_devices", staticmethod(lambda: ("cpu",)))
    cache_dir = offline_tmp_path / "models"
    runner = PyTorchRunner(model_config, cache_dir, "CPU")

    assert runner.is_loaded is False
    first = runner.load()
    second = runner.load()

    assert runner.is_loaded is True
    assert first.reused_loaded_model is False
    assert second.reused_loaded_model is True
    assert len(calls["tokenizer"]) == 1
    assert len(calls["model"]) == 1
    assert calls["tokenizer"][0] == (
        (model_config.model_id,),
        {
            "revision": model_config.revision,
            "cache_dir": str(cache_dir),
            "trust_remote_code": False,
        },
    )
    assert calls["model"][0][1]["torch_dtype"] == "auto"
    assert loaded_model.device == "torch-device:cpu"
    assert loaded_model.eval_called is True
    assert cache_dir.is_dir()

    runner.release()
    assert runner.is_loaded is False
    assert runner._torch is None


def _install_openvino_dependency_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[Any]]:
    state: dict[str, list[Any]] = {
        "model_calls": [],
        "tokenizer_calls": [],
        "compiled_models": [],
    }

    class FakeOVModel:
        def __init__(self, exported: bool) -> None:
            self.exported = exported

        def save_pretrained(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "openvino_model.xml").write_text("xml", encoding="utf-8")
            (directory / "openvino_model.bin").write_bytes(b"bin")
            (directory / "config.json").write_text("{}", encoding="utf-8")

        def compile(self) -> None:
            state["compiled_models"].append(self)

    class OVModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> FakeOVModel:
            state["model_calls"].append((args, kwargs))
            return FakeOVModel(exported=bool(kwargs.get("export")))

    class FakeOVTokenizer:
        def save_pretrained(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> FakeOVTokenizer:
            state["tokenizer_calls"].append((args, kwargs))
            return FakeOVTokenizer()

    torch = ModuleType("torch")
    torch.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    optimum = ModuleType("optimum")
    optimum.__path__ = []  # type: ignore[attr-defined]
    optimum_intel = ModuleType("optimum.intel")
    optimum_intel.OVModelForCausalLM = OVModelForCausalLM  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_intel)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


def test_openvino_runner_converts_once_then_uses_persistent_cache(
    offline_tmp_path: Path,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_openvino_dependency_fakes(monkeypatch)
    monkeypatch.setattr(OpenVINORunner, "available_devices", staticmethod(lambda: ("CPU",)))
    model_cache = offline_tmp_path / "models"
    openvino_cache = offline_tmp_path / "openvino"
    runner = OpenVINORunner(model_config, model_cache, openvino_cache, "cpu")

    first = runner.load()
    reused = runner.load()

    assert first.converted_model_cache_hit is False
    assert first.conversion_seconds >= 0.0
    assert first.compilation_seconds >= 0.0
    assert reused.reused_loaded_model is True
    assert runner.converted_model_dir == (openvino_cache / model_config.cache_key / "model")
    assert runner.compiled_model_cache_dir == (
        openvino_cache / model_config.cache_key / "compiled" / "cpu"
    )
    assert runner._converted_model_is_complete() is True
    assert len(state["model_calls"]) == 2
    export_args, export_kwargs = state["model_calls"][0]
    assert export_args == (model_config.model_id,)
    assert export_kwargs["export"] is True
    assert export_kwargs["compile"] is False
    load_args, load_kwargs = state["model_calls"][1]
    assert load_args == (runner.converted_model_dir,)
    assert load_kwargs["local_files_only"] is True
    assert len(state["compiled_models"]) == 1

    runner.release()
    cached_runner = OpenVINORunner(model_config, model_cache, openvino_cache, "CPU")
    cached = cached_runner.load()

    assert cached.converted_model_cache_hit is True
    assert cached.conversion_seconds == 0.0
    assert len(state["model_calls"]) == 3
    assert sum(bool(call[1].get("export")) for call in state["model_calls"]) == 1
    assert len(state["compiled_models"]) == 2


def test_unsupported_devices_fail_before_runtime_imports(
    offline_tmp_path: Path,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PyTorchRunner, "available_devices", staticmethod(lambda: ("cpu",)))
    pytorch = PyTorchRunner(model_config, offline_tmp_path / "models", "cuda")
    with pytest.raises(UnsupportedDeviceError, match="cuda.*unavailable"):
        pytorch.load()

    monkeypatch.setattr(OpenVINORunner, "available_devices", staticmethod(lambda: ("CPU",)))
    openvino = OpenVINORunner(
        model_config,
        offline_tmp_path / "models",
        offline_tmp_path / "openvino",
        "NPU",
    )
    with pytest.raises(UnsupportedDeviceError, match="NPU.*unavailable"):
        openvino.load()


def test_available_device_detection_filters_and_normalizes_supported_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch.xpu = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert PyTorchRunner.available_devices() == ("cpu", "cuda", "xpu", "mps")

    openvino = ModuleType("openvino")
    openvino.Core = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        available_devices=("CPU", "GPU.0", "NPU", "AUTO", "GNA")
    )
    monkeypatch.setitem(sys.modules, "openvino", openvino)
    assert OpenVINORunner.available_devices() == ("CPU", "GPU.0", "NPU")
