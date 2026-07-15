import importlib

from edge_ai_demo import AppConfig, BenchmarkController, __version__
from edge_ai_demo.model.openvino_runner import OpenVINORunner
from edge_ai_demo.model.pytorch_runner import PyTorchRunner


def test_package_imports_and_controller_constructs_without_creating_runners(
    app_config: AppConfig,
) -> None:
    def runner_factory_must_not_run(*_args: object) -> None:
        raise AssertionError("Controller construction must not create a model runner")

    controller = BenchmarkController(
        app_config,
        runner_factory=runner_factory_must_not_run,
    )

    assert __version__
    assert controller.config is app_config
    assert controller.models


def test_streamlit_entry_import_does_not_load_or_download_models(
    monkeypatch,
) -> None:
    def model_load_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Importing the entry point must not load a model")

    monkeypatch.setattr(PyTorchRunner, "load", model_load_must_not_run)
    monkeypatch.setattr(OpenVINORunner, "load", model_load_must_not_run)

    app = importlib.import_module("edge_ai_demo.app")
    app = importlib.reload(app)

    assert callable(app.main)
