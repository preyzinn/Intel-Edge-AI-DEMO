from pathlib import Path

import pytest

from edge_ai_demo.config import AppConfig, project_root
from edge_ai_demo.model.model_config import DEFAULT_MODEL_ID


def test_environment_overrides_are_centralized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGE_AI_MODEL_ID", DEFAULT_MODEL_ID)
    monkeypatch.setenv("EDGE_AI_MODEL_CACHE_DIR", "relative-model-cache")
    monkeypatch.setenv("EDGE_AI_OPENVINO_CACHE_DIR", "relative-openvino-cache")
    monkeypatch.setenv("EDGE_AI_PYTORCH_DEVICE", "xpu")
    monkeypatch.setenv("EDGE_AI_OPENVINO_DEVICE", "GPU")
    monkeypatch.setenv("EDGE_AI_MAX_NEW_TOKENS", "96")
    monkeypatch.setenv("EDGE_AI_BENCHMARK_RUNS", "4")

    config = AppConfig.from_env()

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.model_cache_dir == (project_root() / "relative-model-cache").resolve()
    assert config.openvino_cache_dir == (project_root() / "relative-openvino-cache").resolve()
    assert config.pytorch_device == "xpu"
    assert config.openvino_device == "GPU"
    assert config.max_new_tokens == 96
    assert config.benchmark_runs == 4


def test_invalid_environment_values_fall_back_to_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDGE_AI_MODEL_ID", "unsupported/model")
    monkeypatch.setenv("EDGE_AI_MAX_NEW_TOKENS", "zero")
    monkeypatch.setenv("EDGE_AI_BENCHMARK_RUNS", "99")
    monkeypatch.setenv("EDGE_AI_PYTORCH_DEVICE", "")
    monkeypatch.setenv("EDGE_AI_OPENVINO_DEVICE", "")

    config = AppConfig.from_env()

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.max_new_tokens == 64
    assert config.benchmark_runs == 3
    assert config.pytorch_device == "cpu"
    assert config.openvino_device == "CPU"


def test_cache_directories_are_created(offline_tmp_path: Path) -> None:
    config = AppConfig(
        model_id=DEFAULT_MODEL_ID,
        model_cache_dir=offline_tmp_path / "models",
        openvino_cache_dir=offline_tmp_path / "openvino",
        pytorch_device="cpu",
        openvino_device="CPU",
        max_new_tokens=64,
        benchmark_runs=3,
    )

    config.create_cache_directories()

    assert config.model_cache_dir.is_dir()
    assert config.openvino_cache_dir.is_dir()
