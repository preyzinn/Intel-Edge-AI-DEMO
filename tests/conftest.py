import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from edge_ai_demo.config import AppConfig
from edge_ai_demo.model.model_config import DEFAULT_MODEL_ID


@pytest.fixture
def offline_tmp_path() -> Iterator[Path]:
    """Provide a writable temp directory without relying on the host user-temp ACL."""

    root = Path(__file__).parent / ".test-artifacts"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


@pytest.fixture
def app_config(offline_tmp_path: Path) -> AppConfig:
    return AppConfig(
        model_id=DEFAULT_MODEL_ID,
        model_cache_dir=offline_tmp_path / "models",
        openvino_cache_dir=offline_tmp_path / "openvino",
        pytorch_device="cpu",
        openvino_device="CPU",
        max_new_tokens=32,
        benchmark_runs=2,
    )
