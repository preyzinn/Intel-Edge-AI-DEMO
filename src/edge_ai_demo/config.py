"""Small environment-based application configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from edge_ai_demo.model.model_config import DEFAULT_MODEL_ID, get_model_config

LOGGER = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the repository root when running from the source checkout."""

    return Path(__file__).resolve().parents[2]


def _read_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid integer in %s: %r", name, raw_value)
        return default

    if not minimum <= value <= maximum:
        LOGGER.warning(
            "Ignoring out-of-range value in %s: %s (expected %s..%s)",
            name,
            value,
            minimum,
            maximum,
        )
        return default
    return value


def _read_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _read_model_id() -> str:
    model_id = os.getenv("EDGE_AI_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    try:
        get_model_config(model_id)
    except ValueError:
        LOGGER.warning(
            "Ignoring unsupported EDGE_AI_MODEL_ID %r; using %s",
            model_id,
            DEFAULT_MODEL_ID,
        )
        return DEFAULT_MODEL_ID
    return model_id


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime defaults. Environment variables override these values."""

    model_id: str
    model_cache_dir: Path
    openvino_cache_dir: Path
    pytorch_device: str
    openvino_device: str
    max_new_tokens: int
    benchmark_runs: int

    @classmethod
    def from_env(cls) -> AppConfig:
        root = project_root()
        return cls(
            model_id=_read_model_id(),
            model_cache_dir=_read_path("EDGE_AI_MODEL_CACHE_DIR", root / ".cache" / "models"),
            openvino_cache_dir=_read_path(
                "EDGE_AI_OPENVINO_CACHE_DIR", root / ".cache" / "openvino"
            ),
            pytorch_device=os.getenv("EDGE_AI_PYTORCH_DEVICE", "cpu").strip() or "cpu",
            openvino_device=os.getenv("EDGE_AI_OPENVINO_DEVICE", "CPU").strip() or "CPU",
            max_new_tokens=_read_int("EDGE_AI_MAX_NEW_TOKENS", default=64, minimum=1, maximum=512),
            benchmark_runs=_read_int("EDGE_AI_BENCHMARK_RUNS", default=3, minimum=1, maximum=10),
        )

    def create_cache_directories(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.openvino_cache_dir.mkdir(parents=True, exist_ok=True)
