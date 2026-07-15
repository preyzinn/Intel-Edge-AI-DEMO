"""Central model allowlist and generation settings."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    display_name: str
    revision: str

    @property
    def cache_key(self) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", self.model_id.lower()).strip("-")
        return f"{normalized}-{self.revision[:12]}"


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42

    def generate_kwargs(self) -> dict[str, int | float | bool]:
        do_sample = self.temperature > 0.0
        kwargs: dict[str, int | float | bool] = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs


SUPPORTED_MODELS: tuple[ModelConfig, ...] = (
    ModelConfig(
        model_id=DEFAULT_MODEL_ID,
        display_name="Qwen2.5 0.5B Instruct",
        revision=DEFAULT_MODEL_REVISION,
    ),
)


def get_model_config(model_id: str) -> ModelConfig:
    for model in SUPPORTED_MODELS:
        if model.model_id == model_id:
            return model
    supported = ", ".join(model.model_id for model in SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model_id}'. Supported model: {supported}.")
