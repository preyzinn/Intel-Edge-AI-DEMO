"""Shared runner mechanics for equivalent tokenization and measurement."""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from edge_ai_demo.model.benchmark_result import (
    LoadTimings,
    Runtime,
    SingleRunResult,
)
from edge_ai_demo.model.errors import InferenceError
from edge_ai_demo.model.metrics import ProcessResourceMonitor, tokens_per_second
from edge_ai_demo.model.model_config import GenerationSettings, ModelConfig


class BaseRunner(ABC):
    """Serialize model access and implement the common measured generation path."""

    runtime: Runtime

    def __init__(self, model_config: ModelConfig, model_cache_dir: Path, device: str) -> None:
        self.model_config = model_config
        self.model_cache_dir = model_cache_dir
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_timings: LoadTimings | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @abstractmethod
    def load(self) -> LoadTimings:
        """Load and prepare the model lazily."""

    def run_once(self, settings: GenerationSettings) -> SingleRunResult:
        """Run one warm or measured generation with exact token accounting."""

        with self._lock:
            if not self.is_loaded:
                raise InferenceError(
                    self.runtime,
                    "inference",
                    f"{self.runtime.value} model is not loaded.",
                )

            total_start = time.perf_counter()
            tokenization_start = time.perf_counter()
            formatted_prompt = self._format_prompt(settings.prompt)
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            tokenization_seconds = time.perf_counter() - tokenization_start

            input_ids = inputs["input_ids"]
            input_token_ids = self._tensor_values(input_ids[0])
            input_tokens = len(input_token_ids)
            prepared_inputs = self._prepare_inputs(inputs)

            random.seed(settings.seed)
            self._set_seed(settings.seed)
            resource_monitor = ProcessResourceMonitor()
            resource_monitor.start()

            try:
                self._synchronize()
                generation_start = time.perf_counter()
                with self._inference_context():
                    output = self._model.generate(
                        **prepared_inputs,
                        **settings.generate_kwargs(),
                    )
                self._synchronize()
                generation_seconds = time.perf_counter() - generation_start
            except Exception as exc:
                resource_monitor.stop()
                message = str(exc).lower()
                if "out of memory" in message or "bad allocation" in message:
                    user_message = (
                        f"{self.runtime.value} ran out of memory during generation. "
                        "Reduce the token limit or close other applications."
                    )
                else:
                    user_message = (
                        f"{self.runtime.value} inference failed. Check the application "
                        "log for technical details."
                    )
                raise InferenceError(self.runtime, "inference", user_message) from exc

            resource_usage = resource_monitor.stop()
            sequences = output.sequences if hasattr(output, "sequences") else output
            generated_ids = sequences[0][input_tokens:]
            generated_token_ids = self._tensor_values(generated_ids)
            generated_tokens = len(generated_token_ids)
            generated_text = self._tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            ).strip()
            total_seconds = time.perf_counter() - total_start

            return SingleRunResult(
                generated_text=generated_text,
                input_token_ids=input_token_ids,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                tokenization_seconds=tokenization_seconds,
                generation_seconds=generation_seconds,
                total_seconds=total_seconds,
                tokens_per_second=tokens_per_second(generated_tokens, generation_seconds),
                resource_usage=resource_usage,
            )

    def _format_prompt(self, prompt: str) -> str:
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @staticmethod
    def _tensor_values(values: Any) -> tuple[int, ...]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return tuple(int(value) for value in values)

    @abstractmethod
    def _prepare_inputs(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def _inference_context(self) -> AbstractContextManager[Any]:
        pass

    @abstractmethod
    def _set_seed(self, seed: int) -> None:
        pass

    @abstractmethod
    def _synchronize(self) -> None:
        pass

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._load_timings = None
            self._release_runtime_resources()

    @abstractmethod
    def _release_runtime_resources(self) -> None:
        pass
