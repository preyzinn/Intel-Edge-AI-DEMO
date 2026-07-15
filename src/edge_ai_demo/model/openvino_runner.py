"""Lazy optimum-intel/OpenVINO causal-language-model runner."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from edge_ai_demo.model.benchmark_result import LoadTimings, Runtime
from edge_ai_demo.model.errors import (
    CompilationError,
    ConversionError,
    ModelLoadError,
    UnsupportedDeviceError,
)
from edge_ai_demo.model.model_config import ModelConfig
from edge_ai_demo.model.runner import BaseRunner

LOGGER = logging.getLogger(__name__)


class OpenVINORunner(BaseRunner):
    runtime = Runtime.OPENVINO

    def __init__(
        self,
        model_config: ModelConfig,
        model_cache_dir: Path,
        openvino_cache_dir: Path,
        device: str,
    ) -> None:
        super().__init__(model_config, model_cache_dir, device.upper())
        self.openvino_cache_dir = openvino_cache_dir
        self._torch: Any | None = None

    @staticmethod
    def available_devices() -> tuple[str, ...]:
        try:
            import openvino as ov

            devices = tuple(
                device
                for device in ov.Core().available_devices
                if device.upper().split(".", maxsplit=1)[0] in {"CPU", "GPU", "NPU"}
            )
            return devices
        except Exception as exc:
            LOGGER.warning("OpenVINO device detection failed: %s", exc)
            return ()

    @property
    def converted_model_dir(self) -> Path:
        return self.openvino_cache_dir / self.model_config.cache_key / "model"

    @property
    def compiled_model_cache_dir(self) -> Path:
        safe_device = self.device.lower().replace(".", "-")
        return self.openvino_cache_dir / self.model_config.cache_key / "compiled" / safe_device

    def _converted_model_is_complete(self) -> bool:
        required = (
            "openvino_model.xml",
            "openvino_model.bin",
            "config.json",
            "tokenizer_config.json",
        )
        return all((self.converted_model_dir / filename).is_file() for filename in required)

    def load(self) -> LoadTimings:
        with self._lock:
            if self.is_loaded and self._load_timings is not None:
                return self._load_timings.as_reused()

            available = self.available_devices()
            if self.device not in available:
                detected = ", ".join(available) or "none"
                raise UnsupportedDeviceError(
                    self.runtime,
                    "device",
                    f"OpenVINO device '{self.device}' is unavailable. Detected: {detected}.",
                )

            try:
                import torch
                from optimum.intel import OVModelForCausalLM
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ModelLoadError(
                    self.runtime,
                    "loading",
                    "OpenVINO dependencies are not installed correctly. Run setup again.",
                ) from exc

            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.converted_model_dir.mkdir(parents=True, exist_ok=True)
            self.compiled_model_cache_dir.mkdir(parents=True, exist_ok=True)
            ov_config = {"CACHE_DIR": str(self.compiled_model_cache_dir)}
            cache_hit = self._converted_model_is_complete()
            conversion_seconds = 0.0

            if not cache_hit:
                conversion_start = time.perf_counter()
                try:
                    converted_model = OVModelForCausalLM.from_pretrained(
                        self.model_config.model_id,
                        revision=self.model_config.revision,
                        export=True,
                        compile=False,
                        device=self.device,
                        cache_dir=str(self.model_cache_dir),
                        trust_remote_code=False,
                        load_in_8bit=False,
                        ov_config=ov_config,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.model_id,
                        revision=self.model_config.revision,
                        cache_dir=str(self.model_cache_dir),
                        trust_remote_code=False,
                    )
                    converted_model.save_pretrained(self.converted_model_dir)
                    tokenizer.save_pretrained(self.converted_model_dir)
                    conversion_seconds = time.perf_counter() - conversion_start
                    del converted_model
                    del tokenizer
                    gc.collect()
                except Exception as exc:
                    message = str(exc).lower()
                    if "out of memory" in message or "bad allocation" in message:
                        user_message = (
                            "OpenVINO conversion ran out of memory. Close other applications "
                            "and try again."
                        )
                    else:
                        user_message = (
                            "OpenVINO could not convert the model. Check the internet "
                            "connection, free disk space, and application log."
                        )
                    raise ConversionError(self.runtime, "conversion", user_message) from exc

            loading_start = time.perf_counter()
            try:
                model = OVModelForCausalLM.from_pretrained(
                    self.converted_model_dir,
                    compile=False,
                    device=self.device,
                    local_files_only=True,
                    ov_config=ov_config,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.converted_model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                loading_seconds = time.perf_counter() - loading_start
            except Exception as exc:
                raise ModelLoadError(
                    self.runtime,
                    "loading",
                    "OpenVINO could not load the converted model. The cache may be "
                    "incomplete; check the application log.",
                ) from exc

            compilation_start = time.perf_counter()
            try:
                model.compile()
                compilation_seconds = time.perf_counter() - compilation_start
            except Exception as exc:
                raise CompilationError(
                    self.runtime,
                    "compilation",
                    f"OpenVINO could not compile the model for {self.device}. "
                    "Select another available device.",
                ) from exc

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._load_timings = LoadTimings(
                loading_seconds=loading_seconds,
                conversion_seconds=conversion_seconds,
                compilation_seconds=compilation_seconds,
                converted_model_cache_hit=cache_hit,
            )
            return self._load_timings

    def _prepare_inputs(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return inputs

    def _inference_context(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def _set_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)

    def _synchronize(self) -> None:
        # OVModelForCausalLM.generate returns only after synchronous generation completes.
        return None

    def _release_runtime_resources(self) -> None:
        self._torch = None
        gc.collect()
