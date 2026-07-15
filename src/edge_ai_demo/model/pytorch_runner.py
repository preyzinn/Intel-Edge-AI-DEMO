"""Lazy PyTorch causal-language-model runner."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from edge_ai_demo.model.benchmark_result import LoadTimings, Runtime
from edge_ai_demo.model.errors import ModelLoadError, UnsupportedDeviceError
from edge_ai_demo.model.model_config import ModelConfig
from edge_ai_demo.model.runner import BaseRunner

LOGGER = logging.getLogger(__name__)


class PyTorchRunner(BaseRunner):
    runtime = Runtime.PYTORCH

    def __init__(self, model_config: ModelConfig, model_cache_dir: Path, device: str) -> None:
        super().__init__(model_config, model_cache_dir, device.lower())
        self._torch: Any | None = None
        self._torch_device: Any | None = None

    @staticmethod
    def available_devices() -> tuple[str, ...]:
        try:
            import torch
        except (ImportError, OSError) as exc:
            LOGGER.warning("PyTorch device detection failed: %s", exc)
            return ()

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        return tuple(devices)

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
                    f"PyTorch device '{self.device}' is unavailable. Detected: {detected}.",
                )

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.model_cache_dir.mkdir(parents=True, exist_ok=True)
                loading_start = time.perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config.model_id,
                    revision=self.model_config.revision,
                    cache_dir=str(self.model_cache_dir),
                    trust_remote_code=False,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_id,
                    revision=self.model_config.revision,
                    cache_dir=str(self.model_cache_dir),
                    trust_remote_code=False,
                    torch_dtype="auto",
                )
                torch_device = torch.device(self.device)
                model.to(torch_device)
                model.eval()
                loading_seconds = time.perf_counter() - loading_start
            except Exception as exc:
                message = str(exc).lower()
                if "out of memory" in message or "bad allocation" in message:
                    user_message = (
                        "PyTorch could not load the model because memory is insufficient."
                    )
                else:
                    user_message = (
                        "PyTorch could not load the model. Check the internet connection, "
                        "model cache, and application log."
                    )
                raise ModelLoadError(self.runtime, "loading", user_message) from exc

            self._torch = torch
            self._torch_device = torch_device
            self._model = model
            self._tokenizer = tokenizer
            self._load_timings = LoadTimings(loading_seconds=loading_seconds)
            return self._load_timings

    def _prepare_inputs(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            name: value.to(self._torch_device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }

    def _inference_context(self) -> AbstractContextManager[Any]:
        return self._torch.inference_mode()

    def _set_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)
        if self.device == "cuda":
            self._torch.cuda.manual_seed_all(seed)
        elif self.device == "xpu" and hasattr(self._torch.xpu, "manual_seed_all"):
            self._torch.xpu.manual_seed_all(seed)

    def _synchronize(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize(self._torch_device)
        elif self.device == "xpu":
            self._torch.xpu.synchronize()
        elif self.device == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.synchronize()

    def _release_runtime_resources(self) -> None:
        torch_module = self._torch
        device = self.device
        self._torch = None
        self._torch_device = None
        gc.collect()
        if torch_module is None:
            return
        try:
            if device == "cuda" and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            elif (
                device == "xpu" and hasattr(torch_module, "xpu") and torch_module.xpu.is_available()
            ):
                torch_module.xpu.empty_cache()
        except (AttributeError, RuntimeError) as exc:
            LOGGER.warning("Could not clear the PyTorch device cache: %s", exc)
