# model/ai_engine.py

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
BASE_DIR = Path(__file__).resolve().parent
HF_MODELS_DIR = BASE_DIR / "hf_models"
OPENVINO_MODELS_DIR = BASE_DIR / "openvino_models"


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "ollama-llama3": {
        "id": "ollama-llama3",
        "family_id": "llama3",
        "family_name": "Llama 3",
        "name": "Llama 3",
        "provider": "ollama",
        "backend": "Ollama",
        "optimized": False,
        "inference_device": "Ollama runtime",
        "model": "llama3",
        "description": "Modelo local servido pelo Ollama.",
    },
    "hf-tinyllama": {
        "id": "hf-tinyllama",
        "family_id": "tinyllama",
        "family_name": "TinyLlama 1.1B Chat",
        "name": "TinyLlama 1.1B Chat",
        "provider": "transformers",
        "backend": "Transformers/PyTorch",
        "optimized": False,
        "inference_device": "CPU",
        "hf_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_dir": HF_MODELS_DIR / "tinyllama-1.1b-chat",
        "description": "Modelo base sem OpenVINO, executado com Transformers/PyTorch.",
    },
    "openvino-tinyllama": {
        "id": "openvino-tinyllama",
        "family_id": "tinyllama",
        "family_name": "TinyLlama 1.1B Chat",
        "name": "TinyLlama 1.1B Chat",
        "provider": "openvino",
        "backend": "OpenVINO",
        "optimized": True,
        "inference_device": "CPU",
        "supported_openvino_devices": ["CPU", "GPU"],
        "unsupported_openvino_devices": {
            "NPU": "Este LLM OpenVINO foi exportado com shapes dinamicos; o compilador NPU exige shapes estaticos para este grafo.",
        },
        "hf_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_dir": OPENVINO_MODELS_DIR / "tinyllama-1.1b-chat",
        "description": "Mesma familia, exportada para inferencia otimizada com OpenVINO.",
    },
    "hf-qwen2.5-0.5b": {
        "id": "hf-qwen2.5-0.5b",
        "family_id": "qwen2.5-0.5b",
        "family_name": "Qwen2.5 0.5B Instruct",
        "name": "Qwen2.5 0.5B Instruct",
        "provider": "transformers",
        "backend": "Transformers/PyTorch",
        "optimized": False,
        "inference_device": "CPU",
        "hf_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": HF_MODELS_DIR / "qwen2.5-0.5b-instruct",
        "description": "Modelo base sem OpenVINO, executado com Transformers/PyTorch.",
    },
    "openvino-qwen2.5-0.5b": {
        "id": "openvino-qwen2.5-0.5b",
        "family_id": "qwen2.5-0.5b",
        "family_name": "Qwen2.5 0.5B Instruct",
        "name": "Qwen2.5 0.5B Instruct",
        "provider": "openvino",
        "backend": "OpenVINO",
        "optimized": True,
        "inference_device": "CPU",
        "supported_openvino_devices": ["CPU", "GPU"],
        "unsupported_openvino_devices": {
            "NPU": "Este LLM OpenVINO foi exportado com shapes dinamicos; o compilador NPU exige shapes estaticos para este grafo.",
        },
        "hf_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": OPENVINO_MODELS_DIR / "qwen2.5-0.5b-instruct",
        "description": "Mesma familia, exportada para inferencia otimizada com OpenVINO.",
    },
    "hf-deepseek-r1-qwen-1.5b": {
        "id": "hf-deepseek-r1-qwen-1.5b",
        "family_id": "deepseek-r1-qwen-1.5b",
        "family_name": "DeepSeek R1 Distill Qwen 1.5B",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "provider": "transformers",
        "backend": "Transformers/PyTorch",
        "optimized": False,
        "inference_device": "CPU",
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_dir": HF_MODELS_DIR / "deepseek-r1-distill-qwen-1.5b",
        "description": "DeepSeek base sem OpenVINO, executado com Transformers/PyTorch.",
    },
    "openvino-deepseek-r1-qwen-1.5b": {
        "id": "openvino-deepseek-r1-qwen-1.5b",
        "family_id": "deepseek-r1-qwen-1.5b",
        "family_name": "DeepSeek R1 Distill Qwen 1.5B",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "provider": "openvino",
        "backend": "OpenVINO",
        "optimized": True,
        "inference_device": "CPU",
        "supported_openvino_devices": ["CPU", "GPU"],
        "unsupported_openvino_devices": {
            "NPU": "Este LLM OpenVINO foi exportado com shapes dinamicos; o compilador NPU exige shapes estaticos para este grafo.",
        },
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_dir": OPENVINO_MODELS_DIR / "deepseek-r1-distill-qwen-1.5b",
        "description": "Mesmo DeepSeek, convertido para execucao otimizada com OpenVINO.",
    },
}

_TRANSFORMERS_CACHE: dict[str, tuple[Any, Any]] = {}
_OPENVINO_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def listar_modelos() -> list[dict[str, Any]]:
    return [_serializar_modelo(model_id) for model_id in MODEL_CATALOG]


def obter_hardware() -> dict[str, Any]:
    return {
        "cpu": _obter_cpu_snapshot(),
        "openvino_available_devices": _obter_dispositivos_openvino(),
    }


def instalar_modelo(model_id: str) -> dict[str, Any]:
    config = _obter_config(model_id)

    if config["provider"] == "ollama":
        _executar_comando(["ollama", "pull", config["model"]])
        return _serializar_modelo(model_id)

    if config["provider"] == "transformers":
        _instalar_transformers(config)
        _TRANSFORMERS_CACHE.pop(model_id, None)
        return _serializar_modelo(model_id)

    _instalar_openvino(config)
    for cache_key in list(_OPENVINO_CACHE):
        if cache_key[0] == model_id:
            _OPENVINO_CACHE.pop(cache_key, None)
    return _serializar_modelo(model_id)


def gerar(
    prompt: str,
    model_id: str = "ollama-llama3",
    inference_device: str | None = None,
) -> dict[str, Any]:
    config = _obter_config(model_id)

    if config["provider"] == "openvino":
        return gerar_openvino(prompt, model_id, inference_device)

    if config["provider"] == "transformers":
        return gerar_transformers(prompt, model_id)

    return gerar_ollama(prompt, model_id)


def gerar_ollama(prompt: str, model_id: str = "ollama-llama3") -> dict[str, Any]:
    config = _obter_config(model_id)
    monitor = _iniciar_monitoramento_cpu()
    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": config["model"],
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()
    latency = time.time() - start
    generated_tokens = data.get("eval_count") or _estimar_tokens(data.get("response", ""))
    tokens_per_second = _calcular_tokens_por_segundo(generated_tokens, latency)

    return _montar_resposta(
        data.get("response", ""),
        latency,
        generated_tokens,
        tokens_per_second,
        model_id,
        _finalizar_monitoramento_cpu(monitor, latency),
    )


def gerar_transformers(prompt: str, model_id: str) -> dict[str, Any]:
    config = _obter_config(model_id)

    if not _modelo_transformers_instalado(config):
        raise RuntimeError(
            f"Modelo '{config['name']}' sem OpenVINO ainda nao esta instalado. "
            "Instale pela sidebar antes de usar."
        )

    model, tokenizer = _carregar_transformers(model_id)
    input_text = _formatar_prompt(tokenizer, prompt)
    inputs = tokenizer(input_text, return_tensors="pt")
    monitor = _iniciar_monitoramento_cpu()
    start = time.time()

    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    latency = time.time() - start
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    generated_tokens = int(generated_ids.shape[1])
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    tokens_per_second = _calcular_tokens_por_segundo(generated_tokens, latency)

    return _montar_resposta(
        text,
        latency,
        generated_tokens,
        tokens_per_second,
        model_id,
        _finalizar_monitoramento_cpu(monitor, latency),
    )


def gerar_openvino(prompt: str, model_id: str, inference_device: str | None = None) -> dict[str, Any]:
    config = _obter_config(model_id)
    device = _normalizar_dispositivo_openvino(inference_device or config["inference_device"])
    _validar_dispositivo_modelo_openvino(config, device)

    if not _modelo_openvino_instalado(config):
        raise RuntimeError(
            f"Modelo '{config['name']}' com OpenVINO ainda nao esta instalado. "
            "Instale pela sidebar antes de usar."
        )

    model, tokenizer = _carregar_openvino(model_id, device)
    input_text = _formatar_prompt(tokenizer, prompt)
    inputs = tokenizer(input_text, return_tensors="pt")
    monitor = _iniciar_monitoramento_cpu()
    start = time.time()

    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    latency = time.time() - start
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    generated_tokens = int(generated_ids.shape[1])
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    tokens_per_second = _calcular_tokens_por_segundo(generated_tokens, latency)

    return _montar_resposta(
        text,
        latency,
        generated_tokens,
        tokens_per_second,
        model_id,
        _finalizar_monitoramento_cpu(monitor, latency),
        device,
    )


def _instalar_transformers(config: dict[str, Any]) -> None:
    _validar_dependencias_transformers()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_dir = Path(config["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["hf_model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config["hf_model_id"], trust_remote_code=True)
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)


def _instalar_openvino(config: dict[str, Any]) -> None:
    _validar_dependencias_openvino()

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer

    local_dir = Path(config["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["hf_model_id"], trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(
        config["hf_model_id"],
        export=True,
        compile=False,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)


def _carregar_transformers(model_id: str) -> tuple[Any, Any]:
    if model_id in _TRANSFORMERS_CACHE:
        return _TRANSFORMERS_CACHE[model_id]

    _validar_dependencias_transformers()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = _obter_config(model_id)
    local_dir = Path(config["local_dir"])

    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, trust_remote_code=True)
    model.eval()
    _TRANSFORMERS_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


def _carregar_openvino(model_id: str, device: str) -> tuple[Any, Any]:
    cache_key = (model_id, device)
    if cache_key in _OPENVINO_CACHE:
        return _OPENVINO_CACHE[cache_key]

    _validar_dependencias_openvino()

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer

    config = _obter_config(model_id)
    local_dir = Path(config["local_dir"])

    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(local_dir, device=device, compile=True)
    _OPENVINO_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def _formatar_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"User: {prompt}\nAssistant:"


def _montar_resposta(
    text: str,
    latency: float,
    generated_tokens: int,
    tokens_per_second: float,
    model_id: str,
    cpu_metrics: dict[str, Any] | None = None,
    inference_device: str | None = None,
) -> dict[str, Any]:
    config = _obter_config(model_id)

    return {
        "text": text,
        "latency": latency,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "model_id": model_id,
        "family_id": config["family_id"],
        "family_name": config["family_name"],
        "backend": config["backend"],
        "optimized": config["optimized"],
        "inference_device": inference_device or config["inference_device"],
        "hardware_metrics": {
            "cpu": cpu_metrics or _obter_cpu_snapshot(),
            "openvino_available_devices": _obter_dispositivos_openvino(),
        },
    }


def _serializar_modelo(model_id: str) -> dict[str, Any]:
    config = MODEL_CATALOG[model_id]
    installed = _modelo_instalado(config)
    data = {
        key: value
        for key, value in config.items()
        if key not in {"local_dir"} and not isinstance(value, Path)
    }
    data["installed"] = installed
    data["install_label"] = "Instalado" if installed else "Instalar"
    return data


def _modelo_instalado(config: dict[str, Any]) -> bool:
    if config["provider"] == "ollama":
        return _modelo_ollama_instalado(config["model"])

    if config["provider"] == "transformers":
        return _modelo_transformers_instalado(config)

    return _modelo_openvino_instalado(config)


def _modelo_ollama_instalado(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

    return result.returncode == 0 and model_name in result.stdout


def _modelo_transformers_instalado(config: dict[str, Any]) -> bool:
    local_dir = Path(config["local_dir"])
    return (local_dir / "config.json").exists()


def _modelo_openvino_instalado(config: dict[str, Any]) -> bool:
    local_dir = Path(config["local_dir"])
    return (local_dir / "openvino_model.xml").exists()


def _obter_config(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_CATALOG:
        raise ValueError(f"Modelo desconhecido: {model_id}")

    return MODEL_CATALOG[model_id]


def _validar_dependencias_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Dependencias Transformers nao instaladas. Execute: "
            "pip install transformers torch"
        ) from exc


def _validar_dependencias_openvino() -> None:
    try:
        import optimum.intel.openvino  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Dependencias OpenVINO nao instaladas. Execute: "
            "pip install openvino optimum-intel transformers torch"
        ) from exc


def _calcular_tokens_por_segundo(generated_tokens: int, latency: float) -> float:
    if latency <= 0:
        return 0.0

    return generated_tokens / latency


def _iniciar_monitoramento_cpu() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        return {
            "process": process,
            "start_times": process.cpu_times(),
            "start": time.time(),
        }
    except Exception:
        return {}


def _finalizar_monitoramento_cpu(monitor: dict[str, Any], latency: float) -> dict[str, Any]:
    snapshot = _obter_cpu_snapshot()
    process = monitor.get("process")
    start_times = monitor.get("start_times")
    start = monitor.get("start")

    if not process or not start_times or not start:
        return snapshot

    try:
        import psutil

        end_times = process.cpu_times()
        elapsed = max(time.time() - start, latency, 0.001)
        cpu_time = (end_times.user - start_times.user) + (end_times.system - start_times.system)
        cpu_count = psutil.cpu_count() or 1
        snapshot["backend_process_cpu_percent"] = round((cpu_time / elapsed / cpu_count) * 100, 1)
    except Exception:
        pass

    return snapshot


def _obter_cpu_snapshot() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "system_cpu_percent": psutil.cpu_percent(interval=0.1),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _obter_dispositivos_openvino() -> list[str]:
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:
        return []


def _normalizar_dispositivo_openvino(device: str) -> str:
    normalized = device.upper().strip()
    available_devices = _obter_dispositivos_openvino()

    if normalized not in available_devices:
        detected = ", ".join(available_devices) if available_devices else "nenhum"
        raise RuntimeError(
            f"Dispositivo OpenVINO indisponivel: {normalized}. "
            f"Detectados: {detected}."
        )

    return normalized


def _validar_dispositivo_modelo_openvino(config: dict[str, Any], device: str) -> None:
    supported_devices = config.get("supported_openvino_devices")
    if not supported_devices or device in supported_devices:
        return

    unsupported_reason = config.get("unsupported_openvino_devices", {}).get(
        device,
        "Este modelo nao foi validado para este dispositivo.",
    )
    supported = ", ".join(supported_devices)
    raise RuntimeError(
        f"{device} foi detectado pelo OpenVINO, mas nao esta habilitado para "
        f"{config['name']} neste app. Motivo: {unsupported_reason} "
        f"Use um destes dispositivos para esta variante: {supported}."
    )


def _estimar_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _executar_comando(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Comando nao encontrado: {command[0]}") from exc

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(error or f"Falha ao executar: {' '.join(command)}")
