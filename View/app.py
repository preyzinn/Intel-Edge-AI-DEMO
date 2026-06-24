import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address",
            "localhost",
            "--server.port",
            "8501",
        ],
        check=False,
    )
    sys.exit()


st.set_page_config(
    page_title="Intel Edge AI Demo",
    layout="centered",
)

API_BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{API_BASE_URL}/chat"
MODELS_URL = f"{API_BASE_URL}/models"
HARDWARE_URL = f"{API_BASE_URL}/hardware"


def is_backend_online() -> bool:
    try:
        response = requests.get(API_BASE_URL, timeout=2)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def fetch_models() -> list[dict]:
    try:
        response = requests.get(MODELS_URL, timeout=10)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.exceptions.RequestException:
        return []


def fetch_hardware() -> dict:
    try:
        response = requests.get(HARDWARE_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {}


def install_model(model_id: str) -> tuple[bool, str]:
    try:
        response = requests.post(f"{MODELS_URL}/{model_id}/install", timeout=1800)
        if response.ok:
            return True, "Modelo instalado."

        detail = response.json().get("detail", response.text)
        return False, detail
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def install_model_with_progress(model_id: str) -> tuple[bool, str]:
    progress = st.progress(0, text="Preparando instalacao...")
    status = st.empty()
    progress_value = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(install_model, model_id)

        while not future.done():
            if progress_value < 90:
                progress_value += 2

            if progress_value < 25:
                message = "Baixando arquivos do modelo..."
            elif progress_value < 60:
                message = "Salvando pesos e tokenizer..."
            else:
                message = "Convertendo/validando variante local..."

            progress.progress(progress_value, text=message)
            status.caption(message)
            time.sleep(1)

        ok, message = future.result()

    if ok:
        progress.progress(100, text="Instalacao concluida.")
        status.caption("Instalacao concluida.")
    else:
        progress.progress(progress_value, text="Instalacao falhou.")
        status.caption("Instalacao falhou.")

    return ok, message


def install_missing_models_with_progress(missing_models: list[dict]) -> tuple[bool, str]:
    for index, model in enumerate(missing_models, start=1):
        st.caption(f"Instalando {index}/{len(missing_models)}: {model['backend']}")
        ok, message = install_model_with_progress(model["id"])
        if not ok:
            return False, f"{model['backend']}: {message}"

    return True, "Variantes instaladas."


def call_chat(
    prompt: str,
    model_id: str,
    inference_device: str | None = None,
    messages: list[dict] | None = None,
) -> dict:
    payload = {
        "prompt": prompt,
        "model_id": model_id,
        "messages": format_context_messages(messages or []),
    }
    if inference_device:
        payload["inference_device"] = inference_device

    response = requests.post(
        API_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def format_context_messages(messages: list[dict]) -> list[dict[str, str]]:
    context_messages = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            context_messages.append(
                {
                    "role": str(role),
                    "content": str(content),
                }
            )

    return context_messages


def group_models_by_family(models: list[dict]) -> dict[str, list[dict]]:
    families: dict[str, list[dict]] = {}
    for model in models:
        families.setdefault(model["family_id"], []).append(model)
    return families


def format_variant_label(model: dict) -> str:
    if model["provider"] == "openvino":
        return "OpenVINO"

    if model["provider"] == "transformers":
        return "Transformers/PyTorch"

    return model["backend"]


def get_openvino_device_options(model: dict, detected_devices: list[str]) -> tuple[list[str], dict[str, str]]:
    supported_devices = model.get("supported_openvino_devices") or detected_devices or ["CPU"]
    unsupported_devices = model.get("unsupported_openvino_devices", {})
    device_options = [device for device in detected_devices if device in supported_devices]

    if not device_options:
        device_options = [device for device in supported_devices if device in detected_devices]

    if not device_options:
        device_options = ["CPU"]

    return device_options, unsupported_devices


def render_metrics_chart(metrics: list[dict], family_name: str | None) -> None:
    st.subheader("Performance")

    if not metrics:
        st.caption("Execute um prompt para gerar metricas.")
        return

    df = pd.DataFrame(metrics)
    if family_name:
        df = df[df["family"] == family_name]

    if df.empty:
        st.caption("Ainda nao ha metricas para esta familia.")
        return

    latest = df.sort_values("prompt_id").groupby("label", as_index=False).tail(1)

    st.caption("Atualiza a cada prompt da familia selecionada.")

    latency_df = latest.pivot_table(
        index="label",
        values="latency",
        aggfunc="last",
    )

    tokens_df = latest.pivot_table(
        index="label",
        values="tokens_per_second",
        aggfunc="last",
    )

    tokens_col, latency_col = st.columns(2)

    with tokens_col:
        st.caption("Tokens/s")
        st.bar_chart(tokens_df)

    with latency_col:
        st.caption("Latencia")
        st.bar_chart(latency_df)

@st.fragment(run_every="2s")
def render_live_hardware_status(backend_online: bool) -> None:
    st.subheader("Hardware detectado")

    if not backend_online:
        st.caption("Backend offline.")
        return

    hardware = fetch_hardware()
    system_metrics = hardware.get("cpu", {})
    available_devices = hardware.get("openvino_available_devices", [])
    memory_percent = system_metrics.get("memory_percent")

    if memory_percent is None:
        st.metric("Memoria do sistema", "N/D")
    else:
        st.metric("Memoria do sistema", f"{float(memory_percent):.1f}%")

    if available_devices:
        st.caption("OpenVINO detectou: " + ", ".join(available_devices))
    else:
        st.caption("OpenVINO nao retornou dispositivos.")

    st.caption("Atualiza automaticamente.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "prompt_counter" not in st.session_state:
    st.session_state.prompt_counter = 0

if "selected_family_id" not in st.session_state:
    st.session_state.selected_family_id = "deepseek-r1-qwen-1.5b"

if "compare_variants" not in st.session_state:
    st.session_state.compare_variants = False

if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = "openvino-deepseek-r1-qwen-1.5b"

if "selected_inference_device" not in st.session_state:
    st.session_state.selected_inference_device = "CPU"


st.title("Intel Edge AI Assistant")

backend_online = is_backend_online()
status_label = "Online" if backend_online else "Offline"
status_icon = "Online" if backend_online else "Offline"
st.caption(f"Status: {status_icon} - {status_label}")

st.divider()

models = fetch_models() if backend_online else []
hardware = fetch_hardware() if backend_online else {}
openvino_devices = hardware.get("openvino_available_devices", [])
models_by_id = {model["id"]: model for model in models}
families = group_models_by_family(models)

if st.session_state.selected_family_id not in families and families:
    st.session_state.selected_family_id = next(iter(families))

selected_model = None
metrics_container = None

with st.sidebar:
    st.header("Painel de controle")

    st.subheader("Conexao")
    st.caption(f"Backend: {status_label}")
    st.caption("Endpoint de chat")
    st.code(API_URL)

    st.divider()
    st.subheader("Modelo")

    if not backend_online:
        st.warning("Backend offline. Inicie o FastAPI para carregar os modelos.")
    elif not models:
        st.warning("Nenhum modelo retornado pelo backend.")
    else:
        family_options = list(families.keys())
        family_index = family_options.index(st.session_state.selected_family_id)

        selected_family_id = st.selectbox(
            "Familia do modelo",
            options=family_options,
            index=family_index,
            format_func=lambda family_id: families[family_id][0]["family_name"],
            help="Escolha o modelo base usado para responder suas perguntas.",
        )
        st.session_state.selected_family_id = selected_family_id

        family_models = families[selected_family_id]
        has_openvino = any(model["provider"] == "openvino" for model in family_models)
        has_transformers = any(model["provider"] == "transformers" for model in family_models)
        family_model_ids = [model["id"] for model in family_models]

        st.subheader("Modo de execucao")
        if has_openvino and has_transformers:
            st.session_state.compare_variants = st.toggle(
                "Comparar runtime base com OpenVINO",
                value=st.session_state.compare_variants,
                help="Quando ativo, cada prompt roda nas duas variantes para comparar latencia e tokens/s.",
            )
        elif has_openvino:
            st.session_state.compare_variants = False
            st.info("Esta familia so tem variante OpenVINO neste app.")
        else:
            st.session_state.compare_variants = False
            st.info("Esta familia nao tem variante OpenVINO neste app.")

        if st.session_state.selected_model_id not in family_model_ids:
            default_model = next(
                (model for model in family_models if model["provider"] == "openvino"),
                family_models[0],
            )
            st.session_state.selected_model_id = default_model["id"]

        selected_model_id = st.selectbox(
            "Runtime individual",
            options=family_model_ids,
            index=family_model_ids.index(st.session_state.selected_model_id),
            format_func=lambda model_id: format_variant_label(models_by_id[model_id]),
            disabled=st.session_state.compare_variants,
            help="Usado apenas quando a comparacao esta desligada.",
        )
        st.session_state.selected_model_id = selected_model_id
        selected_model = models_by_id[selected_model_id]

        status = "Instalado" if selected_model["installed"] else "Nao instalado"
        acceleration = "OpenVINO" if selected_model["optimized"] else "Nenhuma"
        selected_inference_device = selected_model.get("inference_device", "Nao informado")

        if selected_model["provider"] == "openvino" or (
            st.session_state.compare_variants and has_openvino
        ):
            openvino_model_for_device = (
                selected_model
                if selected_model["provider"] == "openvino"
                else next((model for model in family_models if model["provider"] == "openvino"), selected_model)
            )
            device_options, unsupported_devices = get_openvino_device_options(
                openvino_model_for_device,
                openvino_devices,
            )
            if st.session_state.selected_inference_device not in device_options:
                st.session_state.selected_inference_device = (
                    "CPU" if "CPU" in device_options else device_options[0]
                )

            st.session_state.selected_inference_device = st.selectbox(
                "Dispositivo para OpenVINO",
                options=device_options,
                index=device_options.index(st.session_state.selected_inference_device),
                help="Define onde a variante OpenVINO sera executada.",
            )
            selected_inference_device = st.session_state.selected_inference_device
            unavailable_detected_devices = [
                device
                for device in openvino_devices
                if device in unsupported_devices and device not in device_options
            ]
            if unavailable_detected_devices:
                for device in unavailable_detected_devices:
                    st.warning(
                        f"{device} detectado, mas indisponivel para esta variante: "
                        f"{unsupported_devices[device]}"
                    )
        elif selected_model["provider"] == "transformers":
            selected_inference_device = "CPU"
            st.caption("Transformers/PyTorch neste app esta configurado para CPU.")
        elif selected_model["provider"] == "ollama":
            selected_inference_device = "Ollama runtime"
            st.caption("O device do Ollama e controlado pelo proprio Ollama.")

        comparison_models = [
            model
            for model in family_models
            if model["provider"] in {"transformers", "openvino"}
        ]
        install_targets = (
            [model for model in comparison_models if not model["installed"]]
            if st.session_state.compare_variants
            else ([selected_model] if not selected_model["installed"] else [])
        )

        st.subheader("Configuracao ativa")
        if st.session_state.compare_variants:
            compared_runtimes = ", ".join(model["backend"] for model in comparison_models)
            installed_count = sum(1 for model in comparison_models if model["installed"])
            st.caption("Modo: comparacao")
            st.caption(f"Runtimes executados: {compared_runtimes}")
            st.caption(f"Instalacao: {installed_count}/{len(comparison_models)} variantes prontas")
        else:
            st.caption("Modo: runtime individual")
            st.caption(f"Runtime selecionado: {format_variant_label(selected_model)}")
            st.caption(f"Aceleracao: {acceleration}")
            st.caption(f"Instalacao: {status}")
        st.caption(f"Dispositivo: {selected_inference_device}")
        if openvino_devices:
            st.caption("Dispositivos OpenVINO: " + ", ".join(openvino_devices))
        st.write(selected_model["description"])

        if st.session_state.compare_variants:
            st.info("Comparacao ativa: cada mensagem executa nas variantes base e OpenVINO.")

        if st.session_state.compare_variants:
            missing_labels = [model["backend"] for model in install_targets]
            if missing_labels:
                st.warning("Faltam variantes para comparar: " + ", ".join(missing_labels))

        if install_targets:
            button_label = (
                "Instalar variantes faltantes"
                if st.session_state.compare_variants
                else "Instalar variante selecionada"
            )
            if st.button(button_label, use_container_width=True):
                ok, message = install_missing_models_with_progress(install_targets)

                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    st.divider()

    st.subheader("Performance")
    metrics_container = st.container()

    st.divider()

    st.subheader("Acoes")
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Limpar metricas", use_container_width=True):
        st.session_state.metrics = []
        st.rerun()

    st.divider()
    st.caption("Intel Edge AI Demo")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            if "latency" in msg and "tokens_per_second" in msg:
                st.caption(
                    f"{msg['latency']:.2f}s | "
                    f"{msg['generated_tokens']} tokens | "
                    f"{msg['tokens_per_second']:.2f} tokens/s"
                )
            if "model_name" in msg:
                st.caption(f"Modelo: {msg['model_name']} ({msg.get('backend', '')})")


prompt = st.chat_input("Digite sua pergunta...")

if prompt:
    st.session_state.prompt_counter += 1
    prompt_id = st.session_state.prompt_counter
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if selected_model is None:
            st.error("Nenhum modelo selecionado.")
        elif selected_model and not selected_model["installed"] and not st.session_state.compare_variants:
            st.error("Instale a variante selecionada pela sidebar antes de enviar mensagens.")
        else:
            family_models = families.get(selected_model["family_id"], [selected_model])
            if st.session_state.compare_variants:
                variants_to_run = [
                    model
                    for model in family_models
                    if model["provider"] in {"transformers", "openvino"}
                ]
            else:
                variants_to_run = [selected_model]

            missing = [model for model in variants_to_run if not model["installed"]]
            if missing:
                missing_names = ", ".join(f"{model['backend']}" for model in missing)
                st.error(f"Instale as variantes antes de comparar: {missing_names}.")
                st.stop()

            with st.spinner("Processando..."):
                try:
                    rendered_answers = []
                    response_containers = (
                        st.columns(len(variants_to_run))
                        if len(variants_to_run) > 1
                        else [st.container()]
                    )

                    for variant, response_container in zip(variants_to_run, response_containers):
                        requested_device = (
                            st.session_state.selected_inference_device
                            if variant["provider"] == "openvino"
                            else None
                        )
                        data = call_chat(
                            prompt,
                            variant["id"],
                            requested_device,
                            st.session_state.messages,
                        )
                        answer = data.get("response", "Nenhuma resposta retornada.")
                        latency = float(data.get("latency", 0.0))
                        generated_tokens = int(data.get("generated_tokens", 0))
                        tokens_per_second = float(data.get("tokens_per_second", 0.0))
                        backend = data.get("backend", variant.get("backend", ""))
                        optimized = bool(data.get("optimized", False))
                        family_name = data.get("family_name", variant["family_name"])
                        inference_device = data.get(
                            "inference_device",
                            variant.get("inference_device", "Nao informado"),
                        )
                        hardware_metrics = data.get("hardware_metrics", {})
                        system_metrics = hardware_metrics.get("cpu", {})
                        openvino_devices = hardware_metrics.get("openvino_available_devices", [])
                        label = (
                            f"{backend} ({inference_device})"
                            if optimized
                            else f"{backend} (base)"
                        )

                        with response_container:
                            if len(variants_to_run) > 1:
                                st.markdown(f"**{label}**")
                            st.write(answer)
                            st.caption(
                                f"{latency:.2f}s | "
                                f"{generated_tokens} tokens | "
                                f"{tokens_per_second:.2f} tokens/s"
                            )
                            st.caption(f"Modelo: {family_name} ({backend})")
                            st.caption(f"Dispositivo: {inference_device}")

                        rendered_answers.append(
                            f"{label}\n{answer}\n"
                            f"{latency:.2f}s | {generated_tokens} tokens | "
                            f"{tokens_per_second:.2f} tokens/s | "
                            f"dispositivo: {inference_device}"
                        )
                        st.session_state.metrics.append(
                            {
                                "prompt_id": prompt_id,
                                "family": family_name,
                                "label": label,
                                "latency": latency,
                                "generated_tokens": generated_tokens,
                                "tokens_per_second": tokens_per_second,
                                "inference_device": inference_device,
                                "memory_percent": system_metrics.get("memory_percent"),
                                "openvino_available_devices": openvino_devices,
                            }
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "\n\n".join(rendered_answers),
                            "latency": st.session_state.metrics[-1]["latency"],
                            "generated_tokens": st.session_state.metrics[-1]["generated_tokens"],
                            "tokens_per_second": st.session_state.metrics[-1]["tokens_per_second"],
                            "model_name": selected_model["family_name"],
                            "backend": "comparacao" if len(variants_to_run) > 1 else variants_to_run[0]["backend"],
                        }
                    )

                except requests.exceptions.ConnectionError:
                    st.error("Nao consegui conectar na API. Veja se o FastAPI esta rodando na porta 8000.")

                except requests.exceptions.Timeout:
                    st.error("A resposta demorou demais e deu timeout.")

                except requests.exceptions.HTTPError as e:
                    error_response = getattr(e, "response", None)
                    if error_response is not None:
                        detail = error_response.json().get("detail", error_response.text)
                    else:
                        detail = str(e)
                    st.error(detail)

                except Exception as e:
                    st.error(f"Erro: {e}")


active_family_name = selected_model["family_name"] if selected_model else None
if metrics_container is not None:
    with metrics_container:
        render_metrics_chart(st.session_state.metrics, active_family_name)
        st.divider()
        render_live_hardware_status(backend_online)
