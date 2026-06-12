import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Intel Edge AI Demo",
    layout="centered",
)

API_BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{API_BASE_URL}/chat"
MODELS_URL = f"{API_BASE_URL}/models"


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


def call_chat(prompt: str, model_id: str) -> dict:
    response = requests.post(
        API_URL,
        json={
            "prompt": prompt,
            "model_id": model_id,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def group_models_by_family(models: list[dict]) -> dict[str, list[dict]]:
    families: dict[str, list[dict]] = {}
    for model in models:
        families.setdefault(model["family_id"], []).append(model)
    return families


def pick_model_variant(family_models: list[dict], use_openvino: bool) -> dict:
    preferred_provider = "openvino" if use_openvino else "transformers"
    for model in family_models:
        if model["provider"] == preferred_provider:
            return model

    return family_models[0]


def render_metrics_chart(metrics: list[dict], family_name: str | None) -> None:
    if not metrics:
        st.info("Execute pelo menos uma pergunta para gerar metricas.")
        return

    df = pd.DataFrame(metrics)
    if family_name:
        df = df[df["family"] == family_name]

    if df.empty:
        st.info("Ainda nao ha metricas para a familia selecionada.")
        return

    latest = df.sort_values("prompt_id").groupby("label", as_index=False).tail(1)

    st.subheader("Comparacao de performance")
    st.caption("Execute uma pergunta sem OpenVINO e outra com OpenVINO para comparar as ultimas medicoes.")

    tokens_df = latest.pivot_table(
        index="label",
        values="tokens_per_second",
        aggfunc="last",
    )
    latency_df = latest.pivot_table(
        index="label",
        values="latency",
        aggfunc="last",
    )

    st.write("Tokens por segundo")
    st.bar_chart(tokens_df)

    st.write("Latencia em segundos")
    st.bar_chart(latency_df)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "prompt_counter" not in st.session_state:
    st.session_state.prompt_counter = 0

if "selected_family_id" not in st.session_state:
    st.session_state.selected_family_id = "deepseek-r1-qwen-1.5b"

if "use_openvino" not in st.session_state:
    st.session_state.use_openvino = True

if "compare_variants" not in st.session_state:
    st.session_state.compare_variants = False


st.title("Intel Edge AI Assistant")

backend_online = is_backend_online()
status_label = "Online" if backend_online else "Offline"
status_icon = "Online" if backend_online else "Offline"
st.caption(f"Status: {status_icon} - {status_label}")

st.divider()

models = fetch_models() if backend_online else []
models_by_id = {model["id"]: model for model in models}
families = group_models_by_family(models)

if st.session_state.selected_family_id not in families and families:
    st.session_state.selected_family_id = next(iter(families))

selected_model = None

with st.sidebar:
    st.header("Configuracao")

    st.caption("API URL")
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
            "Escolha a familia",
            options=family_options,
            index=family_index,
            format_func=lambda family_id: families[family_id][0]["family_name"],
        )
        st.session_state.selected_family_id = selected_family_id

        family_models = families[selected_family_id]
        has_openvino = any(model["provider"] == "openvino" for model in family_models)
        has_transformers = any(model["provider"] == "transformers" for model in family_models)

        if has_openvino and has_transformers:
            st.session_state.use_openvino = st.toggle(
                "Usar modelo otimizado com OpenVINO",
                value=st.session_state.use_openvino,
            )
            st.session_state.compare_variants = st.toggle(
                "Comparar sem e com OpenVINO",
                value=st.session_state.compare_variants,
            )
        elif has_openvino:
            st.session_state.use_openvino = True
            st.session_state.compare_variants = False
            st.info("Esta familia so tem variante OpenVINO neste app.")
        else:
            st.session_state.use_openvino = False
            st.session_state.compare_variants = False
            st.info("Esta familia nao tem variante OpenVINO neste app.")

        selected_model = pick_model_variant(family_models, st.session_state.use_openvino)
        status = "Instalado" if selected_model["installed"] else "Nao instalado"
        optimization = "com OpenVINO" if selected_model["optimized"] else "sem OpenVINO"

        st.caption(f"Execucao: {selected_model['backend']} ({optimization})")
        st.caption(f"Status: {status}")
        st.write(selected_model["description"])

        if not selected_model["installed"]:
            if st.button("Instalar variante selecionada", use_container_width=True):
                ok, message = install_model_with_progress(selected_model["id"])

                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    st.divider()

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

                    for variant in variants_to_run:
                        data = call_chat(prompt, variant["id"])
                        answer = data.get("response", "Nenhuma resposta retornada.")
                        latency = float(data.get("latency", 0.0))
                        generated_tokens = int(data.get("generated_tokens", 0))
                        tokens_per_second = float(data.get("tokens_per_second", 0.0))
                        backend = data.get("backend", variant.get("backend", ""))
                        optimized = bool(data.get("optimized", False))
                        family_name = data.get("family_name", variant["family_name"])
                        label = f"{backend} ({'OpenVINO' if optimized else 'base'})"

                        if len(variants_to_run) > 1:
                            st.markdown(f"**{label}**")
                        st.write(answer)
                        st.caption(
                            f"{latency:.2f}s | "
                            f"{generated_tokens} tokens | "
                            f"{tokens_per_second:.2f} tokens/s"
                        )
                        st.caption(f"Modelo: {family_name} ({backend})")

                        rendered_answers.append(
                            f"{label}\n{answer}\n"
                            f"{latency:.2f}s | {generated_tokens} tokens | "
                            f"{tokens_per_second:.2f} tokens/s"
                        )
                        st.session_state.metrics.append(
                            {
                                "prompt_id": prompt_id,
                                "family": family_name,
                                "label": label,
                                "latency": latency,
                                "generated_tokens": generated_tokens,
                                "tokens_per_second": tokens_per_second,
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
render_metrics_chart(st.session_state.metrics, active_family_name)
