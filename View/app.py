import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Intel Edge AI Demo",
    page_icon="⚡",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/chat"

# Estado inicial
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True

# Botão para mostrar/esconder sidebar
if st.button("☰ Menu", use_container_width=False):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# Header minimalista
st.title("Intel Edge AI Assistant")
st.caption("Status: 🟢 Online")

st.divider()

# Sidebar opcional
if st.session_state.show_sidebar:
    with st.sidebar:
        st.header("Configuração")

        st.caption("API URL")
        st.code(API_URL)

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Intel Edge AI Demo")

# Mostra histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "latency" in msg:
            st.caption(f"⏱️ {msg['latency']:.2f}s")

# Input do chat
prompt = st.chat_input("Digite sua pergunta...")

# Enviar mensagem
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            start = time.time()

            try:
                response = requests.post(
                    API_URL,
                    json={"prompt": prompt},
                    timeout=120
                )

                response.raise_for_status()
                data = response.json()

                answer = data.get("response", "Nenhuma resposta retornada.")
                latency = data.get("latency", time.time() - start)

                st.write(answer)
                st.caption(f"⏱️ {latency:.2f}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "latency": latency
                })

            except requests.exceptions.ConnectionError:
                st.error("Não consegui conectar na API. Veja se o FastAPI está rodando na porta 8000.")

            except requests.exceptions.Timeout:
                st.error("A resposta demorou demais e deu timeout.")

            except Exception as e:
                st.error(f"Erro: {e}")