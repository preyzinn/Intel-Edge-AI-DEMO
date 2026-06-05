import streamlit as st
import requests

st.set_page_config(page_title="Edge AI Assistant", layout="centered")

st.title("🤖 Intel DEMO")

# Input box
prompt = st.text_input("Ask something:")

# Button
if st.button("Send"):
    if prompt:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"prompt": prompt},
                timeout=120
            )
            response.raise_for_status()

            data = response.json()

            st.subheader("Response:")
            st.write(data.get("response", "No response returned."))

            # Optional latency (if you added it)
            if "latency" in data:
                st.caption(f"Latency: {data['latency']:.2f} seconds")

        except Exception as e:
            st.error(f"Error: {e}")
