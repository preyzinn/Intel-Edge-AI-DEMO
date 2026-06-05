# model/ai_engine.py

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def gerar(prompt):
    start = time.time()

    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    response.raise_for_status()

    data = response.json()

    latency = time.time() - start

    return {
        "text": data.get("response", ""),
        "latency": latency
    }
