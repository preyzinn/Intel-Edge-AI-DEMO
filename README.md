# Intel Edge AI Assistant

A local Edge AI demo that connects a Streamlit chat interface to a FastAPI backend and an Ollama-hosted language model.

The project is organized with a simple MVC-style structure:

- `View/`: Streamlit user interface
- `Controller/`: FastAPI route handlers
- `Model/`: AI engine that calls Ollama

## Features

- Chat-style web UI built with Streamlit
- FastAPI backend with a `/chat` endpoint
- Local inference through Ollama
- Latency reporting for each assistant response
- Conversation history stored in Streamlit session state
- Basic error handling for API connection failures and timeouts

## Project Structure

```text
edge ai test/
+-- main.py
+-- Controller/
|   +-- chat_controller.py
+-- Model/
|   +-- ai_engine.py
+-- View/
    +-- app.py
```

## How It Works

1. The user opens the Streamlit app in the browser.
2. The user sends a message from the chat input.
3. `View/app.py` sends a `POST` request to the FastAPI backend at `http://127.0.0.1:8000/chat`.
4. `Controller/chat_controller.py` receives the JSON payload.
5. The controller calls `gerar()` from `Model/ai_engine.py`.
6. `ai_engine.py` sends the prompt to Ollama at `http://localhost:11434/api/generate`.
7. Ollama returns a model response.
8. FastAPI returns the response and latency to Streamlit.
9. Streamlit displays the answer in the chat UI.

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- The `llama3` model available in Ollama

Python packages:

- `fastapi`
- `uvicorn`
- `streamlit`
- `requests`
- `pydantic`

## Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "edge ai test"
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn streamlit requests pydantic
```

### 4. Install and Start Ollama

Install Ollama from:

```text
https://ollama.com
```

Pull the model used by this project:

```bash
ollama pull llama3
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Running the Project

This project needs two local servers running at the same time:

- FastAPI backend on port `8000`
- Streamlit frontend on port `8501`

### 1. Start the Backend

From the project root:

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

FastAPI interactive docs:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit App

Open a second terminal, activate the same virtual environment, then run:

```bash
streamlit run View/app.py
```

Streamlit URL:

```text
http://localhost:8501
```

## API Reference

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "backend ta rodando!!!"
}
```

### Chat

```http
POST /chat
```

Request body:

```json
{
  "prompt": "What is edge AI?"
}
```

Example response:

```json
{
  "response": "Edge AI means running AI models locally on edge devices instead of relying only on cloud servers.",
  "latency": 1.42
}
```

## Main Files

### `View/app.py`

Contains the Streamlit frontend. It renders the chat interface, stores chat history, sends user prompts to the backend, and displays the assistant response.

### `Controller/chat_controller.py`

Defines the FastAPI `/chat` route. It validates the incoming JSON request and calls the AI engine.

### `Model/ai_engine.py`

Handles communication with Ollama. It sends the prompt to the local Ollama API, waits for a response, measures latency, and returns the generated text.

### `main.py`

Creates the FastAPI app and registers the chat controller routes.

## Configuration

The Streamlit frontend calls this backend URL:

```python
API_URL = "http://127.0.0.1:8000/chat"
```

The AI engine calls this Ollama URL:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
```

The model is currently configured as:

```python
"model": "llama3"
```

To use a different Ollama model, update the `model` value in `Model/ai_engine.py` and make sure the model is installed locally with `ollama pull`.

## Troubleshooting

### Streamlit says it cannot connect to the API

Make sure the FastAPI backend is running:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

### FastAPI returns an Ollama connection error

Make sure Ollama is installed and running:

```bash
ollama serve
```

Then confirm the model exists:

```bash
ollama list
```

If `llama3` is missing, run:

```bash
ollama pull llama3
```

### The request times out

The app currently uses a `120` second timeout. Large models or slow hardware may take longer. You can increase the timeout in:

- `View/app.py`
- `Model/ai_engine.py`

### The model returns an empty response

Check the Ollama response format and confirm the model is working directly:

```bash
ollama run llama3
```

## Development Notes

Run a quick syntax check:

```bash
python -m py_compile main.py Controller/chat_controller.py Model/ai_engine.py View/app.py
```

Suggested optional improvement:

```bash
pip freeze > requirements.txt
```

This makes dependency installation easier for other users:

```bash
pip install -r requirements.txt
```

## License

No license has been specified yet. Add a `LICENSE` file before publishing if you want to define how others may use this project.
