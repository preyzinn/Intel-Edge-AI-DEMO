# Intel Edge AI Demo

Local application for demonstrating LLM inference in an Edge AI environment. The project combines a Streamlit chat interface, a FastAPI backend, and an AI engine that can run models through Ollama, Transformers/PyTorch, or OpenVINO.

The app focuses on comparing base model variants with OpenVINO-optimized variants, showing latency, tokens per second, CPU usage, memory usage, and detected OpenVINO devices.

## Key Features

- Local chat UI built with Streamlit.
- FastAPI backend with endpoints for chat, models, installation, and hardware.
- Model catalog defined in `Model/ai_engine.py`.
- Ollama support for `llama3`.
- Hugging Face model support with Transformers/PyTorch.
- OpenVINO-exported model support through `optimum-intel`.
- Model installation from the Streamlit sidebar.
- Side-by-side comparison between base and OpenVINO variants.
- OpenVINO device selection when available, such as CPU or GPU.
- Per-prompt performance metric collection.
- PowerShell launcher to start FastAPI and Streamlit together.

## Architecture

```text
edge ai test/
+-- main.py                  # Creates the FastAPI app and registers routes
+-- run_app.ps1              # PowerShell launcher for API + Streamlit
+-- Start_Edge_AI.exe        # Windows executable that calls the launcher
+-- Start_Edge_AI.cs         # Source code for the executable launcher
+-- Controller/
|   +-- chat_controller.py   # API HTTP routes
+-- Model/
|   +-- ai_engine.py         # Model catalog, installation, and inference
|   +-- hf_models/           # Locally downloaded Transformers models
|   +-- openvino_models/     # Locally exported OpenVINO models
+-- View/
    +-- app.py               # Streamlit interface
```

## How It Works

1. The user opens Streamlit at `http://localhost:8501`.
2. The interface calls the backend at `http://127.0.0.1:8000`.
3. The sidebar loads model families through `GET /models`.
4. The sidebar shows detected hardware through `GET /hardware`.
5. If a variant is not installed yet, the user can install it through `POST /models/{model_id}/install`.
6. When a prompt is submitted, Streamlit calls `POST /chat`.
7. The backend calls `gerar()` in `Model/ai_engine.py`.
8. The engine routes inference to Ollama, Transformers/PyTorch, or OpenVINO.
9. The response returns text, latency, generated tokens, tokens/s, backend, device, and hardware metrics.
10. Streamlit displays the answer and updates the performance charts.

## Supported Models

The catalog is defined in `MODEL_CATALOG` inside `Model/ai_engine.py`.

| ID | Family | Backend | Optimized | Configured device |
| --- | --- | --- | --- | --- |
| `ollama-llama3` | Llama 3 | Ollama | No | Ollama runtime |
| `hf-tinyllama` | TinyLlama 1.1B Chat | Transformers/PyTorch | No | CPU |
| `openvino-tinyllama` | TinyLlama 1.1B Chat | OpenVINO | Yes | CPU/GPU |
| `hf-qwen2.5-0.5b` | Qwen2.5 0.5B Instruct | Transformers/PyTorch | No | CPU |
| `openvino-qwen2.5-0.5b` | Qwen2.5 0.5B Instruct | OpenVINO | Yes | CPU/GPU |
| `hf-deepseek-r1-qwen-1.5b` | DeepSeek R1 Distill Qwen 1.5B | Transformers/PyTorch | No | CPU |
| `openvino-deepseek-r1-qwen-1.5b` | DeepSeek R1 Distill Qwen 1.5B | OpenVINO | Yes | CPU/GPU |

NPU note: the app may detect an NPU through OpenVINO, but the current OpenVINO variants block NPU execution because these LLMs were exported with dynamic shapes. `ai_engine.py` reports that the NPU compiler requires static shapes for this graph.

## Requirements

- Windows with PowerShell to use `run_app.ps1` or `Start_Edge_AI.exe`.
- Python 3.10 or newer.
- Python virtual environment recommended.
- Ollama installed if you want to use `ollama-llama3`.
- Internet connection for the first Hugging Face model download.
- Enough disk space for model weights.

Python packages used by the project:

```bash
pip install fastapi uvicorn streamlit requests pydantic pandas psutil transformers torch openvino optimum-intel
```

If you only use Ollama, the OpenVINO/Transformers packages are not required. To use Hugging Face and OpenVINO variants from the sidebar, install the full package set above.

## Environment Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install fastapi uvicorn streamlit requests pydantic pandas psutil transformers torch openvino optimum-intel
```

### 3. Prepare Ollama, optional

Required only for the `ollama-llama3` variant.

```powershell
ollama pull llama3
ollama serve
```

If Ollama is already running as a service, you do not need to run `ollama serve` manually.

## Running the App

### Option A: Windows launcher

From the project root, run:

```powershell
.\run_app.ps1
```

Or open:

```text
Start_Edge_AI.exe
```

The launcher starts:

- FastAPI at `http://127.0.0.1:8000`.
- Streamlit at `http://localhost:8501`.

It also writes logs to `.logs/` and stops child processes when you press `Ctrl+C`.

### Option B: Manual execution

Terminal 1, backend:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2, frontend:

```powershell
streamlit run View/app.py --server.address localhost --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Using the Interface

In the Streamlit sidebar:

- Choose the model family.
- Choose the execution variant: base, OpenVINO, or Ollama.
- Enable `Compare base vs OpenVINO` when the family has both variants.
- Install the selected variant or the missing variants.
- Choose the OpenVINO device when applicable.
- Track tokens/s, latency, and CPU charts.

In the chat:

- Type a prompt.
- Wait for inference.
- Review the answer, time, generated tokens, tokens/s, backend, and device.

## API

### Health check

```http
GET /
```

Response:

```json
{
  "message": "backend ta rodando!!!"
}
```

### List models

```http
GET /models
```

Returns the catalog models with installation status.

### Get hardware

```http
GET /hardware
```

Returns CPU/memory metrics and detected OpenVINO devices.

### Install model

```http
POST /models/{model_id}/install
```

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/models/openvino-qwen2.5-0.5b/install
```

### Send prompt

```http
POST /chat
```

Request example:

```json
{
  "prompt": "Explain Edge AI in one sentence.",
  "model_id": "openvino-qwen2.5-0.5b",
  "inference_device": "CPU"
}
```

Response example:

```json
{
  "response": "Edge AI is the execution of AI models close to where data is generated, reducing latency and cloud dependency.",
  "latency": 3.12,
  "generated_tokens": 32,
  "tokens_per_second": 10.25,
  "model_id": "openvino-qwen2.5-0.5b",
  "family_id": "qwen2.5-0.5b",
  "family_name": "Qwen2.5 0.5B Instruct",
  "backend": "OpenVINO",
  "optimized": true,
  "inference_device": "CPU",
  "hardware_metrics": {
    "cpu": {},
    "openvino_available_devices": ["CPU"]
  }
}
```

## Important Configuration

Frontend URLs (`View/app.py`):

```python
API_BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{API_BASE_URL}/chat"
MODELS_URL = f"{API_BASE_URL}/models"
HARDWARE_URL = f"{API_BASE_URL}/hardware"
```

Ollama URL (`Model/ai_engine.py`):

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
```

Local model directories:

```text
Model/hf_models/
Model/openvino_models/
```

## Troubleshooting

### Backend offline in Streamlit

Start FastAPI:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then test:

```text
http://127.0.0.1:8000
```

### Error when using Ollama

Verify that Ollama is installed, running, and has the model downloaded:

```powershell
ollama list
ollama pull llama3
ollama serve
```

### Hugging Face or OpenVINO model not installed

Install it from the sidebar or call the installation endpoint:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/models/hf-qwen2.5-0.5b/install
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/models/openvino-qwen2.5-0.5b/install
```

### OpenVINO does not detect GPU or NPU

Check the devices returned by:

```text
http://127.0.0.1:8000/hardware
```

Even if an NPU is detected, the current app variants only allow CPU/GPU for OpenVINO because of the dynamic-shape restriction in the exported LLMs.

### Response timeout

The frontend uses a `120` second timeout for `/chat`. Model installation uses a `1800` second timeout. Larger models or the first OpenVINO compilation may take longer.

### Port 8000 or 8501 already in use

`run_app.ps1` attempts to restart old processes that belong to this app. If the port is used by another process, it prints a warning and does not kill the unknown process.

## Main Files

- `View/app.py`: Streamlit interface, model selection, installation, chat, and charts.
- `Controller/chat_controller.py`: HTTP endpoints and error handling.
- `Model/ai_engine.py`: model catalog, local installation, inference, and metrics.
- `main.py`: FastAPI bootstrap.
- `run_app.ps1`: starts and monitors backend and frontend.
- `Start_Edge_AI.exe`: Windows launcher for easier execution.
