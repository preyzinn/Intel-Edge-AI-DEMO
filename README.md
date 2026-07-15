# Intel Edge AI Demo

Intel Edge AI Demo is a small, local benchmark that runs the same language model through PyTorch and OpenVINO and shows the results side by side. It has one Streamlit process and one benchmark workflow.

> **Screenshot placeholder:** add a screenshot of the completed comparison UI here.

Benchmark results are not universal. They vary with the computer, selected device, model, prompt, generation settings, operating system, driver, and runtime versions.

## What is compared

The application compares two execution paths:

- **PyTorch:** Hugging Face Transformers loads and generates with the native PyTorch model.
- **OpenVINO:** Optimum Intel converts or loads the same pinned model revision, compiles it for an available OpenVINO device, and generates through OpenVINO.

The built-in model is `Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. Pinning avoids silently comparing different upstream model revisions. The allowlist is intentionally small because this project is a benchmark, not a model marketplace.

OpenVINO conversion explicitly sets `load_in_8bit=False` and supplies no quantization configuration. The benchmark therefore does not request implicit weight-only quantization that would confound runtime performance with a different model representation.

Both paths receive the same tokenizer, formatted prompt, maximum new-token count, temperature, top-p value, repetition penalty, and random seed whenever the runtime supports it. Device categories should also match when possible--for example, PyTorch CPU versus OpenVINO CPU. A CPU-versus-NPU result is valid as a hardware comparison, but it is not a runtime-only comparison.

## Architecture

The code uses a lightweight model-view-controller split:

```text
src/
`-- edge_ai_demo/
    |-- app.py
    |-- config.py
    |-- model/
    |   |-- benchmark_result.py
    |   |-- errors.py
    |   |-- metrics.py
    |   |-- model_config.py
    |   |-- openvino_runner.py
    |   |-- pytorch_runner.py
    |   `-- runner.py
    |-- controller/
    |   `-- benchmark_controller.py
    `-- view/
        |-- components.py
        `-- streamlit_view.py
tests/
scripts/
launcher/
Start_Edge_AI.exe
```

The model layer owns loading, inference, token counts, timing, and resource measurements. The controller validates settings, coordinates equivalent runs and warm-up, and calculates comparisons. The Streamlit view only collects input and presents progress, errors, results, tables, and charts.

## Requirements

- A 64-bit Python version from 3.10 through 3.14
- Windows 10/11 with PowerShell, or a current Linux/macOS shell
- Enough free disk space and RAM for both cached model formats
- Internet access for the first dependency installation and first model download

Python 3.12 is a conservative choice when installing on a new machine because AI-package wheels and hardware integrations can lag behind the newest Python release.

## Windows setup

### One-click launcher

Double-click `Start_Edge_AI.exe` in the project root. The launcher verifies the setup fingerprint, dependency consistency, and required imports. If setup is missing, stale, or incomplete, it runs `scripts/setup.ps1`; otherwise it skips installation and starts the Streamlit application immediately through `scripts/run.ps1`.

Keep the launcher window open while the application is running. To use a different port from PowerShell:

```powershell
.\Start_Edge_AI.exe -Port 8502
```

The launcher is a small .NET Framework console application. Its readable source is `launcher/StartEdgeAIDemo.cs`; rebuild it with Windows' built-in compiler after changing the launcher source:

```powershell
.\scripts\build-launcher.ps1
```

No .NET SDK or third-party executable builder is required.

### PowerShell setup

From PowerShell:

```powershell
git clone https://github.com/preyzinn/Intel-Edge-AI-DEMO
cd Intel-Edge-AI-DEMO
.\scripts\setup.ps1
.\scripts\run.ps1
```

The setup script finds a supported Python, creates `.venv`, upgrades pip, installs the application and required dependencies, runs `pip check`, validates imports, creates the cache directories, and prints the run command. It never installs packages from the Streamlit interface.

Use a different Streamlit port if necessary:

```powershell
.\scripts\run.ps1 -Port 8502
```

The run script starts only this Streamlit process. It does not kill or replace a process already using the selected port.

### Manual Windows commands

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run src/edge_ai_demo/app.py
```

Install development tools when you need to run tests or linting:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Linux and macOS setup

```bash
git clone https://github.com/preyzinn/Intel-Edge-AI-DEMO
cd Intel-Edge-AI-DEMO
chmod +x scripts/setup.sh
./scripts/setup.sh
.venv/bin/python -m streamlit run src/edge_ai_demo/app.py
```

Equivalent manual commands are:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m streamlit run src/edge_ai_demo/app.py
```

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`.

## Configuration

Configuration uses environment variables directly; no `.env` parser is required. Unset variables use safe local defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `EDGE_AI_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Selects a compatible allowlisted model. |
| `EDGE_AI_MODEL_CACHE_DIR` | `.cache/models` | Stores Hugging Face tokenizer and PyTorch model files. |
| `EDGE_AI_OPENVINO_CACHE_DIR` | `.cache/openvino` | Stores converted and compiled OpenVINO artifacts. |
| `EDGE_AI_PYTORCH_DEVICE` | `cpu` | Default PyTorch device. |
| `EDGE_AI_OPENVINO_DEVICE` | `CPU` | Default OpenVINO device. |
| `EDGE_AI_MAX_NEW_TOKENS` | `64` | Default generation limit. |
| `EDGE_AI_BENCHMARK_RUNS` | `3` | Number of measured runs after warm-up. |

For one PowerShell session, set a value before starting the app:

```powershell
$env:EDGE_AI_BENCHMARK_RUNS = "5"
.\scripts\run.ps1
```

Relative cache paths are resolved from the project directory. Keep caches out of Git; model weights and generated OpenVINO files can be large.

## Benchmark methodology

For each runtime the controller follows the same sequence:

1. Validate the prompt, selected model, devices, and generation parameters.
2. Lazily load the tokenizer and runtime model. OpenVINO converts once when its cached representation is absent and reuses the converted/compiled model afterward.
3. Run one complete, unmeasured warm-up generation.
4. Run the configured number of measured generations, alternating which runtime goes first on successive runs to reduce order and thermal bias.
5. Average measured values and compare the exact input token IDs before calculating a performance comparison. If the token IDs differ, both raw results remain visible but the comparison is withheld.

Model download, OpenVINO conversion, model loading/compilation, and warm-up are excluded from measured generation time. Loading is reported separately. Timers use `time.perf_counter()`. PyTorch synchronizes an accelerator before stopping its generation timer when the selected backend requires synchronization.

On an empty cache, the first PyTorch loading measurement can include the initial model download. That time is still isolated from tokenization, generation, throughput, and warm-model total time. Populate both caches before comparing cold-load timings.

The generated response shown for a runtime is from a measured run. With sampling enabled, seeded generation is used where the runtime supports it; runtime kernels can still produce small output differences.

## Metrics

| Metric | Meaning |
| --- | --- |
| Generated response | Decoded text from a measured generation. |
| Input tokens | Exact prompt token count produced by the shared tokenizer. |
| Generated tokens | Average output-token count across measured runs, excluding prompt token IDs. This is not a character or word estimate. |
| Model loading time | One-time lazy tokenizer/model load. For PyTorch this includes moving the model to its device; for OpenVINO it is the local IR/tokenizer load. It is not averaged into warm-model execution. |
| OpenVINO conversion time | One-time conversion and cache-write duration when no complete converted model is cached; zero on a converted-model cache hit. |
| OpenVINO compilation time | Time to compile the loaded OpenVINO model for the selected device, reported separately from loading and generation. |
| Tokenization time | Average measured prompt-formatting and tokenization duration. |
| Generation time | Average duration of `model.generate`, including required device synchronization. |
| Total execution time | Average prompt-format/tokenization-through-decode duration for a warm model. It excludes download, conversion, load/compile, and warm-up. |
| Tokens per second | Total generated tokens across measured runs divided by their total generation seconds (equivalently, average generated tokens divided by average generation time), with a zero-duration guard. |
| CPU usage | Estimated average process CPU utilization sampled during generation and normalized by logical CPU count. |
| RAM usage | Estimated average per-run peak increase in process resident memory (RSS) above the RSS at the start of that generation. It is not GPU or NPU memory. |
| Device | Requested and validated runtime device used for the result. |
| Runtime/model | PyTorch or OpenVINO plus the pinned model identifier. |

The comparison reports the faster runtime by average generation time, absolute generation-time difference, speedup or slowdown percentage, throughput difference, and RAM difference. Native Streamlit bar charts visualize time, throughput, and RAM without extra chart dependencies.

When OpenVINO is faster, its reported speedup is `(PyTorch time / OpenVINO time - 1) * 100`. When it is slower, the slowdown magnitude is `(OpenVINO time / PyTorch time - 1) * 100`. Zero or non-positive timing inputs are guarded rather than producing an infinite or fabricated percentage.

## Model storage

By default, downloaded Hugging Face files are stored under `.cache/models`, and the reusable OpenVINO representation is stored under `.cache/openvino`. Override either location with its environment variable before setup or launch.

Do not commit either cache. Deleting a cache is safe but forces the corresponding model download or OpenVINO conversion on the next run. Model download time is never treated as inference time.

## Hardware and device notes

- **PyTorch CPU** and **OpenVINO CPU** are the most portable like-for-like choice.
- PyTorch accelerator choices appear only when the installed PyTorch build reports them as usable.
- OpenVINO lists only devices reported by the installed runtime. `CPU`, `GPU`, and `NPU` availability depends on the machine, operating system, driver, and OpenVINO installation.
- An Intel GPU or NPU being present in the computer does not guarantee that OpenVINO can compile this model for it. Dynamic shapes, precision support, and driver versions can prevent compilation.
- CPU usage is a process estimate and RAM is process RSS. Neither metric is accelerator utilization or accelerator memory.
- Both loaded runtime models coexist in the one Streamlit process. RAM therefore reports the incremental peak during each generation, not either model's total standalone footprint.
- First conversion can be slow and memory intensive. Subsequent runs reuse the cache.
- Low-memory machines may need to close other applications or use fewer generated tokens. An out-of-memory failure is reported as an error rather than a benchmark result.

## Common errors

### No internet or model download failure

Dependencies and the model require internet access the first time. Check the connection, corporate proxy, certificate configuration, and Hugging Face availability. After the caches are populated, ordinary benchmark runs do not redownload the model.

### Setup reports an unsupported or incomplete `.venv`

Remove only the repository's `.venv` directory and rerun the setup script with a supported Python installed. The script does not delete an existing environment automatically.

### PyTorch device unavailable

Choose CPU or install the correct PyTorch build and hardware driver. The application does not silently substitute a different device because that would invalidate the comparison.

### OpenVINO GPU or NPU unavailable

Choose a device shown by the application. Update the Intel graphics/NPU driver and OpenVINO packages if the hardware should be available. Use `CPU` for the most reliable baseline.

### OpenVINO conversion or compilation fails

Confirm the selected model is allowlisted, there is sufficient disk/RAM, and the cache directory is writable. Remove only that model's incomplete OpenVINO cache before retrying. Technical details are logged; the UI shows a concise message rather than a traceback.

### Streamlit port already in use

Stop the application that owns the port or select another one:

```powershell
.\scripts\run.ps1 -Port 8502
```

## Development checks

Install the development extras, then run the same checks used before release:

```powershell
.\.venv\Scripts\python.exe -m compileall src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Tests use fakes and mocks and must not download a model or require internet access. Importing the package and Streamlit entry point also must not trigger a model download.

## Direct dependencies

- `streamlit` provides the single-page local UI and native charts.
- `torch` runs the PyTorch inference path.
- `transformers` provides the pinned model, tokenizer, and shared generation interface.
- `openvino` provides device discovery and the OpenVINO runtime.
- `optimum-intel` is the single OpenVINO/Transformers integration used for conversion and generation.
- `psutil` samples process CPU and resident memory.
- `pytest` and `ruff` are optional development dependencies for tests, formatting, and linting.

No web backend, alternate model runtime, database, orchestration system, or plotting library is a direct dependency. Some required libraries install their own transitive dependencies; those are managed by pip and are not application features.
