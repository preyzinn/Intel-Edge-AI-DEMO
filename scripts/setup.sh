#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

is_supported_python() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' >/dev/null 2>&1
}

find_python() {
    for candidate in python3.12 python3.11 python3.13 python3.14 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && is_supported_python "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

cd "$PROJECT_ROOT"
printf 'Project: %s\n' "$PROJECT_ROOT"

if [ -x "$VENV_PYTHON" ]; then
    if ! is_supported_python "$VENV_PYTHON"; then
        printf 'The existing .venv uses an unsupported Python. Remove %s and run this script again.\n' "$VENV_DIR" >&2
        exit 1
    fi
    printf 'Reusing the existing .venv.\n'
elif [ -e "$VENV_DIR" ]; then
    printf 'The existing %s is incomplete. Remove it and run this script again.\n' "$VENV_DIR" >&2
    exit 1
else
    if ! PYTHON=$(find_python); then
        printf 'Python 3.10 through 3.14 was not found. Install a supported 64-bit Python and try again.\n' >&2
        exit 1
    fi
    printf 'Creating .venv with %s (%s)...\n' "$PYTHON" "$("$PYTHON" --version 2>&1)"
    "$PYTHON" -m venv "$VENV_DIR"
fi

printf 'Upgrading pip...\n'
"$VENV_PYTHON" -m pip install --upgrade pip

printf 'Installing the benchmark and its required dependencies...\n'
"$VENV_PYTHON" -m pip install --editable "$PROJECT_ROOT"

printf 'Checking installed dependency consistency...\n'
"$VENV_PYTHON" -m pip check

printf 'Validating imports...\n'
"$VENV_PYTHON" - <<'PY'
from importlib import metadata, util

packages = {
    "openvino": "openvino",
    "optimum-intel": "optimum.intel",
    "psutil": "psutil",
    "streamlit": "streamlit",
    "torch": "torch",
    "transformers": "transformers",
}
for distribution, module in packages.items():
    if util.find_spec(module) is None:
        raise SystemExit(f"Missing import: {module}")
    print(f"  {distribution} {metadata.version(distribution)}")

import edge_ai_demo  # noqa: F401, E402
print("  edge_ai_demo import OK")
PY

MODEL_CACHE=${EDGE_AI_MODEL_CACHE_DIR:-"$PROJECT_ROOT/.cache/models"}
OPENVINO_CACHE=${EDGE_AI_OPENVINO_CACHE_DIR:-"$PROJECT_ROOT/.cache/openvino"}
mkdir -p "$MODEL_CACHE" "$OPENVINO_CACHE"

printf '\nSetup complete.\n'
printf 'Model cache: %s\n' "$MODEL_CACHE"
printf 'OpenVINO cache: %s\n' "$OPENVINO_CACHE"
printf 'Run the application with:\n'
printf '  .venv/bin/python -m streamlit run src/edge_ai_demo/app.py\n'
