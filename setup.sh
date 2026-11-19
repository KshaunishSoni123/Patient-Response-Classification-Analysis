#!/usr/bin/env bash
# One-time environment setup for Patient Response Classification.
#
#   ./setup.sh           create .venv (if missing) and install requirements.txt
#   ./setup.sh --force   delete and recreate .venv
#
# Override the interpreter with PYTHON=/path/to/python3.13 ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"

if [[ "${1:-}" == "--force" ]]; then
    echo "==> Removing existing $VENV"
    rm -rf "$VENV"
fi

# --- 1. Find a Python 3.13 interpreter -------------------------------------
is_py313() { "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))' 2>/dev/null; }

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    for candidate in python3.13 python3 /opt/homebrew/bin/python3.13 /opt/anaconda3/bin/python; do
        if command -v "$candidate" >/dev/null 2>&1 && is_py313 "$candidate"; then
            PY="$candidate"
            break
        fi
    done
fi
if [[ -z "$PY" ]] || ! is_py313 "$PY"; then
    echo "ERROR: Python 3.13 not found. Install it (e.g. 'brew install python@3.13') or set PYTHON=..." >&2
    exit 1
fi
echo "==> Using $("$PY" --version) at $(command -v "$PY")"

# --- 2. Virtual environment + packages --------------------------------------
if [[ ! -d "$VENV" ]]; then
    echo "==> Creating $VENV"
    "$PY" -m venv "$VENV"
else
    echo "==> $VENV already exists (use --force to recreate)"
fi

echo "==> Installing requirements (first run downloads ~1 GB, mostly PyTorch)"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements.txt

# --- 3. Project folders -----------------------------------------------------
mkdir -p data/raw data/processed data/synthetic models

# --- 4. Smoke test ----------------------------------------------------------
echo "==> Checking imports"
KERAS_BACKEND=torch "$VENV/bin/python" - <<'EOF'
import sys
sys.modules["tensorflow"] = None  # same TensorFlow block the project scripts use
import anthropic, gensim, keras, optuna, pandas, sklearn, torch, transformers
print(f"    keras {keras.__version__} (backend: {keras.backend.backend()}), torch {torch.__version__}, "
      f"gensim {gensim.__version__}, transformers {transformers.__version__}, anthropic {anthropic.__version__}")
EOF

# --- 5. Things setup can't do for you ---------------------------------------
echo "==> Checks"
if [[ -f data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv ]]; then
    echo "    ok   healthcare dataset present"
else
    echo "    TODO place the healthcare dataset at data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv"
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1; }; then
    echo "    ok   Claude API credentials found"
else
    echo "    TODO for synthetic data: 'ant auth login' or export ANTHROPIC_API_KEY"
fi

cat <<'EOF'

Setup complete. Activate the environment, then follow README.md:

    source .venv/bin/activate
    python data_ingestion.py
    python healthcare_ingestion.py
EOF
