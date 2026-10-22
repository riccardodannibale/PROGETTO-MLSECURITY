#!/usr/bin/env bash

set -euo pipefail

# Root directory of the project
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# uv executable
UV="$HOME/.local/bin/uv"

# Local virtual environment
VENV="$HOME/mlsec-local/venvs/modelxray"

echo "===================================="
echo "Machine Learning Security Setup"
echo "===================================="

cd "$PROJECT_ROOT"

# 1. Verify uv
if [ ! -x "$UV" ]; then
    echo "ERROR: uv is not installed."
    echo "Install uv before running this script."
    exit 1
fi

# 2. Initialize Git submodules
echo "[1/5] Initializing Model X-Ray submodule..."

git submodule update --init --recursive

# 3. Install Python 3.11
echo "[2/5] Checking Python 3.11..."

"$UV" python install 3.11

# 4. Create virtual environment
echo "[3/5] Creating Python environment..."

if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$(dirname "$VENV")"
    "$UV" venv --python 3.11 "$VENV"
fi

# 5. Install locked dependencies
echo "[4/5] Installing dependencies..."

"$UV" pip install \
    --python "$VENV/bin/python" \
    -r "$PROJECT_ROOT/configs/modelxray-ubuntu24-py311.lock.txt"

# 6. Install Model X-Ray package
echo "[5/5] Installing Model X-Ray..."

"$UV" pip install \
    --python "$VENV/bin/python" \
    -e "$PROJECT_ROOT/external/ModelXRay"

# Verify dependencies
"$UV" pip check --python "$VENV/bin/python"

echo ""
echo "===================================="
echo "Setup completed successfully!"
echo "===================================="
echo ""
echo "Python:"
"$VENV/bin/python" --version
echo ""
echo "Model X-Ray environment: $VENV"


echo "Installing GHRP..."

env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NEURIPS_2022_GENERATIVE_HYPER_REPRESENTATIONS=0.0.0 \
  "$UV" pip install \
  --python "$VENV/bin/python" \
  --no-deps \
  -e "$PROJECT_ROOT/external/ModelXRay/external_code/ghrp/"
