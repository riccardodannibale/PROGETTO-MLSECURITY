#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$HOME/mlsec-local/venvs/modelxray/bin/python"

echo "===================================="
echo "Checking Model X-Ray Environment"
echo "===================================="

# Check Python
"$PYTHON" --version

# Check scientific libraries
"$PYTHON" -c "
import numpy
import pandas
import sklearn
import torch
import tensorflow as tf
import model_xray

print('NumPy:', numpy.__version__)
print('PyTorch:', torch.__version__)
print('TensorFlow:', tf.__version__)
print('Model X-Ray: import OK')
"

# Reproduce paper figures
echo ""
echo "Generating Model X-Ray figures..."

cd "$PROJECT_ROOT/external/ModelXRay"

"$PYTHON" scripts/plots/plot_all.py

echo ""
echo "Generated files:"
ls -lh out/plots/

echo ""
echo "===================================="
echo "Smoke test completed!"
echo "===================================="

