#!/usr/bin/env bash
# E1 v2 validation matrix: one Python/TensorFlow process per fit to avoid OOM.
# Run only AFTER run_e1_final_v2.py has passed its smoke test and been committed.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${MODELXRAY_PYTHON:-$HOME/mlsec-local/venvs/modelxray/bin/python}"
ZOOS="${MODELXRAY_GHRP_DIR:-$HOME/mlsec-local/datasets/ghrp_zoos}"
OUT="${E1_V2_OUT_DIR:-$HOME/mlsec-local/runs/e1_v2_validation}"
unset MODELXRAY_PAYLOAD_FILE
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$OUT"

for repeat in {0..9}; do
  for x in 1 8 16 23; do
    dir=$(printf '%s/repeat_%02d_x_%02d' "$OUT" "$repeat" "$x")
    if [[ -s "$dir/metrics.json" && -s "$dir/predictions_validation.csv" && -s "$dir/payload_manifest.csv" ]]; then
      echo "[matrix] already completed repeat=$repeat X=$x"
      continue
    fi
    echo "[matrix] START repeat=$repeat X=$x"
    "$PYTHON" -u scripts/run_e1_final_v2.py \
      --weights "$ZOOS/d1_uniform_real/weights.npy" \
      --split-dir manifests/e1_uniform_v1 \
      --stage validation \
      --x-range 1 8 16 23 \
      --n-repeats 10 \
      --mode ub \
      --seed 42 \
      --only-repeat "$repeat" \
      --only-x "$x" \
      --out-dir "$OUT"
    echo "[matrix] DONE repeat=$repeat X=$x"
  done
done

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
p = Path.home() / 'mlsec-local/runs/e1_v2_validation'
import os
p = Path(os.environ.get('E1_V2_OUT_DIR', p))
a = pd.read_csv(p/'metrics_per_run.csv')
assert len(a[['repeat','X']].drop_duplicates()) == 40, 'Incomplete E1 v2 validation'
print('\n=== E1 v2 VALIDATION COMPLETE 40/40 ===')
print(pd.read_csv(p/'summary.csv').to_string(index=False))
PY
