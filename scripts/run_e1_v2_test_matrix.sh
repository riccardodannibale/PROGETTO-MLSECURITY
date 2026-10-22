#!/usr/bin/env bash
# E1 v2 test finale: protocollo congelato; una nuova istanza Python per fit.
# Uso: bash scripts/run_e1_v2_test_matrix.sh --preflight
#      bash scripts/run_e1_v2_test_matrix.sh --run
# Non altera la validation, non cambia i parametri e non modifica Model X-Ray.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${MODELXRAY_PYTHON:-$HOME/mlsec-local/venvs/modelxray/bin/python}"
ZOOS="${MODELXRAY_GHRP_DIR:-$HOME/mlsec-local/datasets/ghrp_zoos}"
VALIDATION="$HOME/mlsec-local/runs/e1_v2_validation"
TEST="$HOME/mlsec-local/runs/e1_v2_test"

if [[ $# -ne 1 || ( "$1" != "--preflight" && "$1" != "--run" ) ]]; then
  echo "Uso: bash scripts/run_e1_v2_test_matrix.sh --preflight|--run" >&2
  exit 2
fi

# --preflight non legge i pesi dei modelli riservati e non esegue training.
"$PYTHON" - "$VALIDATION" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

root = Path.cwd()
validation = Path(sys.argv[1])
config = json.loads((validation / 'run_config.json').read_text())
expected = {
    'stage': 'validation', 'x_range': [1, 8, 16, 23], 'n_repeats': 10,
    'n_train_identities': 3, 'mode': 'ub', 'seed': 42,
    'runner_revision': 'v2_deterministic_synthetic_file_per_model',
}
for k, v in expected.items():
    if config.get(k) != v:
        raise SystemExit(f'STOP: validation config {k}={config.get(k)!r}, atteso {v!r}')
source = root / 'scripts/run_e1_final_v2.py'
original = subprocess.check_output([
    'git', 'show', f'{config["project_commit"]}:scripts/run_e1_final_v2.py'
])
if hashlib.sha256(original).digest() != hashlib.sha256(source.read_bytes()).digest():
    raise SystemExit('STOP: runner attuale differente da quello della validation v2')
run_dirs = sorted(validation.glob('repeat_*_x_*/metrics.json'))
if len(run_dirs) != 40:
    raise SystemExit(f'STOP: trovati {len(run_dirs)} training di validation invece di 40')
for p in run_dirs:
    base = p.parent
    for name in ('payload_manifest.csv', 'predictions_validation.csv'):
        if not (base / name).is_file() or not (base / name).stat().st_size:
            raise SystemExit(f'STOP: manca {base / name}')
    m = json.loads(p.read_text())
    if len(m['metrics']) != 4 or {x['stage'] for x in m['metrics']} != {'validation'}:
        raise SystemExit(f'STOP: metriche di validation non valide in {p}')
df = pd.read_csv(validation / 'metrics_per_run.csv')
if len(df[['repeat', 'X']].drop_duplicates()) != 40:
    raise SystemExit('STOP: metrics_per_run.csv della validation non contiene 40 fit')
if set(df['stage']) != {'validation'}:
    raise SystemExit('STOP: il CSV della validation contiene altri stage')
print('PREFLIGHT OK: 40/40 validation, protocollo e sorgente runner invariati.')
print('Scelta congelata: centroid / AUROC primaria; X=1,8,16,23; 10 repliche;')
print('soglia originale 0 e Youden appresa dalla validation della stessa replica.')
print('Il test riservato NON e\' stato eseguito in questo controllo.')
PY

if [[ "$1" == "--preflight" ]]; then
  exit 0
fi

if [[ "${E1_TEST_UNBLIND_CONFIRM:-}" != "UNBLIND_E1_V2_TEST" ]]; then
  echo 'STOP: per aprire il test serve E1_TEST_UNBLIND_CONFIRM=UNBLIND_E1_V2_TEST' >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != 'ricky' ]]; then
  echo 'STOP: branch diverso da ricky' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'STOP: prima di aprire il test bisogna committare lo script e avere Git pulito' >&2
  exit 2
fi
if [[ ! -s "$ZOOS/d1_uniform_real/weights.npy" ]]; then
  echo 'STOP: weights.npy mancante' >&2
  exit 2
fi

unset MODELXRAY_PAYLOAD_FILE
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TEST"
echo '[test matrix] Si apre il test riservato: 40 fit isolati.'
echo '[test matrix] In caso di discrepanza numerica, STOP senza retry selettivo.'
for repeat in {0..9}; do
  for x in 1 8 16 23; do
    dir=$(printf '%s/repeat_%02d_x_%02d' "$TEST" "$repeat" "$x")
    if [[ -s "$dir/metrics.json" && -s "$dir/predictions_validation.csv" && \
          -s "$dir/predictions_test.csv" && -s "$dir/payload_manifest.csv" ]]; then
      echo "[test matrix] already completed repeat=$repeat X=$x"
      continue
    fi
    echo "[test matrix] START repeat=$repeat X=$x"
    "$PYTHON" -u scripts/run_e1_final_v2.py \
      --weights "$ZOOS/d1_uniform_real/weights.npy" \
      --split-dir manifests/e1_uniform_v1 \
      --stage test --unblind-test \
      --x-range 1 8 16 23 \
      --n-repeats 10 --n-train-identities 3 \
      --mode ub --seed 42 \
      --only-repeat "$repeat" --only-x "$x" \
      --out-dir "$TEST"
    echo "[test matrix] DONE repeat=$repeat X=$x"
  done
done

"$PYTHON" - "$TEST" <<'PY'
import sys
from pathlib import Path
import pandas as pd
p = Path(sys.argv[1])
df = pd.read_csv(p / 'metrics_per_run.csv')
if len(df[['repeat', 'X']].drop_duplicates()) != 40:
    raise SystemExit('STOP: matrice test incompleta')
summary = pd.read_csv(p / 'summary.csv')
for stage in ('validation', 'test'):
    group = summary[summary['stage'] == stage]
    if len(group) != 16 or not (group['n_runs'] == 10).all():
        raise SystemExit(f'STOP: riepilogo incompleto per stage {stage}')
print('\n=== E1 v2 TEST COMPLETE 40/40 ===')
print(summary[summary['stage'] == 'test'].to_string(index=False))
PY
