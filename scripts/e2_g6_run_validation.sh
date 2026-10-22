#!/usr/bin/env bash
# E2 official validation: ten separate OS processes, no OOD or test checkpoint vectors.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${E2_PY:-$HOME/mlsec-local/venvs/modelxray/bin/python}"
WEIGHTS="$HOME/mlsec-local/datasets/ghrp_zoos/d1_uniform_real/weights.npy"
VAL_DIR="$HOME/mlsec-local/runs/e2_validation_official_v1"
LOG_DIR="$HOME/mlsec-local/runs/e2_validation_official_v1_logs"
RUNNER_SHA='4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac'

[[ -x "$PY" ]] || { echo 'STOP: expected Python venv not found'; exit 2; }
[[ "$(git branch --show-current)" == ricky ]] || { echo 'STOP: wrong git branch'; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'STOP: tracked git files are dirty'; exit 2; }
[[ -f docs/E2_PROTOCOL_FROZEN.md ]] || { echo 'STOP: missing frozen protocol'; exit 2; }
[[ "$(sha256sum scripts/run_e2_paired_v0.py | cut -d' ' -f1)" == "$RUNNER_SHA" ]] || { echo 'STOP: runner SHA differs'; exit 2; }
git show HEAD:docs/E2_PROTOCOL_FROZEN.md | cmp -s - docs/E2_PROTOCOL_FROZEN.md || {
  echo 'STOP: frozen protocol must be committed unchanged'; exit 2;
}
git show HEAD:scripts/run_e2_paired_v0.py | cmp -s - scripts/run_e2_paired_v0.py || {
  echo 'STOP: runner must be committed unchanged'; exit 2;
}
[[ ! -e "$VAL_DIR" ]] || { echo "STOP: output directory already exists: $VAL_DIR"; exit 2; }
[[ ! -e "$LOG_DIR" ]] || { echo "STOP: log directory already exists: $LOG_DIR"; exit 2; }

"$PY" scripts/run_e2_paired_v0.py --weights "$WEIGHTS" \
    --split-dir manifests/e1_uniform_v1 --out-dir "$VAL_DIR" --stage preflight
mkdir -p "$LOG_DIR"
{
 echo "Freeze commit: $(git rev-parse HEAD)"
 echo "Freeze doc sha256: $(sha256sum docs/E2_PROTOCOL_FROZEN.md | cut -d' ' -f1)"
 echo "Runner sha256: $RUNNER_SHA"
 echo 'Scope: E2 ten-fit validation LSB-only, no test approval'
} > "$LOG_DIR/freeze_manifest.txt"
for rep in $(seq 0 9); do
  printf -v rr '%02d' "$rep"
  printf '\n=== E2 VALIDATION repeat=%s/09 ===\n' "$rr"
  "$PY" scripts/run_e2_paired_v0.py --weights "$WEIGHTS" \
       --split-dir manifests/e1_uniform_v1 --out-dir "$VAL_DIR" \
       --stage validation --only-repeat "$rep" 2>&1 | tee "$LOG_DIR/repeat_${rr}.log"
done
printf '\nE2 G6 VALIDATION RUNS FINISHED; audit before proceeding.\n'
