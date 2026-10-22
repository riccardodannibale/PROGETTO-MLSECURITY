#!/usr/bin/env bash
# G7 E2 official paired ID/OOD test — identical G5 detector, 10 fresh OS processes.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${E2_PY:-$HOME/mlsec-local/venvs/modelxray/bin/python}"
WEIGHTS="$HOME/mlsec-local/datasets/ghrp_zoos/d1_uniform_real/weights.npy"
EXPECTED='4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac'
case "$(uname -s)" in
  Darwin) PLATFORM=mac_m4;;
  Linux) PLATFORM=ubuntu;;
  *) echo 'STOP: unsupported OS'; exit 2;;
esac
if [[ "$PLATFORM" == mac_m4 ]]; then
  VAL="$HOME/mlsec-local/runs/e2_validation_official_mac_m4_v1"
else
  VAL="$HOME/mlsec-local/runs/e2_validation_official_v1"
fi
OUT="$HOME/mlsec-local/runs/e2_test_official_${PLATFORM}_v1"
LOG="$HOME/mlsec-local/runs/e2_test_official_${PLATFORM}_v1_logs"
APP="$HOME/mlsec-local/runs/e2_test_release_${PLATFORM}_v1.json"
[[ -x "$PY" && -f "$WEIGHTS" && -f "$APP" ]] || { echo 'STOP: missing venv/weights/test approval'; exit 2; }
[[ "$(git branch --show-current)" == ricky ]] || { echo 'STOP: wrong branch'; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'STOP: tracked files dirty'; exit 2; }
[[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'STOP: official output/log already exist; do not overwrite'; exit 2; }
"$PY" - "$EXPECTED" "$APP" "$VAL" "$PLATFORM" <<'PYCODE'
import hashlib, json, pathlib, subprocess, sys
expected, approval_path, val, platform = sys.argv[1:]
root=pathlib.Path.cwd()
runner=root/'scripts/run_e2_paired_v0.py'
freeze=root/'docs/E2_PROTOCOL_FROZEN.md'
notice=root/'docs/E2_G7_TEST_RELEASE_NOTICE.md'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(runner)==expected, 'STOP: detector runner SHA altered'
for rel in ('scripts/run_e2_paired_v0.py','docs/E2_PROTOCOL_FROZEN.md','docs/E2_G7_TEST_RELEASE_NOTICE.md'):
    assert subprocess.check_output(['git','show',f'HEAD:{rel}'])==(root/rel).read_bytes(),f'STOP: uncommitted {rel}'
d=json.loads(pathlib.Path(approval_path).read_text())
assert d['platform']==platform and d['validation_dir']==str(pathlib.Path(val).resolve()), 'STOP: wrong local approval'
assert d['runner_sha256']==expected and d['release_notice_sha256']==sha(notice), 'STOP: release mismatch'
assert d['validation_run_config_sha256']==sha(pathlib.Path(val)/'run_config.json'), 'STOP: validation config changed'
sys.path.insert(0,str(root/'scripts'))
import run_e2_paired_v0 as e2
e2.check_test_unlock(approval=pathlib.Path(approval_path),validation_dir=pathlib.Path(val),
                     runner_path=runner,expected_runner_sha=expected)
print('E2 TEST UNLOCK CHECK OK: local approval, committed freeze, valid prior fits', flush=True)
PYCODE
"$PY" scripts/e2_g6_audit_validation_v2.py --validation-dir "$VAL"
"$PY" scripts/run_e2_paired_v0.py --weights "$WEIGHTS" \
  --split-dir manifests/e1_uniform_v1 --out-dir "$OUT" --stage preflight
mkdir "$LOG"
{
  echo "E2 test platform=$PLATFORM"
  echo "Release commit=$(git rev-parse HEAD)"
  echo "Frozen SHA=$($PY -c 'from pathlib import Path; import hashlib; print(hashlib.sha256(Path("docs/E2_PROTOCOL_FROZEN.md").read_bytes()).hexdigest())')"
  echo "Runner SHA=$EXPECTED"
  echo "Approval JSON=$APP"
  echo "Validation directory=$VAL"
  echo "Scope=10 paired E2 ID/OOD fits; same Siamese and validation-only thresholds"
} > "$LOG/release_manifest.txt"
for rep in $(seq 0 9); do
  rr=$(printf '%02d' "$rep")
  echo "=== E2 TEST ${PLATFORM} REPEAT $rr/09 ==="
  "$PY" scripts/run_e2_paired_v0.py --weights "$WEIGHTS" \
      --split-dir manifests/e1_uniform_v1 --out-dir "$OUT" \
      --stage test --unblind-test --test-approval "$APP" \
      --validation-dir "$VAL" --only-repeat "$rep" 2>&1 | tee "$LOG/repeat_${rr}.log"
done
echo 'E2 10/10 TEST FITS FINISHED. Proceed to independent G7 test audit; do NOT tune.'
