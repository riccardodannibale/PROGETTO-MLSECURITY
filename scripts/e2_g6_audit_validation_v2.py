#!/usr/bin/env python3
"""Audit E2 validation with exact CSV float round-trip parsing. NO test inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

RUNNER_SHA = "4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac"
WEIGHTS_SHA = "5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad"
SPLITS_SHA = "3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f"


def check(condition, message):
    if not condition:
        raise RuntimeError(f"STOP E2 G6 QA: {message}")


def audit(base: Path, split_csv: Path):
    cfg = json.loads((base / 'run_config.json').read_text())
    check(cfg['stage'] == 'validation' and cfg['runner_sha256'] == RUNNER_SHA, 'config/stage/runner mismatch')
    check(cfg['weights_sha256'] == WEIGHTS_SHA and cfg['splits_sha256'] == SPLITS_SHA, 'source hashes mismatch')
    check(cfg['x'] == 23 and cfg['n_repeats'] == 10 and cfg['alphas'] == [0.24, 0.16], 'protocol params mismatch')
    check(hashlib.sha256(split_csv.read_bytes()).hexdigest() == SPLITS_SHA, 'split CSV SHA mismatch')
    split = pd.read_csv(split_csv)
    train = set(split.loc[split['split'] == 'train', 'row_index'].astype(int))
    val = set(split.loc[split['split'] == 'val', 'row_index'].astype(int))
    check(len(train) == 600 and len(val) == 200 and train.isdisjoint(val), 'bad train/val split')
    check(len(list(base.glob('repeat_*_x_23'))) == 10, 'unexpected fit count')
    check(not (base / 'paired_auroc_per_repeat.csv').exists(), 'test paired delta file exists in validation')
    all_rows = []
    for rep in range(10):
        p = base / f'repeat_{rep:02d}_x_23'
        check(p.is_dir(), f'missing repeat {rep}')
        check(not (p / 'predictions_test.csv').exists(), f'test predictions in repeat {rep}')
        m = json.loads((p / 'metrics.json').read_text())
        check(m['stage'] == 'validation' and m['repeat'] == rep, f'bad metrics header {rep}')
        check(m['fit_seed'] == 42 + rep * 1000 + 23, f'fit seed mismatch {rep}')
        check(len(m['train_source_indices']) == 3 and set(m['train_source_indices']).issubset(train), f'train IDs {rep}')
        check(set(m['validation_source_indices']) == val and not m['test_source_indices'], f'validation/test IDs {rep}')
        check(len(m['metrics']) == 4, f'expected 4 method/policy metrics {rep}')
        combos = {(r['condition'], r['method'], r['policy']) for r in m['metrics']}
        want = {('LSB_ID', method, policy) for method in ('centroid', 'nn')
                for policy in ('original_zero', 'val_youden')}
        check(combos == want, f'bad metric combinations {rep}')
        # CRITICAL: preserve 1-ULP strict-inequality semantics of Youden/nextafter.
        # The pandas default parser can round a stored score onto its threshold.
        pred = pd.read_csv(p / 'predictions_validation.csv', float_precision='round_trip')
        check(len(pred) == 800 and set(pred['method']) == {'centroid', 'nn'}, f'predictions count/method {rep}')
        check(set(pred['stage']) == {'validation'} and set(pred['condition']) == {'LSB_ID'}, f'stage/condition {rep}')
        check(set(pred['repeat']) == {rep}, f'repeat in predictions {rep}')
        check(set(pred['row_index'].astype(int)) == val, f'prediction validation IDs {rep}')
        for method, group in pred.groupby('method'):
            check(len(group) == 400 and int((group.label == 0).sum()) == 200
                  and int((group.label == 1).sum()) == 200, f'class balance {rep}/{method}')
            check(bool(np.isfinite(group['score_margin'].to_numpy()).all()), f'nonfinite score {rep}/{method}')
            thr = float(m['thresholds_val_youden'][method])
            check(bool(np.allclose(group['validation_threshold'].to_numpy(), thr, atol=1e-12, rtol=0)),
                  f'threshold not calibrated once per fit {rep}/{method}')
            check(bool(np.array_equal((group.score_margin.to_numpy() > thr).astype(int),
                                      group.prediction_calibrated.to_numpy())), f'bad threshold application {rep}/{method}')
        payload = pd.read_csv(p / 'payload_manifest.csv')
        check(len(payload) == 203 and set(payload['source_split']) == {'train', 'val'}, f'payload rows/splits {rep}')
        check(len(payload[payload.source_split == 'train']) == 3
              and len(payload[payload.source_split == 'val']) == 200, f'payload split sizes {rep}')
        all_rows.extend(m['metrics'])
    raw = pd.read_csv(base / 'metrics_per_run.csv')
    summary = pd.read_csv(base / 'summary.csv')
    check(len(raw) == 40 and set(raw['repeat']) == set(range(10)), 'raw metrics count')
    check(len(summary) == 4 and set(summary['n_runs']) == {10}, 'summary count')
    for _, row in summary.iterrows():
        subset = raw[(raw['stage'] == row['stage']) & (raw['condition'] == row['condition'])
                     & (raw['method'] == row['method']) & (raw['policy'] == row['policy'])]
        check(len(subset) == 10, f'bad aggregate subset {row["method"]}/{row["policy"]}')
        check(math.isclose(float(subset['auroc'].mean()), float(row['auroc_mean']), abs_tol=1e-10),
              f'AUROC mean mismatch {row["method"]}/{row["policy"]}')
    print('E2 G6 OFFICIAL VALIDATION QA OK: 10/10 fits, 40 metrics, 8,000 prediction rows, 2,030 LSB payload manifest rows')
    print('E2 TEST STILL LOCKED: no ID/OOD test scores examined')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation-dir', type=Path, required=True)
    ap.add_argument('--split-csv', type=Path, default=Path('manifests/e1_uniform_v1/splits.csv'))
    a = ap.parse_args()
    audit(a.validation_dir.expanduser().resolve(), a.split_csv.expanduser().resolve())
