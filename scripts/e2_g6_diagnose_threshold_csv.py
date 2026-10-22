#!/usr/bin/env python3
"""Read-only, validation-only floating-point CSV parsing diagnosis for E2 G6.

Compare pandas default CSV parser and round-trip parser using the original
JSON threshold (never rederive it from validation labels or change threshold).
Preserve original files; do not use test data, train models or tune anything.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def check(condition, message):
    if not condition:
        raise RuntimeError(f'STOP G6 PARSER DIAGNOSIS: {message}')


def run(base: Path):
    cfg = json.loads((base / 'run_config.json').read_text())
    check(cfg.get('stage') == 'validation', 'not a validation directory')
    total_def = total_rt = 0
    for rep in range(10):
        folder = base / f'repeat_{rep:02d}_x_23'
        metrics = json.loads((folder / 'metrics.json').read_text())
        check(metrics['stage'] == 'validation' and metrics['repeat'] == rep, f'bad repeat {rep}')
        csv = folder / 'predictions_validation.csv'
        def_df = pd.read_csv(csv)
        rt_df = pd.read_csv(csv, float_precision='round_trip')
        check(len(def_df) == len(rt_df) == 800, f'bad row count {rep}')
        for method in ('centroid', 'nn'):
            d = def_df.loc[def_df.method == method]
            r = rt_df.loc[rt_df.method == method]
            check(len(d) == len(r) == 400, f'bad method count {rep}/{method}')
            check(np.array_equal(d.row_index.to_numpy(), r.row_index.to_numpy()), 'row ordering changed')
            threshold = float(metrics['thresholds_val_youden'][method])
            check(bool(np.isfinite(threshold)), f'nonfinite threshold {rep}/{method}')
            check(bool(np.isfinite(r.score_margin.to_numpy()).all()), f'nonfinite scores {rep}/{method}')
            check(bool(np.array_equal(r.validation_threshold.to_numpy(),
                                     np.full(400, threshold, dtype=np.float64))),
                  f'round-trip CSV threshold differs from JSON {rep}/{method}')
            stored = r.prediction_calibrated.to_numpy(dtype=np.int64)
            md = np.flatnonzero((d.score_margin.to_numpy() > threshold).astype(int) != stored)
            mr = np.flatnonzero((r.score_margin.to_numpy() > threshold).astype(int) != stored)
            total_def += len(md); total_rt += len(mr)
            print(f'repeat={rep:02d} method={method:8s} default_mismatch={len(md)} roundtrip_mismatch={len(mr)}')
            for i in list(md[:3]) + [j for j in mr[:3] if j not in md[:3]]:
                print('  sample row_index=', int(r.iloc[i].row_index),
                      ' label=',int(r.iloc[i].label),'stored=',int(stored[i]))
                print('    score_default=', repr(float(d.iloc[i].score_margin)),
                      'score_roundtrip=', repr(float(r.iloc[i].score_margin)))
                print('    threshold_json=', repr(threshold),
                      'score_gt_threshold_roundtrip=', bool(r.iloc[i].score_margin > threshold))
    print(f'TOTAL: default_mismatches={total_def} roundtrip_mismatches={total_rt}')
    if total_rt:
        raise SystemExit('STOP: round-trip parsing did NOT resolve all mismatches. Preserve data and report anomalies.')
    print('E2 G6 ROUNDTRIP PARSER DIAGNOSIS OK: no discrepancy at binary64 precision.')
    if total_def:
        print('CONFIRMED: pandas default float parser caused apparent threshold mismatch(s).')
    else:
        print('No default-parser mismatch in this data. Original failure requires further investigation.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--validation-dir', type=Path, required=True)
    a=p.parse_args()
    run(a.validation_dir.expanduser().resolve())
