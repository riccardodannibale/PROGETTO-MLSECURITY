#!/usr/bin/env python3
"""Read-only audit of E2 G2 (X-LSB vs spread) Ubuntu/Mac per-model outputs.

Checks sources are 20 TRAIN-only checkpoints and 60 paired conditions, same
configuration and bit recovery, with explicit numeric tolerance for floats.
No data/model/attack execution; stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

CONF_KEYS = (
    'scope', 'e1_runner_sha256', 'spread_sha256', 'upstream_commit',
    'weights_sha256', 'splits_sha256', 'script_sha256', 'seed', 'repeat',
    'n_models', 'lsb_x', 'spread_alphas', 'spread_bits',
    'identity_row_indices', 'method',
)
EXACT_FIELDS = (
    'source_split', 'row_index', 'attack', 'alpha', 'message_bits',
    'payload_sha256', 'changed_weights', 'all_finite', 'exact_recovery',
)
FLOAT_FIELDS = ('relative_l2', 'max_abs_delta', 'changed_fraction', 'ber')
SUMMARY_FLOATS = (
    'mean_relative_l2', 'median_relative_l2', 'min_relative_l2',
    'max_relative_l2', 'mean_changed_fraction', 'mean_max_abs_delta',
)


def csvload(path: Path) -> list[dict[str,str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def keyed(records: list[dict[str,str]], cols: tuple[str,...]):
    result = {}
    for r in records:
        k = tuple(r[c] for c in cols)
        if k in result:
            raise ValueError(f'Duplicate key: {k}')
        result[k] = r
    return result


def same_float(a: str, b: str, tol: float) -> bool:
    if not a or not b:
        return a == b
    x, y = float(a), float(b)
    return math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def audit(a: Path, b: Path, tol: float) -> None:
    configs = [json.loads((d / 'audit_config.json').read_text(encoding='utf-8')) for d in (a,b)]
    for k in CONF_KEYS:
        if k not in configs[0] or configs[0][k] != configs[1].get(k):
            raise ValueError(f'Config differs or missing: {k}')
    cfg = configs[0]
    assert cfg['n_models'] == 20 and cfg['spread_alphas'] == [0.16, 0.24]
    assert cfg['spread_bits'] == 16 and cfg['lsb_x'] == 23
    ids = cfg['identity_row_indices']
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('Unexpected row index count in config')
    expected = {(str(i), 'X_LSB_23','') for i in ids}
    expected |= {(str(i),'SPREAD_v0',str(alpha)) for i in ids for alpha in (0.16,0.24)}
    rows = [keyed(csvload(d/'intensity_per_model.csv'),('row_index','attack','alpha')) for d in (a,b)]
    if set(rows[0]) != expected or set(rows[1]) != expected:
        raise ValueError('Missing/extra conditions or row indices in G2 CSV')
    for k in sorted(expected):
        ra, rb = rows[0][k], rows[1][k]
        for r in (ra,rb):
            if r['source_split'] != 'train' or r['all_finite'] != 'True':
                raise ValueError(f'Invalid sample flags: {k}')
        for fld in EXACT_FIELDS:
            if ra[fld] != rb[fld]:
                raise ValueError(f'Per-model mismatch {k}: {fld} = {ra[fld]} / {rb[fld]}')
        for fld in FLOAT_FIELDS:
            if not same_float(ra[fld], rb[fld], tol):
                raise ValueError(f'Per-model numeric mismatch {k}: {fld} = {ra[fld]} / {rb[fld]}')
    summaries = [keyed(csvload(d/'intensity_summary.csv'),('attack','alpha')) for d in (a,b)]
    expected_summary = {('X_LSB_23',''),('SPREAD_v0','0.16'),('SPREAD_v0','0.24')}
    if set(summaries[0]) != expected_summary or set(summaries[1]) != expected_summary:
        raise ValueError('Unexpected summary conditions')
    for k in expected_summary:
        ra,rb = summaries[0][k],summaries[1][k]
        if ra['n_models'] != '20' or rb['n_models'] != '20':
            raise ValueError('n_models mismatch')
        for fld in SUMMARY_FLOATS:
            if not same_float(ra[fld],rb[fld],tol):
                raise ValueError(f'Summary numeric mismatch {k}: {fld}')
        # Check summary was actually computed from the per-model observations.
        for tab, srow in ((rows[0],ra),(rows[1],rb)):
            subset = [r for key,r in tab.items() if (key[1],key[2]) == k]
            if len(subset) != 20:
                raise ValueError(f'Unexpected row count for {k}')
            avg = sum(float(r['relative_l2']) for r in subset)/20
            if not same_float(str(avg),srow['mean_relative_l2'],tol):
                raise ValueError(f'Summary inconsistent with per-model CSV: {k}')
    print('E2 G2 CROSS-PLATFORM AUDIT OK: 60 paired rows, exact fields, floats, configs and summaries aligned')
    print('Scope: 20 TRAIN IDs only, no E2 detector or test results.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ubuntu-dir',type=Path,required=True)
    p.add_argument('--mac-dir',type=Path,required=True)
    p.add_argument('--tol',type=float,default=1e-6)
    args=p.parse_args()
    try:
        if not math.isfinite(args.tol) or args.tol <= 0 or args.tol > 1e-3:
            raise ValueError('Tolerance must be (0,1e-3]')
        audit(args.ubuntu_dir,args.mac_dir,args.tol)
    except (ValueError,KeyError,AssertionError,OSError) as exc:
        p.exit(1,f'E2 G2 CROSS-PLATFORM AUDIT FAILED: {exc}\n')

if __name__ == '__main__':
    main()
