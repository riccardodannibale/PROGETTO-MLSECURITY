#!/usr/bin/env python3
"""Read-only checks of already-published E2/E3 *aggregate* CSVs.

Uses only Python's standard library. Does not reconstruct AUROC from raw predictions,
run a detector, open model weights or execute upstream code. A failure is always fatal.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

FAMILIES = ('LSB_ID', 'SPREAD_0.24', 'SPREAD_0.16')
PLATFORMS = ('ubuntu', 'mac_m4')
METHODS = ('centroid', 'nn')


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError('PUBLISHED RESULT QA FAIL: ' + message)


def rows(path: Path) -> list[dict[str, str]]:
    check(path.is_file(), f'missing {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def num(x: str, context: str) -> float:
    v = float(x)
    check(math.isfinite(v), f'nonfinite {context}')
    return v


def near(a: float, b: float, context: str, atol: float = 1e-9) -> None:
    check(abs(a - b) <= atol, f'{context}: {a:.12g} != {b:.12g}')


def single(table: list[dict[str, str]], context: str, **filters: str) -> dict[str, str]:
    found = [r for r in table if all(r.get(k) == v for k, v in filters.items())]
    check(len(found) == 1, f'{context}: expected one record for {filters}, found {len(found)}')
    return found[0]


def verify(root: Path) -> None:
    e2 = root / 'results/e2_final'
    e3 = root / 'results/e3_final'
    pairs_by_platform: dict[str, dict[str, float]] = {}
    by_repeat: dict[str, list[dict[str, str]]] = {}
    print('=== E2: paired test AUROCs ===')
    for platform in PLATFORMS:
        paired = rows(e2 / platform / 'paired_auroc_per_repeat.csv')
        summary = rows(e2 / platform / 'summary.csv')
        check(len(paired) == 10, f'E2 {platform}: expected ten repeats')
        check({int(r['repeat']) for r in paired} == set(range(10)), f'E2 {platform}: repeat IDs')
        by_repeat[platform] = paired
        means: dict[str, float] = {}
        for family in FAMILIES:
            vals = [num(r[family], f'{platform}/{family}') for r in paired]
            check(all(0 <= x <= 1 for x in vals), f'E2 {platform}/{family}: AUROC outside [0,1]')
            mean = statistics.mean(vals)
            record = single(summary, f'E2 {platform}/{family}', stage='test', condition=family,
                            method='centroid', policy='val_youden')
            check(int(record['n_runs']) == 10, f'E2 {platform}/{family} n_runs')
            near(mean, num(record['auroc_mean'], 'summary AUROC'), f'E2 {platform}/{family}: paired vs summary')
            means[family] = mean
        for r in paired:
            near(num(r['delta_auroc_ood24_minus_id'], 'delta24'), num(r['SPREAD_0.24'], 'spread24') -
                 num(r['LSB_ID'], 'lsb'), f'E2 {platform}: repeat {r["repeat"]} delta24')
            near(num(r['delta_auroc_ood16_minus_id'], 'delta16'), num(r['SPREAD_0.16'], 'spread16') -
                 num(r['LSB_ID'], 'lsb'), f'E2 {platform}: repeat {r["repeat"]} delta16')
        means['delta_ood24_minus_id'] = means['SPREAD_0.24'] - means['LSB_ID']
        pairs_by_platform[platform] = means
        print(f'{platform:8} ID={means["LSB_ID"]:.6f} spread24={means["SPREAD_0.24"]:.6f} '
              f'spread16={means["SPREAD_0.16"]:.6f} Δ24={means["delta_ood24_minus_id"]:+.6f}')

    print('=== E2: cross-platform published table ===')
    cross = rows(e2 / 'e2_cross_platform_paired_per_repeat.csv')
    check(len(cross) == 10, 'E2 cross-platform: expected ten repeat rows')
    check({int(r['repeat']) for r in cross} == set(range(10)), 'E2 cross-platform repeats')
    for r in cross:
        rep = int(r['repeat'])
        for platform in PLATFORMS:
            for family in FAMILIES:
                expected = next(t for t in by_repeat[platform] if int(t['repeat']) == rep)
                near(num(r[f'{platform}_{family}'], 'cross value'), num(expected[family], 'paired value'),
                     f'E2 cross-platform repeat {rep} {platform}/{family}')
    stats = rows(e2 / 'e2_cross_platform_statistics.csv')
    for platform in PLATFORMS:
        for family in FAMILIES:
            rec = single(stats, 'E2 stats', platform=platform, measure=family)
            check(int(rec['n_fit']) == 10, 'E2 stats count')
            near(num(rec['mean'], 'E2 cross stats mean'), pairs_by_platform[platform][family],
                 f'E2 cross stats mean {platform}/{family}')
        rec = single(stats, 'E2 stats delta', platform=platform, measure='delta_auroc_ood24_minus_id')
        near(num(rec['mean'], 'E2 delta mean'), pairs_by_platform[platform]['delta_ood24_minus_id'],
             f'E2 cross stats delta {platform}')

    print('=== E3: aggregate controls (not independently rescoring predictions) ===')
    for platform in PLATFORMS:
        summary = rows(e3 / platform / 'e3_controls_summary.csv')
        perfit = rows(e3 / platform / 'e3_controls_per_fit.csv')
        check(len(summary) == 6, f'E3 {platform}: six summary rows')
        check(len(perfit) == 60, f'E3 {platform}: sixty per-fit rows')
        seen = {(int(r['repeat']), r['method'], r['condition']) for r in perfit}
        check(len(seen) == 60 and seen == {(i, m, c) for i in range(10) for m in METHODS for c in FAMILIES},
              f'E3 {platform}: invalid per-fit keys')
        for method in METHODS:
            fprs = []
            for family in FAMILIES:
                r = single(summary, f'E3 {platform}', platform=platform, method=method, condition=family)
                check(int(r['n_fits']) == 10, f'E3 {platform}: n_fits')
                near(num(r['metadata_only_majority_accuracy'], 'metadata baseline'), 0.5,
                     f'E3 {platform}/{method}/{family}: balance')
                check(num(r['max_benign_score_difference'], 'benign diff') <= 1e-6,
                      f'E3 {platform}/{method}/{family}: benign score mismatch')
                per = [x for x in perfit if x['method'] == method and x['condition'] == family]
                for col_sum, col_perfit in (('auroc_mean','auroc'),('tpr_mean','tpr_val_youden'),
                                            ('fpr_mean','fpr_val_youden'),('fnr_mean','fnr_val_youden')):
                    mean = statistics.mean(num(x[col_perfit], f'E3/{col_perfit}') for x in per)
                    near(mean, num(r[col_sum], 'E3 summary mean'), f'E3 {platform}/{method}/{family}/{col_sum}')
                e2_row = single(rows(e2 / platform / 'summary.csv'), f'E3 ↔ E2 {platform}', stage='test',
                                condition=family, method=method, policy='val_youden')
                near(num(r['auroc_mean'], 'E3 AUROC'), num(e2_row['auroc_mean'], 'E2 AUROC'),
                     f'E3↔E2 {platform}/{method}/{family}')
                fprs.append(num(r['fpr_mean'], 'FPR'))
            near(max(fprs), min(fprs), f'E3 {platform}/{method}: FPR not constant', atol=1e-12)
        print(f'{platform:8} 10 fits × 2 methods × 3 conditions; metadata balanced, FPR constant')
    cross_e3 = rows(e3 / 'cross_platform/e3_cross_platform_per_fit.csv')
    check(len(cross_e3) == 60, 'E3 cross-platform paired table should have 60 rows')
    check(len({(int(r['repeat']), r['method'], r['condition']) for r in cross_e3}) == 60,
          'E3 cross-platform duplicate keys')
    print('PUBLISHED E2/E3 SUMMARY QA OK (read-only; original E1/G7/E3 audits remain authoritative)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    a = p.parse_args()
    try:
        verify(a.root.resolve())
    except (RuntimeError, ValueError, KeyError, OSError, StopIteration) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
