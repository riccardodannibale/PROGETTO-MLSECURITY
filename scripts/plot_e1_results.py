#!/usr/bin/env python3
"""Rigenera le quattro figure E1 dai CSV versionati, SENZA riaddestrare.

Uso dalla root della repository:
    ~/mlsec-local/venvs/modelxray/bin/python scripts/plot_e1_results.py

Requisiti: pandas, numpy, matplotlib. Input: summary.csv della validation
sviluppo e del test finale (che include la validation dei nuovi fit).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XS = [1, 8, 16, 23]


def row(frame: pd.DataFrame, *, stage: str, x: int, method: str, policy: str) -> pd.Series:
    d = frame[
        (frame.stage == stage) & (frame.X == x)
        & (frame.method == method) & (frame.policy == policy)
    ]
    if len(d) != 1:
        raise ValueError(f"Riga duplicata o assente: {stage} X={x} {method} {policy}")
    if int(d.iloc[0].n_runs) != 10:
        raise ValueError(f"n_runs != 10 per X={x}")
    return d.iloc[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--validation', type=Path, default=ROOT / 'results/validation/e1_uniform_v2/summary.csv')
    p.add_argument('--test', type=Path, default=ROOT / 'results/test/e1_uniform_v2/summary.csv')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results/figures/e1')
    args = p.parse_args()
    val = pd.read_csv(args.validation)
    tes = pd.read_csv(args.test)
    if set(val.stage) != {'validation'} or set(tes.stage) != {'validation', 'test'}:
        raise ValueError('CSV stage inattesi; non mescolare versioni o split')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def vals(data, stage, method, policy, metric):
        return [float(row(data, stage=stage, x=x, method=method, policy=policy)[metric]) for x in XS]

    def save(fig, name):
        fig.savefig(args.out_dir / name, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print('OK', args.out_dir / name)

    # 1 - AUROC. Le due validation sono distinte.
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for source, stage, label, marker in [
        (val, 'validation', 'Validation di sviluppo (v2)', 'o'),
        (tes, 'validation', 'Validation nei nuovi fit del test', 's'),
        (tes, 'test', 'Test interno riservato', '^'),
    ]:
        ax.plot(XS, vals(source, stage, 'centroid', 'original_zero', 'auroc_mean'),
                marker=marker, linewidth=2, label=label)
    ax.axhline(.5, linestyle='--', linewidth=1, label='Riferimento casuale: 0,5')
    ax.set(xticks=XS, xlabel='Numero di bit di mantissa sovrascritti (X)', ylabel='AUROC media (10 fit)',
           title='E1 — AUROC centroid: sviluppo e test')
    ax.set_ylim(.44, .98)
    ax.grid(alpha=.25)
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'e1_auroc_val_test.png')

    # 2 - Accuracy threshold vs test.
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    width = .37
    at = np.arange(len(XS), dtype=float)
    orig = vals(tes, 'test', 'centroid', 'original_zero', 'accuracy_mean')
    youd = vals(tes, 'test', 'centroid', 'val_youden', 'accuracy_mean')
    sd_orig = vals(tes, 'test', 'centroid', 'original_zero', 'accuracy_std')
    sd_youd = vals(tes, 'test', 'centroid', 'val_youden', 'accuracy_std')
    ax.bar(at-width/2, orig, width, yerr=sd_orig, capsize=3, label='Soglia originale 0')
    ax.bar(at+width/2, youd, width, yerr=sd_youd, capsize=3, label='Youden calibrata su validation')
    ax.axhline(.5, linestyle='--', linewidth=1)
    ax.set(xticks=at, xticklabels=[str(x) for x in XS], xlabel='X-LSB',
           ylabel='Accuracy media ± SD fra 10 fit', title='E1 — Accuracy centroid sul test')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=.22)
    ax.legend(fontsize=9)
    save(fig, 'e1_accuracy_thresholds.png')

    # 3 - FPR sui dati test.
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(XS, vals(tes, 'test', 'centroid', 'original_zero', 'fpr_mean'),
            marker='o', linewidth=2, label='Soglia originale 0')
    ax.plot(XS, vals(tes, 'test', 'centroid', 'val_youden', 'fpr_mean'),
            marker='s', linewidth=2, label='Youden calibrata su validation')
    ax.set(xticks=XS, xlabel='X-LSB', ylabel='False positive rate medio',
           title='E1 — Falsi positivi centroid sul test')
    ax.set_ylim(0, .65)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    save(fig, 'e1_fpr_thresholds.png')

    # 4 - Centroid vs 1-NN AUROC test.
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(XS, vals(tes, 'test', 'centroid', 'original_zero', 'auroc_mean'),
            marker='o', linewidth=2, label='Centroid (primario)')
    ax.plot(XS, vals(tes, 'test', 'nn', 'original_zero', 'auroc_mean'),
            marker='s', linewidth=2, label='1-NN (secondario)')
    ax.axhline(.5, linestyle='--', linewidth=1)
    ax.set(xticks=XS, xlabel='X-LSB', ylabel='AUROC media (10 fit)',
           title='E1 — Centroid e 1-NN sul test')
    ax.set_ylim(.44, .99)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    save(fig, 'e1_auroc_centroid_nn.png')


if __name__ == '__main__':
    main()
