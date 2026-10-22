#!/usr/bin/env python3
"""Synthetic smoke test for the portfolio-only CSV consistency checker.

Tests the small, standalone checker; does not train or inspect any model weights.
"""
from __future__ import annotations
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

CHECKER = Path(__file__).resolve().parents[1] / 'scripts/verify_published_results.py'
CONDS = ('LSB_ID', 'SPREAD_0.24', 'SPREAD_0.16')
AUROCS = {'ubuntu': {'LSB_ID': 0.9042725, 'SPREAD_0.24': 0.47762, 'SPREAD_0.16': 0.49470125},
          'mac_m4': {'LSB_ID': 0.9043, 'SPREAD_0.24': 0.4745, 'SPREAD_0.16': 0.4910425}}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        pair_stat = []
        e3_cross = []
        cross_e2 = []
        for p in AUROCS:
            a = AUROCS[p]
            paired = []
            for i in range(10):
                line = {'repeat': i, 'LSB_ID': a['LSB_ID'], 'SPREAD_0.24': a['SPREAD_0.24'],
                        'SPREAD_0.16': a['SPREAD_0.16'],
                        'delta_auroc_ood24_minus_id': a['SPREAD_0.24']-a['LSB_ID'],
                        'delta_auroc_ood16_minus_id': a['SPREAD_0.16']-a['LSB_ID']}
                paired.append(line)
                if p == 'ubuntu':
                    cross_e2.append({'repeat': i})
                for fam in CONDS:
                    cross_e2[i][f'{p}_{fam}'] = a[fam]
            write(root/f'results/e2_final/{p}/paired_auroc_per_repeat.csv', paired)
            for fam in CONDS:
                pair_stat.append({'platform':p,'measure':fam,'n_fit':10,'mean':a[fam]})
            pair_stat.append({'platform':p,'measure':'delta_auroc_ood24_minus_id','n_fit':10,
                              'mean':a['SPREAD_0.24']-a['LSB_ID']})
            summary = []
            e3summary = []
            perfit = []
            for method in ('centroid','nn'):
                for fam in CONDS:
                    auroc = a[fam] if method == 'centroid' else (0.82 if fam=='LSB_ID' else 0.48)
                    summary.append({'stage':'test','condition':fam,'method':method,'policy':'val_youden',
                                    'n_runs':10,'auroc_mean':auroc})
                    tpr = 0.8 if fam=='LSB_ID' else 0.2
                    fpr = 0.18 if method=='centroid' else 0.23
                    e3summary.append({'platform':p,'method':method,'condition':fam,'n_fits':10,
                                      'metadata_only_majority_accuracy':0.5,
                                      'max_benign_score_difference':0,'auroc_mean':auroc,
                                      'tpr_mean':tpr,'fnr_mean':1-tpr,'fpr_mean':fpr})
                    for i in range(10):
                        perfit.append({'repeat':i,'platform':p,'method':method,'condition':fam,
                                       'auroc':auroc,'tpr_val_youden':tpr,'fnr_val_youden':1-tpr,
                                       'fpr_val_youden':fpr})
                        if p=='ubuntu':
                            e3_cross.append({'repeat':i,'method':method,'condition':fam})
            write(root/f'results/e2_final/{p}/summary.csv', summary)
            write(root/f'results/e3_final/{p}/e3_controls_summary.csv', e3summary)
            write(root/f'results/e3_final/{p}/e3_controls_per_fit.csv', perfit)
        write(root/'results/e2_final/e2_cross_platform_paired_per_repeat.csv', cross_e2)
        write(root/'results/e2_final/e2_cross_platform_statistics.csv', pair_stat)
        write(root/'results/e3_final/cross_platform/e3_cross_platform_per_fit.csv', e3_cross)
        proc = subprocess.run([sys.executable, str(CHECKER), '--root', str(root)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'SUMMARY QA OK' in proc.stdout
        # Tamper: corrupt a published value; checker MUST fail.
        rows = list(csv.DictReader((root/'results/e2_final/ubuntu/summary.csv').open(newline='')))
        rows[0]['auroc_mean'] = 0.01
        write(root/'results/e2_final/ubuntu/summary.csv', rows)
        proc = subprocess.run([sys.executable, str(CHECKER), '--root', str(root)], capture_output=True, text=True)
        assert proc.returncode != 0, 'corrupted AUROC should fail'
        print('PORTFOLIO VERIFIER SELF-TEST OK: valid summaries accepted, corrupted summary rejected')

if __name__ == '__main__':
    main()
