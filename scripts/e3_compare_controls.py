#!/usr/bin/env python3
"""E3 retrospective control replication: read-only per-fit Ubuntu/Mac audit comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONDS = ('LSB_ID','SPREAD_0.24','SPREAD_0.16')
METHODS = ('centroid','nn')
KEY = ['repeat','method','condition']
METRICS = ['auroc','paired_concordance','paired_score_gap_mean','fpr_val_youden','tpr_val_youden','fnr_val_youden']


def load(path:Path,platform:str):
    df=pd.read_csv(path/'e3_controls_per_fit.csv',float_precision='round_trip')
    source=json.loads((path/'e3_source_manifest.json').read_text())
    if source['platform']!=platform or len(df)!=60 or set(df.platform)!={platform}:
        raise RuntimeError(f'STOP E3 CROSS-PLATFORM: wrong platform or record count {platform}')
    if len(df.duplicated(KEY)[lambda s:s]) or set(map(tuple,df[KEY].itertuples(index=False,name=None))) != {
            (r,m,c) for r in range(10) for m in METHODS for c in CONDS}:
        raise RuntimeError(f'STOP E3 CROSS-PLATFORM: incomplete/duplicate pairs {platform}')
    if not df.n_checkpoint_pairs.eq(200).all() or not df.metadata_only_majority_accuracy.eq(.5).all():
        raise RuntimeError(f'STOP E3 CROSS-PLATFORM: pairing/metadata fail {platform}')
    if df['benign_score_max_abs_diff_vs_id'].max()>1e-6:
        raise RuntimeError(f'STOP E3 CROSS-PLATFORM: benign identity fail {platform}')
    if not np.isfinite(df[METRICS].to_numpy(dtype=float)).all():
        raise RuntimeError(f'STOP E3 CROSS-PLATFORM: nonfinite value {platform}')
    return df.sort_values(KEY).reset_index(drop=True)


def run(ubuntu:Path,mac:Path,out:Path):
    if out.exists():
        raise RuntimeError('STOP E3 CROSS-PLATFORM: output path already exists; refuse overwrite')
    a=load(ubuntu,'ubuntu'); b=load(mac,'mac_m4')
    ma=json.loads((ubuntu/'e3_source_manifest.json').read_text())
    mb=json.loads((mac/'e3_source_manifest.json').read_text())
    for field in ('e3_script_sha256', 'runner_sha256', 'splits_sha256'):
        if ma.get(field)!=mb.get(field):
            raise RuntimeError(f'STOP E3 CROSS-PLATFORM: {field} differs between machines')
    if not a[KEY].equals(b[KEY]):
        raise RuntimeError('STOP E3 CROSS-PLATFORM: fit/method/condition mismatch')
    merged=a[KEY].copy()
    for measure in METRICS:
        merged[f'ubuntu_{measure}']=a[measure]
        merged[f'mac_m4_{measure}']=b[measure]
        merged[f'mac_minus_ubuntu_{measure}']=b[measure]-a[measure]
    out.mkdir(parents=True,exist_ok=False)
    merged.to_csv(out/'e3_cross_platform_per_fit.csv',index=False)
    rows=[]
    for method in METHODS:
        for condition in CONDS:
            x=merged[(merged.method==method)&(merged.condition==condition)]
            for measure in METRICS:
                diff=x[f'mac_minus_ubuntu_{measure}']
                rows.append({'method':method,'condition':condition,'measure':measure,
                             'n_fits':len(x),'ubuntu_mean':float(x[f'ubuntu_{measure}'].mean()),
                             'mac_m4_mean':float(x[f'mac_m4_{measure}'].mean()),
                             'mac_minus_ubuntu_mean':float(diff.mean()),
                             'mac_minus_ubuntu_sd':float(diff.std(ddof=1))})
    stat=pd.DataFrame(rows)
    stat.to_csv(out/'e3_cross_platform_summary.csv',index=False)
    lines=['# E3 — Controlli su benigni appaiati e metadati: confronto Ubuntu / Mac M4','',
           '> Analisi retrospettiva di predizioni E2 già viste, NON secondo blind test indipendente.',
           '> L’intensità e la capacità dei payload differiscono ancora: E3 non identifica causalmente la famiglia di attacco.', '',
           '## Controlli strutturali','',
           '- Entrambe le piattaforme: 10 fit, 200 checkpoint appaiati per fit/condizione/metodo.',
           '- Metadata-only majority baseline 50% per identità checkpoint.',
           '- Score benigni e FPR costanti al variare di ID/OOD, a metodo e soglia congelati.', '',
           '## Confronti medi sui 10 fit', '',
           '| Metodo | Condizione | AUROC Ubuntu / Mac | Concordanza entro coppia Ubuntu / Mac | TPR Ubuntu / Mac | FPR Ubuntu / Mac |',
           '|---|---|---:|---:|---:|---:|']
    for method in METHODS:
        for condition in CONDS:
            r=merged[(merged.method==method)&(merged.condition==condition)]
            def pair(metric):
                return f'{r[f"ubuntu_{metric}"].mean():.4f} / {r[f"mac_m4_{metric}"].mean():.4f}'
            lines.append(f'| {method} | {condition} | {pair("auroc")} | {pair("paired_concordance")} | {pair("tpr_val_youden")} | {pair("fpr_val_youden")} |')
    lines += ['', 'Le 10 ripetizioni riutilizzano i medesimi 200 checkpoint test. Le statistiche sui fit sono descrittive, non 2000 unità indipendenti.',
              '', '## Cosa E3 non dimostra', '',
              '- Che tutte le informazioni spurie siano assenti: controlliamo gli identificatori e score benigni, non ogni possibile artefatto a livello di rappresentazione.',
              '- Che lo shift di famiglia *da solo* causi il calo: il payload spread è da 16 bit e il relativo L2 differisce da X-LSB X=23.',
              '- Che questo test costituisca un nuovo holdout esterno; la stessa zoo e test identity sono già state analizzate in E2.', '']
    (out/'E3_CROSS_PLATFORM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sub=stat[(stat.method=='centroid') & (stat.measure=='tpr_val_youden')].set_index('condition').reindex(CONDS)
        xx=np.arange(3); width=0.36
        fig,ax=plt.subplots(figsize=(8,4.7))
        ax.bar(xx-width/2,sub.ubuntu_mean.to_numpy(),width,label='Ubuntu')
        ax.bar(xx+width/2,sub.mac_m4_mean.to_numpy(),width,label='Mac M4')
        ax.set_xticks(xx,CONDS)
        ax.set_ylim(0,1);ax.set_ylabel('TPR — soglia Youden da validation LSB')
        ax.set_title('E3: variazione del TPR con benigni e FPR controllati')
        ax.legend();fig.tight_layout()
        fig.savefig(out/'e3_centroid_tpr_by_condition.png',dpi=180)
        plt.close(fig)
    except ImportError:
        print('Plot skipped (matplotlib missing); all CSV and report produced.')
    print('E3 CROSS-PLATFORM QA OK: 60 paired fit/method/condition records per platform')
    print('Saved to:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--ubuntu',required=True,type=Path)
    p.add_argument('--mac',required=True,type=Path);p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args();run(a.ubuntu.expanduser().resolve(),a.mac.expanduser().resolve(),a.out_dir.expanduser().resolve())
if __name__=='__main__':main()
