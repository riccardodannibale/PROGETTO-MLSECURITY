#!/usr/bin/env python3
"""Read-only paired E2 Ubuntu vs Mac result comparison, after independent G7 QA.

Inputs are test output directories or exported CSV-only directories. Never changes
thresholds, alpha, selection, labels, test predictions or detector parameters.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

METHOD='centroid'
POLICY='val_youden'
CONDITIONS=['LSB_ID','SPREAD_0.24','SPREAD_0.16']


def read(base:Path,platform:str):
    m=pd.read_csv(base/'metrics_per_run.csv',float_precision='round_trip')
    require={(r,c) for r in range(10) for c in CONDITIONS}
    select=m[(m.stage=='test') & (m.method==METHOD) & (m.policy==POLICY)]
    if {(int(r['repeat']),r['condition']) for _,r in select.iterrows()}!=require or len(select)!=30:
        raise ValueError(f'Incomplete primary data for {platform}')
    p=select.pivot(index='repeat',columns='condition',values='auroc').sort_index()
    p['delta_ood24_minus_id']=p['SPREAD_0.24']-p['LSB_ID']
    p['delta_ood16_minus_id']=p['SPREAD_0.16']-p['LSB_ID']
    return p


def run(ubuntu:Path,mac:Path,out:Path):
    if out.exists():
        raise RuntimeError('STOP: comparison output directory exists; no overwrites')
    a=read(ubuntu,'ubuntu');b=read(mac,'mac_m4')
    combined=pd.DataFrame({'repeat':a.index.astype(int)})
    metrics=CONDITIONS+['delta_ood24_minus_id','delta_ood16_minus_id']
    for col in metrics:
        combined[f'ubuntu_{col}']=a[col].to_numpy()
        combined[f'mac_m4_{col}']=b[col].to_numpy()
        combined[f'mac_minus_ubuntu_{col}']=(b[col]-a[col]).to_numpy()
    out.mkdir(parents=True,exist_ok=False)
    combined.to_csv(out/'e2_hardware_paired_per_repeat.csv',index=False)
    rows=[]
    for col in metrics:
        for plat,mat in (('ubuntu',a),('mac_m4',b)):
            rows.append({'platform':plat,'measure':col,'n_fits':10,
                         'mean':float(mat[col].mean()),'sd_repeat':float(mat[col].std(ddof=1)),
                         'minimum':float(mat[col].min()),'maximum':float(mat[col].max())})
        dif=b[col]-a[col]
        rows.append({'platform':'mac_minus_ubuntu','measure':col,'n_fits':10,
                     'mean':float(dif.mean()),'sd_repeat':float(dif.std(ddof=1)),
                     'minimum':float(dif.min()),'maximum':float(dif.max())})
    stat=pd.DataFrame(rows)
    stat.to_csv(out/'e2_hardware_summary.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot(a.index.to_numpy(),a['delta_ood24_minus_id'].to_numpy(),marker='o',label='Ubuntu')
        ax.plot(b.index.to_numpy(),b['delta_ood24_minus_id'].to_numpy(),marker='s',label='Mac M4')
        ax.axhline(0,linestyle='--',linewidth=1)
        ax.set(xlabel='Fit repeat',ylabel='OOD α=0.24 − LSB ID AUROC',title='E2: primary paired ΔAUROC by repeat')
        ax.set_xticks(range(10));ax.legend();fig.tight_layout()
        fig.savefig(out/'e2_primary_delta_by_repeat.png',dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for plat,mat in (('Ubuntu',a),('Mac M4',b)):
            ax.scatter(mat['LSB_ID'],mat['SPREAD_0.24'],label=plat,alpha=0.85)
        ax.plot([0,1],[0,1],linestyle='--',linewidth=1)
        ax.set(xlabel='ID LSB X=23 AUROC',ylabel='OOD spread α=0.24 AUROC',title='E2: same-fit ID vs OOD centroid AUROC')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend();fig.tight_layout()
        fig.savefig(out/'e2_id_vs_ood_auroc.png',dpi=170)
        plt.close(fig)
        graph_note='Due grafici PNG generati.'
    except ImportError:
        graph_note='matplotlib assente: CSV e sintesi creati senza grafici.'
    lines=[
        '# E2 — risultati del test interno ID/OOD e replica hardware',
        '',
        '> Test interno sul medesimo zoo GHRP/STL10 già usato in E1, non nuovo blind holdout esterno.',
        '> OOD spread ispirato allo spread-spectrum; NON MaleficNet originale. Intensità/capacità non matched.',
        '> Le 10 ripetizioni condividono identità di test: non sono dieci dataset indipendenti.',
        '',
        '| Misura | Ubuntu media ± SD (10 fit) | Mac M4 media ± SD (10 fit) |',
        '|---|---:|---:|',
    ]
    for label, col in [('AUROC ID X-LSB X=23','LSB_ID'),
                      ('AUROC OOD spread α=0.24','SPREAD_0.24'),
                      ('AUROC OOD spread α=0.16','SPREAD_0.16'),
                      ('**ΔAUROC primario (α=0.24 − ID)**','delta_ood24_minus_id'),
                      ('ΔAUROC secondario (α=0.16 − ID)','delta_ood16_minus_id')]:
        lines.append(f'| {label} | {a[col].mean():.6f} ± {a[col].std(ddof=1):.6f} | {b[col].mean():.6f} ± {b[col].std(ddof=1):.6f} |')
    lines+=['','## Interpretazione preregistrata','',
             'ΔAUROC < 0 indica una riduzione della discriminazione sullo spread rispetto a ID; ΔAUROC > 0 un aumento.',
             'Le SD sono fra fit, non fra checkpoint indipendenti. I 200 checkpoint test sono ripetuti fra fit.',
             'Non attribuire causalmente ogni differenza alla famiglia dell’attacco: LSB e spread differiscono in intensità e payload.',
             'Ogni detector è riaddestrato da zero con LSB per il proprio fit e valuta entrambe le famiglie senza cambiare soglia.',
             'Usare audit G7 OK su entrambe le piattaforme come prerequisito di questo report.',
             '',graph_note,'']
    (out/'E2_G7_CROSS_PLATFORM_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('E2 G7 CROSS PLATFORM SUMMARY READY:',out)
    print(stat.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--ubuntu',type=Path,required=True)
    p.add_argument('--mac',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();run(a.ubuntu.expanduser().resolve(),a.mac.expanduser().resolve(),a.out_dir.expanduser().resolve())
