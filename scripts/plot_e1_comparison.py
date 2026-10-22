#!/usr/bin/env python3
"""Rigenera figure comparative E1 v2 Ubuntu vs Mac dai summary CSV versionati.

Non carica checkpoint, non esegue training, non apre mai i singoli dati test.
Lanciarlo dalla repo con: python scripts/plot_e1_comparison.py
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
XS=(1,8,16,23)

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as fp:
        rows=list(csv.DictReader(fp))
    if len(rows)!=32 or {(r['stage']) for r in rows} != {'validation','test'}:
        raise ValueError(f'Expected 32 records including validation and test: {path}')
    if any(int(r['n_runs'])!=10 for r in rows):
        raise ValueError('Expected exactly 10 repetitions per configuration')
    return {(r['stage'],int(r['X']),r['method'],r['policy']):r for r in rows}

def v(data,x,stage,method,policy,metric):
    return float(data[(stage,x,method,policy)][metric])

def get(data,stage,method,policy,metric):
    return np.array([v(data,x,stage,method,policy,metric) for x in XS])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ubuntu',type=Path,default=ROOT/'results/test/e1_uniform_v2/summary.csv')
    parser.add_argument('--mac',type=Path,default=ROOT/'results/replicas/e1_mac_m4/summary.csv')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/figures/e1')
    a=parser.parse_args()
    ubu=read(a.ubuntu); mac=read(a.mac)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    def save(name):
        plt.savefig(a.out_dir/name,bbox_inches='tight',dpi=220)
        plt.close()
        print('OK',a.out_dir/name)
    def base(title,ylabel,ylim):
        fig,ax=plt.subplots(figsize=(8.8,4.9))
        ax.set(title=title,xlabel='X-LSB (bit di mantissa)',ylabel=ylabel,ylim=ylim,xticks=XS)
        ax.grid(alpha=.23)
        return fig,ax

    fig,ax=base('E1 v2 | AUROC centroid sul test | confronto CPU','AUROC media su 10 fit',(.46,1))
    ax.plot(XS,get(ubu,'test','centroid','original_zero','auroc_mean'),marker='o',linewidth=2,label='Ubuntu x86_64 (VirtualBox)')
    ax.plot(XS,get(mac,'test','centroid','original_zero','auroc_mean'),marker='s',linewidth=2,label='Mac M4 ARM64 (CPU)')
    ax.axhline(.5,ls='--',lw=1,label='Random baseline 0,5')
    ax.legend(loc='upper left');save('e1_platform_auroc.png')

    fig,ax=base('E1 v2 | Accuracy centroid, soglia Youden dalla validation','Accuracy media ± SD tra 10 fit',(0,1))
    ix=np.arange(4); width=.38
    for i,(d,name) in enumerate(((ubu,'Ubuntu x86_64'),(mac,'Mac M4 ARM64'))):
        ax.bar(ix+(i-.5)*width,get(d,'test','centroid','val_youden','accuracy_mean'),width,
               yerr=get(d,'test','centroid','val_youden','accuracy_std'),capsize=3,label=name)
    ax.set_xticks(ix,[str(x) for x in XS]); ax.set_xlabel('X-LSB');ax.axhline(.5,ls='--',lw=1)
    ax.legend();save('e1_platform_accuracy_youden.png')

    fig,ax=base('E1 v2 | Falsi positivi centroid, soglia validation Youden','FPR medio sul test',(0,.65))
    ax.plot(XS,get(ubu,'test','centroid','val_youden','fpr_mean'),marker='o',linewidth=2,label='Ubuntu x86_64')
    ax.plot(XS,get(mac,'test','centroid','val_youden','fpr_mean'),marker='s',linewidth=2,label='Mac M4 ARM64')
    ax.legend();save('e1_platform_fpr_youden.png')

    fig,ax=base('E1 v2 | AUROC centroid vs 1-NN, due piattaforme','AUROC media sul test',(.46,1))
    for d,name,mark in ((ubu,'Ubuntu','o'),(mac,'Mac M4','s')):
        for method,ls in (('centroid','-'),('nn','--')):
            ax.plot(XS,get(d,'test',method,'original_zero','auroc_mean'),marker=mark,ls=ls,linewidth=2,
                    label=f'{name} · {method}')
    ax.axhline(.5,ls=':',lw=1);ax.legend(fontsize=9,loc='upper left');save('e1_platform_methods.png')

    fig,ax=base('E1 v2 | Validation dei fit test vs test (NO validation sviluppo)','AUROC media centroid',(.46,1))
    for d,name,mark in ((ubu,'Ubuntu','o'),(mac,'Mac M4','s')):
        for stage,ls in (('validation','--'),('test','-')):
            ax.plot(XS,get(d,stage,'centroid','original_zero','auroc_mean'),marker=mark,ls=ls,
                    linewidth=1.9,label=f'{name} · {stage}')
    ax.axhline(.5,ls=':',lw=1);ax.legend(fontsize=9,loc='upper left');save('e1_platform_validation_test.png')

    for x in XS:
        u=v(ubu,x,'test','centroid','original_zero','auroc_mean')
        m=v(mac,x,'test','centroid','original_zero','auroc_mean')
        print(f'X={x:2} Ubuntu={u:.6f} Mac={m:.6f} delta={m-u:+.6f}')
if __name__=='__main__':main()
