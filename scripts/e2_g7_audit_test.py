#!/usr/bin/env python3
"""Independent read-only QA of E2 paired ID/OOD test outputs. Strict CSV round-trip."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

RUNNER_SHA='4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac'
WEIGHTS_SHA='5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad'
SPLITS_SHA='3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f'
CONDITIONS={'LSB_ID','SPREAD_0.24','SPREAD_0.16'}
METHODS={'centroid','nn'}
POLICIES={'original_zero','val_youden'}


def ck(ok, message):
    if not ok:
        raise RuntimeError(f'STOP E2 G7 TEST QA: {message}')


def near(a, b, msg, atol=1e-9):
    ck(bool(np.isclose(float(a), float(b), rtol=0, atol=atol)), msg)


def predictions(df, *, rep, split, allowed, ids, thresholds):
    ck(len(df)==400*len(allowed)*2, f'{split} row count repeat={rep}')
    ck(set(df['condition'])==allowed and set(df['method'])==METHODS, f'{split} method/condition repeat={rep}')
    ck(set(df['stage'])=={split} and set(df['repeat'])=={rep} and set(df['fit_seed'])=={42+rep*1000+23}, f'{split} stage/seed repeat={rep}')
    ck(set(df['row_index'].astype(int))==ids, f'{split} source ids repeat={rep}')
    ck(df['score_margin'].notna().all() and bool(np.isfinite(df['score_margin'].to_numpy()).all()), f'{split} score nonfinite repeat={rep}')
    for condition in allowed:
        for method in METHODS:
            group=df[(df.condition==condition)&(df.method==method)]
            ck(len(group)==400, f'{split} 400 rows repeat={rep}/{condition}/{method}')
            ck(group['row_index'].nunique()==200, f'{split} identities repeat={rep}/{condition}/{method}')
            ck(int((group.label==0).sum())==200 and int((group.label==1).sum())==200, f'{split} balance repeat={rep}/{condition}/{method}')
            cnt=group.groupby('row_index')['label'].apply(lambda s:tuple(sorted(s.astype(int).tolist())))
            ck(all(v==(0,1) for v in cnt), f'{split} paired identities repeat={rep}/{condition}/{method}')
            thr=float(thresholds[method])
            vals=group['validation_threshold'].to_numpy(dtype=np.float64)
            ck(bool(np.array_equal(vals,np.full(len(group),thr))), f'{split} CSV/JSON threshold mismatch repeat={rep}/{condition}/{method}')
            expected=(group['score_margin'].to_numpy(dtype=np.float64)>thr).astype(int)
            ck(bool(np.array_equal(expected,group['prediction_calibrated'].to_numpy(dtype=int))), f'{split} threshold decision mismatch repeat={rep}/{condition}/{method}')
            ck(set(group['X'].astype(int))=={23}, f'{split} expected X metadata repeat={rep}/{condition}/{method}')


def audit_metric(m, pred, rep):
    cond=m['condition']; method=m['method']; policy=m['policy']; stage=m['stage']
    grp=pred[(pred.condition==cond)&(pred.method==method)]
    ck(len(grp)==400, f'metric source {rep}/{stage}/{cond}/{method}')
    y=grp['label'].to_numpy(dtype=int)
    score=grp['score_margin'].to_numpy(dtype=np.float64)
    thr=float(m['threshold'])
    predicted=(score>thr).astype(int)
    ck(m['n']==400 and m['n_benign']==m['n_stego']==200, f'metric support mismatch {rep}/{cond}/{method}/{policy}')
    near(accuracy_score(y,predicted),m['accuracy'],f'accuracy mismatch {rep}/{stage}/{cond}/{method}/{policy}')
    near(f1_score(y,predicted,zero_division=0),m['f1'],f'F1 mismatch {rep}/{stage}/{cond}/{method}/{policy}')
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    ck([int(tn),int(fp),int(fn),int(tp)]==[m['tn'],m['fp'],m['fn'],m['tp']], f'confusion mismatch {rep}/{stage}/{cond}/{method}/{policy}')
    near(fp/200,m['fpr'],f'FPR mismatch {rep}/{stage}/{cond}/{method}/{policy}')
    near(roc_auc_score(y,score),m['auroc'],f'AUROC mismatch {rep}/{stage}/{cond}/{method}/{policy}',atol=1e-8)
    near(average_precision_score(y,score),m['auprc'],f'AUPRC mismatch {rep}/{stage}/{cond}/{method}/{policy}',atol=1e-8)


def audit(base:Path,split_csv:Path):
    ck(base.is_dir(),'missing test directory')
    config=json.loads((base/'run_config.json').read_text())
    ck(config['stage']=='test' and config['runner_sha256']==RUNNER_SHA, 'test stage/runner SHA mismatch')
    ck(config['weights_sha256']==WEIGHTS_SHA and config['splits_sha256']==SPLITS_SHA,'dataset/split digest mismatch')
    ck(config['x']==23 and config['n_repeats']==10 and config['n_train_identities']==3,'unexpected number of runs/train')
    ck(config['alphas']==[0.24,0.16] and config['spread_bits']==16, 'OOD params mismatch')
    ck(hashlib.sha256(split_csv.read_bytes()).hexdigest()==SPLITS_SHA,'split manifest hash mismatch')
    split=pd.read_csv(split_csv)
    ids={name:set(split.loc[split['split']==name,'row_index'].astype(int)) for name in ('train','val','test')}
    ck(all(len(ids[name])==length for name,length in (('train',600),('val',200),('test',200))),'split cardinalities')
    ck(not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']),'source leakage between splits')
    ck(len(list(base.glob('repeat_*_x_23')))==10, 'unexpected repeat directory count')
    raw_auroc=[]; all_metrics=[]; recovery={0.24:[],0.16:[]}
    for rep in range(10):
        d=base/f'repeat_{rep:02d}_x_23'
        ck(d.is_dir(), f'missing repeat {rep}')
        metrics=json.loads((d/'metrics.json').read_text())
        ck(metrics['repeat']==rep and metrics['stage']=='test' and metrics['fit_seed']==42+rep*1000+23,f'fit metadata {rep}')
        ck(len(metrics['train_source_indices'])==3 and set(metrics['train_source_indices'])<=ids['train'],f'training identities {rep}')
        ck(set(metrics['validation_source_indices'])==ids['val'] and set(metrics['test_source_indices'])==ids['test'], f'validation/test identities {rep}')
        ck(len(metrics['metrics'])==16,f'16 metric combos {rep}')
        combos={(m['stage'],m['condition'],m['method'],m['policy']) for m in metrics['metrics']}
        wanted={(stage,condition,method,policy)
                for stage,conds in (('validation',{'LSB_ID'}),('test',CONDITIONS))
                for condition in conds for method in METHODS for policy in POLICIES}
        ck(combos==wanted,f'incomplete unique metric combos {rep}')
        thresholds=metrics['thresholds_val_youden']
        ck(set(thresholds)==METHODS and all(np.isfinite(list(thresholds.values()))),f'thresholds {rep}')
        val=pd.read_csv(d/'predictions_validation.csv',float_precision='round_trip')
        test=pd.read_csv(d/'predictions_test.csv',float_precision='round_trip')
        predictions(val,rep=rep,split='validation',allowed={'LSB_ID'},ids=ids['val'],thresholds=thresholds)
        predictions(test,rep=rep,split='test',allowed=CONDITIONS,ids=ids['test'],thresholds=thresholds)
        # As checked at representation level in the detector, benign examples are identical
        # across families; compare scores by checkpoint identity, not by row order.
        for method in METHODS:
            pivot=(test[(test.method==method)&(test.label==0)]
                   .pivot(index='row_index',columns='condition',values='score_margin'))
            ck(pivot.shape==(200,3),f'benign comparison missing {rep}/{method}')
            diff=np.max(np.abs(pivot['LSB_ID'].to_numpy()[:,None]-pivot[['SPREAD_0.24','SPREAD_0.16']].to_numpy()))
            ck(diff<=1e-6,f'benign score different across attack families repeat={rep}/{method}: max={diff}')
        for m in metrics['metrics']:
            near(m['threshold'],0.0 if m['policy']=='original_zero' else thresholds[m['method']],f'metric threshold differs repeat={rep}/{m["condition"]}/{m["method"]}',atol=0)
            audit_metric(m,val if m['stage']=='validation' else test,rep)
            all_metrics.append(dict(repeat=rep,**m))
            if m['stage']=='test' and m['method']=='centroid' and m['policy']=='val_youden':
                raw_auroc.append((rep,m['condition'],float(m['auroc'])))
        payload=pd.read_csv(d/'payload_manifest.csv',float_precision='round_trip')
        ck(len(payload)==803,f'803 payloads repeat={rep}')
        ck(int((payload.source_split=='train').sum())==3 and int((payload.source_split=='val').sum())==200 and int((payload.source_split=='test').sum())==600,f'payload split counts {rep}')
        test_p=payload.loc[payload.source_split=='test']
        ck(set(test_p.row_index.astype(int))==ids['test'],f'payload test identities {rep}')
        ck(int((test_p.family=='LSB_ID').sum())==200 and int((test_p.family=='SPREAD_v0').sum())==400,f'payload OOD family counts {rep}')
        for alpha in (0.24,0.16):
            part=test_p[(test_p.family=='SPREAD_v0')&np.isclose(test_p.alpha.astype(float),alpha,atol=1e-12)]
            ck(len(part)==200 and set(part.row_index.astype(int))==ids['test'],f'spread metadata alpha={alpha} repeat={rep}')
            ck(set(part.n_bits.astype(int))=={16},f'spread bit count {rep}/{alpha}')
            ck(bool((part.bit_errors.between(0,16)).all()),f'spread bit error bounds {rep}/{alpha}')
            ck(bool(np.allclose(part.ber.to_numpy(dtype=float),part.bit_errors.to_numpy(dtype=float)/16,rtol=0,atol=1e-12)),f'BER mismatch {rep}/{alpha}')
            ck(bool(np.array_equal((part.bit_errors.to_numpy(dtype=int)==0).astype(int),part.exact_recovery.to_numpy(dtype=int))),f'exact recovery mismatch {rep}/{alpha}')
            recovery[alpha].extend(part.exact_recovery.astype(int).tolist())
        print(f'G7 TEST QA repeat={rep:02d} OK: 800 validation preds; 2400 paired test preds; 803 payloads')
    raw=pd.read_csv(base/'metrics_per_run.csv',float_precision='round_trip')
    summary=pd.read_csv(base/'summary.csv',float_precision='round_trip')
    paired=pd.read_csv(base/'paired_auroc_per_repeat.csv',float_precision='round_trip')
    ck(len(raw)==160 and len(summary)==16 and set(summary.n_runs)=={10},'aggregate metrics count')
    ck(len(paired)==10 and set(paired['repeat'].astype(int))==set(range(10)),'paired delta count')
    ck(set(paired.columns)>=CONDITIONS|{'delta_auroc_ood24_minus_id','delta_auroc_ood16_minus_id'},'paired CSV columns')
    actual=pd.DataFrame(raw_auroc,columns=['repeat','condition','auroc']).pivot(index='repeat',columns='condition',values='auroc').sort_index()
    check=paired.set_index('repeat').sort_index()
    for condition in CONDITIONS:
        ck(bool(np.allclose(actual[condition].to_numpy(),check[condition].to_numpy(),rtol=0,atol=1e-12)),f'paired AUROC {condition}')
    for alpha,suf in ((0.24,'24'),(0.16,'16')):
        col=f'delta_auroc_ood{suf}_minus_id'
        expected=actual[f'SPREAD_0.{suf}'].to_numpy()-actual['LSB_ID'].to_numpy()
        ck(bool(np.allclose(expected,check[col].to_numpy(),rtol=0,atol=1e-12)),f'paired delta alpha {alpha}')
    for _,row in summary.iterrows():
        subset=raw[(raw.stage==row.stage)&(raw.condition==row.condition)&(raw.method==row.method)&(raw.policy==row.policy)]
        ck(len(subset)==10,f'aggregate subset {row.method}/{row.condition}/{row.policy}')
        near(subset.auroc.mean(),row.auroc_mean,'aggregate AUROC mean')
    print('\nE2 G7 OFFICIAL PAIRED TEST QA OK: 10/10 fits, 160 metrics, 8000 validation and 24000 test predictions, 8030 payload rows')
    print('Primary per-fit paired delta OOD alpha=.24 minus ID:\n', check['delta_auroc_ood24_minus_id'].to_string())
    print('PRIMARY ΔAUROC mean:',float(check['delta_auroc_ood24_minus_id'].mean()))
    print('PRIMARY ΔAUROC SD (repeat-level):',float(check['delta_auroc_ood24_minus_id'].std(ddof=1)))
    for alpha in (0.24,0.16):
        values=recovery[alpha]
        print(f'OOD alpha={alpha:.2f} exact bit recovery {sum(values)}/{len(values)} checkpoint×repeat embeds')
    print('NOT new independent external holdout; no post-test tuning.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--split-csv',type=Path,default=Path('manifests/e1_uniform_v1/splits.csv'))
    a=p.parse_args();audit(a.test_dir.expanduser().resolve(),a.split_csv.expanduser().resolve())
