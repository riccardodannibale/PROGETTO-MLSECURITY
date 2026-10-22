#!/usr/bin/env python3
"""E3: read-only post-hoc matched-benign, metadata and threshold controls on E2 test runs.

The E2 detector, attack encoders, test labels, parameters and thresholds are
never changed. This is a retrospective diagnostic after E2 test was opened,
NOT an independent confirmatory experiment, nor matched attack intensity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNNER_SHA = '4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac'
SPLITS_SHA = '3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f'
WEIGHTS_SHA = '5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad'
CONDITIONS = ('LSB_ID', 'SPREAD_0.24', 'SPREAD_0.16')
METHODS = ('centroid', 'nn')
META = ('model_id', 'checkpoint_sha256', 'fit_seed', 'X', 'repeat', 'stage')


def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(f'STOP E3 CONTROLS: {msg}')


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_group(group: pd.DataFrame, *, expected_ids: set[int], context: str) -> dict:
    """Require one label=0 and one label=1 with identical source metadata per identity."""
    require(len(group) == 2 * len(expected_ids), f'{context}: row count')
    require(group['row_index'].notna().all(), f'{context}: missing row index')
    require(set(group['row_index'].astype(int)) == expected_ids, f'{context}: checkpoint identities')
    require(set(group['label'].astype(int)) == {0, 1}, f'{context}: label support')
    # Validate exact pairing; metadata cannot predict label within a checkpoint.
    p = group.set_index(['row_index', 'label']).sort_index()
    require(p.index.is_unique, f'{context}: duplicate source+label')
    expected = pd.MultiIndex.from_product([sorted(expected_ids), [0, 1]], names=['row_index', 'label'])
    require(p.index.equals(expected), f'{context}: not precisely 1:1 matched')
    for col in META:
        require(col in p.columns, f'{context}: missing metadata {col}')
        benign = p.xs(0, level='label')[col]
        stego = p.xs(1, level='label')[col]
        require(benign.notna().all() and stego.notna().all(), f'{context}: missing {col}')
        require(bool((benign == stego).all()), f'{context}: {col} predicts label / mispaired')
    scores0 = p.xs(0, level='label')['score_margin'].to_numpy(dtype=np.float64)
    scores1 = p.xs(1, level='label')['score_margin'].to_numpy(dtype=np.float64)
    require(bool(np.isfinite(scores0).all() and np.isfinite(scores1).all()), f'{context}: nonfinite scores')
    values = scores1 - scores0
    higher = int(np.count_nonzero(values > 0))
    equal = int(np.count_nonzero(values == 0))
    lower = int(np.count_nonzero(values < 0))
    require(higher + equal + lower == len(expected_ids), f'{context}: sign count')
    return dict(n_pairs=len(expected_ids), higher=higher, ties=equal, lower=lower,
                paired_concordance=(higher + 0.5 * equal) / len(expected_ids),
                paired_score_gap_mean=float(np.mean(values)),
                benign_scores=scores0, stego_scores=scores1, group_sorted=p)


def compute_controls(base: Path, split_path: Path, *, platform: str):
    require(base.is_dir(), 'missing E2 test output directory')
    require(split_path.is_file() and digest(split_path) == SPLITS_SHA, 'unexpected frozen split checksum')
    conf = json.loads((base / 'run_config.json').read_text())
    require(conf['stage'] == 'test' and conf['runner_sha256'] == RUNNER_SHA, 'wrong test stage/runner')
    require(conf['weights_sha256'] == WEIGHTS_SHA and conf['splits_sha256'] == SPLITS_SHA, 'weights/splits digest')
    require(conf['n_repeats'] == 10 and conf['x'] == 23 and conf['alphas'] == [0.24, 0.16], 'altered experiment')
    table = pd.read_csv(split_path)
    for col in ('split', 'row_index', 'model_id', 'checkpoint_sha256'):
        require(col in table, f'split missing {col}')
    splits = {s: table[table.split == s].set_index('row_index') for s in ('train','val','test')}
    require([len(splits[s]) for s in ('train','val','test')] == [600,200,200], 'wrong split cardinalities')
    ids = set(splits['test'].index.astype(int))
    require(len(ids) == 200, 'nonunique test identities')
    require(not(set(splits['train'].index) & ids or set(splits['val'].index) & ids), 'checkpoint leakage')
    require(len(list(base.glob('repeat_*_x_23'))) == 10, 'unexpected repeat directories')
    expected_meta = splits['test'][['model_id', 'checkpoint_sha256']].sort_index()
    result, manifest = [], {'platform': platform, 'source_dir': str(base), 'splits_sha256': SPLITS_SHA,
                              'runner_sha256': RUNNER_SHA, 'e3_script_sha256': digest(Path(__file__)), 'inputs_sha256': {}, 'n_repeats': 10,
                              'kind': 'read-only retrospective E3 diagnostic; NOT independent blind test'}
    for rep in range(10):
        run = base / f'repeat_{rep:02d}_x_23'
        require(run.is_dir(), f'missing fit {rep}')
        pred_path = run / 'predictions_test.csv'
        met_path = run / 'metrics.json'
        payload_path = run / 'payload_manifest.csv'
        for path in (pred_path,met_path,payload_path):
            require(path.is_file(), f'missing source {path}')
            manifest['inputs_sha256'][str(path.relative_to(base))] = digest(path)
        m = json.loads(met_path.read_text())
        require(m['repeat'] == rep and m['stage'] == 'test' and m['fit_seed'] == 42+1000*rep+23, f'fit config {rep}')
        require(len(m['train_source_indices']) == 3 and set(m['train_source_indices']) <= set(splits['train'].index), f'train identities {rep}')
        require(set(m['validation_source_indices']) == set(splits['val'].index) and set(m['test_source_indices']) == ids, f'val/test identities {rep}')
        thresholds = m['thresholds_val_youden']
        require(set(thresholds) == set(METHODS), f'calibration methods {rep}')
        pred = pd.read_csv(pred_path, float_precision='round_trip', dtype={'model_id': str, 'checkpoint_sha256': str})
        require(len(pred) == 2400 and set(pred.condition) == set(CONDITIONS) and set(pred.method) == set(METHODS), f'prediction coverage {rep}')
        require(set(pred.stage) == {'test'} and set(pred['repeat']) == {rep} and set(pred['fit_seed']) == {42+1000*rep+23}, f'prediction metadata {rep}')
        require(pred['score_margin'].notna().all() and bool(np.isfinite(pred['score_margin'].to_numpy(dtype=float)).all()), f'nonfinite predictions {rep}')
        benign_by_method = {}
        fpr_by_method = {}
        for method in METHODS:
            for condition in CONDITIONS:
                context = f'repeat={rep:02d}/{method}/{condition}'
                g = pred[(pred.method == method) & (pred.condition == condition)].copy()
                r = inspect_group(g, expected_ids=ids, context=context)
                p = r.pop('group_sorted')
                for col in ('model_id','checkpoint_sha256'):
                    actual = p.xs(0,level='label')[col].sort_index()
                    expected = expected_meta[col].astype(str)
                    require(bool((actual.astype(str).to_numpy() == expected.to_numpy()).all()), f'{context}: {col} differs from frozen manifest')
                require(set(g.X.astype(int)) == {23}, f'{context}: representation metadata X mismatch')
                # X is legacy E1 column, NOT a proof that spread attacks used LSB.
                t = float(thresholds[method])
                require(np.isfinite(t), f'{context}: nonfinite val threshold')
                require(bool(np.array_equal(g['validation_threshold'].to_numpy(dtype=np.float64),
                                            np.full(len(g),t,dtype=np.float64))),f'{context}: changed threshold')
                calc = (g['score_margin'].to_numpy(dtype=np.float64)>t).astype(int)
                require(bool(np.array_equal(calc,g['prediction_calibrated'].to_numpy(dtype=int))),f'{context}: threshold decisions mismatch')
                benign = r.pop('benign_scores')
                stego = r.pop('stego_scores')
                bpred = (benign>t).astype(int)
                spred = (stego>t).astype(int)
                fpr = float(np.mean(bpred))
                tpr = float(np.mean(spred))
                auc = float(roc_auc_score(g['label'].astype(int), g['score_margin'].astype(float)))
                recorded = [z for z in m['metrics'] if z['stage']=='test' and z['condition']==condition and
                            z['method']==method and z['policy']=='val_youden']
                require(len(recorded)==1 and np.isclose(auc,float(recorded[0]['auroc']),rtol=0,atol=1e-9),f'{context}: E2 AUROC conflict')
                require(np.isclose(fpr,float(recorded[0]['fpr']),rtol=0,atol=1e-12),f'{context}: E2 FPR conflict')
                if method not in benign_by_method:
                    benign_by_method[method]=benign
                    fpr_by_method[method]=fpr
                else:
                    max_diff=float(np.max(np.abs(benign_by_method[method]-benign)))
                    require(max_diff<=1e-6, f'{context}: benign scores differ from LSB; {max_diff}')
                    require(abs(fpr_by_method[method]-fpr)<1e-12, f'{context}: FPR differs across families')
                result.append({'platform':platform,'repeat':rep,'method':method,'condition':condition,
                               'n_checkpoint_pairs':r['n_pairs'], 'metadata_only_majority_accuracy':0.5,
                               'benign_score_max_abs_diff_vs_id':float(np.max(np.abs(benign_by_method[method]-benign))),
                               'fpr_val_youden':fpr,'tpr_val_youden':tpr,'fnr_val_youden':1-tpr,
                               'auroc':auc,**r})
        payload=pd.read_csv(payload_path,float_precision='round_trip')
        require(len(payload)==803 and (payload.source_split=='test').sum()==600, f'payload count {rep}')
        for a in (0.16,0.24):
            part=payload[(payload.source_split=='test') & (payload.family=='SPREAD_v0') & np.isclose(payload.alpha.astype(float),a,atol=1e-12)]
            require(len(part)==200 and set(part.row_index.astype(int)) == ids,f'payload alpha={a} {rep}')
            require(bool(np.allclose(part.relative_l2.astype(float),a,rtol=0,atol=1e-5)),f'alpha/norm {a} {rep}')
        print(f'E3 matched controls repeat={rep:02d} OK: 6 condition/method groups, 200 matched identities each')
    df=pd.DataFrame(result)
    require(len(df)==60 and df['metadata_only_majority_accuracy'].eq(0.5).all(), 'E3 completeness/balance')
    for method in METHODS:
        pivot=df[df.method==method].pivot(index='repeat',columns='condition',values='fpr_val_youden')
        require(bool(np.allclose(pivot['LSB_ID'],pivot['SPREAD_0.24'],rtol=0,atol=1e-12)) and
                bool(np.allclose(pivot['LSB_ID'],pivot['SPREAD_0.16'],rtol=0,atol=1e-12)),f'FPR control {method}')
    return df,manifest


def write_outputs(df:pd.DataFrame, manifest:dict, out:Path):
    require(not out.exists(),'output directory already exists (never overwrite an E3 report)')
    out.mkdir(parents=True,exist_ok=False)
    df.to_csv(out/'e3_controls_per_fit.csv',index=False)
    summary=(df.groupby(['platform','method','condition'],as_index=False)
             .agg(n_fits=('repeat','size'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),
                  matched_concordance_mean=('paired_concordance','mean'),
                  tpr_mean=('tpr_val_youden','mean'),fnr_mean=('fnr_val_youden','mean'),
                  fpr_mean=('fpr_val_youden','mean'),metadata_only_majority_accuracy=('metadata_only_majority_accuracy','mean'),
                  max_benign_score_difference=('benign_score_max_abs_diff_vs_id','max')))
    summary.to_csv(out/'e3_controls_summary.csv',index=False)
    (out/'e3_source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=[f'# E3 — Controlli retrospettivi, piattaforma {manifest["platform"]}', '',
      '> Sono analisi sui test E2 già consultati; NON sono un secondo blind holdout né una prova causale dello shift.',
      '> Detector, soglie e attacchi rimangono quelli congelati in E2; nessun nuovo training e nessun tuning.', '',
      '## Identità appaiate e negative controls', '',
      '- Per ogni repeat × metodo × condizione: 200 checkpoint originali, uno benigno e uno modificato per ciascuno.',
      '- Identificatori di checkpoint, model_id e SHA sono uguali entro la coppia e coincidono con lo split originale.',
      '- **Baseline metadata-only: 50%** se si usano esclusivamente questi campi appaiati, dato che per ciascuna identità ci sono entrambe le etichette con la stessa frequenza. Non è un classificatore addestrato.',
      '- Gli score benigni sono identici fra condizioni ID/OOD (entro 1e-6), così come l’FPR a soglia fissa per metodo e repeat.',
      '- Il campo CSV `X=23` è un campo ereditato dal runner per gli score, non descrive l’algoritmo spread o il suo valore alpha.', '',
      '## Misure descrittive', '',
      '| Metodo | Condizione | AUROC media | Concordanza entro coppia | TPR | FNR | FPR |',
      '|---|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
      lines.append(f"| {r['method']} | {r['condition']} | {r['auroc_mean']:.4f} | {r['matched_concordance_mean']:.4f} | {r['tpr_mean']:.4f} | {r['fnr_mean']:.4f} | {r['fpr_mean']:.4f} |")
    lines += ['', 'La concordanza entro coppia è P(score(modificato)>score(benigno)) con pareggi pesati a 0,5. È distinta dall’AUROC.',
     '', '## Limiti', '',
     '- Il controllo sui metadati non esclude scorciatoie di basso livello nella rappresentazione delle immagini.',
     '- L’esposizione precedente al test E2 rende questi risultati retrospettivi: non usarli per nuove scelte di modello/alpha.',
     '- X-LSB X=23 e spread α=0,16/0,24 non hanno perturbazione L2 e capacità di payload equivalenti.',
     '- Le stesse 200 identità di test ricorrono nei 10 fit; non trattare 2.000 checkpoint×repeat come indipendenti.',
     '- I controlli matched non dimostrano la causa del calo OOD; limitano alcune spiegazioni spurie.', '']
    (out/'E3_CONTROLS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'E3 MATCHED/METADATA CONTROLS QA OK: 10/10 fits, 60 method/condition records, 12000 original benign/stego pairs assessed')
    print(summary.to_string(index=False))
    print('Saved to:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--split-csv',type=Path,default=Path('manifests/e1_uniform_v1/splits.csv'))
    p.add_argument('--platform',choices=['ubuntu','mac_m4'],required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    frame,metadata=compute_controls(a.test_dir.expanduser().resolve(), a.split_csv.expanduser().resolve(),platform=a.platform)
    write_outputs(frame,metadata,a.out_dir.expanduser().resolve())

if __name__=='__main__':
    main()
