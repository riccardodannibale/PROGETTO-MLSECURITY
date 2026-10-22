#!/usr/bin/env python3
"""QA sorgenti aggiuntive Mac E1: config, summary, payload manifest. Non esegue training."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / 'results/replicas/e1_mac_m4'
AUD = MAC / 'audit'
UBU = ROOT / 'results/test/e1_uniform_v2'
EXPECTED_RUNNER_SHA = 'a75aa3d844bb695a6d6155fe808394025b04c480d74d1e1b856eec5ca6f9c037'
PAYLOAD_SCHEME = 'sha256-keyed-uint8-pcg64-per-model-v1'


def readcsv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def check(cond, message):
    if not cond:
        raise SystemExit('QA FAIL: ' + message)


def main():
    mac_test = json.loads((AUD / 'run_config_test.json').read_text())
    mac_val = json.loads((AUD / 'run_config_validation.json').read_text())
    ubu_test = json.loads((UBU / 'run_config.json').read_text())
    comparable = ('mode', 'imsize', 'n_repeats', 'n_train_identities', 'seed',
                  'weights_sha256', 'splits_sha256', 'upstream_commit',
                  'payload_nbytes', 'payload_scheme', 'x_range')
    for field in comparable:
        check(mac_test[field] == mac_val[field] == ubu_test[field],
              f'config non corrispondente: {field}')
    check(mac_test['stage'] == 'test' and mac_val['stage'] == 'validation', 'stage config')
    check(mac_test['project_commit'].startswith('59f556c'), 'Mac project_commit test')
    check(ubu_test['project_commit'].startswith('f5bf86b'), 'Ubuntu project_commit test')

    runner = ROOT / 'scripts/run_e1_final_v2.py'
    check(runner.is_file(), 'runner E1 v2 non presente')
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    check(runner_sha == EXPECTED_RUNNER_SHA, 'hash runner diverso dallo SHA-256 comunicato')

    manifest = readcsv(AUD / 'payload_manifest_r00_x23.csv')
    check(len(manifest) == 403, 'payload manifest numero righe')
    counts = {split: sum(r['split'] == split for r in manifest)
              for split in ('train', 'val', 'test')}
    check(counts == {'train': 3, 'val': 200, 'test': 200}, f'split nel manifest: {counts}')
    check(len({(r['split'], r['row_index']) for r in manifest}) == 403,
          'identità duplicate nello split del payload manifest')
    for r in manifest:
        check(int(r['repeat']) == 0 and int(r['X']) == 23 and
              int(r['payload_nbytes']) == 16384 and
              r['payload_scheme'] == PAYLOAD_SCHEME, 'schema payload non coerente')
        key = f'{PAYLOAD_SCHEME}|42|0|23|{int(r["row_index"])}'.encode('ascii')
        seed = int.from_bytes(hashlib.sha256(key).digest()[:16], 'big')
        rng = np.random.Generator(np.random.PCG64(seed))
        digest = hashlib.sha256(rng.integers(0, 256, size=16384,
                                         dtype=np.uint8).tobytes()).hexdigest()
        check(digest == r['payload_sha256'], f'payload SHA errato row {r["row_index"]}')

    valrows = readcsv(AUD / 'validation_standalone_summary.csv')
    check(len(valrows) == 16, 'validation stand-alone deve contenere 16 righe')
    combos = {(int(r['X']), r['method'], r['policy']) for r in valrows}
    expected = {(x, m, p) for x in (1, 8, 16, 23)
                for m in ('centroid', 'nn')
                for p in ('original_zero', 'val_youden')}
    check(combos == expected, 'validation stand-alone combinazioni')
    check(all(r['stage'] == 'validation' and int(r['n_runs']) == 10
              for r in valrows), 'validation stand-alone stage/repeats')

    mac_summary = readcsv(MAC / 'summary.csv')
    perrun = readcsv(MAC / 'metrics_per_run.csv')
    check(len(mac_summary) == 32 and len(perrun) == 320, 'summary/per-run test Mac')
    for aggr in mac_summary:
        subset = [r for r in perrun if
                  all(r[k] == aggr[k] for k in ('stage','X','method','policy'))]
        check(len(subset) == 10, 'repliche mancanti nel summary Mac')
        for name in ('auroc', 'accuracy', 'fpr', 'auprc', 'f1'):
            computed = statistics.mean(float(r[name]) for r in subset)
            check(abs(computed - float(aggr[name+'_mean'])) < 1e-9,
                  f'aggregato Mac differente {name}')
    val_x23 = next(r for r in valrows if r['X'] == '23'
                   and r['method'] == 'centroid' and r['policy'] == 'val_youden')
    print('E1 HARDWARE QA OK')
    print(f'Runner SHA-256: {runner_sha}')
    print(f'Configurazioni Mac/Ubuntu: identiche nei campi {len(comparable)} controllati')
    print(f'Mac payload manifest r00 X23: {len(manifest)}/403 SHA-256 OK')
    print('Mac validation stand-alone: 16 aggregate (10 runs ciascuna)')
    print('Mac campagna test: 40 fit, 320 record, medie OK')
    print(f'Mac validation stand-alone X23 AUROC centroid: {float(val_x23["auroc_mean"]):.6f}')
    print('Nota: il diff Git fra due commit è riferito da Fabrizio, non verificato da questo script.')


if __name__ == '__main__':
    main()
