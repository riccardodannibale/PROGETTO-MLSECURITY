#!/usr/bin/env python3
"""Audit DESCRITTIVO cross-platform E1; legge solo CSV versionati, niente training.

Output locale per evitare modifiche involontarie a report/test ufficiali.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import statistics as stats
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XS = (1, 8, 16, 23)
REPEATS = tuple(range(10))


def read_records(path: Path):
    if not path.is_file():
        raise SystemExit(f"ERRORE: non trovo {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 320:
        raise SystemExit(f"ERRORE: {path} contiene {len(rows)} record, attesi 320")
    expected = {(st, rep, x, method, policy)
                for st in ('validation', 'test') for rep in REPEATS for x in XS
                for method in ('centroid', 'nn')
                for policy in ('original_zero', 'val_youden')}
    keyed = {}
    for row in rows:
        try:
            k = (row['stage'], int(row['repeat']), int(row['X']), row['method'], row['policy'])
        except (ValueError, KeyError) as err:
            raise SystemExit(f"ERRORE: riga/colonna non valida in {path}: {err}")
        if k in keyed:
            raise SystemExit(f"ERRORE: duplicato {k} in {path}")
        keyed[k] = row
    if set(keyed) != expected:
        raise SystemExit(f"ERRORE: configurazioni mancanti o impreviste in {path}: "
                         f"missing={len(expected - set(keyed))}, extra={len(set(keyed) - expected)}")
    for rep in REPEATS:
        for x in XS:
            expected_seed = 42 + 1000 * rep + x
            subset = [r for k, r in keyed.items() if k[1] == rep and k[2] == x]
            seeds = {int(r['fit_seed']) for r in subset}
            if seeds != {expected_seed}:
                raise SystemExit(f"ERRORE: fit_seed errato ({rep}, {x}): {seeds}")
            durations = [float(r['elapsed_sec']) for r in subset]
            if max(durations) - min(durations) > 1e-8:
                raise SystemExit(f"ERRORE: elapsed_sec differisce tra le 8 metriche di ({rep},{x})")
            for method in ('centroid', 'nn'):
                for policy in ('original_zero', 'val_youden'):
                    va = keyed[('validation', rep, x, method, policy)]
                    te = keyed[('test', rep, x, method, policy)]
                    if abs(float(va['threshold']) - float(te['threshold'])) > 1e-12:
                        raise SystemExit(f"ERRORE: soglia val/test diversa per ({rep},{x},{method},{policy})")
    return keyed


def single(runs, rep, x, metric):
    """AUROC e tempo univoci per fit; accuracy/FPR con soglia validation Youden."""
    policy = 'original_zero' if metric in ('auroc', 'elapsed_sec') else 'val_youden'
    record = runs[('test', rep, x, 'centroid', policy)]
    return float(record[metric])


def write_csv(path: Path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(v):
    return {'mean': stats.mean(v), 'median': stats.median(v),
            'std': stats.stdev(v), 'min': min(v), 'max': max(v)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ubuntu', type=Path,
                    default=ROOT / 'results/test/e1_uniform_v2/metrics_per_run.csv')
    ap.add_argument('--mac', type=Path,
                    default=ROOT / 'results/replicas/e1_mac_m4/metrics_per_run.csv')
    ap.add_argument('--out-dir', type=Path,
                    default=Path.home() / 'mlsec-local/audits/e1_hardware_comparison')
    args = ap.parse_args()
    if args.ubuntu.resolve() == args.mac.resolve():
        raise SystemExit('ERRORE: i due percorsi puntano allo stesso CSV; servono Ubuntu e Mac distinti.')
    if args.ubuntu.is_file() and args.mac.is_file() and digest(args.ubuntu) == digest(args.mac):
        raise SystemExit('ERRORE: i due CSV sono identici byte-per-byte; verificare che non siano due copie dello stesso run.')
    ubu = read_records(args.ubuntu)
    mac = read_records(args.mac)
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    per_run = []
    metrics = ('auroc', 'accuracy', 'fpr')
    byx = []
    timings = []
    for x in XS:
        for rep in REPEATS:
            row = {'X': x, 'repeat': rep, 'fit_seed': 42 + 1000 * rep + x}
            for metric in (*metrics, 'elapsed_sec'):
                u = single(ubu, rep, x, metric)
                m = single(mac, rep, x, metric)
                row[f'ubuntu_{metric}'] = f'{u:.10g}'
                row[f'mac_{metric}'] = f'{m:.10g}'
                row[f'delta_mac_minus_ubuntu_{metric}'] = f'{m-u:.10g}'
            per_run.append(row)

        for metric in metrics:
            uv = [single(ubu, rep, x, metric) for rep in REPEATS]
            mv = [single(mac, rep, x, metric) for rep in REPEATS]
            delta = [m-u for m, u in zip(mv, uv)]
            ustat, mstat, dstat = map(summary, (uv, mv, delta))
            byx.append({'X': x, 'metric': metric, 'method': 'centroid',
                        'policy': 'original_zero' if metric == 'auroc' else 'val_youden',
                        'n_paired_fits': 10,
                        'ubuntu_mean': ustat['mean'], 'ubuntu_std': ustat['std'],
                        'mac_mean': mstat['mean'], 'mac_std': mstat['std'],
                        'paired_delta_mean': dstat['mean'], 'paired_delta_std': dstat['std'],
                        'paired_delta_min': dstat['min'], 'paired_delta_max': dstat['max'],
                        'paired_abs_delta_mean': stats.mean(map(abs, delta))})
        for platform, runs in (('ubuntu', ubu), ('mac', mac)):
            times = [single(runs, rep, x, 'elapsed_sec') for rep in REPEATS]
            timing_stat = summary(times)
            timings.append({'X': x, 'platform': platform, 'n_fits': 10,
                            'mean_sec': timing_stat['mean'],
                            'median_sec': timing_stat['median'],
                            'std_sec': timing_stat['std'],
                            'min_sec': timing_stat['min'],
                            'max_sec': timing_stat['max']})

    fields = list(per_run[0])
    write_csv(out / 'paired_per_run.csv', per_run, fields)
    write_csv(out / 'metrics_by_x.csv', byx, list(byx[0]))
    write_csv(out / 'timing_by_x.csv', timings, list(timings[0]))

    mac_times = [single(mac, rep, x, 'elapsed_sec') for x in XS for rep in REPEATS]
    ubu_times = [single(ubu, rep, x, 'elapsed_sec') for x in XS for rep in REPEATS]
    x23 = next(row for row in byx if row['X'] == 23 and row['metric'] == 'auroc')
    ratio = stats.mean(ubu_times) / stats.mean(mac_times)
    lines = [
        '# E1 — Audit descrittivo Ubuntu vs Mac M4', '',
        f"Generato (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        '', '## Input (integrità)',
        f'- Ubuntu: `{args.ubuntu}` — SHA-256 `{digest(args.ubuntu)}`',
        f'- Mac: `{args.mac}` — SHA-256 `{digest(args.mac)}`',
        '', '## Controlli',
        '- 320 metriche/stage per piattaforma? **320 metriche totali**, '
        '160 validation + 160 test per piattaforma (40 fit distinti).',
        '- I fit sono abbinabili per X, repeat e fit_seed, subordinatamente all’audit del codice e degli ID di training.',
        '- Soglie applicate al test uguali a quelle della validation dello stesso fit.',
        '- I tempi sono deduplicati per (repeat,X); non 320 durate.',
        '', '## Risultati descrittivi',
        f"- X=23 AUROC centroid: Ubuntu {x23['ubuntu_mean']:.6f}; "
        f"Mac {x23['mac_mean']:.6f}; differenza media abbinata "
        f"(Mac−Ubuntu) {x23['paired_delta_mean']:+.6f}.",
        f'- `elapsed_sec` test, 40 fit: Ubuntu media {stats.mean(ubu_times):.3f}s; '
        f'Mac media {stats.mean(mac_times):.3f}s.',
        f'- Rapporto delle **medie del tratto cronometrato** Ubuntu/Mac: {ratio:.3f}.',
        '', '## Limiti di interpretazione',
        '- Non test di equivalenza, non CI inferenziale e non misura causale della CPU.',
        '- `elapsed_sec` inizia DOPO la costruzione delle immagini train/validation '
        'e NON include l’avvio Python o la scrittura dei CSV.',
        '- Ubuntu è in una VM: OS, virtualizzazione, CPU, build TensorFlow e RAM '
        'sono fattori concomitanti.',
        '- Nessun training eseguito; nessun valore dei CSV originali modificato.',
        '', '## Tabelle generate',
        '- `paired_per_run.csv`: 40 coppie X/repeat e delta Mac−Ubuntu.',
        '- `metrics_by_x.csv`: differenze e variabilità per X e metrica.',
        '- `timing_by_x.csv`: tempi descrittivi per X e piattaforma.',
        '',
    ]
    (out / 'audit_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'AUDIT OK — 40 fit per piattaforma, 320 righe per-run ciascuna.')
    print(f"X=23 centroid AUROC: Ubuntu {x23['ubuntu_mean']:.6f}, "
          f"Mac {x23['mac_mean']:.6f}, delta {x23['paired_delta_mean']:+.6f}")
    print(f'Elapsed test medio: Ubuntu {stats.mean(ubu_times):.2f}s, '
          f'Mac {stats.mean(mac_times):.2f}s; rapporto {ratio:.3f}')
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
