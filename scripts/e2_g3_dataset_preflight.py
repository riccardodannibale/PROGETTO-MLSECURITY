#!/usr/bin/env python3
"""Read-only G3 dataset/provenance inventory. Never deserializes .pt files.

Intended only to locate the *original* preprocessed STL10 dataset used by
ModelZoos' 28x28 CNNs, before assessing model utility on real images.
This is NOT an accuracy measurement. Only TRAIN model IDs are accessed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_WEIGHTS = '5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad'
EXPECTED_SPLITS = '3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for piece in iter(lambda: file.read(1<<20), b''):
            h.update(piece)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def describe(path: Path):
    # File metadata only; no torch.load/pickle.
    if path.is_file():
        print(f'    EXISTS file={path} bytes={path.stat().st_size}')
    elif path.exists():
        print(f'    EXISTS NON-FILE: {path}')
    else:
        print(f'    missing: {path}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--weights', required=True, type=Path)
    cli.add_argument('--split-dir', required=True, type=Path)
    cli.add_argument('--manifest-dir', required=True, type=Path)
    cli.add_argument('--count', type=int, default=3)
    args = cli.parse_args()
    if not 1 <= args.count <= 10:
        cli.error('--count must be 1..10')
    wp = args.weights.expanduser().resolve(strict=True)
    split = args.split_dir.expanduser().resolve(strict=True)
    manifest_dir = args.manifest_dir.expanduser().resolve(strict=True)
    if sha256(wp) != EXPECTED_WEIGHTS or sha256(split/'splits.csv') != EXPECTED_SPLITS:
        raise SystemExit('STOP: E1 weights or split SHA mismatch')
    info = json.loads((manifest_dir/'extraction_summary.json').read_text(encoding='utf-8'))
    if info.get('weights_sha256') != EXPECTED_WEIGHTS or info.get('n_models') != 1000 or info.get('n_parameters_per_model') != 2864:
        raise SystemExit('STOP: extraction summary mismatch')
    root = Path(info['source_root']).expanduser().resolve(strict=True)
    manifest_rows = rows(manifest_dir/'checkpoint_manifest.csv')
    manifest = {int(r['row_index']): r for r in manifest_rows}
    split_rows = rows(split/'splits.csv')
    selected = sorted(int(r['row_index']) for r in split_rows if r['split'] == 'train')
    if len(split_rows) != 1000 or len(manifest_rows) != 1000 or len(manifest) != 1000 or len(selected) != 600:
        raise SystemExit('STOP: invalid split/manifest size')
    if set(int(r['row_index']) for r in split_rows) != set(range(1000)):
        raise SystemExit('STOP: invalid row IDs')

    print('E2 G3 DATASET/PREPROCESSING PREFLIGHT (READ-ONLY)')
    print(f'E1 weights SHA-256: {EXPECTED_WEIGHTS}')
    print(f'E1 split SHA-256:   {EXPECTED_SPLITS}')
    print(f'Original model zoo: {root}')
    print('Common possible dataset dump locations (metadata only):')
    for p in (root/'dataset.pt', root.parent/'dataset.pt', root/'train_data.pt', root/'test_data.pt', root.parent/'train_data.pt', root.parent/'test_data.pt'):
        describe(p)
    print('')
    dump_hints = set()
    for idx in selected[:args.count]:
        r = manifest[idx]
        model_dir = (root/r['model_id']).resolve(strict=True)
        ckpt = (root/r['checkpoint_relpath']).resolve(strict=True)
        if not model_dir.is_relative_to(root) or not ckpt.is_relative_to(model_dir):
            raise SystemExit('STOP: model/checkpoint escapes original root')
        if sha256(ckpt) != r['checkpoint_sha256']:
            raise SystemExit(f'STOP: source SHA mismatch for TRAIN row {idx}')
        param_path = model_dir/'params.json'
        if not param_path.is_file():
            raise SystemExit(f'STOP: params.json not found at {param_path}')
        p = json.loads(param_path.read_text(encoding='utf-8'))
        print(f'TRAIN row {idx}: {r["model_id"]}')
        print('  model:', {k:p.get(k) for k in ('model::type','model::channels_in','model::nlin','model::dropout')})
        hints = {k:v for k,v in p.items() if ('dataset' in k.lower() or 'data_path' in k.lower() or 'training::task'==k)}
        print('  dataset / preprocessing hints:', json.dumps(hints, ensure_ascii=False, sort_keys=True, default=str))
        hint = p.get('dataset::dump')
        if isinstance(hint,str) and hint:
            dump_hints.add(hint)
            q = Path(hint).expanduser()
            # Original absolute path can be stale (path on dataset author's machine).
            print('  dataset::dump original reference:')
            describe(q)
            print('  same basename locally:')
            for candidate in (root/q.name, root.parent/q.name, model_dir/q.name):
                describe(candidate)
        data_dir = p.get('training::data_path')
        if isinstance(data_dir, str) and data_dir:
            q = Path(data_dir).expanduser()
            print('  training::data_path original reference:')
            describe(q)
            for base in (q, root, root.parent):
                for name in ('train_data.pt','train_labels.pt','test_data.pt','test_labels.pt'):
                    if (base/name).is_file(): describe(base/name)
        print('')
    print(f'Distinct dump references across inspected TRAIN models: {len(dump_hints)}')
    for h in sorted(dump_hints): print(f'  {h}')
    print('PREFLIGHT OK: verified TRAIN checkpoint provenance, inspected params.json and file metadata only.')
    print('NEXT: confirm actual preprocessed image tensors/labels and normalization before utility evaluation.')
    print('NO torch.load, no pickle, no malware, no training, no VAL/TEST checkpoint access.')

if __name__ == '__main__':
    main()
