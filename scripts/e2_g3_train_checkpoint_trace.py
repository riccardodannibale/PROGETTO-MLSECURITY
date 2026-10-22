#!/usr/bin/env python3
"""E2 G3 read-only provenance check: original PyTorch -> E1 float32 rows.

Uses only a small set of TRAIN identities. Validates SHA-256 before loading
checkpoint tensor state_dict with torch.load(weights_only=True).
Does not construct or evaluate a network, and uses no malware or training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch

EXPECTED_WEIGHTS='5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad'
EXPECTED_SPLITS='3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f'

def sha256_file(p:Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--weights',type=Path,required=True)
    a.add_argument('--split-dir',type=Path,required=True)
    a.add_argument('--manifest-dir',type=Path,required=True)
    a.add_argument('--count',type=int,default=3)
    args=a.parse_args()
    if not 1 <= args.count <= 20:
        a.error('--count must be 1..20 for a small train-only check')
    root=args.manifest_dir.expanduser().resolve(strict=True)
    wp=args.weights.expanduser().resolve(strict=True)
    splits=args.split_dir.expanduser().resolve(strict=True)
    if sha256_file(wp)!=EXPECTED_WEIGHTS or sha256_file(splits/'splits.csv')!=EXPECTED_SPLITS:
        raise SystemExit('STOP: E1 weights/split hash mismatch')
    info=json.loads((root/'extraction_summary.json').read_text())
    if info.get('weights_sha256')!=EXPECTED_WEIGHTS:
        raise SystemExit('STOP: unexpected extraction summary')
    with (root/'checkpoint_manifest.csv').open(newline='',encoding='utf-8') as f:
        manifest={int(r['row_index']): r for r in csv.DictReader(f)}
    with (splits/'splits.csv').open(newline='',encoding='utf-8') as f:
        train=sorted(int(r['row_index']) for r in csv.DictReader(f) if r['split']=='train')
    if len(manifest)!=1000 or len(train)!=600:
        raise SystemExit('STOP: split or manifest cardinality mismatch')
    source=Path(info['source_root']).expanduser().resolve(strict=True)
    signature=[(x['name'],tuple(x['shape'])) for x in info['parameter_signature']]
    w=np.load(wp,mmap_mode='r',allow_pickle=False)
    if w.shape!=(1000,2864) or w.dtype!=np.float32:
        raise SystemExit('STOP: weights shape/dtype mismatch')
    print('Source root:',source)
    for rid in train[:args.count]:
        row=manifest[rid]
        path=(source/row['checkpoint_relpath']).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file():
            raise SystemExit(f'STOP: unexpected source path at row {rid}')
        if sha256_file(path)!=row['checkpoint_sha256']:
            raise SystemExit(f'STOP: checkpoint SHA mismatch at row {rid}')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(state,Mapping) or not state:
            raise SystemExit(f'STOP: invalid checkpoint state at row {rid}')
        actual=[];parts=[]
        for name, t in state.items():
            if not isinstance(name,str) or not isinstance(t,torch.Tensor) or t.dtype!=torch.float32:
                raise SystemExit(f'STOP: unexpected tensor in row {rid}')
            actual.append((name,tuple(t.shape)))
            parts.append(t.detach().cpu().contiguous().numpy().ravel())
        if actual!=signature:
            raise SystemExit(f'STOP: parameter signature mismatch at row {rid}')
        vec=np.concatenate(parts).astype(np.float32,copy=False)
        if not np.array_equal(vec,w[rid]):
            raise SystemExit(f'STOP: float32 vector mismatch at row {rid}')
        print(f'  TRAIN row {rid}: SHA and {len(vec)} weights match (checkpoint {path.name})')
    print(f'E2 G3 CHECKPOINT TRACE OK: {args.count}/{args.count} TRAIN original checkpoints exactly match E1 weights')
    print('NOT a model reconstruction or STL10 accuracy measurement; model forward/utility still to verify.')

if __name__=='__main__':
    main()
