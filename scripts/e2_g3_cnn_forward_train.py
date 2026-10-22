#!/usr/bin/env python3
"""E2 G3: read-only reconstruction/forward smoke on genuine TRAIN checkpoints.

Does NOT measure model utility or STL10 accuracy. Never accesses VAL/TEST IDs.
The architecture follows ModelZoos/ModelZooDataset's original CNN in def_net.py;
the supplied model-specific params.json determines activation/dropout.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHTS_SHA='5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad'
SPLITS_SHA='3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f'
ACT={'leakyrelu':nn.LeakyReLU,'relu':nn.ReLU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,'silu':nn.SiLU,'gelu':nn.GELU}


def sha256_file(p: Path) -> str:
    d=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20),b''): d.update(chunk)
    return d.hexdigest()


class SmallZooCNN(nn.Module):
    """Structural reconstruction of the upstream CNN, based on parameter signature.

    Original reference: https://github.com/ModelZoos/ModelZooDataset/blob/main/code/model_definitions/def_net.py
    Avoids importing Ray, torchvision, or the original training code.
    """
    def __init__(self, *, activation: str, dropout: float):
        super().__init__()
        if activation not in ACT:
            raise ValueError(f'Unknown model::nlin {activation!r}')
        if not 0 <= dropout < 1:
            raise ValueError(f'Invalid model::dropout {dropout!r}')
        a=ACT[activation]
        layers=[nn.Conv2d(3,8,5), nn.MaxPool2d(2,2), a()]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.extend([nn.Conv2d(8,6,5), nn.MaxPool2d(2,2), a()])
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.extend([nn.Conv2d(6,4,2), a(), nn.Flatten(), nn.Linear(36,20), a()])
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(20,10))
        self.module_list=nn.ModuleList(layers)

    def forward(self,x):
        for layer in self.module_list: x=layer(x)
        return x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--split-dir',type=Path,required=True)
    parser.add_argument('--manifest-dir',type=Path,required=True)
    parser.add_argument('--count',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.count<=10: parser.error('--count must be 1..10')
    wp=args.weights.expanduser().resolve(strict=True)
    sd=args.split_dir.expanduser().resolve(strict=True)
    md=args.manifest_dir.expanduser().resolve(strict=True)
    if sha256_file(wp)!=WEIGHTS_SHA or sha256_file(sd/'splits.csv')!=SPLITS_SHA:
        raise SystemExit('STOP: E1 weights/split hash mismatch')
    info=json.loads((md/'extraction_summary.json').read_text())
    if info.get('weights_sha256')!=WEIGHTS_SHA: raise SystemExit('STOP: provenance summary mismatch')
    original_root=Path(info['source_root']).expanduser().resolve(strict=True)
    with (md/'checkpoint_manifest.csv').open(newline='',encoding='utf8') as f:
        manifest={int(r['row_index']):r for r in csv.DictReader(f)}
    with (sd/'splits.csv').open(newline='',encoding='utf8') as f:
        train=sorted(int(r['row_index']) for r in csv.DictReader(f) if r['split']=='train')
    if len(train)!=600 or len(manifest)!=1000: raise SystemExit('STOP: unexpected split/manifest size')
    W=np.load(wp,mmap_mode='r',allow_pickle=False)
    if W.shape!=(1000,2864) or W.dtype!=np.float32: raise SystemExit('STOP: weight array mismatch')
    sig=[(r['name'],tuple(r['shape'])) for r in info['parameter_signature']]
    g=torch.Generator(device='cpu').manual_seed(2026)
    sample=torch.randn((2,3,28,28),generator=g,dtype=torch.float32)
    print('G3 CNN forward smoke: synthetic inputs ONLY; no training or accuracy measurement')
    for rid in train[:args.count]:
        m=manifest[rid]
        model_dir=(original_root/m['model_id']).resolve(strict=True)
        checkpoint=(original_root/m['checkpoint_relpath']).resolve(strict=True)
        if not model_dir.is_relative_to(original_root) or not checkpoint.is_relative_to(model_dir):
            raise SystemExit(f'STOP: unexpected checkpoint path for TRAIN row {rid}')
        if sha256_file(checkpoint)!=m['checkpoint_sha256']:
            raise SystemExit(f'STOP: source checkpoint hash mismatch for row {rid}')
        params_file=model_dir/'params.json'
        if not params_file.is_file():
            raise SystemExit(f'STOP: {params_file} missing; do not guess activation/dropout')
        params=json.loads(params_file.read_text())
        if 'model::nlin' not in params or 'model::dropout' not in params:
            raise SystemExit(f'STOP: params.json missing model::nlin/model::dropout; keys: {sorted(params)}')
        nlin=params['model::nlin']
        dropout=float(params['model::dropout'])
        net=SmallZooCNN(activation=nlin,dropout=dropout)
        signature=[(name,tuple(p.shape)) for name,p in net.state_dict().items()]
        if signature!=sig:
            raise SystemExit(f'STOP: model signature differs from E1 at row {rid}: {signature}')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or [(name,tuple(t.shape)) for name,t in state.items()]!=sig:
            raise SystemExit(f'STOP: original state_dict is unexpected at row {rid}')
        flattened=np.concatenate([t.detach().cpu().contiguous().numpy().ravel() for t in state.values()])
        if not np.array_equal(flattened,W[rid]):
            raise SystemExit(f'STOP: checkpoint/weights mismatch at row {rid}')
        net.load_state_dict(state,strict=True)
        net.eval()
        with torch.no_grad(): prediction=net(sample)
        if prediction.shape!=(2,10) or not torch.isfinite(prediction).all():
            raise SystemExit(f'STOP: invalid forward result for row {rid}')
        print(f'  TRAIN row {rid}: nlin={nlin}, dropout={dropout}, '
              f'forward_shape={tuple(prediction.shape)}, finite=YES')
    print(f'E2 G3 CNN FORWARD OK: {args.count}/{args.count} TRAIN checkpoints')
    print('NOT STL10 accuracy or evidence that stego model utility is preserved.')

if __name__=='__main__': main()
