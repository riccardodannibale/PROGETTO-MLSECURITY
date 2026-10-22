#!/usr/bin/env python3
"""Lightweight structural tests; no dataset, original checkpoints, or malware."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_g3_train_utility as utility

def check():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Linear(2,1)
        def forward(self,x):
            # Model utility accuracy() requires 10 logits, use wrapper instead.
            return self.a(x)
    toy=Toy()
    # Reuse general mapper for expected dimension via a 2864-element template.
    template={"layer.weight":torch.zeros(2864,dtype=torch.float32)}
    flat=np.arange(2864,dtype=np.float32)/2864
    rebuilt=utility.reshaped_state_from_flat(template,flat)
    assert list(rebuilt)==["layer.weight"]
    assert rebuilt["layer.weight"].shape == (2864,)
    assert np.array_equal(rebuilt["layer.weight"].numpy(),flat)
    # Defensive rejection of incorrect length/dtype and NaNs.
    for bad in (flat[:100],flat.astype(np.float64),np.full(2864,np.nan,dtype=np.float32)):
        try:
            utility.reshaped_state_from_flat(template,bad)
        except ValueError: pass
        else: raise AssertionError("invalid carrier accepted")
    assert utility.relative_l2(flat,flat)==0
    dummy=nn.Sequential(nn.Linear(4,10,bias=True))
    with torch.no_grad():
        dummy[0].weight.zero_()
        dummy[0].bias.zero_();dummy[0].bias[3]=1
    imgs=torch.ones((7,4),dtype=torch.float32)
    labs=torch.full((7,),3,dtype=torch.long)
    good,total=utility.accuracy(dummy,dummy.state_dict(),imgs,labs,batch_size=3)
    assert (good,total)==(7,7)
    print('G3 UTILITY HELPER TEST OK: tensor mapping, invalid dtype/shape, accuracy')

if __name__=='__main__': check()
