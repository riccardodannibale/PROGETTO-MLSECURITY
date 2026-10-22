#!/usr/bin/env python3
"""Regression test: E3 metadata pairing and benign-match controls fail closed."""
from pathlib import Path
import importlib.util
import pandas as pd

src=Path(__file__).resolve().parent.parent/'scripts/e3_matched_controls.py'
spec=importlib.util.spec_from_file_location('e3',src)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def frame():
    rows=[]
    for checkpoint in (3,7):
        for label in (0,1):
            rows.append(dict(row_index=checkpoint,label=label,method='centroid',condition='LSB_ID',
                             stage='test',repeat=0,fit_seed=65,X=23,model_id=f'm{checkpoint}',
                             checkpoint_sha256=f'checksum{checkpoint}',score_margin=label*0.5+checkpoint/100))
    return pd.DataFrame(rows)

ok=frame()
x=m.inspect_group(ok,expected_ids={3,7},context='unit')
assert x['n_pairs']==2 and x['paired_concordance']==1 and x['higher']==2
for col,val in [('model_id','tamper'),('checkpoint_sha256','tamper'),('fit_seed',999)]:
    bad=frame()
    bad.loc[(bad.row_index==3)&(bad.label==1),col]=val
    try:
        m.inspect_group(bad,expected_ids={3,7},context='unit')
        raise AssertionError(f'Failed to catch {col} confound')
    except RuntimeError as e:
        assert 'STOP E3 CONTROLS' in str(e)
bad=frame()
bad.loc[(bad.row_index==3)&(bad.label==1),'label']=0
try:
    m.inspect_group(bad,expected_ids={3,7},context='unit')
    raise AssertionError('Failed to catch duplicate benign labels')
except RuntimeError as e:
    assert 'STOP E3 CONTROLS' in str(e)
print('E3 CONTROLS UNIT TEST OK: 1:1 pairing, perfect metadata balance, corrupt-pair fail-closed')
