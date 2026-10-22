#!/usr/bin/env python3
"""QA non invasivo dei risultati E1 Mac; non accede a pesi/test individuali."""
from __future__ import annotations
import csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'results/replicas/e1_mac_m4'
def rows(name):
 with (BASE/name).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def main():
 s=rows('summary.csv');r=rows('metrics_per_run.csv')
 assert len(s)==32 and len(r)==320,(len(s),len(r))
 assert {a['stage'] for a in s}=={'validation','test'}
 for stage in ('validation','test'):
  keys={(int(a['repeat']),int(a['X'])) for a in r if a['stage']==stage}
  assert keys=={(i,x) for i in range(10) for x in (1,8,16,23)}
 for row in s:
  sub=[a for a in r if all(row[k]==a[k] for k in ('stage','X','method','policy'))]
  assert len(sub)==10
  assert int(row['n_runs'])==10
  for left,right in [('accuracy_mean','accuracy'),('accuracy_std','accuracy'),('auroc_mean','auroc'),('auprc_mean','auprc'),('f1_mean','f1'),('fpr_mean','fpr')]:
   vals=[float(x[right]) for x in sub]
   val=statistics.stdev(vals) if left=='accuracy_std' else statistics.mean(vals)
   assert abs(float(row[left])-val)<1e-10,(row,left,val)
 for stage in ('validation','test'):
  assert len([a for a in r if a['stage']==stage])==160
 for repeat in range(10):
  for x in (1,8,16,23):
   for method in ('centroid','nn'):
    for policy in ('original_zero','val_youden'):
     t=[a for a in r if int(a['repeat'])==repeat and int(a['X'])==x and a['method']==method and a['policy']==policy]
     assert len(t)==2 and {a['stage'] for a in t}=={'validation','test'}
     assert abs(float(t[0]['threshold'])-float(t[1]['threshold']))<1e-12
 print('E1 MAC QA OK: 40 validation + 40 test; 320 record, 32 aggregate; soglie uguali val/test per fit; medie e SD verificate.')
if __name__=='__main__':main()
