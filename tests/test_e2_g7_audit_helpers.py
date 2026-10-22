"""Synthetic, test-free G7 audit helper regression: exact threshold and metadata."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_g7_audit_test as qa

rows=[]
for method in ('centroid','nn'):
    for condition in ('LSB_ID','SPREAD_0.24','SPREAD_0.16'):
        for label in (0,1):
            for row_index in range(200):
                score=float((label*2-1)*(0.5+row_index/3000.0))
                threshold=0.0
                rows.append({'condition':condition,'method':method,'stage':'test','repeat':0,
                             'fit_seed':65,'row_index':row_index,'X':23,'label':label,
                             'score_margin':score,'validation_threshold':threshold,
                             'prediction_calibrated':int(score>threshold)})
df=pd.DataFrame(rows)
qa.predictions(df,rep=0,split='test',allowed=qa.CONDITIONS,ids=set(range(200)),thresholds={'centroid':0.0,'nn':0.0})
grp=df[(df.method=='nn')&(df.condition=='SPREAD_0.16')]
y=grp.label.to_numpy();score=grp.score_margin.to_numpy();pred=(score>0).astype(int)
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
metric={'stage':'test','condition':'SPREAD_0.16','method':'nn','policy':'val_youden',
        'threshold':0.0,'n':400,'n_benign':200,'n_stego':200,
        'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),
        'fpr':float(fp/200),'auroc':float(roc_auc_score(y,score)),
        'auprc':float(average_precision_score(y,score)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
qa.audit_metric(metric,df,0)
bad=df.copy()
index=int(bad.index[0]);bad.loc[index,'prediction_calibrated']=1-int(bad.loc[index,'prediction_calibrated'])
try:
    qa.predictions(bad,rep=0,split='test',allowed=qa.CONDITIONS,ids=set(range(200)),thresholds={'centroid':0.0,'nn':0.0})
except RuntimeError as err:
    assert 'threshold decision mismatch' in str(err)
else:
    raise AssertionError('Corrupt calibrated prediction must fail closed')
print('E2 G7 TEST AUDIT HELPERS OK: complete paired data, score-derived metrics, fail-closed corrupted threshold')
