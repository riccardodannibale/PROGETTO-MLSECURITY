#!/usr/bin/env python3
"""Regression test: precision of threshold at one binary64 ULP in CSV."""
from pathlib import Path
import tempfile
import json
import numpy as np
import pandas as pd

value = float(-9.997999799979999)
thr = float(np.nextafter(value, -np.inf))
assert value > thr
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / 'predictions_validation.csv'
    pd.DataFrame([{
        'method': 'nn', 'score_margin': value, 'validation_threshold': thr,
        'prediction_calibrated': 1,
    }]).to_csv(p, index=False)
    q = pd.read_csv(p, float_precision='round_trip')
    t = json.loads(json.dumps({'t': thr}))['t']
    assert q.score_margin[0] == value
    assert q.validation_threshold[0] == t
    assert int(q.score_margin[0] > t) == int(q.prediction_calibrated[0]) == 1
    default = pd.read_csv(p)
    print('default parser:', repr(float(default.score_margin[0])))
    print('round_trip parser:', repr(float(q.score_margin[0])))
print('E2 G6 CSV ROUNDTRIP REGRESSION TEST OK: strict threshold semantics intact')
