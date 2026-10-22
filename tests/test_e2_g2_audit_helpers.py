#!/usr/bin/env python3
"""Self-contained tests for E2 G2 helper and interception logic, no real data."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

code = Path(__file__).resolve().parents[1] / 'scripts' / 'e2_intensity_train_audit.py'
spec = importlib.util.spec_from_file_location('e2g2', code)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model_xray = types.ModuleType('model_xray')
model_xray.__path__ = []
data = types.ModuleType('model_xray.data')
data.__path__ = []
pipeline = types.ModuleType('model_xray.data.attack_pipeline')


def fake_lsb(ws, **kwargs):
    arr = np.array(ws, copy=True)
    arr[0, 0] += 1.0
    return arr


def fake_image(ws, *, imsize, x, payload_filepath):
    assert x == 23 and imsize == 100 and Path(payload_filepath).read_bytes() == b'hello'
    arr = pipeline.x_lsb_attack(ws)
    assert arr.shape == (1, 2864)
    return np.zeros((1, 100, 100), dtype=np.float32)

pipeline.x_lsb_attack = fake_lsb
pipeline.img_pp_xlsb_attack = fake_image
data.attack_pipeline = pipeline
model_xray.data = data
sys.modules.update({'model_xray': model_xray,
                    'model_xray.data': data,
                    'model_xray.data.attack_pipeline': pipeline})

weights = np.ones(2864, dtype=np.float32)
before = weights.copy()
attacked = module.capture_canonical_lsb(weights, b'hello')
assert attacked.dtype == np.float32
assert attacked[0] == 2 and np.array_equal(attacked[1:], weights[1:])
assert np.array_equal(weights, before)
assert pipeline.x_lsb_attack is fake_lsb, 'Upstream function not restored'
stat = module.metrics(weights, attacked)
assert stat['changed_weights'] == 1 and stat['all_finite']
assert stat['max_abs_delta'] == 1.0
assert abs(stat['relative_l2'] - 1.0 / np.sqrt(2864)) < 1e-10
assert module.compare_float(0.240000001, 0.24)
assert not module.compare_float(0.32, 0.24)

# Interceptor should fail closed when pipeline does not call the canonical function.
pipeline.img_pp_xlsb_attack = lambda *_a, **_kw: np.zeros((1, 100, 100), dtype=np.float32)
try:
    module.capture_canonical_lsb(weights, b'hello')
    raise AssertionError('Expected failure when upstream hook is unused')
except RuntimeError:
    pass
assert pipeline.x_lsb_attack is fake_lsb

print('E2 G2 SELF-TEST OK: canonical hook capture, metrics, fail-closed and restoration')
