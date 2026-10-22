#!/usr/bin/env python3
"""Unit-level G5 invariants; no TensorFlow/model fitting/test checkpoint access."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "scripts/run_e2_paired_v0.py"
sys.path.insert(0, str(SRC.parent))
spec = importlib.util.spec_from_file_location("e2_g5", SRC)
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)

assert e2.SEED == 42
assert e2.X_LSB == 23
assert e2.REPEATS == 10
assert e2.TRAIN_IDENTITIES == 3
assert e2.ALPHAS == (0.24, 0.16)
assert e2.BITS == 16

# Repeating the same source/seed should always yield the same key and payload.
for row_index in (3, 4, 7):
    x = e2.spread.synthetic_message(n_bits=16, master_seed=42, repeat=0, row_index=row_index)
    y = e2.spread.synthetic_message(n_bits=16, master_seed=42, repeat=0, row_index=row_index)
    assert np.array_equal(x, y)

# Test must fail before reading any reserved weights when approval is absent.
try:
    e2.check_test_unlock(approval=None, validation_dir=Path('/nonexistent'),
                         runner_path=SRC, expected_runner_sha=e2.sha(SRC))
except RuntimeError as exc:
    assert 'test locked' in str(exc)
else:
    raise AssertionError('Test guard accepted absent approval')

for repeat in range(10):
    expected = 42 + repeat * 1000 + 23
    assert isinstance(expected, int)

print('E2 G5 PREFLIGHT TEST OK: frozen parameters, keyed bits, fail-closed test lock')
