#!/usr/bin/env python3
"""Fast self-tests on artificial vectors only, no user dataset required."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_synthetic_spread import (spreading_codes, synthetic_message,
                                embed_inert_bits, decode_inert_bits)


def test_synthetic_spreading() -> None:
    n, b = 2864, 16
    rng = np.random.default_rng(100)
    w = rng.normal(0.0, 1.0, n).astype(np.float32)
    bits = synthetic_message(n_bits=b, master_seed=42, repeat=0, row_index=27)
    c = spreading_codes(n_weights=n, n_bits=b, master_seed=42, repeat=0, row_index=27)
    np.testing.assert_allclose(c.T @ c, np.eye(b), atol=1e-12)
    assert set(bits.tolist()).issubset({0, 1})
    assert np.array_equal(bits, synthetic_message(n_bits=b, master_seed=42, repeat=0, row_index=27))
    a = embed_inert_bits(w, bits, alpha=0.5, master_seed=42, repeat=0, row_index=27)
    other = embed_inert_bits(w, bits, alpha=0.5, master_seed=42, repeat=0, row_index=27)
    assert a.dtype == np.float32 and a.shape == (n,) and np.isfinite(a).all()
    assert np.array_equal(a, other)
    # Recovery is from stego weights + secret key/row id, NOT the clean host.
    decoded = decode_inert_bits(a, n_bits=b, master_seed=42, repeat=0, row_index=27)
    np.testing.assert_array_equal(decoded, bits)
    delta = np.linalg.norm(a.astype(np.float64) - w) / np.linalg.norm(w)
    assert abs(delta - 0.5) < 1e-5, delta
    # Use of 0 strength must preserve the input bitwise.
    none = embed_inert_bits(w, bits, alpha=0, master_seed=42, repeat=0, row_index=27)
    np.testing.assert_array_equal(none, w)
    print("SELF-TEST OK: deterministic / orthogonal / blind-to-cover decoding / rel L2")


if __name__ == "__main__":
    test_synthetic_spreading()
