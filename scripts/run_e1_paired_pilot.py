#!/usr/bin/env python3
"""E1 pilot with identity-matched benign/stego training examples.

One checkpoint per source model is required (e.g., d1_uniform_real).
For each training identity, create *both* its cover and its LSB-stego image;
hold out different source identities for testing. This removes source-ID/
class-label confounding present in the upstream 3-cover + 3-stego split.

Exploratory comparison only: changes the upstream training selection protocol.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_xray.data.attack_pipeline import img_pp_xlsb_attack
from model_xray.data.ghrp_zoos import load_mz_weights
from model_xray.fsl.evaluate import evaluate_model
from model_xray.fsl.train import train_fsl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mz-name", default="d1_uniform_real")
    ap.add_argument("--x-range", type=int, nargs="+", default=[16, 23])
    ap.add_argument("--n-repeats", type=int, default=2)
    ap.add_argument("--n-train-identities", type=int, default=3)
    ap.add_argument("--n-test-identities", type=int, default=300)
    ap.add_argument("--imsize", type=int, default=100)
    ap.add_argument("--mode", default="ub")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "mlsec-local/runs/e1_paired_pilot")
    args = ap.parse_args()

    # Unlike a baseline CSV from the paper, this output contains only new observations.
    weights = load_mz_weights(args.mz_name)
    if weights.ndim != 2 or weights.dtype != np.float32:
        raise ValueError(f"Expected 2D float32 weights, received {weights.shape}, {weights.dtype}")
    if args.n_train_identities + args.n_test_identities > len(weights):
        raise ValueError("Not enough distinct source models for this train/test split")
    print(f"Loaded {args.mz_name}: {weights.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    records = []
    for rep in range(args.n_repeats):
        chosen = rng.choice(len(weights), size=args.n_train_identities + args.n_test_identities, replace=False)
        tr = chosen[:args.n_train_identities]
        te = chosen[args.n_train_identities:]
        assert not set(tr).intersection(te)

        for x in args.x_range:
            print(f"[paired E1] repeat={rep} x={x}; train identities={len(tr)}; test identities={len(te)}", flush=True)
            # Pair each original with its attacked counterpart *within* training.
            train_b = img_pp_xlsb_attack(weights[tr], imsize=args.imsize, x=0, payload_filepath=None)
            train_m = img_pp_xlsb_attack(weights[tr], imsize=args.imsize, x=x, payload_filepath=None)
            test_b = img_pp_xlsb_attack(weights[te], imsize=args.imsize, x=0, payload_filepath=None)
            test_m = img_pp_xlsb_attack(weights[te], imsize=args.imsize, x=x, payload_filepath=None)

            X_tr = np.concatenate((train_b, train_m), axis=0)
            y_tr = np.array([0] * len(tr) + [1] * len(tr), dtype=np.int64)
            X_te = np.concatenate((test_b, test_m), axis=0)
            y_te = np.array([0] * len(te) + [1] * len(te), dtype=np.int64)

            # Set both NumPy/Python and TensorFlow RNGs before each fit.
            fit_seed = args.seed + rep * 1000 + x
            random.seed(fit_seed)
            np.random.seed(fit_seed)
            import tensorflow as tf
            tf.keras.utils.set_random_seed(fit_seed)

            model = None
            t0 = time.monotonic()
            try:
                model = train_fsl(X_tr, y_tr, model_arch="osl_siamese_cnn", imsize=args.imsize, mode=args.mode)
                result = evaluate_model(model, X_te, y_te)
            finally:
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            row = {
                "repeat": rep, "X": x, "centroid": float(result["centroid"]),
                "nn": float(result["nn"]), "n_train_identities": len(tr),
                "n_test_identities": len(te), "seed": fit_seed,
                "elapsed_sec": round(time.monotonic() - t0, 1),
            }
            records.append(row)
            print(f"[paired E1] centroid={row['centroid']:.4f}; nn={row['nn']:.4f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "fsl_paired.csv"
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    print(f"Saved {len(records)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
