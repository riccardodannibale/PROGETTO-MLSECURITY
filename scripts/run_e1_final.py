#!/usr/bin/env python3
"""E1 (ID) identity-disjoint few-shot protocol on real CNN weights.

This script runs the ORIGINAL Model X-Ray FSL network without editing the
upstream submodule. It adds fixed train/val/test IDs, identity-matched cover/
stego examples, per-sample score margins, and validation-only calibration.

Default stage is 'validation'. Test is accessed only with --stage test
--unblind-test, after the protocol has been fixed. This is an adapted E1
protocol, *not* a bit-for-bit reproduction of the paper's original 3+3 split.

Only inert pseudo-random payloads are supported (no payload file argument).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import euclidean_distances

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "ModelXRay"
sys.path.insert(0, str(UPSTREAM_ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_ids(csv_path: Path, protocol: dict, weights_path: Path) -> dict[str, pd.DataFrame]:
    if sha256(csv_path) != protocol["splits_sha256"]:
        raise ValueError("Hash degli split differente da protocol.json")
    if sha256(weights_path) != protocol["weights_sha256"]:
        raise ValueError("Hash dei pesi differente da protocol.json; non procedo")
    df = pd.read_csv(csv_path, dtype={"model_id": str, "checkpoint_sha256": str})
    required = {"row_index", "model_id", "checkpoint_sha256", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonne split mancanti: {sorted(required - set(df.columns))}")
    if df["row_index"].tolist() != list(range(protocol["n_models"])):
        raise ValueError("row_index non rispetta l'allineamento a weights.npy")
    if df["model_id"].duplicated().any() or df["checkpoint_sha256"].duplicated().any():
        raise ValueError("Modelli o checkpoint duplicati")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("Split diversi da train/val/test")
    splits = {k: df[df["split"] == k].copy() for k in ("train", "val", "test")}
    for k, v in splits.items():
        if len(v) != protocol["partition_sizes"][k]:
            raise ValueError(f"Numero modelli non corretto in {k}")
    return splits


def make_pair_images(weights: np.ndarray, indices: np.ndarray, *, x: int, imsize: int):
    from model_xray.data.attack_pipeline import img_pp_xlsb_attack
    base = np.asarray(weights[indices], dtype=np.float32)
    covers = img_pp_xlsb_attack(base.copy(), imsize=imsize, x=0, payload_filepath=None)
    stegos = img_pp_xlsb_attack(base.copy(), imsize=imsize, x=x, payload_filepath=None)
    images = np.concatenate([covers, stegos], axis=0)
    labels = np.concatenate([np.zeros(len(indices), dtype=int), np.ones(len(indices), dtype=int)])
    rows = np.concatenate([indices, indices])
    return images, labels, rows


def margins_and_reference_preds(model, images: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute Euclidean (d_benign-d_stego) margins and original FSL labels.

    Positive scores mean closer to the stego class. For 1-NN we use the
    nearest training *example* in each class; centroid uses class means.
    Both are checked against Siamese's own inference methods on every run.
    """
    embeds = np.asarray(model.get_embeddings(images))
    train_embeds = np.asarray(model.train_data["x_train_embeddings"])
    train_y = np.asarray(model.train_data["y"]).astype(int)
    if set(np.unique(train_y)) != {0, 1}:
        raise ValueError("Expected class labels 0/1")
    if not np.isfinite(embeds).all() or not np.isfinite(train_embeds).all():
        raise ValueError("Non-finite FSL embeddings")

    # Match default Siamese calc_centroids(..., apply_transforms='NA'), 1NN.
    centers = np.stack([train_embeds[train_y == cls].mean(axis=0) for cls in (0, 1)])
    dc = euclidean_distances(embeds, centers)
    dr = euclidean_distances(embeds, train_embeds)
    d0 = np.min(dr[:, train_y == 0], axis=1)
    d1 = np.min(dr[:, train_y == 1], axis=1)
    outputs = {
        "centroid": (dc[:, 0] - dc[:, 1], model.inference_centroid(
            images, x_test_embeddings=embeds, apply_transforms="NA")),
        "nn": (d0 - d1, model.inference_nn(
            images, x_test_embeddings=embeds, k=1, metric="euclidean")),
    }
    for method, (score, original_preds) in outputs.items():
        original_preds = np.asarray(original_preds).astype(int)
        score_preds = (score > 0).astype(int)
        # Margin=0 ties can follow different 1-NN tie breaking. Explicitly
        # reject this corner case instead of presenting non-equivalent scores.
        unequal = np.flatnonzero(score_preds != original_preds)
        if unequal.size:
            raise AssertionError(
                f"Score/predizione upstream discordi in {method}: "
                f"{unequal.size}/{len(score)}; non uso queste metriche."
            )
        if not np.isfinite(score).all():
            raise ValueError(f"Non-finite {method} margins")
        outputs[method] = (np.asarray(score, dtype=float), original_preds)
    return outputs


def validation_threshold(y_true: np.ndarray, margins: np.ndarray) -> float:
    """Select threshold on VAL only by maximum Youden J, deterministic ties.

    The score uses a strict '>' decision. roc_curve's '>=' thresholds are
    converted to equivalent strict thresholds with nextafter toward -inf.
    """
    fpr, tpr, t = roc_curve(y_true, margins, drop_intermediate=False)
    keep = np.isfinite(t)
    j = tpr - fpr
    best = np.flatnonzero(keep & (j >= np.max(j[keep]) - 1e-12))
    if not len(best):
        raise ValueError("No finite validation threshold")
    # Tie resolution: threshold closest to the original zero-margin rule.
    candidate = t[best[np.argmin(np.abs(t[best]))]]
    return float(np.nextafter(candidate, -np.inf))


def calc_metrics(y_true: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "n": int(len(y_true)), "n_benign": int(np.sum(y_true == 0)),
        "n_stego": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)), "specificity": float(tn / (tn + fp)),
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def prediction_rows(*, source_rows, y, split_table, outputs, thresholds, repeat, x, stage, fit_seed):
    lookup = split_table.set_index("row_index")
    data = []
    for method, (margin, upstream_pred) in outputs.items():
        threshold = thresholds[method]
        preds = (margin > threshold).astype(int)
        for idx in range(len(y)):
            row_id = int(source_rows[idx])
            data.append({
                "stage": stage, "repeat": repeat, "X": x,
                "fit_seed": fit_seed, "method": method,
                "row_index": row_id, "model_id": lookup.at[row_id, "model_id"],
                "checkpoint_sha256": lookup.at[row_id, "checkpoint_sha256"],
                "label": int(y[idx]), "score_margin": float(margin[idx]),
                "prediction_original": int(upstream_pred[idx]),
                "validation_threshold": float(threshold),
                "prediction_calibrated": int(preds[idx]),
            })
    return pd.DataFrame(data)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "manifests/e1_uniform_v1")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["validation", "test"], default="validation")
    ap.add_argument("--unblind-test", action="store_true")
    ap.add_argument("--x-range", nargs="+", type=int, default=[1, 8, 16, 23])
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-train-identities", type=int, default=3)
    ap.add_argument("--imsize", type=int, default=100)
    ap.add_argument("--mode", choices=["ub", "es", "st"], default="ub")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage == "test" and not args.unblind_test:
        ap.error("Test blind: requires --unblind-test after freezing the protocol")
    if args.unblind_test and args.stage != "test":
        ap.error("--unblind-test only with --stage test")
    if args.n_repeats < 1 or args.n_train_identities < 3:
        ap.error("Need positive repeats and at least 3 paired training identities")
    if not args.x_range or any(not 1 <= x <= 23 for x in args.x_range):
        ap.error("X must be in [1,23]")
    if len(args.x_range) != len(set(args.x_range)):
        ap.error("Duplicate X values")
    if os.environ.get("MODELXRAY_PAYLOAD_FILE"):
        ap.error("Unset MODELXRAY_PAYLOAD_FILE: this protocol uses inert random payloads only")
    return args


def main():
    args = parse_args()
    weights_path = args.weights.expanduser().resolve(strict=True)
    split_dir = args.split_dir.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    protocol = json.loads((split_dir / "protocol.json").read_text())
    w = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if w.shape != (protocol["n_models"], protocol["n_parameters_per_model"]):
        raise ValueError(f"Weights shape mismatch: {w.shape}")
    if w.dtype != np.float32:
        raise ValueError(f"Expected float32 weights, found {w.dtype}")
    splits = split_ids(split_dir / "splits.csv", protocol, weights_path)
    if args.n_train_identities > len(splits["train"]):
        raise ValueError("Too many training identities")
    config = {
        "stage": args.stage, "x_range": args.x_range, "n_repeats": args.n_repeats,
        "n_train_identities": args.n_train_identities, "imsize": args.imsize,
        "mode": args.mode, "seed": args.seed, "weights_sha256": protocol["weights_sha256"],
        "splits_sha256": protocol["splits_sha256"],
        "protocol_id": protocol["protocol_id"],
        "source": "ModelXRay upstream Siamese + separate identity-paired protocol",
        "score_definition": "Euclidean margin nearest benign - nearest stego; higher=stego",
        "threshold_selection": "Validation-only maximum Youden J, original zero margin also reported",
        "important_note": "Earlier exploratory runs used the same zoo; not fully external holdout",
    }
    try:
        config["project_commit"] = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        config["project_commit"] = "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old != config:
            raise ValueError(f"Out dir already used with different config: {out_dir}")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    from model_xray.fsl.train import train_fsl
    import tensorflow as tf

    train_pool = splits["train"]["row_index"].to_numpy(dtype=int)
    val_ids = splits["val"]["row_index"].to_numpy(dtype=int)
    test_ids = splits["test"]["row_index"].to_numpy(dtype=int)
    print(f"E1 {args.stage}: weights={w.shape}, split sizes="
          f"{len(train_pool)}/{len(val_ids)}/{len(test_ids)}", flush=True)

    for rep in range(args.n_repeats):
        train_ids = np.random.default_rng(args.seed + rep).choice(
            train_pool, size=args.n_train_identities, replace=False)
        for x in args.x_range:
            run_dir = out_dir / f"repeat_{rep:02d}_x_{x:02d}"
            required = [run_dir / "metrics.json", run_dir / "predictions_validation.csv"]
            if args.stage == "test":
                required.append(run_dir / "predictions_test.csv")
            if all(p.is_file() for p in required):
                print(f"[E1] skip completed repeat={rep} X={x}", flush=True)
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(f"Partial run exists; inspect before retry: {run_dir}")
            print(f"[E1] repeat={rep} X={x} stage={args.stage}", flush=True)
            if set(train_ids) & (set(val_ids) | set(test_ids)):
                raise AssertionError("Identity leakage from train to val/test")
            fit_seed = args.seed + rep * 1000 + x
            random.seed(fit_seed)
            np.random.seed(fit_seed)
            tf.keras.utils.set_random_seed(fit_seed)
            Xtr, ytr, _ = make_pair_images(w, train_ids, x=x, imsize=args.imsize)
            Xval, yval, val_rows = make_pair_images(w, val_ids, x=x, imsize=args.imsize)
            model = None
            t0 = time.monotonic()
            try:
                model = train_fsl(Xtr, ytr, model_arch="osl_siamese_cnn", imsize=args.imsize, mode=args.mode)
                val_out = margins_and_reference_preds(model, Xval)
                thresholds = {m: validation_threshold(yval, out[0]) for m, out in val_out.items()}
                metrics = []
                for method, (score, _) in val_out.items():
                    for policy, thr in (("original_zero", 0.0), ("val_youden", thresholds[method])):
                        metrics.append({"stage": "validation", "method": method, "policy": policy,
                                        **calc_metrics(yval, score, thr)})
                validation_df = prediction_rows(
                    source_rows=val_rows, y=yval, split_table=splits["val"],
                    outputs=val_out, thresholds=thresholds, repeat=rep,
                    x=x, stage="validation", fit_seed=fit_seed)
                test_df = None
                if args.stage == "test":
                    Xtest, ytest, test_rows = make_pair_images(w, test_ids, x=x, imsize=args.imsize)
                    test_out = margins_and_reference_preds(model, Xtest)
                    for method, (score, _) in test_out.items():
                        for policy, thr in (("original_zero", 0.0), ("val_youden", thresholds[method])):
                            metrics.append({"stage": "test", "method": method, "policy": policy,
                                            **calc_metrics(ytest, score, thr)})
                    test_df = prediction_rows(
                        source_rows=test_rows, y=ytest, split_table=splits["test"],
                        outputs=test_out, thresholds=thresholds, repeat=rep,
                        x=x, stage="test", fit_seed=fit_seed)
            finally:
                del model
                tf.keras.backend.clear_session()
                gc.collect()

            run_dir.mkdir(parents=True, exist_ok=False)
            validation_df.to_csv(run_dir / "predictions_validation.csv", index=False)
            if test_df is not None:
                test_df.to_csv(run_dir / "predictions_test.csv", index=False)
            (run_dir / "metrics.json").write_text(json.dumps({
                "repeat": rep, "X": x, "fit_seed": fit_seed,
                "train_source_indices": [int(i) for i in train_ids],
                "validation_source_indices": [int(i) for i in val_ids],
                "test_source_indices": [int(i) for i in test_ids] if args.stage == "test" else [],
                "elapsed_sec": round(time.monotonic() - t0, 2), "metrics": metrics,
            }, indent=2, sort_keys=True) + "\n")
            print(f"[E1] saved repeat={rep} X={x} in {run_dir}; elapsed={time.monotonic()-t0:.1f}s", flush=True)

    runs = sorted(out_dir.glob("repeat_*_x_*/metrics.json"))
    metrics_rows = []
    for p in runs:
        obj = json.loads(p.read_text())
        for m in obj["metrics"]:
            metrics_rows.append({"repeat": obj["repeat"], "X": obj["X"],
                                 "fit_seed": obj["fit_seed"], "elapsed_sec": obj["elapsed_sec"], **m})
    if not metrics_rows:
        raise RuntimeError("No completed runs")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_per_run.csv", index=False)
    summary = metrics_df.groupby(["stage", "X", "method", "policy"], as_index=False).agg(
        n_runs=("accuracy", "size"), accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"), auroc_mean=("auroc", "mean"),
        auprc_mean=("auprc", "mean"), f1_mean=("f1", "mean"),
        fpr_mean=("fpr", "mean"))
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nSUMMARY\n" + summary.to_string(index=False), flush=True)
    expected = args.n_repeats * len(args.x_range)
    print(f"Completed runs: {len(runs)}/{expected}; files: {out_dir}", flush=True)
    if len(runs) != expected:
        raise RuntimeError("Incomplete experimental matrix")


if __name__ == "__main__":
    main()
