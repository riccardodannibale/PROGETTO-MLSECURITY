#!/usr/bin/env python3
"""E2 G5: paired LSB-ID / synthetic-spread-OOD Model X-Ray experiment.

Stage 'validation' trains Siamese on X-LSB (X=23) and calibrates on LSB only.
Stage 'test' trains separate, new fits on LSB, calibrates on LSB validation,
and evaluates ID and both OOD attacks with the SAME detector per fit.

TEST IS LOCKED: requires full 10/10 validation, reviewed frozen protocol,
explicit opt-in, matched runner SHA, clean Git, and a separate output dir.
Never use OOD/test scores to tune alpha, thresholds, or detector training.
No malware is read or executed. This is NOT upstream MaleficNet.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
UPSTREAM = ROOT / "external/ModelXRay"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(UPSTREAM))

import run_e1_final_v2 as e1
import e2_synthetic_spread as spread

E1_SHA = "a75aa3d844bb695a6d6155fe808394025b04c480d74d1e1b856eec5ca6f9c037"
SPREAD_SHA = "70d9e0c02fd26dffc7d69f3fbbdb70cd3219844200c625891797eb13ee544edd"
UPSTREAM_COMMIT = "a880e332f556fc71faf8d3753931005d5d019339"
WEIGHTS_SHA = "5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad"
SPLITS_SHA = "3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f"
ALPHAS = (0.24, 0.16)  # pre-selected from TRAIN-only pilot
BITS = 16
SEED = 42
X_LSB = 23
REPEATS = 10
MODE = "ub"
IMSIZE = 100
TRAIN_IDENTITIES = 3


def sha(path: Path) -> str:
    return e1.sha256(path)


def git(*cmd: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *cmd], text=True).strip()


def source_preflight(weights_path: Path, split_dir: Path):
    for relative, expected in (("scripts/run_e1_final_v2.py", E1_SHA),
                               ("scripts/e2_synthetic_spread.py", SPREAD_SHA)):
        current = sha(ROOT / relative)
        if current != expected:
            raise RuntimeError(f"STOP: {relative} SHA unexpected: {current}")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"STOP: ModelXRay submodule SHA mismatch: {commit}")
    if os.environ.get("MODELXRAY_PAYLOAD_FILE"):
        raise RuntimeError("STOP: external payload override forbidden")
    if sha(weights_path) != WEIGHTS_SHA or sha(split_dir / "splits.csv") != SPLITS_SHA:
        raise RuntimeError("STOP: dataset/split hash mismatch")
    protocol = json.loads((split_dir / "protocol.json").read_text())
    if (protocol["weights_sha256"] != WEIGHTS_SHA or protocol["splits_sha256"] != SPLITS_SHA or
        protocol["n_models"] != 1000 or protocol["n_parameters_per_model"] != 2864 or
        protocol["partition_sizes"] != {"train": 600, "val": 200, "test": 200}):
        raise RuntimeError("STOP: unexpected E1 frozen protocol")
    splits = e1.split_ids(split_dir / "splits.csv", protocol, weights_path)
    return protocol, splits


def check_test_unlock(*, approval: Path | None, validation_dir: Path,
                      runner_path: Path, expected_runner_sha: str):
    """Fail closed before the first test checkpoint is read. No TF imports."""
    if approval is None or not approval.is_file():
        raise RuntimeError("STOP: test locked; approved E2 freeze JSON missing")
    approved = json.loads(approval.read_text())
    needed = {"state": "approved_for_e2_test", "validation_count": 10,
              "runner_sha256": expected_runner_sha}
    for key, value in needed.items():
        if approved.get(key) != value:
            raise RuntimeError(f"STOP: approval {key} not matched")
    frozen_path = ROOT / "docs/E2_PROTOCOL_FROZEN.md"
    if (not frozen_path.is_file() or
        approved.get("frozen_protocol_sha256") != sha(frozen_path)):
        raise RuntimeError("STOP: frozen protocol doc absent/different")
    if approved.get("riccardo_approved") is not True or approved.get("fabrizio_approved") is not True:
        raise RuntimeError("STOP: missing explicit joint approval")
    if not validation_dir.is_dir():
        raise RuntimeError("STOP: validation campaign missing")
    val_config = json.loads((validation_dir / "run_config.json").read_text())
    if val_config.get("stage") != "validation" or val_config.get("runner_sha256") != expected_runner_sha:
        raise RuntimeError("STOP: validation used different runner")
    if val_config.get("n_repeats") != 10 or val_config.get("alphas") != list(ALPHAS):
        raise RuntimeError("STOP: validation config differs")
    for rep in range(REPEATS):
        d = validation_dir / f"repeat_{rep:02d}_x_23"
        required = ("metrics.json", "predictions_validation.csv", "payload_manifest.csv")
        if any(not (d / name).is_file() or (d / name).stat().st_size == 0 for name in required):
            raise RuntimeError(f"STOP: incomplete validation fit {rep}")
        metrics = json.loads((d / "metrics.json").read_text())
        if metrics["repeat"] != rep or metrics["stage"] != "validation" or len(metrics["metrics"]) != 4:
            raise RuntimeError(f"STOP: malformed validation run {rep}")
        if {m["condition"] for m in metrics["metrics"]} != {"LSB_ID"}:
            raise RuntimeError("STOP: unexpected OOD metrics in validation")
    if git("branch", "--show-current") != "ricky":
        raise RuntimeError("STOP: not on branch ricky")
    # Changed tracked code or protocol is forbidden. Untracked scratch files are
    # tolerated: they do not change reviewed committed sources.
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("STOP: tracked files have uncommitted changes")
    if subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:scripts/run_e2_paired_v0.py"]) != runner_path.read_bytes():
        raise RuntimeError("STOP: E2 runner not identical to committed version")
    if subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:docs/E2_PROTOCOL_FROZEN.md"]) != frozen_path.read_bytes():
        raise RuntimeError("STOP: frozen E2 protocol not committed unchanged")


def make_spread_pairs(weights: np.ndarray, indices: np.ndarray, *, alpha: float,
                      repeat: int, imsize: int = IMSIZE):
    """Image representations for same original covers and keyed spread stegos.

    No test data is accessed until the explicit stage=test branch.
    """
    from model_xray.data.attack_pipeline import img_pp_xlsb_attack
    ids = np.asarray(indices, dtype=int)
    base = np.asarray(weights[ids], dtype=np.float32)
    if base.shape != (len(ids), 2864):
        raise RuntimeError(f"STOP: invalid spread weight shape: {base.shape}")
    covers = img_pp_xlsb_attack(base.copy(), imsize=imsize, x=0, payload_filepath=None)
    modifications = []
    metadata = []
    for row, w in zip(ids, base):
        rid = int(row)
        bits = spread.synthetic_message(n_bits=BITS, master_seed=SEED, repeat=repeat, row_index=rid)
        encoded = spread.embed_inert_bits(w, bits, alpha=alpha, master_seed=SEED, repeat=repeat, row_index=rid)
        decoded = spread.decode_inert_bits(encoded, n_bits=BITS, master_seed=SEED, repeat=repeat, row_index=rid)
        errors = int(np.count_nonzero(bits != decoded))
        delta = encoded.astype(np.float64) - w.astype(np.float64)
        relative = float(np.linalg.norm(delta) / np.linalg.norm(w.astype(np.float64)))
        if not np.isfinite(relative) or not np.isfinite(encoded).all():
            raise RuntimeError("STOP: non-finite OOD representation")
        modifications.append(encoded)
        metadata.append({"source_split": "test", "row_index": rid, "family": "SPREAD_v0",
                         "alpha": alpha, "repeat": repeat, "n_bits": BITS,
                         "payload_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                         "bit_errors": errors, "ber": errors / BITS,
                         "exact_recovery": int(errors == 0), "relative_l2": relative,
                         "max_abs_delta": float(np.max(np.abs(delta))),
                         "changed_weights": int(np.count_nonzero(delta))})
    modified = np.stack(modifications, axis=0)
    stegos = img_pp_xlsb_attack(modified, imsize=imsize, x=0, payload_filepath=None)
    if covers.shape != stegos.shape or covers.shape != (len(ids), imsize, imsize):
        raise RuntimeError("STOP: wrong shape after canonical representation")
    if not np.isfinite(covers).all() or not np.isfinite(stegos).all():
        raise RuntimeError("STOP: non-finite canonical representations")
    return (np.concatenate([covers, stegos], axis=0),
            np.concatenate([np.zeros(len(ids), dtype=int), np.ones(len(ids), dtype=int)]),
            np.concatenate([ids, ids]), metadata, covers)


def validation_evaluation(model, w, val_ids, splits, rep):
    xval, yval, val_rows, payloads = e1.make_pair_images(
        w, val_ids, x=23, imsize=IMSIZE, master_seed=SEED, repeat=rep)
    outputs = e1.margins_and_reference_preds(model, xval)
    thresholds = {method: e1.validation_threshold(yval, data[0]) for method, data in outputs.items()}
    metrics = [{"stage": "validation", "condition": "LSB_ID", "method": method, "policy": policy,
                **e1.calc_metrics(yval, scores, thr)}
               for method, (scores, _pred) in outputs.items()
               for policy, thr in (("original_zero", 0.0), ("val_youden", thresholds[method]))]
    prediction = e1.prediction_rows(source_rows=val_rows, y=yval, split_table=splits["val"],
                                      outputs=outputs, thresholds=thresholds,
                                      repeat=rep, x=23, stage="validation", fit_seed=SEED + rep * 1000 + 23)
    prediction.insert(0, "condition", "LSB_ID")
    return thresholds, metrics, prediction, payloads


def test_evaluation(model, w, test_ids, splits, rep, thresholds):
    frames, all_metrics, payloads = [], [], []
    reference_covers = None
    # ID and both OOD conditions are evaluated on exactly the same trained model.
    for condition, alpha in (("LSB_ID", None), ("SPREAD_0.24", 0.24), ("SPREAD_0.16", 0.16)):
        if alpha is None:
            x, y, rows, payload_meta = e1.make_pair_images(
                w, test_ids, x=23, imsize=IMSIZE, master_seed=SEED, repeat=rep)
            payloads.extend([dict(source_split="test", family="LSB_ID", **p) for p in payload_meta])
            reference_covers = x[:len(test_ids)].copy()
        else:
            x, y, rows, payload_meta, covers = make_spread_pairs(w, test_ids, alpha=alpha, repeat=rep)
            if not np.array_equal(covers, reference_covers):
                raise RuntimeError("STOP: ID and OOD benign images not identical")
            payloads.extend(payload_meta)
        outputs = e1.margins_and_reference_preds(model, x)
        if set(outputs) != set(thresholds):
            raise RuntimeError("STOP: missing expected methods in test")
        for method, (score, _pred) in outputs.items():
            for policy, thr in (("original_zero", 0.0), ("val_youden", thresholds[method])):
                all_metrics.append({"stage": "test", "condition": condition,
                                    "method": method, "policy": policy,
                                    **e1.calc_metrics(y, score, thr)})
        df = e1.prediction_rows(source_rows=rows, y=y, split_table=splits["test"],
                                outputs=outputs, thresholds=thresholds, repeat=rep,
                                x=23, stage="test", fit_seed=SEED + rep * 1000 + 23)
        df.insert(0, "condition", condition)
        frames.append(df)
        del x, y, rows, outputs
    return pd.concat(frames, ignore_index=True), all_metrics, payloads


def aggregate(out_dir: Path):
    paths = sorted(out_dir.glob("repeat_*_x_23/metrics.json"))
    entries = []
    for p in paths:
        entry = json.loads(p.read_text())
        for metric in entry["metrics"]:
            entries.append({"repeat": entry["repeat"], "fit_seed": entry["fit_seed"],
                            "elapsed_sec": entry["elapsed_sec"], **metric})
    if not entries:
        raise RuntimeError("STOP: no completed fits")
    df = pd.DataFrame(entries)
    df.to_csv(out_dir / "metrics_per_run.csv", index=False)
    summary = df.groupby(["stage", "condition", "method", "policy"], as_index=False).agg(
        n_runs=("auroc", "size"), accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"), auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"), auprc_mean=("auprc", "mean"),
        f1_mean=("f1", "mean"), fpr_mean=("fpr", "mean"))
    summary.to_csv(out_dir / "summary.csv", index=False)
    # In test stage, compute primary paired AUROC delta for SAME repeat/model.
    test = df[(df.stage == "test") & (df.method == "centroid") & (df.policy == "val_youden")]
    if not test.empty:
        pivot = test.pivot(index="repeat", columns="condition", values="auroc")
        if set(pivot.columns) != {"LSB_ID", "SPREAD_0.24", "SPREAD_0.16"}:
            raise RuntimeError("STOP: incomplete paired ID/OOD score conditions")
        pivot["delta_auroc_ood24_minus_id"] = pivot["SPREAD_0.24"] - pivot["LSB_ID"]
        pivot["delta_auroc_ood16_minus_id"] = pivot["SPREAD_0.16"] - pivot["LSB_ID"]
        pivot.to_csv(out_dir / "paired_auroc_per_repeat.csv")
    return len(paths)


def make_config(stage: str, runner_sha: str):
    return {
        "stage": stage, "runner_sha256": runner_sha,
        "weights_sha256": WEIGHTS_SHA, "splits_sha256": SPLITS_SHA,
        "upstream_commit": UPSTREAM_COMMIT, "e1_runner_sha256": E1_SHA,
        "spread_runner_sha256": SPREAD_SHA,
        "x": X_LSB, "n_repeats": REPEATS, "n_train_identities": TRAIN_IDENTITIES,
        "seed": SEED, "fit_seed": "seed+repeat*1000+x", "mode": MODE,
        "imsize": IMSIZE, "alphas": list(ALPHAS), "spread_bits": BITS,
        "payload_scheme": e1.PAYLOAD_SCHEME,
        "primary": "paired AUROC centroid OOD0.24 minus LSB ID",
        "notes": "E2 internal attack-family shift; E1 checkpoint test was already inspected",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, default=ROOT / "manifests/e1_uniform_v1")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["preflight", "validation", "test"], default="preflight")
    ap.add_argument("--only-repeat", type=int)
    ap.add_argument("--unblind-test", action="store_true")
    ap.add_argument("--test-approval", type=Path)
    ap.add_argument("--validation-dir", type=Path)
    args = ap.parse_args()
    if args.only_repeat is not None and not 0 <= args.only_repeat < REPEATS:
        ap.error("--only-repeat must be in [0,9]")
    if args.stage != "test" and (args.unblind_test or args.test_approval or args.validation_dir):
        ap.error("test flags are only accepted with stage=test")
    if args.stage == "test" and (not args.unblind_test or args.test_approval is None or args.validation_dir is None):
        ap.error("TEST LOCKED: --unblind-test --test-approval --validation-dir required")

    weights_path = args.weights.expanduser().resolve(strict=True)
    split_dir = args.split_dir.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    runner_sha = sha(Path(__file__).resolve())
    protocol, splits = source_preflight(weights_path, split_dir)
    if args.stage == "test":
        check_test_unlock(approval=args.test_approval.expanduser().resolve(),
                          validation_dir=args.validation_dir.expanduser().resolve(),
                          runner_path=Path(__file__).resolve(), expected_runner_sha=runner_sha)
    if args.stage == "preflight":
        print("E2 G5 PREFLIGHT OK: source+dataset+split hashes; test locked; no training, no test checkpoint vectors materialized")
        return

    if args.stage == "validation" and args.only_repeat is None:
        ap.error("G5 validation requires --only-repeat: one fresh OS process per fit")
    if args.stage == "test" and args.only_repeat is None:
        ap.error("G6 test requires --only-repeat: one fresh OS process per fit")
    config = make_config(args.stage, runner_sha)
    config_path = out_dir / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("STOP: output directory reused with different config")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    rep = args.only_repeat
    run_dir = out_dir / f"repeat_{rep:02d}_x_23"
    if run_dir.exists():
        raise RuntimeError(f"STOP: run directory already exists; NO selective rerun: {run_dir}")

    # A full-file SHA was computed in source_preflight, but validation never
    # materializes/interprets a TEST checkpoint vector or generates test representations.
    w = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if w.shape != (1000, 2864) or w.dtype != np.float32:
        raise RuntimeError("STOP: wrong D1 weights array shape/dtype")
    train_pool = splits["train"]["row_index"].to_numpy(dtype=int)
    val_ids = splits["val"]["row_index"].to_numpy(dtype=int)
    test_ids = splits["test"]["row_index"].to_numpy(dtype=int) if args.stage == "test" else None
    tr_ids = np.random.default_rng(SEED + rep).choice(train_pool, size=TRAIN_IDENTITIES, replace=False)
    if set(tr_ids) & (set(val_ids) | (set(test_ids) if test_ids is not None else set())):
        raise RuntimeError("STOP: checkpoint identity leakage")

    fit_seed = SEED + rep * 1000 + X_LSB
    random.seed(fit_seed); np.random.seed(fit_seed)
    import tensorflow as tf
    from model_xray.fsl.train import train_fsl
    tf.keras.utils.set_random_seed(fit_seed)
    xtr, ytr, _, train_payloads = e1.make_pair_images(
        w, tr_ids, x=23, imsize=IMSIZE, master_seed=SEED, repeat=rep)
    xval, yval, val_rows, val_payloads = e1.make_pair_images(
        w, val_ids, x=23, imsize=IMSIZE, master_seed=SEED, repeat=rep)
    payloads = [dict(source_split="train", family="LSB_ID", **x) for x in train_payloads]
    payloads += [dict(source_split="val", family="LSB_ID", **x) for x in val_payloads]

    start = time.monotonic()
    model = None
    try:
        model = train_fsl(xtr, ytr, model_arch="osl_siamese_cnn", imsize=IMSIZE, mode=MODE)
        val_outputs = e1.margins_and_reference_preds(model, xval)
        thresholds = {method: e1.validation_threshold(yval, pair[0]) for method, pair in val_outputs.items()}
        metrics = [{"stage": "validation", "condition": "LSB_ID", "method": method, "policy": policy,
                    **e1.calc_metrics(yval, score, thr)}
                   for method, (score, _pred) in val_outputs.items()
                   for policy, thr in (("original_zero", 0.0), ("val_youden", thresholds[method]))]
        val_df = e1.prediction_rows(source_rows=val_rows, y=yval, split_table=splits["val"],
                                     outputs=val_outputs, thresholds=thresholds,
                                     repeat=rep, x=23, stage="validation", fit_seed=fit_seed)
        val_df.insert(0, "condition", "LSB_ID")
        del val_outputs, xval, yval, val_rows
        test_df = None
        if args.stage == "test":
            test_df, test_metrics, test_payloads = test_evaluation(model, w, test_ids, splits, rep, thresholds)
            metrics.extend(test_metrics)
            payloads.extend(test_payloads)
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Output is materialized after a whole fit completed successfully.
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(payloads).to_csv(run_dir / "payload_manifest.csv", index=False)
    val_df.to_csv(run_dir / "predictions_validation.csv", index=False)
    if test_df is not None:
        test_df.to_csv(run_dir / "predictions_test.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps({
        "repeat": rep, "stage": args.stage, "fit_seed": fit_seed,
        "train_source_indices": [int(i) for i in tr_ids],
        "validation_source_indices": [int(i) for i in val_ids],
        "test_source_indices": [int(i) for i in test_ids] if test_ids is not None else [],
        "thresholds_val_youden": thresholds,
        "metrics": metrics, "elapsed_sec": round(time.monotonic() - start, 2),
    }, sort_keys=True, indent=2) + "\n")
    finished = aggregate(out_dir)
    print(f"E2 G5 {args.stage} repeat={rep} OK: {finished}/{REPEATS} fits. OUT={out_dir}", flush=True)
    print("WARNING: NOT MaleficNet; test not novel blind external holdout", flush=True)


if __name__ == "__main__":
    main()
