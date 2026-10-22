#!/usr/bin/env python3
"""E2 G2: read-only TRAIN-pool audit of canonical E1 X-LSB vs spread v0.

Uses exactly E1's deterministic payload helper and its high-level upstream
img_pp_xlsb_attack. Temporarily intercepts that pipeline's x_lsb_attack
return value to measure PRE-image attacked float32 weights. The monkeypatch
is local to this Python process and is restored immediately after each call.
No detector training, checkpoint loading, validation or test samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT / "external/ModelXRay"
EXPECTED_E1 = "a75aa3d844bb695a6d6155fe808394025b04c480d74d1e1b856eec5ca6f9c037"
EXPECTED_SPREAD = "70d9e0c02fd26dffc7d69f3fbbdb70cd3219844200c625891797eb13ee544edd"
EXPECTED_UPSTREAM = "a880e332f556fc71faf8d3753931005d5d019339"
EXPECTED_WEIGHTS = "5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad"
EXPECTED_SPLITS = "3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_float(got: float, expected: float, tolerance: float = 1e-6) -> bool:
    return math.isfinite(got) and math.isfinite(expected) and math.isclose(
        got, expected, rel_tol=tolerance, abs_tol=tolerance
    )


def verify_environment(weight_path: Path, split_dir: Path, smoke_dir: Path) -> tuple[list[int], dict[tuple[int, float], dict]]:
    if os.environ.get("MODELXRAY_PAYLOAD_FILE"):
        raise ValueError("STOP: unset MODELXRAY_PAYLOAD_FILE, no external payloads")
    if digest(PROJECT / "scripts/run_e1_final_v2.py") != EXPECTED_E1:
        raise ValueError("STOP: E1 runner SHA-256 differs; inspect before continuing")
    if digest(PROJECT / "scripts/e2_synthetic_spread.py") != EXPECTED_SPREAD:
        raise ValueError("STOP: spread starter SHA-256 differs; inspect before continuing")
    if digest(weight_path) != EXPECTED_WEIGHTS or digest(split_dir / "splits.csv") != EXPECTED_SPLITS:
        raise ValueError("STOP: E1 weights/splits SHA-256 mismatch")
    config = json.loads((split_dir / "protocol.json").read_text(encoding="utf-8"))
    if config.get("weights_sha256") != EXPECTED_WEIGHTS or config.get("splits_sha256") != EXPECTED_SPLITS:
        raise ValueError("STOP: E1 protocol.json hash mismatch")
    smoke_cfg = json.loads((smoke_dir / "smoke_config.json").read_text(encoding="utf-8"))
    for field, expected in {"seed":42, "repeat":0, "n_bits":16, "n_models":20,
                            "weights_sha256":EXPECTED_WEIGHTS, "splits_sha256":EXPECTED_SPLITS,
                            "code_sha256":EXPECTED_SPREAD}.items():
        if smoke_cfg.get(field) != expected:
            raise ValueError(f"STOP: unexpected smoke config {field}: {smoke_cfg.get(field)}")
    expected_alphas = [0.0, 0.05, 0.1, 0.16, 0.24, 0.32]
    if smoke_cfg.get("alphas") != expected_alphas:
        raise ValueError("STOP: unexpected smoke alpha grid")
    rev = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if rev != EXPECTED_UPSTREAM:
        raise ValueError(f"STOP: upstream commit differs: {rev}")
    split_rows = csv_rows(split_dir / "splits.csv")
    train_ids = sorted(int(r["row_index"]) for r in split_rows if r["split"] == "train")
    if len(split_rows) != 1000 or len(train_ids) != 600 or len(set(train_ids)) != 600:
        raise ValueError("STOP: invalid E1 split structure")
    rng = np.random.Generator(np.random.PCG64(42))
    expected_ids = sorted(int(x) for x in rng.choice(train_ids, size=20, replace=False))
    smoke_rows = csv_rows(smoke_dir / "smoke_per_model.csv")
    lookup = {}
    for r in smoke_rows:
        k = (int(r["row_index"]), float(r["alpha"]))
        if k in lookup or r["source_split"] != "train" or int(r["repeat"]) != 0 or int(r["seed"]) != 42 or int(r["n_bits"]) != 16:
            raise ValueError(f"STOP: malformed/duplicate smoke row {k}")
        lookup[k] = r
    if set(lookup) != {(i, a) for i in expected_ids for a in expected_alphas}:
        raise ValueError("STOP: smoke rows/IDs/alphas are not exactly the train-only run")
    return expected_ids, lookup


def capture_canonical_lsb(weights: np.ndarray, payload: bytes) -> np.ndarray:
    """Execute E1's SAME image pipeline but capture float weights before imaging.

    Do not reproduce X-LSB logic ourselves: the canonical upstream function is
    called exactly as in E1. Assert the intercept was hit precisely once.
    """
    from model_xray.data import attack_pipeline as ap
    original_attack = ap.x_lsb_attack
    captured = []

    def audit_hook(*args, **kwargs):
        out = original_attack(*args, **kwargs)
        captured.append(np.array(out, copy=True))
        return out

    with tempfile.TemporaryDirectory(prefix="e2_g2_inert_lsb_") as temp:
        binary_path = Path(temp) / "synthetic.bin"
        binary_path.write_bytes(payload)
        ap.x_lsb_attack = audit_hook
        try:
            rep = ap.img_pp_xlsb_attack(weights[np.newaxis, :].copy(), imsize=100,
                                       x=23, payload_filepath=str(binary_path))
        finally:
            ap.x_lsb_attack = original_attack
    if len(captured) != 1:
        raise RuntimeError(f"STOP: canonical pipeline called upstream X-LSB {len(captured)} times, expected 1")
    arr = captured[0]
    if arr.shape not in ((1, 2864), (2864,)) or arr.dtype != np.float32:
        raise RuntimeError(f"STOP: unexpected raw attacked shape/dtype {arr.shape}/{arr.dtype}")
    if np.asarray(rep).shape != (1, 100, 100):
        raise RuntimeError(f"STOP: pipeline representation shape {np.asarray(rep).shape}")
    stego = arr.reshape(-1)
    if not np.isfinite(stego).all() or np.array_equal(stego, weights):
        raise RuntimeError("STOP: attacked weights invalid or unchanged")
    return stego


def metrics(original: np.ndarray, modified: np.ndarray) -> dict:
    delta = modified.astype(np.float64) - original.astype(np.float64)
    orig_norm = float(np.linalg.norm(original.astype(np.float64)))
    if not orig_norm or not np.isfinite(delta).all():
        raise RuntimeError("STOP: zero original or non-finite delta")
    changed = int(np.count_nonzero(delta))
    return {
        "relative_l2": float(np.linalg.norm(delta) / orig_norm),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "changed_weights": changed,
        "changed_fraction": changed / len(delta),
        "all_finite": bool(np.isfinite(modified).all()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, default=PROJECT / "manifests/e1_uniform_v1")
    ap.add_argument("--smoke-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    wpath = args.weights.expanduser().resolve(strict=True)
    split_dir = args.split_dir.expanduser().resolve(strict=True)
    smoke_dir = args.smoke_dir.expanduser().resolve(strict=True)
    outdir = args.out_dir.expanduser().resolve()
    ids, smoke = verify_environment(wpath, split_dir, smoke_dir)
    if outdir.exists():
        raise SystemExit("STOP: output directory already exists; do not overwrite historical audits")
    w = np.load(wpath, mmap_mode="r", allow_pickle=False)
    if w.shape != (1000, 2864) or w.dtype != np.float32:
        raise SystemExit("STOP: wrong weights shape/dtype")
    # Import after verifying source hashes; E1 helper is the single source of truth.
    sys.path.insert(0, str(UPSTREAM))
    import run_e1_final_v2 as e1
    import e2_synthetic_spread as spread
    if e1.PAYLOAD_NBYTES != 16384 or e1.PAYLOAD_SCHEME != "sha256-keyed-uint8-pcg64-per-model-v1":
        raise RuntimeError("STOP: unexpected E1 payload protocol")
    records: list[dict] = []
    for row_index in ids:
        original = np.array(w[row_index], dtype=np.float32, copy=True)
        if not np.isfinite(original).all():
            raise RuntimeError(f"STOP: non-finite original row {row_index}")
        payload = e1.deterministic_payload(master_seed=42, repeat=0, x=23, row_index=row_index)
        before = original.copy()
        lsb = capture_canonical_lsb(original, payload)
        if not np.array_equal(original, before) or not np.array_equal(original, w[row_index]):
            raise RuntimeError(f"STOP: source original unexpectedly modified for row {row_index}")
        # X=23 must leave sign+exponent intact (uint32 mask 0xff800000).
        if not np.array_equal(original.view(np.uint32) & np.uint32(0xFF800000),
                              lsb.view(np.uint32) & np.uint32(0xFF800000)):
            raise RuntimeError(f"STOP: X-LSB changed sign/exponent at train row {row_index}")
        lsb_stats = metrics(original, lsb)
        records.append({"source_split": "train", "row_index": row_index, "attack": "X_LSB_23",
                        "alpha": "", "message_bits": 2864*23, "ber": "", "exact_recovery": "",
                        "payload_sha256": hashlib.sha256(payload).hexdigest(), **lsb_stats})
        bits = spread.synthetic_message(n_bits=16, master_seed=42, repeat=0, row_index=row_index)
        for alpha in (0.16, 0.24):
            encoded = spread.embed_inert_bits(original, bits, alpha=alpha, master_seed=42, repeat=0,
                                              row_index=row_index)
            decoded = spread.decode_inert_bits(encoded, n_bits=16, master_seed=42, repeat=0,
                                               row_index=row_index)
            stats = metrics(original, encoded)
            recorded = smoke[row_index, alpha]
            errors = int(np.count_nonzero(bits != decoded))
            if not compare_float(stats["relative_l2"], float(recorded["relative_l2"])) or \
               errors != int(recorded["bit_errors"]) or \
               not compare_float(stats["max_abs_delta"], float(recorded["max_abs_delta"])):
                raise RuntimeError(f"STOP: spread regeneration differs from smoke row={row_index}, alpha={alpha}")
            records.append({"source_split": "train", "row_index": row_index, "attack": "SPREAD_v0",
                            "alpha": alpha, "message_bits": 16, "ber": errors / 16,
                            "exact_recovery": int(errors == 0),
                            "payload_sha256": hashlib.sha256(bits.tobytes()).hexdigest(), **stats})
    summaries = []
    for attack, alpha in (("X_LSB_23", ""), ("SPREAD_v0", 0.16), ("SPREAD_v0", 0.24)):
        sub = [r for r in records if r["attack"] == attack and r["alpha"] == alpha]
        measures = np.array([r["relative_l2"] for r in sub], dtype=float)
        fractions = np.array([r["changed_fraction"] for r in sub], dtype=float)
        summaries.append({"attack": attack, "alpha": alpha, "n_models": len(sub),
                          "mean_relative_l2": float(measures.mean()),
                          "median_relative_l2": float(np.median(measures)),
                          "min_relative_l2": float(measures.min()),
                          "max_relative_l2": float(measures.max()),
                          "mean_changed_fraction": float(fractions.mean()),
                          "mean_max_abs_delta": float(np.mean([r["max_abs_delta"] for r in sub]))})
    # Write only after all validations succeeded.
    outdir.mkdir(parents=True)
    write_csv(outdir / "intensity_per_model.csv", records)
    write_csv(outdir / "intensity_summary.csv", summaries)
    (outdir / "audit_config.json").write_text(json.dumps({
        "scope": "E2 G2 exploratory TRAIN ONLY; no ML detector, no validation/test vectors",
        "e1_runner_sha256": EXPECTED_E1, "spread_sha256": EXPECTED_SPREAD,
        "upstream_commit": EXPECTED_UPSTREAM,
        "weights_sha256": EXPECTED_WEIGHTS, "splits_sha256": EXPECTED_SPLITS,
        "script_sha256": digest(Path(__file__).resolve()),
        "seed": 42, "repeat": 0, "n_models": 20,
        "lsb_x": 23, "spread_alphas": [0.16, 0.24], "spread_bits": 16,
        "identity_row_indices": ids,
        "method": "intercept x_lsb_attack return in canonical img_pp_xlsb_attack (before imaging)",
    }, indent=2) + "\n", encoding="utf-8")
    print("E2 G2 TRAIN-ONLY AUDIT OK: 20 sources × (LSB X23, spread .16, spread .24)")
    for item in summaries:
        print(f"  {item['attack']:<10} alpha={str(item['alpha']):>4} "
              f"mean rel_L2={item['mean_relative_l2']:.6f}, "
              f"mean changed_fraction={item['mean_changed_fraction']:.4f}")
    print(f"Saved {outdir}")
    print("Do not interpret as OOD detection, stealth, model accuracy, or matched payload capacity.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"E2 G2 AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
