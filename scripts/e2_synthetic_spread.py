#!/usr/bin/env python3
"""E2 research prototype: synthetic keyed spread-spectrum on float32 weights.

SAFE SCOPE: inert random bits only; no malware; no model inference/training.
Original implementation for course research. NOT MaleficNet and NOT a port.

Model: w' = float32(w + (alpha * ||w||_2 / sqrt(B)) * C @ b)
C: keyed N-by-B orthonormal real-valued random spreading columns.
Decoder: sign(C.T @ w') -> bits. Requires row_index and secret master_seed,
           but NOT the original benign weights. No ECC, LDPC, or fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ALGORITHM = "e2-keyed-orthogonal-spread-v0-educational"


def _seed(tag: str, master_seed: int, repeat: int, row_index: int) -> int:
    key = f"{ALGORITHM}|{tag}|{master_seed}|{repeat}|{row_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big")


def synthetic_message(*, n_bits: int, master_seed: int, repeat: int, row_index: int) -> np.ndarray:
    if not 1 <= n_bits <= 32:
        raise ValueError("Educational prototype supports 1..32 bits, not arbitrary files")
    rng = np.random.Generator(np.random.PCG64(_seed("payload", master_seed, repeat, row_index)))
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def spreading_codes(*, n_weights: int, n_bits: int, master_seed: int, repeat: int, row_index: int) -> np.ndarray:
    if n_weights < n_bits or n_bits < 1 or n_bits > 32:
        raise ValueError("Need N >= B, and B <= 32")
    rng = np.random.Generator(np.random.PCG64(_seed("code", master_seed, repeat, row_index)))
    # Full model dimension -> orthonormal pseudo-random spreading vectors.
    x = rng.standard_normal((n_weights, n_bits), dtype=np.float64)
    q, r = np.linalg.qr(x, mode="reduced")
    # Fix the arbitrary sign convention of QR columns for cross-run stability.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[np.newaxis, :]


def embed_inert_bits(weights: np.ndarray, bits: np.ndarray, *, alpha: float, master_seed: int, repeat: int, row_index: int) -> np.ndarray:
    w = np.asarray(weights)
    b = np.asarray(bits)
    if w.ndim != 1 or w.dtype != np.float32 or not np.isfinite(w).all():
        raise ValueError("Expected a finite 1D float32 weight vector")
    if b.ndim != 1 or b.dtype != np.uint8 or not np.isin(b, [0, 1]).all():
        raise ValueError("Expected a uint8 synthetic bit vector")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    norm = float(np.linalg.norm(w.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Zero/invalid carrier norm")
    codes = spreading_codes(n_weights=w.size, n_bits=b.size, master_seed=master_seed, repeat=repeat, row_index=row_index)
    symbols = 2.0 * b.astype(np.float64) - 1.0
    delta = (float(alpha) * norm / math.sqrt(b.size)) * (codes @ symbols)
    encoded = (w.astype(np.float64) + delta).astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("Encoded weights contain non-finite values")
    return encoded


def decode_inert_bits(encoded_weights: np.ndarray, *, n_bits: int, master_seed: int, repeat: int, row_index: int) -> np.ndarray:
    """Blind-to-cover decoder; needs key and row ID, NOT clean original weights."""
    w2 = np.asarray(encoded_weights)
    if w2.ndim != 1 or w2.dtype != np.float32 or not np.isfinite(w2).all():
        raise ValueError("Expected a finite 1D float32 encoded vector")
    codes = spreading_codes(n_weights=w2.size, n_bits=n_bits, master_seed=master_seed, repeat=repeat, row_index=row_index)
    return (codes.T @ w2.astype(np.float64) >= 0.0).astype(np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="E2 TRAIN-ONLY proof-of-concept; no detector training")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--split-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-models", default=8, type=int)
    p.add_argument("--bits", default=16, type=int)
    p.add_argument("--alphas", default="0,0.01,0.03,0.10,0.32", help="Comma-separated relative L2 strengths; exploratory, train-only")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--repeat", default=0, type=int)
    args = p.parse_args()

    if not 1 <= args.num_models <= 20:
        p.error("Safety gate: exploratory smoke uses at most 20 train IDs")
    if args.bits not in (8, 16, 32):
        p.error("Choose message lengths 8, 16, or 32 bits")
    try:
        alphas = [float(x.strip()) for x in args.alphas.split(",")]
    except ValueError:
        p.error("Invalid --alphas")
    if not alphas or len(alphas) > 8 or len(alphas) != len(set(alphas)) or any(not np.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        p.error("Expected 1-8 unique relative strengths, all between 0 and 1")

    wp = args.weights.expanduser().resolve(strict=True)
    sd = args.split_dir.expanduser().resolve(strict=True)
    protocol = json.loads((sd / "protocol.json").read_text(encoding="utf-8"))
    csvp = sd / "splits.csv"
    if sha256_file(wp) != protocol["weights_sha256"]:
        raise SystemExit("STOP: weights SHA-256 does not match frozen E1 protocol")
    if sha256_file(csvp) != protocol["splits_sha256"]:
        raise SystemExit("STOP: split SHA-256 does not match frozen E1 protocol")
    weights = np.load(wp, mmap_mode="r", allow_pickle=False)
    if weights.shape != (1000, 2864) or weights.dtype != np.float32:
        raise SystemExit(f"STOP: unexpected weights {weights.shape}/{weights.dtype}")
    with csvp.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    if len(split_rows) != 1000 or set(r["split"] for r in split_rows) != {"train", "val", "test"}:
        raise SystemExit("STOP: invalid split table")
    # Identity-disjoint, TRAIN ONLY. NEVER read validation or test weight vectors here.
    train_ids = np.array(sorted(int(r["row_index"]) for r in split_rows if r["split"] == "train"), dtype=int)
    if train_ids.size != 600 or any(i < 0 or i >= 1000 for i in train_ids):
        raise SystemExit("STOP: unexpected train pool")
    rng = np.random.Generator(np.random.PCG64(args.seed + args.repeat))
    ids = np.sort(rng.choice(train_ids, size=args.num_models, replace=False))
    out = args.out_dir.expanduser().resolve()
    if (out / "smoke_per_model.csv").exists() or (out / "smoke_summary.csv").exists():
        raise SystemExit("STOP: output exists, choose a NEW directory (never overwrite smoke results)")
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for rid in ids:
        original = np.asarray(weights[int(rid)], dtype=np.float32).copy()
        if not np.isfinite(original).all() or not np.linalg.norm(original):
            raise SystemExit(f"STOP: non-finite/zero original at train row {rid}")
        bits = synthetic_message(n_bits=args.bits, master_seed=args.seed, repeat=args.repeat, row_index=int(rid))
        for alpha in alphas:
            encoded = embed_inert_bits(original, bits, alpha=alpha, master_seed=args.seed, repeat=args.repeat, row_index=int(rid))
            found = decode_inert_bits(encoded, n_bits=args.bits, master_seed=args.seed, repeat=args.repeat, row_index=int(rid))
            delta = encoded.astype(np.float64) - original.astype(np.float64)
            errors = int(np.count_nonzero(bits != found))
            records.append({
                "source_split": "train", "row_index": int(rid), "repeat": args.repeat,
                "seed": args.seed, "n_bits": args.bits, "alpha": alpha,
                "bit_errors": errors, "ber": errors / args.bits,
                "exact_recovery": int(errors == 0),
                "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(original.astype(np.float64))),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "changed_weights": int(np.count_nonzero(delta)),
                "all_finite": bool(np.isfinite(encoded).all()),
                "message_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
            })
    with (out / "smoke_per_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (out / "smoke_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "n_models", "mean_ber", "exact_recovery_rate", "mean_relative_l2"])
        writer.writeheader()
        for alpha in alphas:
            rs = [r for r in records if r["alpha"] == alpha]
            writer.writerow({
                "alpha": alpha, "n_models": len(rs),
                "mean_ber": float(np.mean([r["ber"] for r in rs])),
                "exact_recovery_rate": float(np.mean([r["exact_recovery"] for r in rs])),
                "mean_relative_l2": float(np.mean([r["relative_l2"] for r in rs])),
            })
    (out / "smoke_config.json").write_text(json.dumps({
        "purpose": "E2 GATE E2-0 only; preliminary/train only; NO OOD test/detector",
        "algorithm": ALGORITHM, "no_real_malware": True, "no_checkpoint_written": True,
        "seed": args.seed, "repeat": args.repeat, "n_bits": args.bits,
        "n_models": args.num_models, "alphas": alphas,
        "weights_sha256": protocol["weights_sha256"], "splits_sha256": protocol["splits_sha256"],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"E2 TRAIN-ONLY SMOKE OK: {len(ids)} IDs × {len(alphas)} strengths; output={out}")
    for alpha in alphas:
        rs = [r for r in records if r["alpha"] == alpha]
        print(f"  alpha={alpha:0.4f} BER={np.mean([r['ber'] for r in rs]):.4f} "
              f"exact={np.mean([r['exact_recovery'] for r in rs]):.3f} "
              f"rel_L2={np.mean([r['relative_l2'] for r in rs]):.5f}")
    print("NOT a validation or test result. Not equivalent to original MaleficNet.")


if __name__ == "__main__":
    main()
