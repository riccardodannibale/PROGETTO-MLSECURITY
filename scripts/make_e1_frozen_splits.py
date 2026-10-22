#!/usr/bin/env python3
"""Freeze checkpoint-identity-disjoint E1 train/validation/test splits.

Standalone helper for the project repo (does not modify upstream Model X-Ray).
Input: weights.npy and checkpoint_manifest.csv from extract_d1_real_weights.py.
Output: immutable-by-default split CSV and protocol JSON, suitable for Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    args = p.parse_args()

    weights_path = args.weights.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()

    if min(args.n_train, args.n_val, args.n_test) <= 0:
        p.error("Train, val e test devono contenere almeno un modello")

    w = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if w.ndim != 2 or w.dtype != np.float32:
        p.error(f"Atteso weights.npy float32 2D, trovato shape={w.shape}, dtype={w.dtype}")

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"row_index", "model_id", "checkpoint_number", "checkpoint_relpath", "checkpoint_sha256"}
        if not required.issubset(reader.fieldnames or []):
            p.error(f"Manifest senza colonne richieste: {sorted(required)}")
        records = list(reader)

    n = len(records)
    if n != w.shape[0]:
        p.error(f"Manifest con {n} righe, weights.npy con {w.shape[0]}")
    if args.n_train + args.n_val + args.n_test != n:
        p.error(f"Split richiesto {args.n_train}+{args.n_val}+{args.n_test}, ma i modelli sono {n}")

    indices = [int(row["row_index"]) for row in records]
    if indices != list(range(n)):
        p.error("row_index non è consecutivo in ordine 0..N-1: controllare l'allineamento ai pesi")
    model_ids = [row["model_id"] for row in records]
    if len(set(model_ids)) != n:
        p.error("model_id duplicato: impossibile garantire split disgiunti")
    hashes = [row["checkpoint_sha256"] for row in records]
    if any(len(h) != 64 for h in hashes) or len(set(hashes)) != n:
        p.error("Hash dei checkpoint mancanti/non validi/duplicati")

    csv_dest = out_dir / "splits.csv"
    json_dest = out_dir / "protocol.json"
    if csv_dest.exists() or json_dest.exists():
        p.error(f"Split già congelato in {out_dir}: non sovrascrivo file esistenti")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    mapping = {}
    for partition, subset in (
        ("train", perm[:args.n_train]),
        ("val", perm[args.n_train: args.n_train + args.n_val]),
        ("test", perm[args.n_train + args.n_val:]),
    ):
        mapping.update((int(i), partition) for i in subset)
    assert Counter(mapping.values()) == {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    tmp_csv = out_dir / ".splits.csv.tmp"
    tmp_json = out_dir / ".protocol.json.tmp"
    try:
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[*required_order(), "split"])
            writer.writeheader()
            for row in records:
                writer.writerow({**row, "split": mapping[int(row["row_index"])]})
        metadata = {
            "protocol_id": "e1_uniform_real_v1",
            "purpose": "FSL E1 pilot-to-confirmatory protocol; freeze model origins before final runs",
            "seed": args.seed,
            "n_models": n,
            "n_parameters_per_model": int(w.shape[1]),
            "partition_sizes": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
            "partition_unit": "model_id (one final checkpoint per independent training directory)",
            "weights_sha256": file_sha256(weights_path),
            "checkpoint_manifest_sha256": file_sha256(manifest_path),
            "splits_sha256": file_sha256(tmp_csv),
            "leakage_check": "model_id and checkpoint_sha256 unique; each row assigned to exactly one split",
            "methodology_note": "Earlier pilot testing explored the same 1000-model zoo; thus this split is frozen prospectively but not a wholly untouched external test set.",
        }
        with tmp_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_csv, csv_dest)
        os.replace(tmp_json, json_dest)
    finally:
        for tmp in (tmp_csv, tmp_json):
            if tmp.exists():
                tmp.unlink()

    print(f"Split congelato: {csv_dest}")
    print(f"Protocollo: {json_dest}")
    print(f"Modelli: {n} | parametri: {w.shape[1]}")
    print(f"Train={args.n_train}, Val={args.n_val}, Test={args.n_test}")
    print("IDENTITA' DISGIUNTE: OK")
    print("NOTA: il file non viene sovrascritto automaticamente; versione da committare nel repo.")


def required_order() -> list[str]:
    return ["row_index", "model_id", "checkpoint_number", "checkpoint_relpath", "checkpoint_sha256"]


if __name__ == "__main__":
    main()
