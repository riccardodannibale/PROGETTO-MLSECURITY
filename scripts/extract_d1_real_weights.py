#!/usr/bin/env python3
"""Estrarre un checkpoint finale per modello da un GHRP Small CNN Zoo.

Il file dataset.pt presente nel modello zoo contiene IMMAGINI di training,
non i pesi delle CNN. I pesi sono i file NN_tune_trainable_*/checkpoint_N/checkpoints.

Produce weights.npy (righe = modelli distinti), checkpoint_manifest.csv e
extraction_summary.json; non modifica i dati di origine né il submodule.

Carica soltanto state dict di tensori con torch.load(weights_only=True).
Usare esclusivamente checkpoint di provenienza verificata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from collections.abc import Mapping

import numpy as np
import torch

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_checkpoint(model_dir: Path) -> tuple[int, Path]:
    options = []
    for item in model_dir.iterdir():
        if not item.is_dir():
            continue
        match = CHECKPOINT_RE.fullmatch(item.name)
        if match and (item / "checkpoints").is_file():
            options.append((int(match.group(1)), item / "checkpoints"))
    if not options:
        raise RuntimeError(f"Nessun file di checkpoint trovato per {model_dir}")
    return max(options, key=lambda entry: entry[0])


def flatten_checkpoint(path: Path, *, expected_signature=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"Atteso state dict non vuoto, trovato {type(state).__name__}: {path}")

    signature = []
    values = []
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Chiave o valore non valido nel checkpoint: {path}")
        if value.dtype != torch.float32:
            raise TypeError(f"Parametro {name} di tipo {value.dtype}, atteso float32: {path}")
        signature.append((name, tuple(value.shape)))
        values.append(value.detach().cpu().contiguous().numpy().ravel())

    signature = tuple(signature)
    if expected_signature is not None and signature != expected_signature:
        raise ValueError(
            f"Architettura / ordine dei parametri diverso in {path}. "
            "Non concatenare CNN con architetture differenti nello stesso weights.npy."
        )
    return np.concatenate(values).astype(np.float32, copy=False), signature


def atomic_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".summary_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Directory estratta tune_zoo_stl_small_uniform/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory nuova, distinta dai dataset sorgente")
    parser.add_argument("--expected-models", type=int, default=None,
                        help="Se specificato, interrompe se il numero di modelli è diverso")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sovrascrive file precedentemente generati nella directory di output")
    args = parser.parse_args()

    root = args.input_root.expanduser().resolve(strict=True)
    out = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error("--input-root deve essere una directory")
    if out == root or root in out.parents:
        parser.error("--output-dir deve essere esterna alla directory del Model Zoo originale")

    model_dirs = sorted(p for p in root.iterdir()
                        if p.is_dir() and p.name.startswith("NN_tune_trainable_"))
    if not model_dirs:
        parser.error(f"Nessuna directory NN_tune_trainable_* trovata in {root}")
    if args.expected_models is not None and len(model_dirs) != args.expected_models:
        parser.error(f"Trovati {len(model_dirs)} modelli, attesi {args.expected_models}")

    selected = [(p, *selected_checkpoint(p)) for p in model_dirs]
    out.mkdir(parents=True, exist_ok=True)
    npy = out / "weights.npy"
    csv_path = out / "checkpoint_manifest.csv"
    json_path = out / "extraction_summary.json"
    for dest in (npy, csv_path, json_path):
        if dest.exists() and not args.overwrite:
            parser.error(f"Output già esistente: {dest} (usa --overwrite se intenzionale)")

    first, signature = flatten_checkpoint(selected[0][2])
    n_models, n_params = len(selected), first.size
    print(f"Origine: {root}", flush=True)
    print(f"Modelli distinti: {n_models}; checkpoint finale per modello: 1", flush=True)
    print(f"Pesi per modello: {n_params}; numero tensori: {len(signature)}", flush=True)
    print(f"Output previsto: {n_models} x {n_params} float32 "
          f"({n_models * n_params * 4 / 1024**2:.2f} MiB)", flush=True)

    fd, temp_npy = tempfile.mkstemp(prefix=".weights_", suffix=".npy", dir=out)
    os.close(fd)
    fd, temp_csv = tempfile.mkstemp(prefix=".manifest_", suffix=".csv", dir=out)
    os.close(fd)
    hashes = set()
    duplicate_hashes = 0
    try:
        mm = np.lib.format.open_memmap(temp_npy, mode="w+", dtype=np.float32,
                                       shape=(n_models, n_params))
        fieldnames = ["row_index", "model_id", "checkpoint_number", "checkpoint_relpath", "checkpoint_sha256"]
        with open(temp_csv, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for index, (model_dir, number, ckpt) in enumerate(selected):
                vector, _ = (first, signature) if index == 0 else flatten_checkpoint(
                    ckpt, expected_signature=signature)
                if vector.shape != (n_params,) or not np.isfinite(vector).all():
                    raise ValueError(f"Pesi invalidi al modello {model_dir.name}: "
                                     f"shape={vector.shape}")
                mm[index] = vector
                digest = sha256(ckpt)
                duplicate_hashes += digest in hashes
                hashes.add(digest)
                writer.writerow({
                    "row_index": index,
                    "model_id": model_dir.name,
                    "checkpoint_number": number,
                    "checkpoint_relpath": str(ckpt.relative_to(root)),
                    "checkpoint_sha256": digest,
                })
                if (index + 1) % 100 == 0 or index + 1 == n_models:
                    print(f"Estratti {index + 1}/{n_models} modelli", flush=True)
        mm.flush()
        del mm
        summary = {
            "dataset_type": "ghrp_small_cnn_model_weights",
            "source_root": str(root),
            "checkpoint_policy": "one_last_numeric_checkpoint_per_model_directory",
            "n_models": n_models,
            "n_parameters_per_model": n_params,
            "dtype": "float32",
            "n_checkpoint_sha256_duplicates": duplicate_hashes,
            "parameter_signature": [
                {"name": name, "shape": list(shape)} for name, shape in signature
            ],
            "weights_sha256": sha256(Path(temp_npy)),
        }
        if duplicate_hashes:
            print(f"ATTENZIONE: {duplicate_hashes} checkpoint con hash duplicato", flush=True)
        os.replace(temp_npy, npy)
        os.replace(temp_csv, csv_path)
        atomic_json(json_path, summary)
        print(f"Salvato: {npy}", flush=True)
        print(f"Salvato: {csv_path}", flush=True)
        print(f"Salvato: {json_path}", flush=True)
        print("Estrazione checkpoint reali: OK", flush=True)
        return 0
    finally:
        for tmp in (temp_npy, temp_csv):
            if os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        sys.exit(1)
