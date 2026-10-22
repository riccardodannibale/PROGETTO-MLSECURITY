#!/usr/bin/env python3
"""Diagnostica di sola lettura del pareggio centroid E1 v2 (repeat=8, X=1).

Replica unicamente il fit incriminato, SENZA scrivere file, modificare il
runner originale, ricalibrare soglie o accedere al test riservato.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "external" / "ModelXRay"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from scripts.run_e1_final_v2 import make_pair_images, split_ids  # noqa: E402


def main():
    import json
    import tensorflow as tf
    from model_xray.fsl.train import train_fsl

    rep, x, seed = 8, 1, 42
    data_dir = Path.home() / "mlsec-local/datasets/ghrp_zoos/d1_uniform_real"
    weights_path = data_dir / "weights.npy"
    split_dir = ROOT / "manifests/e1_uniform_v1"
    proto = json.loads((split_dir / "protocol.json").read_text())
    splits = split_ids(split_dir / "splits.csv", proto, weights_path)
    w = np.load(weights_path, mmap_mode="r", allow_pickle=False)

    train_pool = splits["train"]["row_index"].to_numpy(dtype=int)
    val_ids = splits["val"]["row_index"].to_numpy(dtype=int)
    train_ids = np.random.default_rng(seed + rep).choice(
        train_pool, size=3, replace=False
    )
    fit_seed = seed + rep * 1000 + x
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    tf.keras.utils.set_random_seed(fit_seed)

    Xtr, ytr, _, _ = make_pair_images(
        w, train_ids, x=x, imsize=100, master_seed=seed, repeat=rep
    )
    Xval, yval, val_rows, _ = make_pair_images(
        w, val_ids, x=x, imsize=100, master_seed=seed, repeat=rep
    )
    print(f"DIAGNOSTICA repeat={rep}, X={x}, fit_seed={fit_seed}", flush=True)
    print(f"TRAIN source rows: {train_ids.tolist()}", flush=True)
    print(f"VAL: {len(val_ids)} identita', {len(yval)} osservazioni", flush=True)
    model = train_fsl(Xtr, ytr, model_arch="osl_siamese_cnn", imsize=100, mode="ub")

    embeds = np.asarray(model.get_embeddings(Xval))
    train_embeds = np.asarray(model.train_data["x_train_embeddings"])
    train_y = np.asarray(model.train_data["y"]).astype(int)
    np_centers = np.stack(
        [train_embeds[train_y == cls].mean(axis=0) for cls in (0, 1)]
    )
    ref_preds = np.asarray(model.inference_centroid(
        Xval, x_test_embeddings=embeds, apply_transforms="NA"
    )).astype(int)
    ref_knn = model.train_data["centroid_knn_NA"]
    ref_centers = np.asarray(ref_knn._fit_X)
    ref_classes = np.asarray(ref_knn._y)
    if not np.array_equal(ref_classes, [0, 1]):
        raise RuntimeError(f"Ordine classi dei centroidi inatteso: {ref_classes}")

    # Le due scelte di centroidi: NumPy nel runner, TF nell'upstream.
    d_np = euclidean_distances(embeds, np_centers)
    d_ref = euclidean_distances(embeds, ref_centers)
    margin_np = d_np[:, 0] - d_np[:, 1]
    margin_ref = d_ref[:, 0] - d_ref[:, 1]
    pred_np = (margin_np > 0).astype(int)
    pred_ref = (margin_ref > 0).astype(int)
    nearest_preds = np.asarray(ref_knn.predict(embeds)).astype(int)
    print("\n=== ORIGINE DELLA DISCORDANZA ===")
    print("Dtype embedding:", embeds.dtype)
    print("Dtype centri NumPy / upstream:", np_centers.dtype, ref_centers.dtype)
    print("Max differenza assoluta tra centri:",
          float(np.max(np.abs(np_centers.astype('float64') - ref_centers.astype('float64')))))
    print("Discordanze rispetto a upstream:")
    print("  margine NumPy:     ", int(np.sum(pred_np != ref_preds)), "/", len(ref_preds))
    print("  margine centri TF:", int(np.sum(pred_ref != ref_preds)), "/", len(ref_preds))
    print("  ref_knn.predict:  ", int(np.sum(nearest_preds != ref_preds)), "/", len(ref_preds))

    bad = np.flatnonzero(pred_np != ref_preds)
    closest = np.argsort(np.abs(margin_np))[:5]
    inspect = np.unique(np.r_[bad, closest])
    print("\n=== CASI DISCORDANTI E 5 MARGINI PIU' PICCOLI ===")
    print("idx,row_id,etichetta,reale,score_np,score_ref,d_np0,d_np1,d_ref0,d_ref1")
    for i in inspect:
        print(f"{i},{int(val_rows[i])},{yval[i]},{ref_preds[i]},"
              f"{margin_np[i]:.12g},{margin_ref[i]:.12g},"
              f"{d_np[i, 0]:.12g},{d_np[i, 1]:.12g},"
              f"{d_ref[i, 0]:.12g},{d_ref[i, 1]:.12g}")
    print("\nNUMERO CASI CON MARGINE ORIGINALE ESATTAMENTE ZERO:",
          int(np.sum(margin_np == 0)))
    print("\nFINE DIAGNOSTICA: nessun risultato E1 e' stato modificato.")


if __name__ == "__main__":
    main()
