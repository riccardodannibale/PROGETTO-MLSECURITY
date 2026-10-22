#!/usr/bin/env python3
"""E2 G3-C: exploratory model-utility audit, strictly TRAIN checkpoint IDs.

Reads the genuine GHRP CNN checkpoint tensors and original (already preprocessed)
STL10 IMAGE valset, both with torch.load(weights_only=True). Evaluates each
same CNN on identical images: benign, upstream canonical X-LSB X23, synthetic
spread .16/.24. No training, no checkpoint writes, no VAL/TEST checkpoint IDs.
Do not interpret the image valset as the held-out detector/model checkpoint test.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "external/ModelXRay"))

import e2_g3_cnn_forward_train as g3
import e2_intensity_train_audit as g2
import e2_synthetic_spread as spread
import run_e1_final_v2 as e1

EXPECTED = {
    "scripts/run_e1_final_v2.py": "a75aa3d844bb695a6d6155fe808394025b04c480d74d1e1b856eec5ca6f9c037",
    "scripts/e2_synthetic_spread.py": "70d9e0c02fd26dffc7d69f3fbbdb70cd3219844200c625891797eb13ee544edd",
    "scripts/e2_intensity_train_audit.py": "e3bb2fb669ff3aae36f2a3412fae5e9e3a8d69f4217f9833ec37735c0500e51b",
    "scripts/e2_g3_cnn_forward_train.py": "0595e8f3b09a2bdbc8269cf63b4647db43d9ffb8acc0bd21f938c31dde37cbbe",
}
DATASET_SHA = "7afdccfe4d5418d3645e1a42f149c377cdc9088a9283cfd8288924c3a2e0b434"
UPSTREAM_COMMIT = "a880e332f556fc71faf8d3753931005d5d019339"


def assert_code_hashes():
    for filename, expected in EXPECTED.items():
        actual = g3.sha256_file(PROJECT / filename)
        if actual != expected:
            raise RuntimeError(f"STOP: unexpected source hash {filename}: {actual}")
    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT / "external/ModelXRay"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("STOP: upstream ModelXRay submodule commit mismatch")
    if e1.PAYLOAD_NBYTES != 16_384 or e1.PAYLOAD_SCHEME != "sha256-keyed-uint8-pcg64-per-model-v1":
        raise RuntimeError("STOP: E1 payload scheme mismatch")


def load_val_images_only(path: Path, *, count: int, seed: int):
    """Safely whitelist TensorDataset in 2.4 API; always reset globals."""
    if g3.sha256_file(path) != DATASET_SHA:
        raise RuntimeError("STOP: original STL10 dataset SHA-256 mismatch")
    serialization = torch.serialization
    if not all(hasattr(serialization, name) for name in
               ("add_safe_globals", "get_safe_globals", "clear_safe_globals")):
        raise RuntimeError("STOP: required safe serialization API unavailable")
    previous = list(serialization.get_safe_globals())
    try:
        serialization.add_safe_globals([TensorDataset])
        obj = torch.load(path, map_location="cpu", weights_only=True)
    finally:
        serialization.clear_safe_globals()
        serialization.add_safe_globals(previous)
    if type(obj) is not dict or "valset" not in obj or type(obj["valset"]) is not TensorDataset:
        raise RuntimeError("STOP: original dataset has no expected TensorDataset valset")
    valset = obj["valset"]
    if len(valset.tensors) != 2:
        raise RuntimeError("STOP: valset must contain images and labels")
    images, labels = valset.tensors
    if (images.ndim != 4 or tuple(images.shape[1:]) != (3,28,28) or
        images.dtype != torch.float32 or labels.dtype != torch.int64 or
        labels.ndim != 1 or images.shape[0] != labels.shape[0] or images.shape[0] < count):
        raise RuntimeError(f"STOP: valset unexpected shape/dtype: {tuple(images.shape)} / {tuple(labels.shape)}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(images), generator=g)[:count].sort().values
    x, y = images[idx].contiguous(), labels[idx].contiguous()
    if (not bool(torch.isfinite(x).all()) or float(x.min()) < -1.00001 or float(x.max()) > 1.00001 or
        not bool(((y>=0)&(y<10)).all())):
        raise RuntimeError("STOP: image normalization/labels invalid")
    if x.shape[0] != count:
        raise RuntimeError("STOP: incorrect sampled image count")
    # Return only the sampled tensors: validation/test checkpoint vectors are never read.
    return x, y, idx, len(images)


def reshaped_state_from_flat(template: dict, flat: np.ndarray) -> OrderedDict:
    """Preserve the original named-tensor order, shapes and float32 values."""
    if flat.shape != (2864,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("Expected finite 2864-dimensional float32 weights")
    out = OrderedDict()
    offset = 0
    for name, tensor in template.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("Unexpected original state_dict entry")
        n = tensor.numel()
        value = flat[offset:offset+n].reshape(tuple(tensor.shape)).copy()
        out[name] = torch.from_numpy(value)
        offset += n
    if offset != len(flat):
        raise ValueError(f"State tensors contain {offset} parameters, expected {len(flat)}")
    return out


def accuracy(net: torch.nn.Module, state: dict, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> tuple[int, int]:
    net.load_state_dict(state, strict=True)
    net.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            pred = net(x[start:start+batch_size])
            if pred.ndim != 2 or pred.shape[1] != 10 or not bool(torch.isfinite(pred).all()):
                raise RuntimeError("STOP: network produced invalid logits")
            correct += int((pred.argmax(dim=1) == y[start:start+batch_size]).sum())
    return correct, len(y)


def relative_l2(base: np.ndarray, mod: np.ndarray) -> float:
    a = base.astype(np.float64)
    return float(np.linalg.norm(mod.astype(np.float64)-a)/np.linalg.norm(a))


def run(args):
    if not 1 <= args.count <= 10 or not 16 <= args.n_images <= 512:
        raise RuntimeError("STOP: G3 train-only pilot allows 1..10 models and 16..512 images")
    if args.image_seed < 0:
        raise RuntimeError("STOP: image seed must be nonnegative")
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise RuntimeError("STOP: output directory already exists; never overwrite audits")
    assert_code_hashes()
    weights_path = args.weights.expanduser().resolve(strict=True)
    split_dir = args.split_dir.expanduser().resolve(strict=True)
    manifest_dir = args.manifest_dir.expanduser().resolve(strict=True)
    if g3.sha256_file(weights_path) != g3.WEIGHTS_SHA or g3.sha256_file(split_dir / "splits.csv") != g3.SPLITS_SHA:
        raise RuntimeError("STOP: frozen E1 hashes mismatch")
    protocol = json.loads((split_dir / "protocol.json").read_text())
    if protocol.get("weights_sha256") != g3.WEIGHTS_SHA or protocol.get("splits_sha256") != g3.SPLITS_SHA:
        raise RuntimeError("STOP: frozen E1 protocol mismatch")
    with (split_dir / "splits.csv").open(newline="", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    train = sorted(int(r["row_index"]) for r in split_rows if r["split"] == "train")
    if len(split_rows) != 1000 or len(train) != 600 or len(set(train)) != 600:
        raise RuntimeError("STOP: unexpected split table")
    ids = train[:args.count]  # same 3 used by the earlier G3 forward smoke
    info = json.loads((manifest_dir / "extraction_summary.json").read_text())
    if info.get("weights_sha256") != g3.WEIGHTS_SHA:
        raise RuntimeError("STOP: provenance summary mismatch")
    original_root = Path(info["source_root"]).resolve(strict=True)
    data_file = (original_root / "dataset.pt").resolve(strict=True)
    if not data_file.is_relative_to(original_root):
        raise RuntimeError("STOP: unexpected dataset path")
    x, y, sample_ids, val_count = load_val_images_only(data_file, count=args.n_images, seed=args.image_seed)
    with (manifest_dir / "checkpoint_manifest.csv").open(newline="",encoding="utf-8") as f:
        manifest = {int(r["row_index"]):r for r in csv.DictReader(f)}
    if len(manifest) != 1000:
        raise RuntimeError("STOP: checkpoint manifest length")
    W = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if W.shape != (1000,2864) or W.dtype != np.float32:
        raise RuntimeError("STOP: E1 weights matrix differs")
    signature = [(r["name"], tuple(r["shape"])) for r in info["parameter_signature"]]
    rows=[]
    print(f"E2 G3 model utility pilot: {len(ids)} TRAIN checkpoints, {len(y)} preprocessed image-valset items")
    for rid in ids:
        m=manifest[rid]
        model_dir=(original_root / m["model_id"]).resolve(strict=True)
        checkpoint=(original_root / m["checkpoint_relpath"]).resolve(strict=True)
        if not model_dir.is_relative_to(original_root) or not checkpoint.is_relative_to(model_dir):
            raise RuntimeError("STOP: unexpected checkpoint path")
        if g3.sha256_file(checkpoint) != m["checkpoint_sha256"]:
            raise RuntimeError(f"STOP: checkpoint hash mismatch row {rid}")
        params = json.loads((model_dir / "params.json").read_text())
        if params.get("model::type") != "CNN" or params.get("model::channels_in") != 3 or \
           "model::nlin" not in params or "model::dropout" not in params:
            raise RuntimeError(f"STOP: missing or unexpected CNN params row {rid}")
        net = g3.SmallZooCNN(activation=params["model::nlin"], dropout=float(params["model::dropout"]))
        if [(k,tuple(v.shape)) for k,v in net.state_dict().items()] != signature:
            raise RuntimeError(f"STOP: reconstructed CNN signature differs row {rid}")
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        if type(state) not in (dict, OrderedDict) or [(k,tuple(v.shape)) for k,v in state.items()] != signature:
            raise RuntimeError(f"STOP: unexpected checkpoint state row {rid}")
        flat = np.concatenate([t.detach().cpu().contiguous().numpy().reshape(-1) for t in state.values()])
        original = np.asarray(W[rid],dtype=np.float32).copy()
        if not np.array_equal(flat,original):
            raise RuntimeError(f"STOP: checkpoint/E1 weights differ row {rid}")
        variants = [("benign",original,None,None)]
        # Exact E1 payload scheme, upstream x_lsb_attack and canonical high-level path.
        payload = e1.deterministic_payload(master_seed=42,repeat=0,x=23,row_index=rid)
        lsb = g2.capture_canonical_lsb(original.copy(),payload)
        variants.append(("X_LSB_23",lsb,None,None))
        bits=spread.synthetic_message(n_bits=16,master_seed=42,repeat=0,row_index=rid)
        for alpha in (0.16, 0.24):
            modified = spread.embed_inert_bits(original,bits,alpha=alpha,master_seed=42,repeat=0,row_index=rid)
            decoded = spread.decode_inert_bits(modified,n_bits=16,master_seed=42,repeat=0,row_index=rid)
            errors = int(np.count_nonzero(bits != decoded))
            variants.append((f"SPREAD_{alpha:.2f}",modified,alpha,errors))
        by_name={}
        for name, weights, alpha, errors in variants:
            correct, total = accuracy(net, reshaped_state_from_flat(state,weights), x, y)
            item={
                "checkpoint_split":"train", "row_index":rid,"model_id":m["model_id"],
                "image_split":"valset", "n_images":total,"correct":correct,
                "accuracy": correct/total,"condition":name,"alpha":alpha if alpha is not None else "",
                "relative_l2": relative_l2(original,weights),
                "payload_bit_errors":errors if errors is not None else "",
            }
            rows.append(item);by_name[name]=item
            print(f"  TRAIN row={rid} {name:<13} accuracy={correct}/{total}={correct/total:.4f} rel_L2={item['relative_l2']:.4f}")
        if by_name["benign"]["accuracy"] < 0.01:
            print(f"  NOTE: near-zero benign accuracy at TRAIN row {rid}; check labels and preprocessing before interpretation")
    # Write only after successful pilot; file destination must not exist.
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "utility_per_model.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
    summary=[]
    for name in ("benign","X_LSB_23","SPREAD_0.16","SPREAD_0.24"):
        sub=[r for r in rows if r["condition"]==name]
        summary.append({"condition":name,"n_checkpoints":len(sub),"n_images_per_checkpoint":args.n_images,
                        "mean_accuracy":float(np.mean([r["accuracy"] for r in sub])),
                        "mean_relative_l2":float(np.mean([r["relative_l2"] for r in sub])),
                        "mean_accuracy_delta_vs_benign":float(np.mean([
                            r["accuracy"] - next(b["accuracy"] for b in rows
                                if b["row_index"]==r["row_index"] and b["condition"]=="benign") for r in sub]))})
    with (out_dir / "utility_summary.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0])); wr.writeheader();wr.writerows(summary)
    (out_dir / "audit_config.json").write_text(json.dumps({
        "scope":"G3-C exploratory model utility: TRAIN checkpoint IDs, image-valset only; no detector training",
        "original_dataset_sha256":DATASET_SHA, "e1_weights_sha256":g3.WEIGHTS_SHA,
        "e1_splits_sha256":g3.SPLITS_SHA, "upstream_commit":UPSTREAM_COMMIT,
        "source_script_hashes":EXPECTED,
        "g3_utility_script_sha256":g3.sha256_file(Path(__file__).resolve()),
        "image_sample_seed":args.image_seed,"image_sample_indices_sha256":hashlib.sha256(
            np.asarray(sample_ids.tolist(),dtype="<i8").tobytes()).hexdigest(),
        "image_valset_total_size":val_count,"image_sample_count":args.n_images,
        "checkpoint_ids":ids,"spread_seed":42,"spread_repeat":0,"spread_bits":16,
        "spread_alphas":[0.16,0.24],"lsb_x":23,
    },indent=2)+"\n",encoding="utf-8")
    print(f"E2 G3 UTILITY PILOT OK: {len(ids)} TRAIN checkpoints × 4 conditions; {len(y)} image valset items each")
    print(f"Saved output: {out_dir}")
    print("NOT E2 OOD detector outcome, not final utility benchmark; results exploratory TRAIN-only.")


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights",type=Path,required=True)
    ap.add_argument("--split-dir",type=Path,required=True)
    ap.add_argument("--manifest-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--count",type=int,default=3)
    ap.add_argument("--n-images",type=int,default=256)
    ap.add_argument("--image-seed",type=int,default=2026)
    args=ap.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"E2 G3 UTILITY FAILED: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(1)

if __name__=="__main__":
    main()
