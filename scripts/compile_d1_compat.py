#!/usr/bin/env python3
"""Convert a GHRP TensorDataset zoo into Model X-Ray's weights.npy cache.

This is a compatibility pilot, not evidence that the source partitions contain
independent checkpoints or that their semantics match the paper's exact D1.

Usage from project root (with MODELXRAY_GHRP_DIR set):
    ~/mlsec-local/venvs/modelxray/bin/python scripts/compile_d1_compat.py

By default, loads with weights_only=True and explicitly allowlists TensorDataset.
If the trusted Zenodo file contains further unsupported objects, inspect the
failure before opting into --trusted-pickle (unsafe on untrusted files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import TensorDataset


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mz-name', default='stl10', help='Name of zoo subdirectory (default: stl10)')
    parser.add_argument('--force', action='store_true', help='Replace existing weights.npy')
    parser.add_argument('--trusted-pickle', action='store_true',
                        help='Permit unsafe pickle loading ONLY for a provenance-verified dataset')
    args = parser.parse_args()

    base_str = os.environ.get('MODELXRAY_GHRP_DIR')
    if not base_str:
        parser.error('Set MODELXRAY_GHRP_DIR to the parent of your GHRP zoo directories')
    base = Path(base_str).expanduser().resolve()
    zoo_dir = (base / args.mz_name).resolve()
    if not zoo_dir.is_relative_to(base):
        parser.error('mz-name must resolve inside MODELXRAY_GHRP_DIR')

    source = zoo_dir / 'dataset.pt'
    output = zoo_dir / 'weights.npy'
    provenance = zoo_dir / 'weights_compat_manifest.json'
    if not source.is_file():
        parser.error(f'Input not found: {source}')
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; use --force only if you intend to replace it')

    print(f'Source: {source}', flush=True)
    if args.trusted_pickle:
        print('WARNING: loading with weights_only=False; only do this for a verified, trusted source.', flush=True)
        data = torch.load(source, map_location='cpu', weights_only=False)
    else:
        try:
            # PyTorch 2.4 provides add_safe_globals but not safe_globals.
            # The allowlist affects only this short-lived converter process.
            torch.serialization.add_safe_globals([TensorDataset])
            data = torch.load(source, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise SystemExit(
                f'Safer load failed: {exc}\n'
                'Inspect the unsupported types. Only for a source whose origin and checksum '
                'you verified, you may rerun with --trusted-pickle.'
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(f'Expected a dict, got {type(data).__name__}')

    parts = []
    descriptions = []
    for split in ('trainset', 'testset', 'valset'):
        ds = data.get(split)
        if ds is None:
            if split == 'valset':
                continue
            raise ValueError(f'Missing required partition: {split}')
        if not isinstance(ds, TensorDataset) or not ds.tensors:
            raise TypeError(f'{split}: expected TensorDataset with tensors, got {type(ds).__name__}')
        weights = ds.tensors[0]
        if weights.ndim < 2 or not weights.is_floating_point():
            raise TypeError(f'{split}: unexpected weight tensor shape/dtype {tuple(weights.shape)} / {weights.dtype}')
        n, per_sample = weights.shape[0], weights[0].numel()
        if len(ds.tensors) > 1 and ds.tensors[1].shape[0] != n:
            raise ValueError(f'{split}: label count does not match sample count')
        descriptions.append({'split': split, 'samples': n,
                             'tensor_shape': list(weights.shape), 'dtype': str(weights.dtype),
                             'num_values_per_sample': per_sample})
        parts.append((split, weights))
        print(f'{split}: {n} samples x {per_sample} values; shape={tuple(weights.shape)}', flush=True)

    lengths = {tensor[0].numel() for _, tensor in parts}
    if len(lengths) != 1:
        raise ValueError(f'Incompatible flattened sizes across splits: {lengths}')
    n_weights = lengths.pop()
    total = sum(tensor.shape[0] for _, tensor in parts)
    estimated_mib = total * n_weights * np.dtype('float32').itemsize / 1024**2
    print(f'Output: {total} rows x {n_weights} float32 values, {estimated_mib:.1f} MiB', flush=True)

    # Atomic write: do not leave a partially written weights.npy after interruption.
    with tempfile.NamedTemporaryFile(dir=zoo_dir, prefix='.weights-', suffix='.npy', delete=False) as f:
        temp_path = Path(f.name)
    try:
        mm = np.lib.format.open_memmap(temp_path, mode='w+', dtype=np.float32,
                                       shape=(total, n_weights))
        offset = 0
        for split, tensor in parts:
            rows = tensor.shape[0]
            # The first TensorDataset tensor contains the numeric GHRP representation.
            flat = tensor.detach().cpu().reshape(rows, n_weights)
            mm[offset:offset + rows, :] = flat.numpy()
            offset += rows
            print(f'Wrote {split} ({rows} rows)', flush=True)
        mm.flush()
        del mm
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    manifest = {
        'status': 'compatibility_pilot_unvalidated_semantics',
        'source': str(source),
        'source_sha256': sha256_file(source),
        'output': str(output),
        'output_sha256': sha256_file(output),
        'shape': [total, n_weights],
        'dtype': 'float32',
        'splits_in_concatenation_order': descriptions,
        'warning': 'GHRP train/test/val sets were concatenated. Labels were not included. '
                   'Do not use these original splits as independent evaluation evidence '
                   'without documenting model/checkpoint provenance and leakage checks.'
    }
    provenance.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {output}', flush=True)
    print(f'Wrote {provenance}', flush=True)


if __name__ == '__main__':
    main()
