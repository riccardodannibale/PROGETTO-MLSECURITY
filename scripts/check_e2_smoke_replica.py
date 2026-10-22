#!/usr/bin/env python3
"""Read-only comparison of Ubuntu and Mac E2 train-only smoke artifacts.

No third-party dependencies. Read-only. Never accesses E2 val/test weights.
Exit 0 iff configs, rows, message hashes, recovery and summary metrics agree.
Tiny platform-dependent float differences are tolerated at explicit thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CONFIG_KEYS = (
    "algorithm", "seed", "repeat", "n_bits", "n_models", "alphas",
    "weights_sha256", "splits_sha256", "code_sha256", "no_real_malware",
)
ROW_KEYS = ("row_index", "repeat", "seed", "n_bits", "alpha")
EXACT_FIELDS = (
    "source_split", "bit_errors", "ber", "exact_recovery", "all_finite",
    "message_sha256",
)
APPROX_FIELDS = ("relative_l2", "max_abs_delta")
SUMMARY_COLS = ("mean_ber", "exact_recovery_rate", "mean_relative_l2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_dir(folder: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    config = json.loads((folder / "smoke_config.json").read_text(encoding="utf-8"))
    return config, read_csv(folder / "smoke_per_model.csv"), read_csv(folder / "smoke_summary.csv")


def key(row: dict[str, str]) -> tuple:
    return tuple(int(row[c]) if c in ("row_index", "repeat", "seed", "n_bits") else float(row[c]) for c in ROW_KEYS)


def indexed_rows(rows: list[dict[str, str]]) -> dict[tuple, dict]:
    data = {}
    for row in rows:
        k = key(row)
        if k in data:
            raise ValueError(f"Duplicate per-model key: {k}")
        if row["source_split"] != "train":
            raise ValueError(f"Unexpected split for {k}: {row['source_split']}")
        data[k] = row
    return data


def indexed_summary(rows: list[dict[str, str]]) -> dict[float, dict]:
    data = {}
    for row in rows:
        a = float(row["alpha"])
        if a in data:
            raise ValueError(f"Duplicate alpha in summary: {a}")
        data[a] = row
    return data


def float_same(a: str | float, b: str | float, tol: float) -> bool:
    try:
        aa, bb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(aa) and math.isfinite(bb) and math.isclose(aa, bb, abs_tol=tol, rel_tol=tol)


def audit(ubuntu_dir: Path, mac_dir: Path, tolerance: float) -> tuple[list[str], list[str]]:
    errors = []
    notes = []
    uc, ur, us = read_dir(ubuntu_dir)
    mc, mr, ms = read_dir(mac_dir)
    for field in CONFIG_KEYS:
        if field not in uc or field not in mc or uc[field] != mc[field]:
            errors.append(f"CONFIG mismatch {field}: Ubuntu={uc.get(field)!r}, Mac={mc.get(field)!r}")
    urows, mrows = indexed_rows(ur), indexed_rows(mr)
    if set(urows) != set(mrows):
        missing = sorted(set(urows) - set(mrows))
        extra = sorted(set(mrows) - set(urows))
        errors.append(f"PER-MODEL keys differ: absent-on-Mac={missing[:4]} extra-on-Mac={extra[:4]}")
    for k in sorted(set(urows) & set(mrows)):
        u, m = urows[k], mrows[k]
        for field in EXACT_FIELDS:
            if field == "ber":
                if not float_same(u[field], m[field], 1e-12):
                    errors.append(f"row={k} {field}: {u[field]} vs {m[field]}")
            elif u[field] != m[field]:
                errors.append(f"row={k} {field}: {u[field]} vs {m[field]}")
        for field in APPROX_FIELDS:
            if not float_same(u[field], m[field], tolerance):
                errors.append(f"row={k} {field}: {u[field]} vs {m[field]} (tol={tolerance})")
        if u.get("changed_weights") != m.get("changed_weights"):
            notes.append(f"row={k} changed_weights: Ubuntu {u.get('changed_weights')} vs Mac {m.get('changed_weights')}")
    u_sum, m_sum = indexed_summary(us), indexed_summary(ms)
    if set(u_sum) != set(m_sum):
        errors.append(f"SUMMARY alpha sets differ: {set(u_sum)} vs {set(m_sum)}")
    for alpha in sorted(set(u_sum) & set(m_sum)):
        a, b = u_sum[alpha], m_sum[alpha]
        if a.get("n_models") != b.get("n_models"):
            errors.append(f"SUMMARY alpha={alpha} n_models mismatch")
        for field in SUMMARY_COLS:
            if not float_same(a[field], b[field], tolerance):
                errors.append(f"SUMMARY alpha={alpha} {field}: {a[field]} vs {b[field]}")
        # Verify Ubuntu and Mac summaries against their own raw rows, not only against each other.
        for side_name, by_row, summary in (("Ubuntu", urows, a), ("Mac", mrows, b)):
            eligible = [r for k, r in by_row.items() if k[-1] == alpha]
            if len(eligible) != int(summary["n_models"]):
                errors.append(f"{side_name} SUMMARY alpha={alpha} n_models != raw-count")
            elif eligible:
                quantities = {
                    "mean_ber": sum(float(r["ber"]) for r in eligible) / len(eligible),
                    "exact_recovery_rate": sum(float(r["exact_recovery"]) for r in eligible) / len(eligible),
                    "mean_relative_l2": sum(float(r["relative_l2"]) for r in eligible) / len(eligible),
                }
                for field, computed in quantities.items():
                    if not float_same(computed, summary[field], tolerance):
                        errors.append(f"{side_name} SUMMARY alpha={alpha} {field} != raw mean")
    expected = int(uc.get("n_models", 0)) * len(uc.get("alphas", []))
    if expected != len(urows) or expected != len(mrows):
        errors.append(f"Unexpected record count: expected={expected}, Ubuntu={len(urows)}, Mac={len(mrows)}")
    return errors, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ubuntu-dir", type=Path, required=True)
    p.add_argument("--mac-dir", type=Path, required=True)
    p.add_argument("--float-tolerance", type=float, default=1e-6)
    a = p.parse_args()
    if not math.isfinite(a.float_tolerance) or a.float_tolerance < 0 or a.float_tolerance > 1e-3:
        p.error("float-tolerance must be finite and between 0 and 1e-3")
    try:
        errors, notes = audit(a.ubuntu_dir.expanduser(), a.mac_dir.expanduser(), a.float_tolerance)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        p.exit(2, f"E2 REPLICA AUDIT ERROR: {exc}\n")
    if errors:
        print(f"E2 REPLICA AUDIT FAILED: {len(errors)} issues")
        for item in errors[:35]:
            print("  -", item)
        if len(errors) > 35:
            print(f"  ...and {len(errors) - 35} more")
        raise SystemExit(1)
    print("E2 REPLICA AUDIT OK: configs, all per-model keys, bit recovery, summaries and floats aligned")
    if notes:
        print(f"NOTE: {len(notes)} minor changed_weights count differences (inspect if scientifically relevant)")
        for item in notes[:5]:
            print("  -", item)


if __name__ == "__main__":
    main()
