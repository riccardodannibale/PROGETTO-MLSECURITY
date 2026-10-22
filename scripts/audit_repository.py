#!/usr/bin/env python3
"""Nondestructive local Git inventory for an E1–E3 portfolio release.

Never calls git add/rm/reset/clean, never writes any files. No remote network.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

CORE = (
    'README.md', 'REPRODUCING.md', 'exam/PROJECT_REPORT.md',
    'exam/PROJECT_REPORT.docx', 'exam/E3_REPORT.md',
    'scripts/run_e1_final_v2.py', 'scripts/run_e2_paired_v0.py',
    'scripts/e2_synthetic_spread.py', 'scripts/e3_matched_controls.py',
    'manifests/e1_uniform_v1/splits.csv',
    'docs/E2_PROTOCOL_FROZEN.md',
    'results/e2_final/e2_cross_platform_statistics.csv',
    'results/e3_final/cross_platform/e3_cross_platform_summary.csv',
)


def cmd(root: Path, *args: str) -> tuple[bool, str]:
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout if p.returncode == 0 else p.stderr).strip()


def run(root: Path) -> int:
    ok, inside = cmd(root, 'rev-parse', '--show-toplevel')
    if not ok:
        print('ERROR: not a Git working tree:', inside, file=sys.stderr)
        return 2
    print('=== READ-ONLY GIT INVENTORY ===')
    print('Repository:', inside)
    for label, args in [
        ('Current branch', ('branch', '--show-current')),
        ('Current HEAD', ('log', '-1', '--oneline')),
        ('Remote HEAD (last fetched)', ('symbolic-ref', '-q', 'refs/remotes/origin/HEAD')),
        ('Local main', ('log', '-1', '--oneline', 'main')),
        ('Local ricky', ('log', '-1', '--oneline', 'ricky')),
        ('Remote-tracking origin/main', ('log', '-1', '--oneline', 'origin/main')),
        ('Remote-tracking origin/ricky', ('log', '-1', '--oneline', 'origin/ricky')),
    ]:
        passed, output = cmd(root, *args)
        print(f'{label}: {output if passed else "NOT AVAILABLE (fetch first)"}')
    passed, status = cmd(root, 'status', '--short')
    print('\nWorking-tree changes (including untracked):')
    print(status if passed and status else '(none)' if passed else 'unavailable')
    print('\nExpected release paths:')
    for relative in CORE:
        marker = 'PRESENT' if (root / relative).exists() else 'MISSING'
        print(f'  {marker:7} {relative}')
    sub = root / 'external' / 'ModelXRay'
    passed, submodules = cmd(root, 'submodule', 'status')
    print('\nSubmodules:', submodules if passed and submodules else '(none or unavailable)')
    print('ModelXRay directory:', 'present' if sub.is_dir() else 'absent')
    passed, tracked = cmd(root, 'ls-files')
    files = [x for x in tracked.splitlines() if x] if passed else []
    root_docs = [x for x in files if '/' not in x and (x.endswith('.md') or x.endswith('.docx'))]
    papers = [x for x in files if x.lower().endswith('.pdf') and (x.startswith('docs/references/') or x.startswith('paper/'))]
    print('\nTracked root MD/DOCX files (manual review):')
    for f in root_docs:
        print(' ', f)
    if not root_docs:
        print('  (none)')
    print('\nTracked third-party-looking PDFs (license review; do not delete automatically):')
    for f in papers:
        print(' ', f)
    if not papers:
        print('  (none)')
    print('\nPotentially large tracked working-tree files (>= 10 MiB):')
    count = 0
    for f in files:
        item = root / f
        if item.is_file() and not item.is_symlink() and item.stat().st_size >= 10 * 1024 * 1024:
            print(f'  {item.stat().st_size / 1024**2:.1f} MiB: {f}')
            count += 1
    if not count:
        print('  (none detected)')
    print('\nNOTE: Git history and remote HEAD can differ from this snapshot; run `git fetch origin` separately, then rerun.')
    print('NOTE: this script performs no deletion, branch switch, network, writes or commits.')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    a = p.parse_args()
    sys.exit(run(a.root.resolve()))
