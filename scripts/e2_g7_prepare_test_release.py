#!/usr/bin/env python3
"""Release local E2 paired test only after exact validation QA and committed freeze.

Creates one local, auditable approval JSON. No training or test data access.
Validation artifacts remain untouched. Both collaborators approved this stage;
recorded messages are attestations, not cryptographic signatures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import e2_g6_audit_validation_v2 as qa
import run_e2_paired_v0 as runner

RUNNER_SHA = '4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac'
RELEASE_NOTICE = ROOT / 'docs/E2_G7_TEST_RELEASE_NOTICE.md'
FROZEN = ROOT / 'docs/E2_PROTOCOL_FROZEN.md'
SPLIT = ROOT / 'manifests/e1_uniform_v1/splits.csv'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def committed_exact(relative: str):
    path = ROOT / relative
    if not path.is_file() or subprocess.check_output(['git', '-C', str(ROOT), 'show', f'HEAD:{relative}']) != path.read_bytes():
        raise RuntimeError(f'STOP: file not committed exactly: {relative}')


def prepare(platform: str, validation: Path, output: Path):
    if platform not in ('ubuntu', 'mac_m4'):
        raise RuntimeError('STOP: unsupported platform')
    if not validation.is_dir():
        raise RuntimeError('STOP: missing local validation directory')
    if output.exists():
        raise RuntimeError('STOP: approval output exists; no overwrites')
    if git('branch', '--show-current') != 'ricky':
        raise RuntimeError('STOP: not on ricky branch')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('STOP: tracked working tree is dirty')
    for f in ('scripts/run_e2_paired_v0.py', 'scripts/e2_g6_audit_validation_v2.py',
              'docs/E2_PROTOCOL_FROZEN.md', 'docs/E2_G7_TEST_RELEASE_NOTICE.md'):
        committed_exact(f)
    if digest(ROOT / 'scripts/run_e2_paired_v0.py') != RUNNER_SHA:
        raise RuntimeError('STOP: E2 detector runner SHA changed')
    notice = RELEASE_NOTICE.read_text(encoding='utf-8')
    if 'APPROVAL: E2 TEST AUTHORIZED' not in notice or '22 ottobre 2026' not in notice:
        raise RuntimeError('STOP: release notice has not been formally recorded')
    frozen_sha = digest(FROZEN)
    print('=== E2 local validation audit before test release ===', flush=True)
    qa.audit(validation, SPLIT)
    print('Validation QA passed. Generating local test-release authorization.', flush=True)
    data = {
        'state': 'approved_for_e2_test',
        'validation_count': 10,
        'runner_sha256': RUNNER_SHA,
        'frozen_protocol_sha256': frozen_sha,
        'riccardo_approved': True,
        'fabrizio_approved': True,
        'platform': platform,
        'validation_dir': str(validation),
        'validation_run_config_sha256': digest(validation / 'run_config.json'),
        'release_notice_sha256': digest(RELEASE_NOTICE),
        'release_commit': git('rev-parse', 'HEAD'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'approval_evidence': 'Riccardo general authorization and Fabrizio explicit go-ahead relayed on 2026-10-22; neither is a digital signature.',
        'scope': '10 pre-registered E2 paired ID/OOD test fits; no detector/threshold/alpha tuning',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents an accidental re-issue or replacement of the original approval.
    with output.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')
    # Verify same contract as the untouched G5 runner BEFORE test vector access.
    runner.check_test_unlock(approval=output, validation_dir=validation,
                             runner_path=ROOT / 'scripts/run_e2_paired_v0.py',
                             expected_runner_sha=RUNNER_SHA)
    print(f'E2 TEST RELEASE OK: {platform} / validation 10/10 / original runner locked SHA')
    print('Release JSON:', output)
    print('TEST NOT EXECUTED: run launch script separately.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--platform', required=True, choices=['ubuntu', 'mac_m4'])
    p.add_argument('--validation-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    prepare(args.platform, args.validation_dir.expanduser().resolve(), args.out.expanduser().resolve())
