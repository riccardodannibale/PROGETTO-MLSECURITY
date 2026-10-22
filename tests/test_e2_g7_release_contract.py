"""Non-test regression checks for release guard. Does not unblind anything."""
from pathlib import Path
import ast
import json
import sys
from tempfile import TemporaryDirectory

root=Path(__file__).resolve().parents[1]
script=root/'scripts/e2_g7_prepare_test_release.py'
tree=ast.parse(script.read_text())
assert any(isinstance(node,ast.FunctionDef) and node.name=='prepare' for node in tree.body)
assert any(isinstance(node,ast.FunctionDef) and node.name=='committed_exact' for node in tree.body)
run=(root/'scripts/run_e2_paired_v0.py').read_text()
assert 'check_test_unlock' in run
assert 'approved_for_e2_test' in run
assert 'frozen_protocol_sha256' in run
assert '--unblind-test' in run
notice=(root/'docs/E2_G7_TEST_RELEASE_NOTICE.md').read_text()
assert 'APPROVAL: E2 TEST AUTHORIZED' in notice
assert '10 nuovi fit' in notice
print('E2 G7 RELEASE CONTRACT SELF-TEST OK: locked runner preserved, separate release mechanism')
