"""The fixed issuer suite cannot quietly run only selected numerical checks."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'tools/native/qualification/route_suite.py'


@pytest.mark.parametrize('filtered', [False, True])
def test_route_collection_requires_every_requested_test(tmp_path, filtered):
    tests = tmp_path / 'test_cases.py'
    tests.write_text('def test_first():\n    pass\ndef test_second():\n    pass\n')
    script = '''
import importlib.util
import sys
import pytest
spec = importlib.util.spec_from_file_location('fixed_routes', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
observed = module.TestResults()
code = pytest.main(['-q', sys.argv[2], *sys.argv[3:]], plugins=[observed])
assert code == 0
try:
    observed.require_complete()
except RuntimeError as error:
    assert sys.argv[3:] and 'deselected' in str(error)
else:
    assert not sys.argv[3:], 'filtered evidence was accepted'
'''
    command = [sys.executable, '-c', script, str(RUNNER), str(tests)]
    if filtered:
        command += ['-k', 'first']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('runner', [
    'qualification/route_suite.py', 'check_planar_witness.py',
    'check_ghost_planar.py',
])
def test_qualification_corpus_runners_reject_disabled_assertions(runner):
    result = subprocess.run(
        [sys.executable, '-O', str(ROOT / 'tools/native' / runner), '--help'],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'assertions' in result.stderr
