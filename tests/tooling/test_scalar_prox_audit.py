"""Deterministic gate for the complete R2 certificate audit."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_complete_scalar_prox_certificate_audit() -> None:
    env = os.environ.copy()
    source = str(REPO_ROOT / 'src')
    env['PYTHONPATH'] = os.pathsep.join(
        part for part in (source, env.get('PYTHONPATH', '')) if part
    )
    completed = subprocess.run(
        [sys.executable, 'tools/check_scalar_prox_audit.py'],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert 'algebraic_generated=300 successes=300' in output
    assert 'exponential_generated=61 successes=61' in output
    assert output.count('structured_failures=0 violations=0') == 2
    assert 'derivative_enclosures=600' in output
    assert 'derivative_enclosures=122' in output
    assert 'objective_differences=300' in output
    assert 'objective_differences=61' in output
    assert 'endpoint_selections=300' in output
    assert 'endpoint_selections=61' in output
