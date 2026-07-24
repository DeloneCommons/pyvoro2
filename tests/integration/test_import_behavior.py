"""Fresh-process package import behavior."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_top_level_import_keeps_inverse_and_native_modules_lazy() -> None:
    code = """
import json
import sys

import pyvoro2

inverse_modules = sorted(
    name
    for name in sys.modules
    if name == 'pyvoro2.inverse' or name.startswith('pyvoro2.inverse.')
)
native_modules = sorted(
    {'pyvoro2._core', 'pyvoro2._core2d'}.intersection(sys.modules)
)
print(json.dumps({
    'inverse_modules': inverse_modules,
    'native_modules': native_modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        'inverse_modules': [],
        'native_modules': [],
    }
