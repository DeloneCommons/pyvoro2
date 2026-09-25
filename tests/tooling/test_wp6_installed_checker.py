"""The installation gate must distinguish qualification from refusal."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from pyvoro2 import _core2d


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'wp6_installed_checker', ROOT / 'tools' / 'check_installed_package.py',
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_installed_checker_exposes_explicit_planar_refusal_flag():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools' / 'check_installed_package.py'),
         '--help'], capture_output=True, text=True, check=True,
    )
    assert '--planar-refusal' in result.stdout


def test_installed_profile_gate_cannot_confuse_support_and_refusal():
    assert hasattr(CHECKER, '_check_planar_profile'), 'WP6 profile gate is absent'
    profile = _core2d._planar_witness_profile()
    unsupported = not profile['cohort_supported']
    checked = CHECKER._check_planar_profile(_core2d, refusal=unsupported)
    assert checked == profile
    with pytest.raises(CHECKER.InstalledPackageCheckError):
        CHECKER._check_planar_profile(_core2d, refusal=not unsupported)


def test_installed_pure_helper_inventory_contains_all_wp6_layers():
    expected = {f'pyvoro2._internal.planar.wp6_{name}'
                for name in ('profile', 'ideal', 'certificate', 'numerical')}
    assert expected <= set(CHECKER.INTERNAL_HELPER_MODULES)
