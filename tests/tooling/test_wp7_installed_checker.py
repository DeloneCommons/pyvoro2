"""The installed gate distinguishes ghost qualification from refusal."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'wp7_installed_checker', ROOT / 'tools' / 'check_installed_package.py',
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_installed_checker_exposes_independent_ghost_refusal_flag():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools' / 'check_installed_package.py'),
         '--help'], capture_output=True, text=True, check=True,
    )
    assert '--ghost-refusal' in result.stdout
    assert '--planar-refusal' in result.stdout


def test_installed_inventory_contains_wp7_semantic_and_dimension_helpers():
    assert {'pyvoro2._internal.ghost',
            'pyvoro2._internal.spatial.ghost_certificate',
            'pyvoro2._internal.planar.ghost_certificate'} <= set(
                CHECKER.INTERNAL_HELPER_MODULES)


@pytest.mark.parametrize('dimension, mask, shift', (
    (2, (True, False), (1, 0)),
    (3, (False, True, False), (0, -1, 0)),
))
def test_installed_boundary_guard_keeps_ghost_self_distinct_from_wall(
        dimension, mask, shift):
    boundary = {'boundary_reference': {
        'kind': 'ghost_self', 'generator_id': None,
        'shift': shift, 'wall_id': None,
    }}
    assert CHECKER._check_ghost_reference(boundary, dimension, mask) == 'ghost_self'
    for invalid in ({**boundary, 'adjacent_cell': -1},
                    {'boundary_reference': dict(boundary['boundary_reference'],
                                                shift=(0,) * dimension)}):
        with pytest.raises(CHECKER.InstalledPackageCheckError):
            CHECKER._check_ghost_reference(invalid, dimension, mask)


def test_installed_ghost_call_uses_boundary_selector_without_public_vertices():
    import pyvoro2._core as core3d
    import pyvoro2._core2d as core2d

    from pyvoro2._internal.planar.wp6_profile import SOURCE_SHA256
    from pyvoro2._internal.spatial.ghost_certificate import (
        _QUALIFIED_GHOST_SOURCE_SHA256,
    )

    supported2d = (core2d._planar_witness_profile()['source_sha256'] ==
                   SOURCE_SHA256 and core2d._planar_witness_profile()['qualified'])
    # A stale extension is a build failure, never a platform refusal.
    assert core2d._planar_witness_profile()['source_sha256'] == SOURCE_SHA256
    packet, = core3d._observe_ghost_box(
        np.empty((0, 3)), np.empty((0,), dtype=np.int32),
        ((0., 1.),) * 3, (1, 1, 1), (False,) * 3, 1,
        np.array([[.5, .5, .5]]),
    )
    assert packet['build']['ghost_source_sha256'] == _QUALIFIED_GHOST_SOURCE_SHA256
    if supported2d:
        CHECKER._check_ghost_workflows(refusal=False)
        with pytest.raises(CHECKER.InstalledPackageCheckError):
            CHECKER._check_ghost_workflows(refusal=True)
    else:
        CHECKER._check_ghost_workflows(refusal=True)
