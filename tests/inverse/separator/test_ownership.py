"""Ownership checks for the canonical separator implementation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator


PACKAGE_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'pyvoro2'
CANONICAL_ROOT = PACKAGE_ROOT / 'inverse' / 'separator'


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def test_module_metadata_and_pickle_globals_are_canonical() -> None:
    assert separator.SeparatorObservations.__module__ == (
        'pyvoro2.inverse.separator.constraints'
    )
    assert separator.SeparatorFitProblem.__module__ == (
        'pyvoro2.inverse.separator.problem'
    )
    assert separator.SeparatorFitResult.__module__ == (
        'pyvoro2.inverse.separator.types'
    )
    assert separator.fit_weights_from_separators.__module__ == (
        'pyvoro2.inverse.separator.solver'
    )

    for value in (
        separator.SeparatorObservations,
        separator.SeparatorFitProblem,
        separator.SeparatorFitResult,
        separator.fit_weights_from_separators,
    ):
        assert pickle.loads(pickle.dumps(value)) is value


def test_inverse_package_exposes_only_the_small_high_level_surface() -> None:
    assert inverse.__all__ == [
        'SeparatorObservations',
        'resolve_separator_observations',
        'SeparatorFitResult',
        'fit_weights_from_separators',
        'weights_to_radii',
        'radii_to_weights',
    ]
    for name in inverse.__all__:
        assert getattr(inverse, name) is getattr(separator, name)
    for advanced_name in (
        'SeparatorFitProblem',
        'FitModel',
        'match_realized_pairs',
        'ActiveSetOptions',
        'solve_self_consistent_power_weights',
    ):
        assert not hasattr(inverse, advanced_name)


def test_canonical_sources_have_no_reverse_compatibility_imports() -> None:
    for path in sorted(CANONICAL_ROOT.glob('*.py')):
        for node in ast.walk(_parsed(path)):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith('pyvoro2.powerfit')
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('pyvoro2.powerfit')
                assert (node.module or '') != 'powerfit'


def test_active_set_uses_the_canonical_fixed_observation_solver_name() -> None:
    referenced_names = {
        node.id
        for node in ast.walk(_parsed(CANONICAL_ROOT / 'active.py'))
        if isinstance(node, ast.Name)
    }

    assert 'fit_weights_from_separators' in referenced_names
    assert 'fit_power_weights' not in referenced_names


def test_compatibility_source_trees_are_removed() -> None:
    assert not (PACKAGE_ROOT / 'powerfit').exists()
    assert not (PACKAGE_ROOT / 'planar' / 'result.py').exists()


def test_isolated_canonical_import_is_one_way_and_keeps_cores_lazy() -> None:
    code = """
import importlib.abc
import json
import sys

class BlockNativeExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'pyvoro2._core', 'pyvoro2._core2d'}:
            raise ImportError('native extension intentionally unavailable')
        return None

sys.meta_path.insert(0, BlockNativeExtensions())
import pyvoro2 as pv
import pyvoro2.inverse as inverse
print(json.dumps({
    'callable': callable(inverse.fit_weights_from_separators),
    'powerfit': 'pyvoro2.powerfit' in sys.modules,
    'powerfit_attribute': hasattr(pv, 'powerfit'),
    'core3d': 'pyvoro2._core' in sys.modules,
    'core2d': 'pyvoro2._core2d' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=PACKAGE_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        'callable': True,
        'powerfit': False,
        'powerfit_attribute': False,
        'core3d': False,
        'core2d': False,
    }
