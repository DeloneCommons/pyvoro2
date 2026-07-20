"""Ownership and compatibility checks for the separator implementation move."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pyvoro2
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator
import pyvoro2.powerfit as powerfit


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'pyvoro2'
CANONICAL_ROOT = PACKAGE_ROOT / 'inverse' / 'separator'
COMPATIBILITY_ROOT = PACKAGE_ROOT / 'powerfit'

SUBMODULES = (
    'active',
    'constraints',
    'model',
    'problem',
    'realize',
    'report',
    'solver',
    'types',
)
COMPATIBILITY_SUBMODULES = (*SUBMODULES, 'transforms')


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def test_canonical_and_historical_package_exports_are_identity_aliases() -> None:
    assert set(powerfit.__all__) < set(separator.__all__)
    for name in powerfit.__all__:
        canonical = getattr(separator, name)
        assert getattr(powerfit, name) is canonical
        if name in pyvoro2.__all__:
            assert getattr(pyvoro2, name) is canonical


def test_module_metadata_is_canonical_and_old_pickle_globals_still_resolve() -> None:
    assert separator.SeparatorObservations.__module__ == (
        'pyvoro2.inverse.separator.constraints'
    )
    assert separator.SeparatorObservations.__name__ == 'SeparatorObservations'
    assert separator.SeparatorFitProblem.__module__ == (
        'pyvoro2.inverse.separator.problem'
    )
    assert separator.SeparatorFitProblem.__name__ == 'SeparatorFitProblem'
    assert separator.SeparatorFitResult.__module__ == (
        'pyvoro2.inverse.separator.types'
    )
    assert separator.SeparatorFitResult.__name__ == 'SeparatorFitResult'
    assert separator.fit_weights_from_separators.__module__ == (
        'pyvoro2.inverse.separator.solver'
    )
    assert (
        separator.fit_weights_from_separators.__name__
        == 'fit_weights_from_separators'
    )

    old_class_global = (
        b'cpyvoro2.powerfit.types\nPowerWeightFitResult\n.'
    )
    old_function_global = b'cpyvoro2.powerfit.solver\nfit_power_weights\n.'
    assert pickle.loads(old_class_global) is separator.PowerWeightFitResult
    assert pickle.loads(old_function_global) is separator.fit_power_weights


def test_historical_direct_submodules_forward_the_canonical_objects() -> None:
    for name in SUBMODULES:
        historical = importlib.import_module(f'pyvoro2.powerfit.{name}')
        canonical = importlib.import_module(
            f'pyvoro2.inverse.separator.{name}',
        )
        for export in historical.__all__:
            assert getattr(historical, export) is getattr(canonical, export)

    historical_transforms = importlib.import_module(
        'pyvoro2.powerfit.transforms',
    )
    for export in historical_transforms.__all__:
        assert getattr(historical_transforms, export) is getattr(
            separator,
            export,
        )


def test_inverse_package_exposes_only_the_small_issue_12_surface() -> None:
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
    tree = _parsed(CANONICAL_ROOT / 'active.py')
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    assert 'fit_weights_from_separators' in referenced_names
    assert 'fit_power_weights' not in referenced_names


def test_compatibility_sources_contain_only_explicit_forwarding() -> None:
    for name in ('__init__', *COMPATIBILITY_SUBMODULES):
        path = COMPATIBILITY_ROOT / f'{name}.py'
        tree = _parsed(path)
        assert not any(
            isinstance(
                node,
                (
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.Lambda,
                ),
            )
            for node in ast.walk(tree)
        )

        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module != '__future__'
        ]
        assert imports
        assert all(
            node.module in {'_weight_transforms', 'warnings'}
            or (node.module or '').startswith('inverse.separator')
            for node in imports
        )
        assert all(
            isinstance(node, (ast.Expr, ast.ImportFrom, ast.Assign))
            for node in tree.body
        )


def test_isolated_top_level_powerfit_attribute_is_lazy_compatibility() -> None:
    code = """
import json
import sys
import pyvoro2 as pv

initially_loaded = 'pyvoro2.powerfit' in sys.modules
initially_cached = 'powerfit' in pv.__dict__
discoverable = 'powerfit' in dir(pv)
historical = pv.powerfit
import pyvoro2.inverse.separator as separator
print(json.dumps({
    'initially_loaded': initially_loaded,
    'initially_cached': initially_cached,
    'discoverable': discoverable,
    'excluded_from_all': 'powerfit' not in pv.__all__,
    'loaded_after_access': 'pyvoro2.powerfit' in sys.modules,
    'cached_after_access': pv.__dict__.get('powerfit') is historical,
    'historical_module': historical.__name__,
    'all_exports_identical': all(
        getattr(historical, name) is getattr(separator, name)
        for name in historical.__all__
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=PACKAGE_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout)
    assert state == {
        'initially_loaded': False,
        'initially_cached': False,
        'discoverable': True,
        'excluded_from_all': True,
        'loaded_after_access': True,
        'cached_after_access': True,
        'historical_module': 'pyvoro2.powerfit',
        'all_exports_identical': True,
    }


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
    'powerfit_cached': 'powerfit' in pv.__dict__,
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
    state = json.loads(completed.stdout)
    assert state == {
        'callable': True,
        'powerfit': False,
        'powerfit_cached': False,
        'core3d': False,
        'core2d': False,
    }
