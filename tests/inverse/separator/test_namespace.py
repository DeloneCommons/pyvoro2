"""Canonical separator names and completed v0.8 compatibility removal."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import numpy as np
import pytest

import pyvoro2
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator
import pyvoro2.planar as planar


PACKAGE_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'pyvoro2'
HIGH_LEVEL_NAMES = (
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitResult',
    'fit_weights_from_separators',
    'weights_to_radii',
    'radii_to_weights',
)
REMOVED_ALIASES = (
    'PairBisectorConstraints',
    'resolve_pair_bisector_constraints',
    'PowerFitProblem',
    'PowerWeightFitResult',
    'fit_power_weights',
)
REMOVED_ALIAS_MODULES = {
    'constraints': REMOVED_ALIASES[:2],
    'problem': (REMOVED_ALIASES[2],),
    'types': (REMOVED_ALIASES[3],),
    'solver': (REMOVED_ALIASES[4],),
}


def test_exact_canonical_high_level_exports() -> None:
    assert tuple(inverse.__all__) == HIGH_LEVEL_NAMES
    assert tuple(separator.__all__[:5]) == (
        'SeparatorObservations',
        'resolve_separator_observations',
        'SeparatorFitProblem',
        'SeparatorFitResult',
        'fit_weights_from_separators',
    )
    assert len(separator.__all__) == len(set(separator.__all__))
    for name in inverse.__all__:
        assert getattr(inverse, name) is getattr(separator, name)


def test_removed_aliases_are_absent_from_all_canonical_routes() -> None:
    for name in REMOVED_ALIASES:
        assert name not in separator.__all__
        assert not hasattr(separator, name)
        assert name not in pyvoro2.__all__
        assert not hasattr(pyvoro2, name)

    for module_name, removed_names in REMOVED_ALIAS_MODULES.items():
        module = importlib.import_module(
            f'pyvoro2.inverse.separator.{module_name}'
        )
        for name in removed_names:
            assert name not in getattr(module, '__all__', ())
            assert not hasattr(module, name)


def test_removed_compatibility_package_and_lazy_attribute_are_absent() -> None:
    assert importlib.util.find_spec('pyvoro2.powerfit') is None
    with pytest.raises(ModuleNotFoundError, match='pyvoro2\\.powerfit'):
        importlib.import_module('pyvoro2.powerfit')
    assert 'powerfit' not in pyvoro2.__dict__
    assert 'powerfit' not in dir(pyvoro2)


def test_canonical_types_have_canonical_introspection() -> None:
    assert separator.SeparatorObservations.__name__ == 'SeparatorObservations'
    assert separator.SeparatorFitProblem.__name__ == 'SeparatorFitProblem'
    assert separator.SeparatorFitResult.__name__ == 'SeparatorFitResult'
    assert (
        separator.fit_weights_from_separators.__name__
        == 'fit_weights_from_separators'
    )
    assert tuple(field.name for field in fields(separator.SeparatorObservations))
    assert tuple(field.name for field in fields(separator.SeparatorFitProblem))
    assert tuple(field.name for field in fields(separator.SeparatorFitResult))


def test_separator_observation_validation_uses_canonical_name() -> None:
    observations = inverse.resolve_separator_observations(
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        [(0, 1, 0.5)],
    )

    with pytest.raises(ValueError) as exc_info:
        replace(observations, confidence=np.array([-1.0]))

    assert str(exc_info.value) == (
        'SeparatorObservations.confidence must be non-negative'
    )
    assert REMOVED_ALIASES[0] not in str(exc_info.value)


def test_canonical_fit_preserves_records_reports_and_graph_diagnostics() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (1, 2, 0.60), (0, 2, 0.45)],
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        connectivity_check='diagnose',
    )

    assert isinstance(observations, inverse.SeparatorObservations)
    assert isinstance(fit, inverse.SeparatorFitResult)
    assert fit.status == 'optimal'
    assert fit.weights is not None
    assert fit.radii is not None
    assert fit.edge_diagnostics is not None
    assert fit.connectivity is not None
    assert len(fit.to_records(observations)) == observations.n_constraints
    report = fit.to_report(observations)
    assert report['summary']['status'] == fit.status
    assert report['summary']['n_constraints'] == observations.n_constraints


def test_canonical_solver_annotation_and_raw_input_support_planar_domains() -> None:
    domain = planar.Box(((0.0, 2.0), (-1.0, 1.0)))
    domain_annotation = get_type_hints(
        inverse.fit_weights_from_separators,
    )['domain']
    assert isinstance(domain, domain_annotation)

    result = inverse.fit_weights_from_separators(
        np.array([[0.5, 0.0], [1.5, 0.0]], dtype=float),
        [(0, 1, 0.25)],
        domain=domain,
        connectivity_check='diagnose',
    )

    assert isinstance(result, inverse.SeparatorFitResult)
    assert result.status == 'optimal'
    assert result.weights is not None
    assert result.weights.shape == (2,)


def test_isolated_canonical_imports_keep_native_extensions_lazy() -> None:
    code = """
import importlib.util
import json
import sys

import pyvoro2
import pyvoro2.inverse
import pyvoro2.inverse.separator

print(json.dumps({
    'removed_spec': importlib.util.find_spec('pyvoro2.powerfit'),
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
        'removed_spec': None,
        'core3d': False,
        'core2d': False,
    }
