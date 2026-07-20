"""Canonical separator names and bounded v0.7 compatibility behavior."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints
import warnings

import numpy as np
import pytest

import pyvoro2
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator
import pyvoro2.planar as planar


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'pyvoro2'
CANONICAL_NAMES = (
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitProblem',
    'SeparatorFitResult',
    'fit_weights_from_separators',
)
HIGH_LEVEL_NAMES = (
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitResult',
    'fit_weights_from_separators',
    'weights_to_radii',
    'radii_to_weights',
)
HISTORICAL_TO_CANONICAL = {
    'PairBisectorConstraints': 'SeparatorObservations',
    'resolve_pair_bisector_constraints': 'resolve_separator_observations',
    'PowerFitProblem': 'SeparatorFitProblem',
    'PowerWeightFitResult': 'SeparatorFitResult',
    'fit_power_weights': 'fit_weights_from_separators',
}


def _assert_optional_array_equal(left: object, right: object) -> None:
    if left is None or right is None:
        assert left is right
    else:
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_exact_canonical_package_exports() -> None:
    assert tuple(inverse.__all__) == HIGH_LEVEL_NAMES

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        powerfit = importlib.import_module('pyvoro2.powerfit')

    assert tuple(separator.__all__[:5]) == CANONICAL_NAMES
    assert set(separator.__all__) == set(powerfit.__all__) | set(CANONICAL_NAMES)
    assert len(separator.__all__) == len(set(separator.__all__))
    for canonical_name in CANONICAL_NAMES:
        assert canonical_name not in powerfit.__all__
        assert not hasattr(powerfit, canonical_name)


def test_historical_names_are_identity_aliases_with_canonical_introspection() -> None:
    for historical_name, canonical_name in HISTORICAL_TO_CANONICAL.items():
        historical = getattr(separator, historical_name)
        canonical = getattr(separator, canonical_name)
        assert historical is canonical
        assert canonical.__name__ == canonical_name
        assert inspect.signature(historical) == inspect.signature(canonical)

    assert fields(separator.PairBisectorConstraints) == fields(
        separator.SeparatorObservations
    )
    assert fields(separator.PowerFitProblem) == fields(separator.SeparatorFitProblem)
    assert fields(separator.PowerWeightFitResult) == fields(
        separator.SeparatorFitResult
    )


def test_separator_observation_validation_uses_canonical_name() -> None:
    observations = inverse.resolve_separator_observations(
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        [(0, 1, 0.5)],
    )

    with pytest.raises(ValueError) as exc_info:
        replace(observations, confidence=np.array([-1.0]))

    message = str(exc_info.value)
    assert message == 'SeparatorObservations.confidence must be non-negative'
    assert 'PairBisectorConstraints' not in message


def test_direct_submodule_and_top_level_compatibility_routes() -> None:
    module_pairs = {
        'constraints': (
            ('PairBisectorConstraints', 'SeparatorObservations'),
            ('resolve_pair_bisector_constraints', 'resolve_separator_observations'),
        ),
        'problem': (('PowerFitProblem', 'SeparatorFitProblem'),),
        'types': (('PowerWeightFitResult', 'SeparatorFitResult'),),
        'solver': (('fit_power_weights', 'fit_weights_from_separators'),),
    }
    for module_name, name_pairs in module_pairs.items():
        canonical_module = importlib.import_module(
            f'pyvoro2.inverse.separator.{module_name}'
        )
        historical_module = importlib.import_module(
            f'pyvoro2.powerfit.{module_name}'
        )
        for historical_name, canonical_name in name_pairs:
            canonical = getattr(canonical_module, canonical_name)
            assert getattr(canonical_module, historical_name) is canonical
            assert getattr(historical_module, historical_name) is canonical

    top_level_historical_names = {
        'PairBisectorConstraints',
        'resolve_pair_bisector_constraints',
        'PowerWeightFitResult',
        'fit_power_weights',
    }
    for historical_name, canonical_name in HISTORICAL_TO_CANONICAL.items():
        if historical_name not in top_level_historical_names:
            assert not hasattr(pyvoro2, historical_name)
            assert historical_name not in pyvoro2.__all__
            continue
        assert getattr(pyvoro2, historical_name) is getattr(
            separator, canonical_name
        )
        assert canonical_name not in pyvoro2.__all__
        assert not hasattr(pyvoro2, canonical_name)


def test_canonical_and_historical_calls_are_numerically_equivalent() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    raw = [(0, 1, 0.25), (1, 2, 0.60), (0, 2, 0.45)]

    canonical_observations = inverse.resolve_separator_observations(points, raw)
    historical_observations = separator.resolve_pair_bisector_constraints(
        points, raw
    )
    assert isinstance(canonical_observations, separator.SeparatorObservations)
    assert isinstance(canonical_observations, separator.PairBisectorConstraints)
    assert canonical_observations.to_records() == historical_observations.to_records()
    for name in (
        'i',
        'j',
        'shifts',
        'target',
        'confidence',
        'distance',
        'distance2',
        'delta',
        'target_fraction',
        'target_position',
        'input_index',
        'explicit_shift',
    ):
        np.testing.assert_array_equal(
            getattr(canonical_observations, name),
            getattr(historical_observations, name),
        )

    canonical_fit = inverse.fit_weights_from_separators(
        points,
        canonical_observations,
        connectivity_check='diagnose',
    )
    historical_fit = pyvoro2.fit_power_weights(
        points,
        historical_observations,
        connectivity_check='diagnose',
    )
    assert isinstance(canonical_fit, separator.SeparatorFitResult)
    assert isinstance(canonical_fit, separator.PowerWeightFitResult)
    for name in (
        'status',
        'hard_feasible',
        'weight_shift',
        'measurement',
        'rms_residual',
        'max_residual',
        'solver',
        'n_iter',
        'converged',
        'status_detail',
        'warnings',
    ):
        assert getattr(canonical_fit, name) == getattr(historical_fit, name)
    for name in (
        'weights',
        'radii',
        'target',
        'predicted',
        'predicted_fraction',
        'predicted_position',
        'residuals',
        'used_shifts',
    ):
        _assert_optional_array_equal(
            getattr(canonical_fit, name),
            getattr(historical_fit, name),
        )
    assert canonical_fit.to_records(canonical_observations) == (
        historical_fit.to_records(historical_observations)
    )
    assert canonical_fit.to_report(canonical_observations) == (
        historical_fit.to_report(historical_observations)
    )
    assert canonical_fit.connectivity == historical_fit.connectivity
    assert canonical_fit.objective_breakdown == historical_fit.objective_breakdown
    assert canonical_fit.edge_diagnostics is not None
    assert historical_fit.edge_diagnostics is not None
    for field in fields(canonical_fit.edge_diagnostics):
        _assert_optional_array_equal(
            getattr(canonical_fit.edge_diagnostics, field.name),
            getattr(historical_fit.edge_diagnostics, field.name),
        )


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


def test_deprecated_package_warning_is_hidden_by_interpreter_defaults() -> None:
    env = os.environ.copy()
    env.pop('PYTHONWARNINGS', None)
    completed = subprocess.run(
        [sys.executable, '-c', 'import pyvoro2.powerfit'],
        cwd=PACKAGE_ROOT.parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ''


def test_deprecated_package_emits_one_useful_warning_when_enabled() -> None:
    code = """
import json
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', DeprecationWarning)
    import pyvoro2.powerfit as first
    import pyvoro2.powerfit as second
print(json.dumps({
    'same': first is second,
    'count': len(caught),
    'category': caught[0].category.__name__,
    'message': str(caught[0].message),
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
    assert state['same'] is True
    assert state['count'] == 1
    assert state['category'] == 'DeprecationWarning'
    assert 'pyvoro2.inverse' in state['message']
    assert 'v0.8' in state['message']


def test_plain_and_canonical_imports_emit_no_compatibility_warning() -> None:
    code = """
import json
import sys
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', DeprecationWarning)
    import pyvoro2
    import pyvoro2.inverse
    import pyvoro2.inverse.separator
print(json.dumps({
    'warnings': [str(item.message) for item in caught],
    'powerfit': 'pyvoro2.powerfit' in sys.modules,
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
        'warnings': [],
        'powerfit': False,
        'core3d': False,
        'core2d': False,
    }
