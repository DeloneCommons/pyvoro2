"""Neutral ownership and compatibility tests for weight/radius transforms."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2._weight_transforms as neutral_transforms
import pyvoro2.powerfit as powerfit
import pyvoro2.powerfit.active as powerfit_active
import pyvoro2.powerfit.problem as powerfit_problem
import pyvoro2.powerfit.transforms as compatibility_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'src' / 'pyvoro2'
TRANSFORM_NAMES = {'radii_to_weights', 'weights_to_radii'}


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_parsed(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or '')
    return modules


def test_transform_functions_have_one_neutral_implementation() -> None:
    definitions = {
        (path.relative_to(PACKAGE_ROOT).as_posix(), node.name)
        for path in PACKAGE_ROOT.rglob('*.py')
        for node in ast.walk(_parsed(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in TRANSFORM_NAMES
    }

    assert definitions == {
        ('_weight_transforms.py', 'radii_to_weights'),
        ('_weight_transforms.py', 'weights_to_radii'),
    }


def test_all_public_and_compatibility_routes_share_function_objects() -> None:
    assert pv.radii_to_weights is neutral_transforms.radii_to_weights
    assert pv.weights_to_radii is neutral_transforms.weights_to_radii
    assert powerfit.radii_to_weights is neutral_transforms.radii_to_weights
    assert powerfit.weights_to_radii is neutral_transforms.weights_to_radii
    assert (
        compatibility_transforms.radii_to_weights
        is neutral_transforms.radii_to_weights
    )
    assert (
        compatibility_transforms.weights_to_radii
        is neutral_transforms.weights_to_radii
    )
    assert pv.radii_to_weights.__module__ == 'pyvoro2._weight_transforms'
    assert pv.weights_to_radii.__module__ == 'pyvoro2._weight_transforms'


@pytest.mark.parametrize(
    ('module', 'source_name'),
    (
        (powerfit_problem, 'problem.py'),
        (powerfit_active, 'active.py'),
    ),
)
def test_separator_modules_import_neutral_transform_directly(
    module: object,
    source_name: str,
) -> None:
    assert module.weights_to_radii is neutral_transforms.weights_to_radii

    direct_imports = {
        (node.level, node.module, alias.name)
        for node in ast.walk(_parsed(PACKAGE_ROOT / 'powerfit' / source_name))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (2, '_weight_transforms', 'weights_to_radii') in direct_imports
    assert not any(
        module_name == 'transforms' or module_name.endswith('.transforms')
        for _, module_name, _ in direct_imports
        if module_name is not None
    )


def test_neutral_module_has_no_separator_or_native_dependencies() -> None:
    imports = _imported_modules(PACKAGE_ROOT / '_weight_transforms.py')

    assert imports == {'__future__', 'numpy'}
    assert not any('powerfit' in module for module in imports)
    assert not any('separator' in module for module in imports)
    assert '_core' not in imports
    assert '_core2d' not in imports


def test_exact_and_tolerance_based_round_trips() -> None:
    radii = np.array([0.0, 0.5, 2.0])
    weights = neutral_transforms.radii_to_weights(radii)
    restored_radii, shift = neutral_transforms.weights_to_radii(weights)

    np.testing.assert_array_equal(weights, np.array([0.0, 0.25, 4.0]))
    np.testing.assert_array_equal(restored_radii, radii)
    assert shift == 0.0

    original_weights = np.array([-2.5, 0.25, 4.0])
    shifted_radii, shift = neutral_transforms.weights_to_radii(original_weights)
    restored_weights = neutral_transforms.radii_to_weights(shifted_radii) - shift
    np.testing.assert_allclose(restored_weights, original_weights, atol=1e-14)

    tolerance_radii, tolerance_shift = neutral_transforms.weights_to_radii(
        np.array([-1e-15, 1.0]),
        weight_shift=0.0,
    )
    np.testing.assert_allclose(
        neutral_transforms.radii_to_weights(tolerance_radii) - tolerance_shift,
        np.array([-1e-15, 1.0]),
        atol=1e-14,
    )
    with pytest.raises(ValueError, match='negative values'):
        neutral_transforms.weights_to_radii(
            np.array([-2e-14, 1.0]),
            weight_shift=0.0,
        )


def test_automatic_and_explicit_shift_behavior() -> None:
    automatic_radii, automatic_shift = neutral_transforms.weights_to_radii(
        np.array([-2.0, 2.0]),
        r_min=1.0,
    )
    np.testing.assert_allclose(automatic_radii, np.array([1.0, np.sqrt(5.0)]))
    assert automatic_shift == 3.0

    explicit_radii, explicit_shift = neutral_transforms.weights_to_radii(
        np.array([2.0, 3.0]),
        weight_shift=-2.0,
    )
    np.testing.assert_array_equal(explicit_radii, np.array([0.0, 1.0]))
    assert explicit_shift == -2.0


def test_empty_array_behavior() -> None:
    weights = neutral_transforms.radii_to_weights(np.array([]))
    radii, automatic_shift = neutral_transforms.weights_to_radii(
        np.array([]),
        r_min=2.0,
    )
    explicit_radii, explicit_shift = neutral_transforms.weights_to_radii(
        np.array([]),
        weight_shift=-3.0,
    )

    assert weights.shape == (0,)
    assert radii.shape == (0,)
    assert automatic_shift == 4.0
    assert explicit_radii.shape == (0,)
    assert explicit_shift == -3.0


@pytest.mark.parametrize('bad_value', (np.nan, np.inf, -np.inf))
def test_nonfinite_values_are_rejected(bad_value: float) -> None:
    with pytest.raises(ValueError, match='finite'):
        neutral_transforms.radii_to_weights(np.array([1.0, bad_value]))
    with pytest.raises(ValueError, match='finite'):
        neutral_transforms.weights_to_radii(np.array([0.0, bad_value]))
    with pytest.raises(ValueError, match='weight_shift must be finite'):
        neutral_transforms.weights_to_radii(
            np.array([0.0, 1.0]),
            weight_shift=bad_value,
        )


@pytest.mark.parametrize('bad_r_min', (np.nan, np.inf, -np.inf))
def test_nonfinite_r_min_is_rejected(bad_r_min: float) -> None:
    with pytest.raises(ValueError, match='r_min must be finite'):
        neutral_transforms.weights_to_radii(
            np.array([0.0, 1.0]),
            r_min=bad_r_min,
        )


@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_overflowing_radius_squares_are_rejected() -> None:
    float_max = np.finfo(float).max

    with pytest.raises(ValueError, match='non-finite weights'):
        neutral_transforms.radii_to_weights(np.array([1.0, float_max]))
    with pytest.raises(ValueError, match='r_min squared must be finite'):
        neutral_transforms.weights_to_radii(
            np.array([0.0, 1.0]),
            r_min=float_max,
        )


@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_overflowing_derived_shift_and_shifted_weights_are_rejected() -> None:
    float_max = np.finfo(float).max
    finite_r_min = np.sqrt(float_max / 2.0)

    with pytest.raises(ValueError, match='derived weight shift must be finite'):
        neutral_transforms.weights_to_radii(
            np.array([-float_max, 0.0]),
            r_min=finite_r_min,
        )
    with pytest.raises(ValueError, match='non-finite values'):
        neutral_transforms.weights_to_radii(
            np.array([-float_max, float_max]),
        )
    with pytest.raises(ValueError, match='non-finite values'):
        neutral_transforms.weights_to_radii(
            np.array([0.0, float_max]),
            weight_shift=float_max,
        )


@pytest.mark.parametrize(
    ('weights', 'kwargs'),
    (
        (np.array([-2.0, 0.0, 3.0]), {}),
        (np.array([0.0, 1.0, 4.0]), {'r_min': 0.5}),
        (np.array([-1.0, 3.0]), {'weight_shift': 1.0}),
        (np.array([]), {'r_min': 2.0}),
    ),
)
def test_successful_weight_transforms_return_only_finite_values(
    weights: np.ndarray,
    kwargs: dict[str, float],
) -> None:
    radii, shift = neutral_transforms.weights_to_radii(weights, **kwargs)

    assert np.all(np.isfinite(radii))
    assert np.isfinite(shift)
    assert np.all(np.isfinite(neutral_transforms.radii_to_weights(radii)))


def test_dimension_and_negative_value_validation() -> None:
    with pytest.raises(ValueError, match='radii must be 1D'):
        neutral_transforms.radii_to_weights(np.zeros((1, 1)))
    with pytest.raises(ValueError, match='weights must be 1D'):
        neutral_transforms.weights_to_radii(np.zeros((1, 1)))
    with pytest.raises(ValueError, match='radii must be non-negative'):
        neutral_transforms.radii_to_weights(np.array([0.0, -1.0]))
    with pytest.raises(ValueError, match='r_min must be >= 0'):
        neutral_transforms.weights_to_radii(np.array([0.0]), r_min=-1.0)


def test_conflicting_shift_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match='at most one'):
        neutral_transforms.weights_to_radii(
            np.array([0.0, 1.0]),
            r_min=1.0,
            weight_shift=0.0,
        )


def test_neutral_module_executes_without_package_or_native_imports() -> None:
    code = """
import importlib.abc
import importlib.util
import sys

attempted = []


class Pyvoro2ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyvoro2' or fullname.startswith('pyvoro2.'):
            attempted.append(fullname)
            raise AssertionError(f'unexpected pyvoro2 import: {fullname}')
        return None


sys.meta_path.insert(0, Pyvoro2ImportGuard())
spec = importlib.util.spec_from_file_location(
    '_isolated_weight_transforms',
    sys.argv[1],
)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

radii, shift = module.weights_to_radii([-1.0, 3.0])
assert radii.tolist() == [0.0, 2.0]
assert shift == 1.0
assert attempted == []
assert 'pyvoro2._core' not in sys.modules
assert 'pyvoro2._core2d' not in sys.modules
"""
    subprocess.run(
        [
            sys.executable,
            '-c',
            code,
            str(PACKAGE_ROOT / '_weight_transforms.py'),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_transform_submodule_does_not_broaden_package_native_imports() -> None:
    code = """
import importlib.abc
import json
import sys

native_extensions = {'pyvoro2._core', 'pyvoro2._core2d'}
attempted = []


class NativeImportRecorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in native_extensions:
            attempted.append(fullname)
        return None


sys.meta_path.insert(0, NativeImportRecorder())
__import__(sys.argv[1])
print(json.dumps(attempted))
"""

    def native_import_attempts(module_name: str) -> list[str]:
        completed = subprocess.run(
            [sys.executable, '-c', code, module_name],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    package_attempts = native_import_attempts('pyvoro2')
    transform_attempts = native_import_attempts('pyvoro2._weight_transforms')

    assert set(package_attempts) == {'pyvoro2._core', 'pyvoro2._core2d'}
    assert transform_attempts == package_attempts
