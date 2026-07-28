#!/usr/bin/env python3
"""Check an installed pyvoro2 package and its representative public workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_HELPER_MODULES = (
    'pyvoro2._internal.cell_output',
    'pyvoro2._internal.inputs',
    'pyvoro2._internal.power_input',
    'pyvoro2._internal.weight_transforms',
    'pyvoro2._internal.spatial.domain_geometry',
    'pyvoro2._internal.spatial.domain_utils',
    'pyvoro2._internal.spatial.face_shifts',
    'pyvoro2._internal.planar.domain_geometry',
    'pyvoro2._internal.planar.edge_shifts',
)
OBSOLETE_PRIVATE_MODULES = (
    'pyvoro2._cell_output',
    'pyvoro2._inputs',
    'pyvoro2._power_input',
    'pyvoro2._weight_transforms',
    'pyvoro2._domain_geometry',
    'pyvoro2._face_shifts3d',
    'pyvoro2._util',
    'pyvoro2.planar._domain_geometry',
    'pyvoro2.planar._edge_shifts2d',
)


class InstalledPackageCheckError(RuntimeError):
    """Raised when installed-package provenance or a smoke workflow is invalid."""


def module_location(module: ModuleType, module_name: str) -> Path:
    """Return the resolved filesystem location reported by an imported module."""

    location = getattr(module, '__file__', None)
    if not location:
        raise InstalledPackageCheckError(
            f'{module_name} does not report a filesystem location'
        )
    return Path(location).resolve()


def assert_outside_repository(
    module: ModuleType,
    module_name: str,
    repository_root: Path,
) -> Path:
    """Require an imported module to resolve outside the repository checkout."""

    location = module_location(module, module_name)
    root = repository_root.resolve()
    if location.is_relative_to(root):
        raise InstalledPackageCheckError(
            f'{module_name} was imported from the repository checkout: {location}'
        )
    return location


def _check_scipy(*, require_scipy: bool) -> None:
    spec = importlib.util.find_spec('scipy')
    if require_scipy:
        if spec is None:
            raise InstalledPackageCheckError(
                'SciPy is required for this installed-package check but was not found'
            )
        scipy = importlib.import_module('scipy')
        print(f'scipy: {module_location(scipy, "scipy")}')
        return

    if spec is not None:
        raise InstalledPackageCheckError(
            'SciPy must be absent from this base installation, but it was found at '
            f'{spec.origin}'
        )
    print('scipy: absent (verified with importlib.util.find_spec)')


def _check_removed_compatibility() -> None:
    import inspect
    import pyvoro2 as pv
    import pyvoro2.inverse.separator as separator
    import pyvoro2.planar as pv2

    if importlib.util.find_spec('pyvoro2.powerfit') is not None:
        raise InstalledPackageCheckError(
            'the removed pyvoro2.powerfit package is still importable'
        )
    removed_aliases = (
        'PairBisectorConstraints',
        'resolve_pair_bisector_constraints',
        'PowerFitProblem',
        'PowerWeightFitResult',
        'fit_power_weights',
    )
    for name in removed_aliases:
        if hasattr(pv, name) or hasattr(separator, name):
            raise InstalledPackageCheckError(
                f'the removed compatibility alias {name} is still exported'
            )
    if hasattr(pv, 'powerfit'):
        raise InstalledPackageCheckError(
            'the removed top-level powerfit attribute is still exported'
        )
    if hasattr(pv2, 'PlanarComputeResult'):
        raise InstalledPackageCheckError(
            'the removed PlanarComputeResult alias is still exported'
        )
    if 'return_result' in inspect.signature(pv2.compute).parameters:
        raise InstalledPackageCheckError(
            'the removed planar return_result parameter is still accepted'
        )
    print('removed v0.7 compatibility surfaces: absent')


def _check_private_helper_layout() -> None:
    native_extensions = {'pyvoro2._core', 'pyvoro2._core2d'}
    loaded_early = sorted(native_extensions.intersection(sys.modules))
    if loaded_early:
        raise InstalledPackageCheckError(
            f'native extensions loaded before helper imports: {loaded_early}'
        )

    for module_name in INTERNAL_HELPER_MODULES:
        if importlib.util.find_spec(module_name) is None:
            raise InstalledPackageCheckError(
                f'installed package is missing {module_name}'
            )
        importlib.import_module(module_name)

    loaded_after_helpers = sorted(native_extensions.intersection(sys.modules))
    if loaded_after_helpers:
        raise InstalledPackageCheckError(
            'private pure-Python helpers loaded native extensions: '
            f'{loaded_after_helpers}'
        )

    for module_name in OBSOLETE_PRIVATE_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            raise InstalledPackageCheckError(
                f'obsolete private helper remains importable: {module_name}'
            )
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
        else:
            raise InstalledPackageCheckError(
                f'obsolete private helper imported successfully: {module_name}'
            )

    print('private helper layout: internal modules present, obsolete paths absent')
    print('private helper imports: native extensions remained lazy')


def _run_workflows(repository_root: Path) -> None:
    import numpy as np
    import pyvoro2 as pv
    import pyvoro2.inverse as inverse
    import pyvoro2.planar as pv2

    modules = (
        ('pyvoro2', pv),
        ('pyvoro2._core', importlib.import_module('pyvoro2._core')),
        ('pyvoro2._core2d', importlib.import_module('pyvoro2._core2d')),
    )
    for module_name, module in modules:
        location = assert_outside_repository(
            module,
            module_name,
            repository_root,
        )
        print(f'{module_name}: {location}')

    points3 = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    result3 = pv.compute(
        points3,
        domain=pv.Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))),
        mode='standard',
    )
    if not isinstance(result3, pv.TessellationResult):
        raise InstalledPackageCheckError(
            'the spatial smoke workflow did not return TessellationResult'
        )
    if len(result3.cells) != 2:
        raise InstalledPackageCheckError(
            f'the spatial smoke workflow returned {len(result3.cells)} cells'
        )

    points2 = np.array(
        [[0.25, 0.5], [0.75, 0.5]],
        dtype=float,
    )
    result2 = pv2.compute(
        points2,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_edges=True,
    )
    if not isinstance(result2, pv.TessellationResult):
        raise InstalledPackageCheckError(
            'the planar smoke workflow did not return TessellationResult'
        )
    if len(result2.cells) != 2:
        raise InstalledPackageCheckError(
            f'the planar smoke workflow returned {len(result2.cells)} cells'
        )

    periodic_points = np.array(
        [[0.12, 0.2], [0.75, 0.25], [0.35, 0.72], [0.88, 0.82]],
        dtype=float,
    )
    periodic_result = pv2.compute(
        periodic_points,
        domain=pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True),
        ),
        return_edge_shifts=True,
    )
    if not periodic_result.has_periodic_shifts:
        raise InstalledPackageCheckError(
            'the periodic planar workflow did not expose image shifts'
        )
    if not np.isclose(np.sum(periodic_result.cell_measures), 1.0):
        raise InstalledPackageCheckError(
            'the periodic planar workflow did not cover the unit cell'
        )

    public_weights = np.array([-2.5, 0.25, 4.0])
    backend_radii, representation_shift = pv.weights_to_radii(public_weights)
    restored_weights = pv.radii_to_weights(backend_radii) - representation_shift
    if not np.allclose(restored_weights, public_weights):
        raise InstalledPackageCheckError(
            'the public weight/radius transforms did not round-trip'
        )
    if inverse.weights_to_radii is not pv.weights_to_radii:
        raise InstalledPackageCheckError(
            'the public transform routes do not share one implementation'
        )

    fit = inverse.fit_weights_from_separators(
        points2,
        [(0, 1, 0.25)],
        solver='direct',
        linear_backend='dense',
        connectivity_check='diagnose',
    )
    if fit.status != 'optimal':
        raise InstalledPackageCheckError(
            f'the inverse smoke workflow returned status {fit.status!r}'
        )
    if fit.solver != 'direct':
        raise InstalledPackageCheckError(
            f'the inverse smoke workflow used {fit.solver!r}, not the direct method'
        )
    if fit.linear_backend != 'dense':
        raise InstalledPackageCheckError(
            'the inverse smoke workflow used '
            f'{fit.linear_backend!r}, not the dense backend'
        )
    for field_name in ('weights', 'radii', 'predicted'):
        values = getattr(fit, field_name)
        if values is None or not np.all(np.isfinite(values)):
            raise InstalledPackageCheckError(
                f'the inverse smoke workflow returned non-finite {field_name}'
            )

    print('spatial workflow: TessellationResult with 2 cells')
    print('planar workflow: TessellationResult with 2 cells')
    print('periodic workflow: planar unit-cell coverage with image shifts')
    print('weight/radius transforms: public routes round-trip finite values')
    print('inverse workflow: optimal direct+dense fit with finite values')


def main() -> int:
    """Run provenance, dependency, native-extension, and public-workflow checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=REPO_ROOT,
        help='repository checkout that imports must resolve outside',
    )
    scipy_group = parser.add_mutually_exclusive_group(required=True)
    scipy_group.add_argument(
        '--require-scipy',
        action='store_true',
        help='require and report an installed SciPy package',
    )
    scipy_group.add_argument(
        '--forbid-scipy',
        action='store_true',
        help='require SciPy to be absent from the environment',
    )
    args = parser.parse_args()

    _check_scipy(require_scipy=args.require_scipy)
    _check_removed_compatibility()
    _check_private_helper_layout()
    _run_workflows(args.repo_root)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except InstalledPackageCheckError as exc:
        print(f'ERROR: {exc}')
        raise SystemExit(1)
