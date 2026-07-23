#!/usr/bin/env python3
"""Check an installed pyvoro2 package and its representative public workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    fit = inverse.fit_weights_from_separators(
        points2,
        [(0, 1, 0.25)],
        connectivity_check='diagnose',
    )
    if fit.status != 'optimal':
        raise InstalledPackageCheckError(
            f'the inverse smoke workflow returned status {fit.status!r}'
        )
    if fit.solver != 'analytic':
        raise InstalledPackageCheckError(
            f'the inverse smoke workflow used {fit.solver!r}, not the analytic path'
        )
    for field_name in ('weights', 'radii', 'predicted'):
        values = getattr(fit, field_name)
        if values is None or not np.all(np.isfinite(values)):
            raise InstalledPackageCheckError(
                f'the inverse smoke workflow returned non-finite {field_name}'
            )

    print('spatial workflow: TessellationResult with 2 cells')
    print('planar workflow: TessellationResult with 2 cells')
    print('inverse workflow: optimal analytic fit with finite values')


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
    _run_workflows(args.repo_root)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except InstalledPackageCheckError as exc:
        print(f'ERROR: {exc}')
        raise SystemExit(1)
