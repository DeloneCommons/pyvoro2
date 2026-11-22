"""Isolated P0 replay for #153; exit 1 means a reported contract discrepancy.

Run from the repository root after editable installation. No production fixes,
permanent xfails, or additional runtime dependencies. See the adjacent note for
authority, interpretation, and promotion ownership. --native also runs Voro++.
"""

import argparse
from fractions import Fraction as F
import json
import platform
import subprocess
import sys

import numpy as np

from pyvoro2 import OrthorhombicCell, PeriodicCell, compute
from pyvoro2._internal.spatial.domain_geometry import DomainGeometry3D
from pyvoro2.inverse.separator.constraints import resolve_separator_observations
from pyvoro2.planar import RectangularCell


def exact(value):
    """Serialize every float operand losslessly, including subnormals."""
    if isinstance(value, np.ndarray):
        return exact(value.tolist())
    if isinstance(value, (list, tuple)):
        return [exact(item) for item in value]
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, F):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_solve(vector, basis):
    """Independent rational Gaussian elimination: solve q @ basis = vector."""
    n = len(vector)
    rows = [[F(basis[j][i]) for j in range(n)]
            + [F(vector[i])] for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [entry / divisor for entry in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def determinant(basis):
    a, b, c = [[F(float(x)) for x in row] for row in basis]
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def record(rows, case, route, inputs, observed, checks, oracle):
    rows.append(dict(case=case, route=route, inputs=exact(inputs),
                     observed=exact(observed), oracle=exact(oracle),
                     checks={key: bool(value) for key, value in checks.items()}))


def source_and_remap(rows):
    points = np.array([[2.0**-41, 1/4, 1/4], [1/2, 1/4, 1/4]])
    domain = OrthorhombicCell(((0, 1),) * 3)
    obs = resolve_separator_observations(points, [(0, 1, 1/2)], domain=domain)
    displacement = F(1, 2) - F(1, 2**41)
    # Since 0 < d < 1/2, zero is uniquely closest among ALL integers.
    gap = (displacement - 1)**2 - displacement**2
    assert gap == F(1, 2**40)
    record(rows, '#60', 'Python separator consumer', points,
           dict(shift=obs.shifts, delta=obs.delta),
           dict(source_shift=np.array_equal(obs.shifts, [[0, 0, 0]]),
                source_delta=F(float(obs.delta[0, 0])) == displacement),
           dict(unique_shift=[0, 0, 0], dx=displacement, squared_gap=gap))

    basis = np.array([[1., 0., 0.], [-1/2, 1., 0.], [0., 0., 1.]])
    point = np.array([[3/16, -2.0**-55, 3/8]])
    cell = PeriodicCell(basis)
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    for route, remapper in [('public', cell), ('detached snapshot', snapshot)]:
        wrapped, shift = remapper.remap_cart(point, return_shifts=True)
        # Allow ONLY the existing default y snap. Its x residual is physical:
        # q @ A = (3/16,0,3/8) - wrapped must have integer q.
        snapped = [3/16, 0., 3/8]
        delta = [F(x) - F(float(y)) for x, y in zip(snapped, wrapped[0])]
        # All these differences are exactly representable in this fixture.
        q = row_solve(delta, basis)
        record(rows, '#65', route + ' (Python)', dict(A=basis, point=point),
               dict(wrapped=wrapped, shift=shift, translation_coefficients=q),
               dict(no_physical_residual_loss=all(x.denominator == 1 for x in q)),
               dict(allowed_snapped_point=snapped, exact_source_y=F(-1, 2**55)))
    return cell, point


def endpoints(rows):
    # Exact fixtures from historical PR #67; positive handedness is admitted
    # today. Negative handedness is measured separately below, not bypassed.
    fixtures = [
        ('A-y-to-x', [[1., 0., 0.], [1/4, 1., 0.], [0., 0., 1.]],
         [-2.0**-55, -2.0**-55, 3/8], [0., 0., 3/8]),
        ('B-z-to-y-to-x', [[1., 0., 0.], [-1/2, 1., 0.], [0., 2.0**-54, 1.]],
         [3/16, -2.0**-55, -2.0**-55], [3/16, 0., 0.]),
    ]
    translations = np.array([[0, 0, 0], [2, -1, 3], [-2, 1, -3]])
    for label, basis, point, expected in fixtures:
        basis = np.array(basis)
        points = np.array([point]) + translations @ basis
        cell = PeriodicCell(basis)
        snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
        for route, remapper in [('public', cell), ('snapshot', snapshot)]:
            for eps in (0., None, 1e-12):
                wrapped, shifts = remapper.remap_internal(
                    points, eps=eps, return_shifts=True)
                errors_x = [
                    F(w[0]) + sum(int(s[j]) * F(basis[j, 0]) for j in range(3))
                    - F(p[0]) for w, s, p in zip(wrapped, shifts, points)
                ]
                # A permits its original tiny x seam to snap to zero; B has
                # a physical 3/16 residual that must reconstruct exactly.
                allowed_x = F(2.0**-55) if label == 'A-y-to-x' else F(0)
                record(rows, '#67 ' + label, route + ' (Python)',
                       dict(A=basis, points=points, eps=eps),
                       dict(wrapped=wrapped, shifts=shifts,
                            exact_x_reconstruction_errors=errors_x,
                            historical_target_match=np.array_equal(
                                wrapped, np.tile(expected, (3, 1))),
                            historical_shift_match=np.array_equal(
                                shifts, translations)),
                       dict(half_open=np.all((wrapped >= 0) & (wrapped < 1)),
                            tangential_residual=all(abs(e) <= allowed_x
                                                    for e in errors_x)),
                       dict(historical_backend_target=expected,
                            shifts=translations))

    tiny = float.fromhex('0x0.0000000000001p-1022')
    for length in (1., 2.):
        # PR #67 seams plus ordinary negative quotient and underflow controls.
        seams = [-tiny, 0., tiny, np.nextafter(length, 0.), length,
                 np.nextafter(length, np.inf), -1/8]
        for route in ('planar', 'orthorhombic', 'triclinic', 'snapshot'):
            dim = 2 if route == 'planar' else 3
            bounds = ((0., length),) * dim
            if route == 'planar':
                remapper = RectangularCell(bounds)
            elif route == 'orthorhombic':
                remapper = OrthorhombicCell(bounds)
            else:
                remapper = PeriodicCell(np.eye(3) * length)
                if route == 'snapshot':
                    geometry = DomainGeometry3D(remapper)
                    remapper = geometry.native_periodic_snapshot()
            points = np.repeat(np.array(seams)[:, None], dim, axis=1)
            remap = (remapper.remap_internal if route in ('triclinic', 'snapshot')
                     else remapper.remap_cart)
            wrapped, shifts = remap(points, eps=0., return_shifts=True)
            # Only half-open numerical backend invariants here: do not invent
            # an epsilon policy for the negative minimum subnormal.
            record(rows, '#67 seams', route + ' (Python)',
                   dict(length=length, points=points, eps=0.),
                   dict(wrapped=wrapped, shifts=shifts),
                   dict(half_open=np.all((wrapped >= 0) & (wrapped < length)),
                        positive_subnormal=np.all(wrapped[2] == tiny),
                        ordinary_negative=np.all(wrapped[-1] == length - 1/8)
                        and np.all(shifts[-1] == -1)),
                   'backend half-open range; preserve positive interior; '
                   'floor(-1/8/L)=-1')

    lower = np.array([[2., 0., 0.], [-1/2, 2., 0.], [1/4, -1/2, 2.]])
    cell = PeriodicCell(lower)
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    for axis in range(3):
        points = np.tile([3/16, 3/8, 5/8], (3, 1))
        points[:, axis] = [-tiny, tiny, -1/8]
        for route, remapper in [('public', cell), ('snapshot', snapshot)]:
            wrapped, shifts = remapper.remap_internal(
                points, eps=0., return_shifts=True)
            expected = points[:2].copy()
            expected[0, axis] = 0.
            record(rows, '#67 coupled underflow', route + ' (Python)',
                   dict(A=lower, points=points, axis=axis, eps=0.),
                   dict(wrapped=wrapped, shifts=shifts),
                   dict(half_open=np.all((wrapped >= 0) & (wrapped < 2)),
                        positive_subnormal=wrapped[1, axis] == tiny,
                        ordinary_negative=shifts[2, axis] == -1),
                   dict(historical_underflow_target=expected,
                        ordinary_negative_shift=-1))


def frames(rows):
    bases = dict(near_parallel=[[1., 1., 0.], [1., 1. + 2.0**-30, 0.],
                                [0., 0., 1.]],
                 identity=np.eye(3),
                 right_handed=[[1., 1., 0.], [-1., 1., 0.], [0., 0., 2.]],
                 anisotropic=np.diag([2.0**-16, 1., 2.0**16]),
                 cubic_shear=[[1., 0., 0.], [8., 1., 0.], [0., 0., 1.]])
    for label, basis in bases.items():
        basis = np.array(basis)
        cell = PeriodicCell(basis)
        snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
        # Independent nonpivoted Householder/LAPACK QR, normalized as a pair.
        q, r = np.linalg.qr(basis.T)
        signs = np.sign(np.diag(r))
        q, r = q * signs, signs[:, None] * r
        source_det = determinant(basis)
        for route, rotation, params in [
            ('public', cell._rotation_to_internal(), cell.to_internal_params()),
            ('snapshot', snapshot.rotation_to_internal, snapshot.params),
            ('QR control', q.T, (r[0, 0], r[0, 1], r[1, 1],
                                 r[0, 2], r[1, 2], r[2, 2])),
        ]:
            error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
            volume = F(float(params[0])) * F(float(params[2])) * F(float(params[5]))
            relative_error = abs(volume / source_det - 1)
            # Bounded arithmetic witness, not universal native admission limits.
            record(rows, 'frame ' + label, route + ' (Python/NumPy)', basis,
                   dict(orthogonality_max=error, parameter_volume=volume,
                        relative_volume_error=float(relative_error)),
                   dict(orthogonal=error < 64 * np.finfo(float).eps,
                        volume=float(relative_error) < 1e-6),
                   dict(source_exact_det=source_det))


def native_controls(rows, cell, point):
    from pyvoro2 import _core, _core2d  # Require both when explicitly requested.
    from pyvoro2 import Box
    from pyvoro2.planar import Box as Box2D, compute as compute2d

    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    submitted, shifts = snapshot.remap_internal(point, return_shifts=True)
    raw = _core.compute_periodic_standard(
        submitted, np.array([0], dtype=np.int32), snapshot.params,
        (1, 1, 1), 8, (True, True, True))
    result = compute(point, domain=cell, blocks=(1, 1, 1))
    record(rows, '#65 native', 'compiled 3D standard',
           dict(A=cell.vectors, point=point),
           dict(submitted=submitted, preparation_shift=shifts,
                stored_site=raw[0]['site'], reported_site=result.cells[0]['site'],
                result_source=result.sites),
           dict(physical_x=result.cells[0]['site'][0] == 3/16),
           'x=3/16 survives y snapping; site match is not insertion disposition')
    for dim, run, domain in [(3, compute, Box(((0., 1.),) * 3)),
                             (2, compute2d, Box2D(((0., 1.),) * 2))]:
        source = np.full((1, dim), 1/2)
        result = run(source, domain=domain)
        boundary = 'faces' if dim == 3 else 'edges'
        labels = [item['adjacent_cell'] for item in result.cells[0][boundary]]
        source[:] = 0.
        result.cells[0]['site'][0] = 7/8
        record(rows, 'walls and lifetime', f'compiled {dim}D standard',
               dict(bounds=domain.bounds, source=[[1/2] * dim]),
               dict(labels=labels, sites=result.sites,
                    mutable_raw_site=result.cells[0]['site']),
               dict(walls=all(label < 0 for label in labels),
                    owned_source=np.all(result.sites == 1/2)
                    and not result.sites.flags.writeable,
                    unit_measure=result.cell_measures[0] == 1.),
               'single-site unit box: genuine wall labels, unit measure, '
               'owned source array')
    return [_core.__file__, _core2d.__file__]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    rows = []
    cell, point = source_and_remap(rows)
    endpoints(rows)
    frames(rows)
    modules = native_controls(rows, cell, point) if args.native else []
    report = dict(revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        native_modules=modules, native_skipped=not args.native, cases=rows)
    print(json.dumps(report, indent=2))
    return int(any(not all(row['checks'].values()) for row in rows))


if __name__ == '__main__':
    sys.exit(main())
