"""Reproducible WP4 consumer evidence (also runnable against the WP3 tree).

Run with the selected tree on PYTHONPATH. This is qualification infrastructure,
not package API; instrumentation observes real minimum-image and preparation
calls. No large refused proof box is enumerated merely to obtain a timing.
"""

from dataclasses import asdict
import json
from time import perf_counter
from unittest.mock import patch
import warnings

import numpy as np

import pyvoro2
from pyvoro2._internal import periodic_images as images
from pyvoro2._internal import exact_lattice as exact
from pyvoro2._internal import generator_preparation as prep
from pyvoro2._internal.duplicate_scanning import candidate_pairs, cross_candidate_pairs
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from _lattice_reduction_workload import (
    FROZEN_WORKLOAD_FIXTURES, FROZEN_RANDOM_UNIMODULAR_FIXTURES,
    evaluate_proof_workload,
)


def _max_bits(values):
    return max((abs(value).bit_length() for value in values), default=0)


def measure_minimum(fixture):
    lattice = np.array(fixture.basis)
    rows = 6 if fixture.name == 'thin-1e-3' else 1
    before_formula = evaluate_proof_workload(
        fixture.basis, pi=fixture.pi, pj=fixture.pj, image_search=1,
    )
    images._basis_cache_clear()
    exact._reduction_cache_clear()
    report = dict(source_formula_product=before_formula.box_count,
                  source_formula_widths=before_formula.interval_widths,
                  rows=rows, displacement_evaluations=0, integer_bits=0,
                  proof_widths=[], seed_count=0, seed_truncated=False)
    original_displacement = images._candidate_displacement
    original_prepare = images._prepare_triclinic_row
    original_seed = images._seed_offsets

    def seed(*args, **kwargs):
        offsets, count, truncated = original_seed(*args, **kwargs)
        report['seed_truncated'] |= truncated

        def observed_offsets():
            consumed = 0
            for offset in offsets:
                consumed += 1
                report['seed_count'] += 1
                yield offset
            assert consumed == count

        return observed_offsets(), count, truncated

    def displacement(d, basis, shift):
        result = original_displacement(d, basis, shift)
        # Explicit Cartesian products and partial sums are observed before
        # cancellation; these are sampled semantic operands, not Fraction internals.
        operands = list(d) + list(shift) + list(result)
        for k in range(3):
            total = d[k]
            for j in range(3):
                term = shift[j]*basis[j][k]
                operands += [basis[j][k], term]
                total += term
                operands.append(total)
        operands += [v*v for v in result]
        operands.append(sum(v*v for v in result))
        report['integer_bits'] = max(report['integer_bits'], _max_bits(operands))
        report['displacement_evaluations'] += 1
        return result

    def prepare(*args, **kwargs):
        row = original_prepare(*args, **kwargs)
        report['proof_widths'].append(tuple(
            b-a+1 for a, b in zip(row.lower, row.upper)))
        return row

    start = perf_counter()
    with patch.object(images, '_candidate_displacement', displacement), \
            patch.object(images, '_seed_offsets', seed), \
            patch.object(images, '_prepare_triclinic_row', prepare):
        try:
            result = images.minimum_image_displacements(
                np.tile(fixture.pi, (rows, 1)), np.tile(fixture.pj, (rows, 1)),
                lattice_vectors=lattice, periodic_axes=(True,)*3,
                tie_orientation=[1]*rows, image_search=1,
            )
            report.update(
                status='certified', candidates=result.candidate_count.tolist(),
                user_shift_bits=_max_bits(int(x) for x in result.shift.flat))
            assert report['displacement_evaluations'] == (
                sum(result.candidate_count) + sum(result.seed_count))
        except images.MinimumImageCertificationError as error:
            report.update(status=error.stage, candidate_bound=error.candidate_bound,
                          refusal_widths=error.interval_widths,
                          configured_limit=error.configured_limit)
    report['cold_seconds_instrumented'] = perf_counter()-start
    report['cold_reduction_cache'] = exact._reduction_cache_info()._asdict()
    basis = images._prepare_basis(*images._basis_key(lattice, (True,)*3))
    report['warm_proof_cache'] = images._basis_cache_info()._asdict()
    report['warm_reduction_cache'] = exact._reduction_cache_info()._asdict()
    reduction = getattr(basis, 'reduction', None)
    if reduction is not None:
        report['reduction'] = asdict(reduction.diagnostics)
        report['max_abs_U'] = max(abs(x) for row in reduction.transform for x in row)
        report['max_abs_U_inverse'] = max(
            abs(x) for row in reduction.inverse_transform for x in row)
    displacement, _lattice, exponent = images._aligned_integer_geometry(
        np.array(fixture.pi), np.array(fixture.pj), basis,
    )
    q = images._fractional_coordinates(
        displacement, exponent, basis.inverse,
    )
    observed = [x for row in basis.fractions + basis.inverse for x in row] + list(q)
    report['prepared_rational_bits'] = max(
        max(abs(x.numerator).bit_length(), x.denominator.bit_length())
        for x in observed)
    return report


def measure_preparation():
    # Z^3 in a poor exact representation. All persistent sites are safe.
    # The injected temporary query is close across the physical x seam.
    lattice = ((1., 0., 0.), (float(2**32), 1., 0.), (0., 0., 1.))
    cell = pyvoro2.PeriodicCell(lattice)
    geometry = geometry3d(cell)
    snapshot = geometry.native_periodic_snapshot()
    points = np.array([(k/128, .25, .25) for k in range(128)])
    query = np.array([(1.-2**-20, .25, .25)])
    layout = images._exact_triclinic_bucket_layout(
        np.vstack((points, query)), origin=np.zeros(3),
        lattice_vectors=np.asarray(lattice), radius=1e-5,
    )
    report = dict(bounds=[str(x) for x in layout.coefficient_bounds], bins=layout.bins,
                  unique_keys=len(set(layout.keys)), point_count=129, scans=[])
    report['within_candidates'] = len(list(candidate_pairs(
        points, radius=1e-5, geometry=geometry)))
    report['cross_candidates'] = len(list(cross_candidate_pairs(
        points, query, radius=1e-5, geometry=geometry)))
    original_scan = prep.scan_close_pairs
    original_cross = prep.scan_cross_close_pairs
    original_solve = images._solve_triclinic_plan
    proof_count = 0

    def solve(*args, **kwargs):
        nonlocal proof_count
        result = original_solve(*args, **kwargs)
        proof_count += result.candidate_count
        return result

    def record(call, *args, **kwargs):
        nonlocal proof_count
        proof_count = 0
        try:
            scan = call(*args, **kwargs)
        except images.MinimumImageCertificationError as error:
            report['scans'].append(dict(failure=error.stage,
                                        candidate_bound=error.candidate_bound,
                                        exact_classification_candidates=proof_count))
            raise
        report['scans'].append(dict(
            pairs=scan.pairs, candidate_pairs=scan.candidate_count,
            exact_classification_candidates=proof_count))
        return scan

    policy = dict(duplicate_check='off', duplicate_threshold=1e-5,
                  duplicate_wrap=False, duplicate_max_pairs=10,
                  periodic_snapshot=snapshot, backend_radii=None)
    with patch.object(images, '_solve_triclinic_plan', solve), \
            patch.object(prep, 'scan_close_pairs', lambda *a, **k: record(
                original_scan, *a, **k)), \
            patch.object(prep, 'scan_cross_close_pairs', lambda *a, **k: record(
                original_cross, *a, **k)):
        try:
            persistent = prep.prepare_generators(
                points, geometry=geometry, operation='ghost_cells', external_ids=None,
                **policy,
            )
        except images.MinimumImageCertificationError:
            report['persistent_outcome'] = 'proof-refused'
            # A real successful one-site preparation allows an independent
            # temporary call even when the old sparse-cloud preparation refused.
            persistent = prep.prepare_generators(
                points[:1], geometry=geometry, operation='ghost_cells',
                external_ids=None,
                **policy,
            )
            report['temporary_reference_count'] = 1
        else:
            report['persistent_outcome'] = 'safe'
            report['temporary_reference_count'] = len(points)
        try:
            prep.prepare_temporary_generators(
                query, persistent=persistent, geometry=geometry, **policy,
            )
        except pyvoro2.DuplicateError as error:
            report['temporary_outcome'] = error.kind
    return report


def main():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        report = {
            'source': pyvoro2.__file__,
            'minimum': {f.name: measure_minimum(f) for f in
                        FROZEN_WORKLOAD_FIXTURES + FROZEN_RANDOM_UNIMODULAR_FIXTURES},
            'preparation': measure_preparation(),
        }
    print(json.dumps(report, indent=2, default=lambda x: int(x)))


if __name__ == '__main__':
    main()
