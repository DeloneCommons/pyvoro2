"""Compiled matching controls in the source-indexed numerical reference."""
from dataclasses import replace

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell
from pyvoro2.inverse.separator import (
    ActiveSetOptions, match_realized_pairs, resolve_separator_observations,
    solve_self_consistent_power_weights,
)
from pyvoro2.planar import RectangularCell


def _domain(kind):
    if kind == 'planar':
        return RectangularCell(((0., 1.),) * 2, periodic=(True, False)), 2
    if kind == 'orthorhombic':
        return OrthorhombicCell(
            ((0., 1.),) * 3, periodic=(True, False, False)), 3
    return PeriodicCell(np.eye(3)), 3


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic', 'triclinic'])
@pytest.mark.parametrize('lift', [0, 3])
@pytest.mark.parametrize('explicit', [False, True])
def test_translated_compiled_matching_and_raw_labels(kind, lift, explicit):
    domain, dim = _domain(kind)
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, lift + .375]
    forward = (-lift,) + (0,) * (dim - 1)
    reverse = tuple(-v for v in forward)
    other = (-lift - 1,) + (0,) * (dim - 1)
    absent = (-lift - 2,) + (0,) * (dim - 1)
    rows = ([(0, 1, .5, forward), (1, 0, .5, reverse)] if explicit else
            [(0, 1, .5), (1, 0, .5)])
    rows += [(0, 1, .5, other), (0, 1, .5, absent)]
    obs = resolve_separator_observations(points, rows, domain=domain)
    result = match_realized_pairs(
        points, domain=domain, constraints=obs, weights=np.zeros(2),
        return_cells=True, return_boundary_measure=True,
        return_tessellation_diagnostics=True)
    np.testing.assert_array_equal(result.realized_same_shift,
                                  [True, True, True, False])
    np.testing.assert_array_equal(result.realized_other_shift,
                                  [False, False, False, True])
    assert result.realized_shifts[0] == tuple(sorted([forward, other]))
    assert result.realized_shifts[1] == tuple(sorted(
        [tuple(-v for v in s) for s in [forward, other]]))
    np.testing.assert_allclose(result.boundary_measure, 1., atol=1e-14)
    assert result.unaccounted_pairs == ()
    assert result.tessellation_diagnostics is not None
    boundaries = 'edges' if dim == 2 else 'faces'
    cell = next(c for c in result.cells if c['id'] == 0)
    raw_labels = {tuple(b['adjacent_shift']) for b in cell[boundaries]
                  if b['adjacent_cell'] == 1}
    assert raw_labels == {(0,) * dim, (-1,) + (0,) * (dim - 1)}
    assert obs.shifts[0].tolist() == list(forward)


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic', 'triclinic'])
def test_unaccounted_pair_labels_and_measures_are_transported(kind):
    domain, dim = _domain(kind)
    points = np.full((3, dim), .25)
    points[:, 0] = [.125, 3.375, 2.75]
    obs = resolve_separator_observations(points, [(0, 1, .5)], domain=domain)
    result = match_realized_pairs(
        points, domain=domain, constraints=obs, weights=np.zeros(3),
        return_boundary_measure=True)
    pairs = {(p.site_i, p.site_j): p for p in result.unaccounted_pairs}
    assert pairs[(0, 2)].realized_shifts == ((-3,) + (0,) * (dim - 1),)
    assert pairs[(1, 2)].realized_shifts == ((1,) + (0,) * (dim - 1),)
    assert pairs[(0, 2)].boundary_measure == pytest.approx(1.)
    assert pairs[(1, 2)].boundary_measure == pytest.approx(1.)


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic', 'triclinic'])
def test_translated_compiled_active_state_retains_requested_image(kind):
    domain, dim = _domain(kind)
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, 3.375]
    result = solve_self_consistent_power_weights(
        points, [(0, 1, .5, (-3,) + (0,) * (dim - 1))], domain=domain,
        options=ActiveSetOptions(max_iter=3), return_history=True,
        return_boundary_measure=True, return_cells=True)
    assert result.termination == 'self_consistent'
    assert result.active_mask.tolist() == [True]
    assert result.realized.realized_same_shift.tolist() == [True]
    assert result.constraints.shifts[0, 0] == -3
    assert result.realized.boundary_measure[0] == pytest.approx(1.)


def _corrupt_reference(cells, ref, corruption, boundary_key):
    if corruption == 'missing_shift':
        for cell in cells:
            for boundary in cell[boundary_key]:
                if cell['id'] == 0 and boundary['adjacent_cell'] == 1:
                    del boundary['adjacent_shift']
    elif corruption == 'inactive_shift':
        for cell in cells:
            for boundary in cell[boundary_key]:
                if cell['id'] == 0 and boundary['adjacent_cell'] == 1:
                    boundary['adjacent_shift'] = (
                        [0, 1] + [0] * (ref.submitted.shape[1] - 2))
    elif corruption == 'missing_anchor':
        present = ref.anchor_present.copy()
        present[1] = False
        ref = replace(ref, anchor_present=present)
    else:
        anchors = ref.anchors.copy()
        anchors[1, 0] += .125
        ref = replace(ref, anchors=anchors)
    return ref


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic'])
@pytest.mark.parametrize('corruption,reason', [
    ('nonintegral', 'nonintegral'), ('missing_anchor', 'anchor'),
    ('missing_shift', 'adjacent_shift'), ('inactive_shift', 'inactive'),
])
def test_matching_fails_instead_of_classifying_incompatible_evidence(
    monkeypatch, kind, corruption, reason,
):
    import pyvoro2.inverse.separator.realize as realize

    domain, dim = _domain(kind)
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, 3.375]
    obs = resolve_separator_observations(points, [(0, 1, .5)], domain=domain)
    name = '_compute_planar_cells' if dim == 2 else '_compute_3d_cells'
    actual = getattr(realize, name)

    def compute(*args, **kwargs):
        cells, diag, periodic, ref = actual(*args, **kwargs)
        ref = _corrupt_reference(
            cells, ref, corruption, 'edges' if dim == 2 else 'faces')
        return cells, diag, periodic, ref

    monkeypatch.setattr(realize, name, compute)
    with pytest.raises(ValueError, match=rf'observed pair \([01], [01]\).*{reason}'):
        match_realized_pairs(points, domain=domain, constraints=obs, weights=[0., 0.])


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic'])
@pytest.mark.parametrize('fail_on_call', [1, 2])
def test_active_correspondence_failure_precedes_history_or_final_publication(
    monkeypatch, kind, fail_on_call,
):
    import pyvoro2.inverse.separator.active as active
    import pyvoro2.inverse.separator.realize as realize

    domain, dim = _domain(kind)
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, 3.375]
    name = '_compute_planar_cells' if dim == 2 else '_compute_3d_cells'
    actual = getattr(realize, name)
    count = 0
    recorded = []
    record = active._record_path_iteration

    def compute(*args, **kwargs):
        nonlocal count
        count += 1
        cells, diag, periodic, ref = actual(*args, **kwargs)
        if count == fail_on_call:
            ref = _corrupt_reference(cells, ref, 'nonintegral',
                                     'edges' if dim == 2 else 'faces')
        return cells, diag, periodic, ref

    def record_iteration(*args, **kwargs):
        recorded.append(kwargs['iteration'])
        return record(*args, **kwargs)

    def forbidden_publication(self):
        pytest.fail('a correspondence failure cannot publish an active result')

    monkeypatch.setattr(realize, name, compute)
    monkeypatch.setattr(active, '_record_path_iteration', record_iteration)
    monkeypatch.setattr(
        active._AcceptedActiveSetState, 'to_result', forbidden_publication)
    initial = np.array([True])
    with pytest.raises(ValueError, match='observed pair .*nonintegral'):
        solve_self_consistent_power_weights(
            points, [(0, 1, .5)], domain=domain, active0=initial,
            options=ActiveSetOptions(max_iter=3), return_history=True)
    assert initial.tolist() == [True]
    assert count == fail_on_call
    assert recorded == ([] if fail_on_call == 1 else [1])


def test_wide_matching_maps_preserve_measure_keys_and_reverse_union():
    from types import SimpleNamespace
    from pyvoro2._internal.realization_reference import capture_reference
    from pyvoro2.inverse.separator.realize import _collect_boundary_maps

    maximum = 2**63 - 1
    points = np.array([[.125, .25], [.375, .25]])
    cells = [
        {'id': 0, 'site': points[0], 'vertices': [[0., 0.], [0., 1.]],
         'edges': [{'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'length': 1.}]},
        {'id': 1, 'site': points[1], 'vertices': [[0., 0.], [0., 1.]],
         'edges': [{'adjacent_cell': 0, 'adjacent_shift': (0, 0), 'length': 1.}]},
    ]
    ref = capture_reference(SimpleNamespace(
        native_points=points,
        remap_shifts=np.array([[-maximum, 0], [maximum, 0]], dtype=np.int64),
        periodic_axes=(True, False)), cells, lattice=np.eye(2))
    _, shifts, measures = _collect_boundary_maps(
        cells, boundary_key='edges', shift_dim=2, return_boundary_measure=True,
        measure_field='length', periodic=True, reference=ref)
    assert shifts[(0, 1)] == {(-2 * maximum, 0)}
    assert shifts[(1, 0)] == {(2 * maximum, 0)}
    assert measures[(0, 1, (-2 * maximum, 0))] == 1.
    assert measures[(1, 0, (2 * maximum, 0))] == 1.


def test_empty_boundary_maps_do_not_demand_missing_evidence():
    from pyvoro2.inverse.separator.realize import _collect_boundary_maps

    empty, shifts, measures = _collect_boundary_maps(
        [{'id': 0, 'empty': True, 'vertices': [], 'edges': []}],
        boundary_key='edges', shift_dim=2, return_boundary_measure=True,
        measure_field='length', periodic=True, reference=None)
    assert empty == {0: True}
    assert shifts == measures == {}
