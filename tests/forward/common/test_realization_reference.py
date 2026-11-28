"""Exact membership in the represented native lattice, independent of Voro++."""
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest


def _capture(submitted, anchors, *, lattice=None, shifts=None, axes=None):
    from pyvoro2._internal.realization_reference import capture_reference

    submitted = np.array(submitted, dtype=float)
    n, dim = submitted.shape
    prepared = SimpleNamespace(
        native_points=submitted,
        remap_shifts=np.zeros((n, dim), dtype=np.int64) if shifts is None else shifts,
        periodic_axes=(True,) * dim if axes is None else axes,
    )
    cells = [{'id': i, 'site': a} for i, a in enumerate(anchors) if a is not None]
    # Cell ordering is not the site-ID association.
    return capture_reference(
        prepared, cells[::-1], lattice=np.eye(dim) if lattice is None else lattice)


@pytest.mark.parametrize('anchors,offset', [
    ([[.125, .25], [.375, .25]], (-3, 0)),
    ([[.125, .25], [-.625, .25]], (-4, 0)),
    ([[2.125, -1.75], [2.375, -1.75]], (-3, 0)),
])
def test_relative_membership_and_reversal(anchors, offset):
    ref = _capture([[.125, .25], [.375, .25]], anchors,
                   shifts=np.array([[0, 0], [3, 0]], dtype=np.int64))
    assert ref.shift_offset(0, 1) == offset
    assert ref.shift_offset(1, 0) == tuple(-v for v in offset)


def test_triangular_relative_membership_is_exact():
    lattice = np.array([[1., 0., 0.], [.25, 2., 0.], [-.5, .125, 3.]])
    submitted = np.array([[.125, .25, .5], [.375, .5, .75]])
    # h=(2,-3,4), evaluated by hand: h@L=(-.75,-5.5,12).
    anchors = submitted.copy()
    anchors[1] -= [-.75, -5.5, 12.]
    ref = _capture(submitted, anchors, lattice=lattice)
    assert ref.shift_offset(0, 1) == (-2, 3, -4)
    assert ref.shift_offset(1, 0) == (2, -3, 4)


def test_endpoint_transition_fraction_identity():
    period = 105/1024
    x = np.nextafter(period, 0.)
    anchor = -2.**-56
    assert F(x) - F(anchor) == F(period)
    ref = _capture([[0., .25], [x, .25]], [[0., .25], [anchor, .25]],
                   lattice=np.diag([period, 1.]))
    assert ref.shift_offset(0, 1) == (-1, 0)


@pytest.mark.parametrize('anchors,axes,reason', [
    ([[.125, .25], [.5, .25]], (True, True), 'nonintegral'),
    ([[.125, .25], [.375, 1.25]], (True, False), 'inactive'),
    ([[.125, .25], None], (True, True), 'anchor'),
])
def test_missing_or_incompatible_observed_reference_refuses(anchors, axes, reason):
    ref = _capture([[.125, .25], [.375, .25]], anchors, axes=axes)
    with pytest.raises(ValueError, match=rf'\(0, 1\).*{reason}'):
        ref.shift_offset(0, 1)


def test_rounded_shear_transition_is_not_attributed_by_tolerance():
    shear = float(1/3)
    lattice = [[1., 0., 0.], [shear, 1., 0.], [0., 0., 1.]]
    submitted = np.array([[.125, .25, .25], [.375, .5, .25]])
    anchors = submitted.copy()
    anchors[1] -= lattice[1]
    # Rounded subtraction need not be an exact represented lattice translation.
    assert F(submitted[1, 0]) - F(anchors[1, 0]) == F(shear)
    anchors[1, 0] = np.nextafter(anchors[1, 0], np.inf)
    ref = _capture(submitted, anchors, lattice=lattice)
    with pytest.raises(ValueError, match='nonintegral'):
        ref.shift_offset(0, 1)


def test_coefficients_stay_wide_after_preparation_and_anchor_transport():
    maximum = 2**63 - 1
    ref = _capture([[.125, .25], [.375, .25]],
                   [[.125, .25], [-.625, .25]],
                   shifts=np.array([[-maximum, 0], [maximum, 0]], dtype=np.int64))
    assert ref.shift_offset(0, 1) == (-2 * maximum - 1, 0)
    assert ref.shift_offset(1, 0) == (2 * maximum + 1, 0)
    assert ref.convert_shift(0, 1, (2**80, 0), offset=ref.shift_offset(0, 1)) == (
        2**80 - 2 * maximum - 1, 0)


def test_capture_owns_only_real_anchors_and_detaches_all_arrays():
    from pyvoro2._internal.realization_reference import capture_reference

    points = np.array([[.125, .25], [.375, .25]])
    shifts = np.array([[0, 0], [3, 0]], dtype=np.int64)
    lattice = np.eye(2)
    cells = [{'id': 0, 'site': [.125, .25]},
             {'id': 1, 'site': [.375, .25], 'empty': True}]
    ref = capture_reference(SimpleNamespace(
        native_points=points, remap_shifts=shifts, periodic_axes=(True, True)),
        cells, lattice=lattice)
    assert ref.anchor_present.tolist() == [True, False]
    points[:] = 0
    shifts[:] = 0
    lattice[:] = 0
    cells[0]['site'][0] = 9
    assert ref.submitted[0, 0] == .125
    assert ref.preparation_shifts[1, 0] == 3
    assert ref.lattice[0, 0] == 1
    assert ref.anchors[0, 0] == .125
    for array in (ref.submitted, ref.preparation_shifts, ref.lattice,
                  ref.anchors, ref.anchor_present):
        assert array.flags.owndata
        assert not array.flags.writeable
