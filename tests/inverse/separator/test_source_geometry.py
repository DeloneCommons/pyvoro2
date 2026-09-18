"""Public separator geometry must use caller endpoints, before backend remap."""

import sys

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell
from pyvoro2.inverse import resolve_separator_observations
from pyvoro2.planar import RectangularCell


@pytest.mark.parametrize('image_search', (0, 1, 4, sys.maxsize))
def test_nearest_image_preserves_source_endpoint_below_backend_snap(
    image_search,
):
    epsilon = 2.0**-41
    points = np.array([[epsilon, 0.25, 0.25], [0.5, 0.25, 0.25]])
    observations = resolve_separator_observations(
        points, [(0, 1, 0.5)], domain=PeriodicCell(np.eye(3)),
        image='nearest', image_search=image_search,
    )

    # The alternative image is strictly farther: squared-distance gap 2**-40.
    np.testing.assert_array_equal(observations.shifts, [[0, 0, 0]])
    np.testing.assert_array_equal(observations.delta, [[0.5 - epsilon, 0, 0]])
    assert observations.distance[0] == 0.5 - epsilon
    assert observations.distance2[0] == (0.5 - epsilon)**2
    np.testing.assert_array_equal(points[:, 0], [epsilon, 0.5])


@pytest.mark.parametrize(('basis', 'shift'), (
    (np.eye(3), (-1, -1, 0)),
    (np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]), (-1, 0, 0)),
    (np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]]), (1, 0, 0)),
))
@pytest.mark.parametrize('image_search', (0, 1, 4))
def test_equivalent_user_bases_select_same_physical_tie(basis, shift, image_search):
    points = np.array([[0, 0, 0], [0.5, 0.5, 0]])
    observations = resolve_separator_observations(
        points, [(0, 1, 0.5), (1, 0, 0.5)],
        domain=PeriodicCell(basis), image_search=image_search,
    )

    # All three exact integer bases generate Z^3. The four minimizers are
    # (+/-0.5, +/-0.5, 0); forward orientation selects the Cartesian minimum.
    np.testing.assert_array_equal(observations.shifts[0], shift)
    np.testing.assert_array_equal(observations.shifts[1], -np.array(shift))
    np.testing.assert_array_equal(observations.delta, [
        [-0.5, -0.5, 0], [0.5, 0.5, 0],
    ])
    np.testing.assert_array_equal(observations.distance2, [0.5, 0.5])
    np.testing.assert_array_equal(
        points[1] + observations.shifts[0] @ basis - points[0],
        observations.delta[0],
    )


@pytest.mark.parametrize(('translate_i', 'translate_j'), (
    ((0, 0, 0), (2, -3, 1)),
    ((2, -3, 1), (0, 0, 0)),
    ((2, -3, 1), (2, -3, 1)),
    ((1, 0, -2), (-2, 3, 1)),
))
def test_source_endpoint_translation_covariance(translate_i, translate_j):
    basis = np.array([[2, 0.5, 0], [0, 1, 0.25], [0, 0, 1]])
    cell = PeriodicCell(basis, origin=(0.25, -0.5, 0.125))
    points = np.array([[0.125, 0.25, 0.375], [2, 0.875, 0.5]])
    original = resolve_separator_observations(points, [(0, 1, 0.5)], domain=cell)
    translated = points + np.array([translate_i, translate_j]) @ basis
    observations = resolve_separator_observations(
        translated, [(0, 1, 0.5)], domain=cell,
    )

    # Every coordinate and translation is dyadic and exactly representable.
    # The residual (-1/8, 1/8, 1/8) is uniquely nearest: every nonzero
    # lattice vector has length >= 1, exceeding twice the residual's norm.
    np.testing.assert_array_equal(original.shifts, [[-1, 0, 0]])
    np.testing.assert_array_equal(original.delta, [[-0.125, 0.125, 0.125]])
    np.testing.assert_array_equal(
        observations.shifts,
        original.shifts + np.array(translate_i) - np.array(translate_j),
    )
    np.testing.assert_array_equal(observations.delta, original.delta)
    np.testing.assert_array_equal(observations.distance, original.distance)
    np.testing.assert_array_equal(observations.distance2, original.distance2)


@pytest.mark.parametrize('image', ('nearest', 'given_only'))
def test_explicit_shifts_use_source_endpoints_including_mixed_rows(image):
    epsilon = 2.0**-41
    points = np.array([[epsilon, 0.25, 0.25], [1.5, 0.25, 0.25]])
    rows = [(0, 1, 0.5, (0, 0, 0)), (0, 1, 0.5, (-2, 0, 0))]
    expected_shifts = [[0, 0, 0], [-2, 0, 0]]
    expected_delta = [[1.5 - epsilon, 0, 0], [-0.5 - epsilon, 0, 0]]
    if image == 'nearest':
        rows.append((0, 1, 0.5))
        expected_shifts.append([-1, 0, 0])
        expected_delta.append([0.5 - epsilon, 0, 0])
    observations = resolve_separator_observations(
        points, rows, domain=PeriodicCell(np.eye(3)), image=image,
    )

    np.testing.assert_array_equal(observations.shifts, expected_shifts)
    np.testing.assert_array_equal(observations.delta, expected_delta)
    np.testing.assert_array_equal(
        observations.explicit_shift, [len(row) == 4 for row in rows],
    )
    np.testing.assert_array_equal(
        observations.distance, np.abs(np.array(expected_delta)[:, 0]),
    )


@pytest.mark.parametrize('dimension', (2, 3))
@pytest.mark.parametrize('partial', (False, True))
def test_rectangular_inference_uses_unwrapped_source_points(dimension, partial):
    periodic = (True,) * (dimension - 1) + (not partial,)
    domain_type = RectangularCell if dimension == 2 else OrthorhombicCell
    domain = domain_type(((0, 1),) * dimension, periodic=periodic)
    points = np.array([[0.125] * dimension, [0.875] * dimension])
    points[1, 0] += 2
    observations = resolve_separator_observations(points, [(0, 1, 0.5)], domain=domain)

    expected_shift = [-3] + [-1] * (dimension - 1)
    expected_delta = [-0.25] * dimension
    if partial:
        expected_shift[-1] = 0
        expected_delta[-1] = 0.75
    np.testing.assert_array_equal(observations.shifts, [expected_shift])
    np.testing.assert_array_equal(observations.delta, [expected_delta])
