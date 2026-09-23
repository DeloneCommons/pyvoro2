"""Exact observation cycles with independent, ordered geometric fixtures."""

from fractions import Fraction as F
import math

import pytest

from pyvoro2._internal.spatial.wp5_common import (
    WP5Budget,
    WP5Failure,
    WP5Limits,
)

from pyvoro2._internal.spatial.wp5_cycle import audit_cycle


SQUARE = ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))
Z_NORMAL = (0, 0, 1)


@pytest.mark.parametrize('cycle', [SQUARE, SQUARE[::-1]])
def test_valid_cycle_preserves_both_orientations_and_immutable_order(cycle):
    result = audit_cycle(cycle, Z_NORMAL, 0)
    assert result == cycle
    assert isinstance(result, tuple)
    assert all(isinstance(point, tuple) for point in result)
    assert all(isinstance(value, F) for point in result for value in point)


def test_original_rank_three_cycle_can_have_a_valid_exact_projection():
    # The original three edge vectors have determinant 16, not zero.
    vertices = ((0, 0, 0), (2, 0, 1), (2, 2, -1), (0, 2, 2))
    assert audit_cycle(vertices, Z_NORMAL, 4) == (
        (0, 0, 4), (2, 0, 4), (2, 2, 4), (0, 2, 4),
    )
    assert vertices == ((0, 0, 0), (2, 0, 1), (2, 2, -1), (0, 2, 2))


@pytest.mark.parametrize('normal, offset', [
    ((1, 1, 1), 1), ((2, 2, 2), 2), ((-1, -1, -1), -1),
])
def test_oblique_support_projection_has_exact_rational_coordinates(
        normal, offset):
    assert audit_cycle(((0, 0, 0), (3, 0, 0), (0, 3, 0)), normal, offset) == (
        (F(1, 3), F(1, 3), F(1, 3)),
        (F(7, 3), F(-2, 3), F(-2, 3)),
        (F(-2, 3), F(7, 3), F(-2, 3)),
    )


def test_consecutive_projected_duplicates_and_closing_pair_are_collapsed():
    vertices = ((0, 0, 1), (0, 0, 2), (2, 0, 3), (2, 2, 4),
                (2, 2, 5), (0, 2, 6), (0, 0, 7), (0, 0, 8))
    assert audit_cycle(vertices, Z_NORMAL, 0) == SQUARE


def test_only_strictly_between_collinear_points_are_removed():
    vertices = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0),
                (2, 2, 0), (0, 2, 0))
    assert audit_cycle(vertices, Z_NORMAL, 0) == SQUARE


def test_redundant_start_point_is_removed_without_reordering():
    vertices = ((1, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 0))
    assert audit_cycle(vertices, Z_NORMAL, 0) == (
        (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 0),
    )


@pytest.mark.parametrize('vertices', [
    (), ((0, 0, 0),), ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
    ((0, 0, 0), (0, 0, 1), (0, 0, 2)),
    ((0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0)),
])
def test_observations_of_affine_dimension_below_two_are_collapsed(vertices):
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(vertices, Z_NORMAL, 0)
    assert raised.value.code == 'WP5_NATIVE_CYCLE_COLLAPSED'


@pytest.mark.parametrize('vertices', [
    # Proper crossing (bow tie).
    ((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)),
    # A concavity without a crossing.
    ((0, 0, 0), (3, 0, 0), (1, 1, 0), (3, 3, 0), (0, 3, 0)),
    # Opposed collinear turn: (3, 0) is not between (0, 0) and (2, 0).
    ((0, 0, 0), (3, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)),
    # Repeated nonconsecutive vertex; do not repair the order.
    ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)),
    # Nonadjacent endpoint touching the interior of the first edge.
    ((0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 0, 0), (0, 4, 0)),
    # Nonadjacent collinear overlapping edges.
    ((0, 0, 0), (4, 0, 0), (4, 3, 0), (1, 0, 0),
     (3, 0, 0), (0, 3, 0)),
    # Pentagram order has one orientation of successive turns but crosses.
    ((0, 3, 0), (2, -3, 0), (-3, 1, 0), (3, 1, 0), (-2, -3, 0)),
])
def test_malformed_order_is_invalid_instead_of_repaired(vertices):
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(vertices, Z_NORMAL, 0)
    assert raised.value.code == 'WP5_NATIVE_CYCLE_INVALID'


def test_nonconsecutive_repeat_cannot_be_removed_as_a_redundant_vertex():
    vertices = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 2, 0),
                (1, 0, 0), (0, 2, 0))
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(vertices, Z_NORMAL, 0)
    assert raised.value.code == 'WP5_NATIVE_CYCLE_INVALID'


def test_tiny_nonzero_projected_area_is_not_collapsed_by_a_tolerance():
    small = float.fromhex('0x0.0000000000001p-1022')
    vertices = ((0, 0, 0), (small, 0, 0), (0, small, 0))
    assert audit_cycle(vertices, Z_NORMAL, 0) == (
        (0, 0, 0), (F(1, 2**1074), 0, 0), (0, F(1, 2**1074), 0),
    )


@pytest.mark.parametrize('vertices, normal, offset', [
    (SQUARE, (0, 0, 0), 0), (SQUARE, (0, 1), 0),
    (SQUARE, (0, 0, math.inf), 0), (SQUARE, Z_NORMAL, math.nan),
    (((0, 0), (1, 0, 0), (0, 1, 0)), Z_NORMAL, 0),
    (((0, 0, 0), (math.inf, 0, 0), (0, 1, 0)), Z_NORMAL, 0),
])
def test_malformed_native_support_or_coordinates_are_source_profile_failures(
        vertices, normal, offset):
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(vertices, normal, offset)
    assert raised.value.code == 'WP5_SOURCE_PROFILE_MISMATCH'


def test_cycle_work_limit_refuses_without_returning_a_partial_cycle():
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(SQUARE, Z_NORMAL, 0,
                    budget=WP5Budget(WP5Limits(work_limit=1)))
    assert raised.value.code == 'WP5_RESOURCE_LIMIT'


def test_projection_bit_limit_refuses_exact_fraction_growth():
    with pytest.raises(WP5Failure) as raised:
        audit_cycle(SQUARE, (1, 1, 16), 0,
                    budget=WP5Budget(WP5Limits(bit_limit=8)))
    assert raised.value.code == 'WP5_RESOURCE_LIMIT'
