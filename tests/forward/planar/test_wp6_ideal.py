"""Exact expectations are hand derived or supplied by an independent oracle."""

from fractions import Fraction as F
import importlib
import math

import numpy as np
import pytest

from wp6_interval_oracle import oracle_cell


UNIT = ((0, 1), (0, 1))


@pytest.fixture
def geometry():
    name = 'pyvoro2._internal.planar.wp6_ideal'
    assert importlib.util.find_spec(name) is not None, 'WP6 ideal is not implemented'
    return importlib.import_module(name)


def test_one_site_axial_edges_diagonal_points_and_absent_far_images(geometry):
    cell, = geometry.ideal_cells([(F(1, 4), F(3, 4))], [0], UNIT,
                                 (1, 1), (True, True))
    assert cell.dimension == 2
    assert cell.area == 1
    assert set(cell.vertices) == {(F(x, 2), F(y, 2))
                                  for x in (-1, 1) for y in (-1, 1)}
    axial = ((-1, 0), (1, 0), (0, -1), (0, 1))
    assert set(cell.positive) == {(0, s) for s in axial}
    for contact in cell.positive.values():
        assert contact.length_squared == 1
        assert len(contact.endpoints) == 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            contact = cell.contact(0, (sx, sy))
            assert contact.status == 'point'
            assert contact.endpoints == ((F(sx, 2), F(sy, 2)),)
    assert cell.contact(0, (2, 0)).status == 'absent'
    assert cell.contact(0, (0, 0)).status == 'absent'
    assert cell.wall(-1).status == 'absent'


def test_two_site_periodic_diamond_has_four_other_owner_labels(geometry):
    cell, other = geometry.ideal_cells(
        [(F(1, 4), F(1, 4)), (F(3, 4), F(3, 4))], [0, 0],
        UNIT, (1, 1), (True, True),
    )
    assert cell.area == other.area == F(1, 2)
    assert set(cell.vertices) == {(F(1, 2), 0), (F(-1, 2), 0),
                                  (0, F(1, 2)), (0, F(-1, 2))}
    assert set(cell.positive) == {(1, (x, y)) for x in (-1, 0) for y in (-1, 0)}
    assert all(c.length_squared == F(1, 2) for c in cell.positive.values())
    assert cell.contact(0, (1, 0)).status == 'point'


def test_coincident_wall_generator_labels_and_lower_dimensional_owner(geometry):
    cells = geometry.ideal_cells(
        [(F(1, 4), F(1, 2)), (F(3, 4), F(1, 2))],
        [F(3, 4) ** 2, F(1, 4) ** 2], UNIT, (1, 1), (False, True),
    )
    wall = cells[0].wall(-2)
    generator = cells[0].contact(1, (0, 0))
    assert wall.status == generator.status == 'positive'
    expected = ((F(3, 4), F(-1, 2)), (F(3, 4), F(1, 2)))
    assert wall.endpoints == generator.endpoints == expected
    assert -2 in cells[0].positive and (1, (0, 0)) in cells[0].positive
    assert cells[1].dimension == 1
    assert cells[1].contact(0, (0, 0)).status == 'lower-dimensional'
    assert cells[1].contact(0, (0, 0)).length_squared == 1
    assert not cells[1].positive


def test_point_empty_and_identical_function_are_distinct(geometry):
    cells = geometry.ideal_cells([(F(1, 4), F(1, 4)), (F(3, 4), F(3, 4))],
                                 [1, 0], UNIT, (1, 1), (False, False))
    assert cells[0].contact(1, (0, 0)).status == 'point'
    assert cells[1].dimension == 0
    assert cells[1].contact(0, (0, 0)).status == 'lower-dimensional'
    assert not cells[1].positive
    cells = geometry.ideal_cells([(F(1, 2), F(1, 2))] * 3, [0, 0, 1],
                                 UNIT, (1, 1), (False, False))
    assert [c.dimension for c in cells] == [-1, -1, 2]
    assert cells[0].contact(1, (0, 0)).status == 'identical'
    assert cells[0].contact(1, (0, 0)).dimension == -1
    assert cells[0].contact(2, (0, 0)).status == 'absent'
    assert cells[2].contact(0, (0, 0)).status == 'absent'


def test_nondyadic_vertices_are_exact(geometry):
    points = [(F(1, 8), F(1, 8)), (F(5, 8), F(1, 8)), (F(3, 8), F(7, 8))]
    cells = geometry.ideal_cells(points, [0, 0, 0], UNIT, (1, 1), (False, False))
    assert set(cells[0].vertices) == {
        (F(-1, 8), F(-1, 8)), (F(1, 4), F(-1, 8)),
        (F(1, 4), F(1, 3)), (F(-1, 8), F(11, 24)),
    }


def test_supplied_binary64_period_is_not_exact_endpoint_subtraction(geometry):
    bounds = ((0.1, 1.0), (0, 1))
    period = 1.0 - 0.1
    cell, = geometry.ideal_cells([(0.5, 0.5)], [0], bounds,
                                 (period, 1), (True, False))
    assert cell.area == F(period)
    assert cell.area != F(1.0) - F(0.1)


def test_exactified_radius_square_is_not_a_rounded_float_square(geometry):
    # The square of represented 0.1, independently recorded by the reference.
    squared_radius = F(12980742146337070512478121581609,
                       1298074214633706907132624082305024)
    assert squared_radius != F(0.1 ** 2)
    cells = geometry.ideal_cells([(0, 0), (1, 0)], [squared_radius, 0],
                                 ((0, 2), (0, 1)), (2, 1), (False, False))
    separator = (1 + squared_radius) / 2
    assert cells[0].contact(1, (0, 0)).endpoints == ((separator, 0), (separator, 1))


def test_weights_and_backend_radius_squares_have_distinct_exact_geometry(geometry):
    points = [(2, F(1, 4)), (F(1, 2), 2), (F(7, 2), 2)]
    bounds = ((0, 4), (0, 4))
    semantic = geometry.ideal_cells(points, [-(2 ** 54), 1, 2], bounds,
                                    (4, 4), (True, True))
    effective = geometry.ideal_cells(points, [0, 2 ** 54, 2 ** 54], bounds,
                                     (4, 4), (True, True))
    # The source1/source2 distances are 3 and -1. Their unit weight
    # difference puts global boundaries at 2-1/6 and 0+1/2.
    assert set(semantic[1].vertices) == {
        (0, -2), (0, 2), (F(4, 3), -2), (F(4, 3), 2),
    }
    assert semantic[1].area == F(16, 3)
    assert set(effective[1].vertices) == {
        (F(-1, 2), -2), (F(-1, 2), 2), (F(3, 2), -2), (F(3, 2), 2),
    }
    assert effective[1].area == 8


@pytest.mark.parametrize(
    'periodic', [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize('weights', [[0, 0, 0], [F(1, 16), F(-1, 8), F(1, 4)]])
def test_all_masks_against_independent_larger_interval_family(
    geometry, periodic, weights,
):
    points = [(F(1, 8), F(1, 4)), (F(3, 4), F(7, 8)), (F(1, 2), F(1, 8))]
    cells = geometry.ideal_cells(points, weights, UNIT, (1, 1), periodic)
    for source, cell in enumerate(cells):
        rank, vertices, contacts = oracle_cell(points, weights, UNIT, (1, 1),
                                               periodic, source)
        assert cell.dimension == rank
        assert set(cell.vertices) == vertices
        assert set(cell.positive) == {
            label for label, (status, _) in contacts.items() if status == 'positive'
        }
        for label, (status, endpoints) in contacts.items():
            got = cell.wall(label) if isinstance(label, int) else cell.contact(*label)
            assert (got.status, got.endpoints) == (status, endpoints)


def test_permutations_and_large_representative_translations_preserve_labels(geometry):
    points = [(F(1, 4), F(1, 2)), (F(3, 4), F(1, 2))]
    moved = [(F(3, 4) + 100, F(1, 2) - 200), points[0]]
    before = geometry.ideal_cells(points, [0, 0], UNIT, (1, 1), (True, True))
    after = geometry.ideal_cells(moved, [0, 0], UNIT, (1, 1), (True, True))
    assert after[1].vertices == before[0].vertices
    assert after[1].contact(0, (-100, 200)) == before[0].contact(1, (0, 0))
    assert after[1].contact(0, (-101, 200)) == before[0].contact(1, (-1, 0))
    assert after[1].contact(0, (0, 0)).status == 'absent'
    far = [(2 ** 80 + F(3, 4), F(1, 2)), points[0]]
    large = geometry.ExactIdeal(far, [0, 0], UNIT, (1, 1), (True, True)).cell(1)
    assert large.contact(0, (-(2 ** 80), 0)).status == 'positive'


def test_reciprocal_segments_agree_after_exact_image_translation(geometry):
    points = [(F(1, 8), F(1, 4)), (F(3, 4), F(7, 8)), (F(1, 2), F(1, 8))]
    cells = geometry.ideal_cells(points, [0, 0, 0], UNIT, (1, 1), (True, True))
    for source, cell in enumerate(cells):
        for (owner, shift), contact in cell.positive.items():
            reciprocal = cells[owner].contact(source, tuple(-v for v in shift))
            d = tuple(points[owner][k] + shift[k] - points[source][k] for k in range(2))
            translated = tuple(sorted((p[0] - d[0], p[1] - d[1])
                                      for p in contact.endpoints))
            assert reciprocal.status == 'positive'
            assert reciprocal.endpoints == translated
            assert reciprocal.length_squared == contact.length_squared


def test_resource_budget_preflights_before_image_enumeration(geometry):
    budget = geometry.ExactAuditBudget(max_candidates=7)
    ideal = geometry.ExactIdeal([(F(1, 2), F(1, 2))], [0], UNIT,
                                (1, 1), (True, True), budget=budget)
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        ideal.cell(0)
    assert caught.value.reason == 'resource'
    assert caught.value.resource == 'candidates'
    assert budget.candidate_count == 0
    assert budget.work == 0


def test_all_source_family_is_preflighted_before_a_successful_prefix(geometry):
    budget = geometry.ExactAuditBudget(max_candidates=33)
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        geometry.ideal_cells([(F(1, 4), F(1, 4)), (F(3, 4), F(3, 4))],
                             [0, 0], UNIT, (1, 1), (True, True), budget=budget)
    assert caught.value.observed == 34
    assert budget.candidate_count == budget.work == 0


def test_numpy_scalars_do_not_introduce_fixed_width_exact_arithmetic(geometry):
    points = np.array([[0.25, 0.25], [0.75, 0.75]])
    weights = np.array([-(2 ** 63), 2 ** 63 - 1], dtype=np.int64)
    periodic = np.array([True, True], dtype=np.bool_)
    cells = geometry.ideal_cells(points, weights, UNIT, np.ones(2), periodic)
    assert cells[0].dimension == -1
    assert cells[1].area == 1


def test_exact_bit_guard_and_work_refuse_explicitly(geometry):
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        geometry.ideal_cells([(F(1, 2 ** 200), 0)], [0], UNIT, (1, 1),
                             (False, False),
                             budget=geometry.ExactAuditBudget(max_bits=64))
    assert caught.value.reason == 'resource'
    assert caught.value.resource == 'bits'
    budget = geometry.ExactAuditBudget(max_work=5)
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        geometry.ideal_cells([(F(1, 2), F(1, 2))], [0], UNIT, (1, 1),
                             (True, True), budget=budget)
    assert caught.value.resource == 'work'


def test_checked_sqrt_handles_squared_overflow_underflow_and_rounding(geometry):
    length = geometry.length_from_squared
    assert length(F(2)) == math.sqrt(2)
    assert length(F(2) ** 2046) == math.ldexp(1.0, 1023)
    assert length(F(2) ** -2148) == math.ldexp(1.0, -1074)
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        length(F(2) ** -2150)
    assert caught.value.reason == 'representation'
    with pytest.raises(geometry.ExactAuditRefusal):
        length(F(2) ** 2048)
    assert length(0) == 0


def test_exact_sqrt_rounds_midpoints_to_even_including_subnormal_boundary(geometry):
    length = geometry.length_from_squared
    assert length((1 + F(2) ** -53) ** 2) == 1.0
    assert length((1 + 3 * F(2) ** -53) ** 2) == 1.0 + 2.0 ** -51
    assert length((3 * F(2) ** -1075) ** 2) == math.ldexp(1.0, -1073)
    largest = F(float.fromhex('0x1.fffffffffffffp1023'))
    assert length((largest + F(2) ** 969) ** 2) == float(largest)
    with pytest.raises(geometry.ExactAuditRefusal) as caught:
        length((largest + F(2) ** 970) ** 2)
    assert caught.value.reason == 'representation'


def test_empty_system_and_invalid_geometry(geometry):
    assert geometry.ideal_cells([], [], UNIT, (1, 1), (True, True)) == ()
    for bad in (math.inf, math.nan, True):
        with pytest.raises((ValueError, TypeError)):
            geometry.ideal_cells([(bad, 0)], [0], UNIT, (1, 1), (False, False))
    with pytest.raises(ValueError):
        geometry.ideal_cells([(0, 0)], [0], UNIT, (0, 1), (True, False))
