"""Source-derived image identities, independent of ideal face positivity."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math

import numpy as np
import pytest

from pyvoro2 import _core


def _observe(points, *, blocks=(2, 2, 2), params=None, radii=None):
    points = np.asarray(points, dtype=np.float64)
    ids = np.arange(len(points), dtype=np.int32)
    if params is None:
        packet = _core._observe_box(
            points, ids, ((0.0, 4.0),) * 3, blocks,
            (True, True, True), 8, radii=radii,
        )
    else:
        packet = _core._observe_periodic(
            points, ids, params, blocks, 8, radii=radii,
        )
    return packet, points, ids


def test_packet_has_actual_rectangular_insertion_and_grid_operands():
    packet, _, _ = _observe([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
    context = packet['context']
    assert context['block_widths'] == [2.0, 2.0, 2.0]
    assert context['block_reciprocals'] == [0.5, 0.5, 0.5]
    assert context['mask_shape'] == [5, 5, 5]
    assert packet['sites'][0]['block'] == [1, 1, 1]
    assert packet['sites'][0]['block_index'] == 7
    assert packet['sites'][1]['block'] == [0, 0, 0]
    assert packet['sites'][1]['block_index'] == 0
    assert [row['block_slot'] for row in packet['sites']] == [0, 0]
    assert packet['build']['int_bits'] == 32
    assert packet['build']['int_min'] == -2147483648
    assert packet['build']['int_max'] == 2147483647


def test_packet_retains_actual_triclinic_primary_grid_coordinates():
    packet, _, _ = _observe(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        params=(4.0, 0.0, 4.0, 0.0, 0.0, 4.0),
    )
    assert packet['context']['image_grid'] == {
        'ey': 3, 'ez': 3, 'wy': 5, 'wz': 5,
        'oy': 8, 'oz': 8, 'oxyz': 128,
    }
    assert packet['context']['mask_shape'] == [5, 7, 7]
    assert packet['sites'][0]['block'] == [0, 3, 3]
    assert packet['sites'][0]['block_index'] == 54
    assert packet['sites'][1]['block'] == [1, 4, 4]
    assert packet['sites'][1]['block_index'] == 73


def _producer(packet, points, ids, radii=None, budget=None):
    from pyvoro2._internal.spatial.wp5_producer import Producer
    return Producer(packet, points, ids, radii, budget=budget)


def test_rectangular_particle_occurrences_keep_both_periodic_directions():
    packet, points, ids = _observe([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    producer = _producer(packet, points, ids)
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    labels = {}
    for origin in cell['origins']:
        if origin['kind'] == 'particle' and origin['owner'] == 1:
            result = producer.attribute(cell, origin)
            labels[tuple(origin['normal'])] = result.shift
    assert labels[2.0, 0.0, 0.0] == (0, 0, 0)
    assert labels[-2.0, 0.0, 0.0] == (-1, 0, 0)
    assert producer.removals == ((0, 0, 0), (0, 0, 0))


@pytest.mark.parametrize('normal,expected', [
    ((2.0, 2.0, 2.0), (0, 0, 0)),
    ((-2.0, 2.0, 2.0), (-1, 0, 0)),
    ((1.0, -2.0, 2.0), (0, -1, 0)),
    ((1.5, 0.5, -2.0), (0, 0, -1)),
    ((0.5, -3.5, -2.0), (0, -1, -1)),
    ((-1.5, 4.5, -2.0), (-1, 1, -1)),
])
def test_triclinic_source_images_have_hand_derived_coefficients(normal, expected):
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        params=(4.0, 1.0, 4.0, 0.5, 1.5, 4.0),
    )
    producer = _producer(packet, points, ids)
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    origin = next(row for row in cell['origins']
                  if row['kind'] == 'particle' and row['owner'] == 1
                  and tuple(row['normal']) == normal)
    label = producer.attribute(cell, origin)
    assert label.owner == 1
    assert label.shift == expected
    assert label.routes


def test_same_block_power_replays_the_actual_add_then_subtract_offset():
    radii = np.array([2.0**27 + 1, 2.0**27 + 1])
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        blocks=(1, 1, 1), radii=radii,
    )
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    origin = next(row for row in cell['origins']
                  if row['kind'] == 'particle' and row['owner'] == 1)
    assert origin['offset'] == 0.0
    label = _producer(packet, points, ids, radii).attribute(cell, origin)
    assert label.shift == (0, 0, 0)


def test_orthogonal_periodic_seeds_and_physical_walls_have_separate_identity():
    points = np.array([[1.0, 2.0, 3.0]])
    ids = np.array([0], dtype=np.int32)
    packet = _core._observe_box(
        points, ids, ((0.0, 4.0),) * 3, (1, 1, 1),
        (True, False, True), 8,
    )
    producer = _producer(packet, points, ids)
    cell = packet['cells'][0]
    labels = [producer.attribute(cell, origin) for origin in cell['origins']
              if origin['kind'] == 'orthogonal_seed']
    assert [(label.owner, label.shift) for label in labels] == [
        (0, (-1, 0, 0)), (0, (1, 0, 0)),
        (-3, None), (-4, None),
        (0, (0, 0, -1)), (0, (0, 0, 1)),
    ]


@pytest.mark.parametrize('normal,expected', [
    ((4.0, 0.0, 0.0), (1, 0, 0)),
    ((-4.0, -0.0, -0.0), (-1, 0, 0)),
    ((1.0, 4.0, 0.0), (0, 1, 0)),
    ((0.5, 1.5, 4.0), (0, 0, 1)),
])
def test_seed_replay_preserves_sign_and_source_coefficient(normal, expected):
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0]], params=(4.0, 1.0, 4.0, 0.5, 1.5, 4.0),
    )
    cell = packet['cells'][0]
    origin = next(row for row in cell['origins']
                  if row['kind'] == 'triclinic_seed'
                  and tuple(row['normal']) == normal)
    result = _producer(packet, points, ids).attribute(cell, origin)
    assert result.owner == 0
    assert result.shift == expected


def test_storage_verification_rejects_a_changed_block_operand():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    packet, points, ids = _observe([[1.0, 1.0, 1.0]])
    packet['context']['block_reciprocals'][0] = 1.0
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids)
    assert caught.value.code == 'WP5_SOURCE_PROFILE_MISMATCH'


def test_wrong_finite_support_is_inconsistent_without_an_ideal_tiebreaker():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    packet, points, ids = _observe([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    origin = deepcopy(next(row for row in cell['origins']
                           if row['kind'] == 'particle'))
    origin['normal'][0] = math.nextafter(origin['normal'][0], math.inf)
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids).attribute(cell, origin)
    assert caught.value.code == 'WP5_IMAGE_INCONSISTENT'


def test_upper_y_wrap_keeps_the_separately_rounded_x_correction():
    # k=1, pre-wrap beta=0, alpha=-1, then beta=1 and the right source
    # wraps in x. The source makes ((0.1-4)+0.2)+4, not 0.1+0.2.
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        params=(4.0, 0.2, 4.0, 0.1, 1.5, 4.0),
    )
    cell = next(row for row in packet['cells'] if row['id'] == 1)
    origin = next(row for row in cell['origins']
                  if row['kind'] == 'particle' and row['owner'] == 0
                  and tuple(row['normal']) == (
                      float.fromhex('-0x1.b333333333332p+0'), 3.5, 2.0))
    label = _producer(packet, points, ids).attribute(cell, origin)
    assert label.shift == (0, 1, 1)
    assert any(route.displacement == (0.30000000000000027, 5.5, 4.0)
               and route.construction_block == (0, 5, 5)
               for route in label.routes)


@pytest.mark.parametrize('periodic_container', (False, True))
def test_native_insertion_rounding_removal_is_added_to_the_source_chart(
    periodic_container,
):
    points = np.array([[math.nextafter(0.1, 0.0), 0.02, 0.02]])
    ids = np.array([0], dtype=np.int32)
    if periodic_container:
        packet = _core._observe_periodic(
            points, ids, (0.1, 0.025, 0.1, 0.05, 0.025, 0.1), (5, 1, 1), 8,
        )
    else:
        packet = _core._observe_box(
            points, ids, ((0.0, 0.1),) * 3, (5, 1, 1),
            (True, True, True), 8,
        )
    producer = _producer(packet, points, ids)
    assert producer.removals == ((1, 0, 0),)
    assert packet['sites'][0]['site'][0] == -2.0**-56


def test_candidate_cap_refuses_even_when_a_compatible_direct_prefix_exists():
    from pyvoro2._internal.spatial.wp5_common import (
        WP5Budget, WP5Failure, WP5Limits,
    )
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], blocks=(1, 1, 1),
    )
    cell = packet['cells'][0]
    origin = next(row for row in cell['origins'] if row['kind'] == 'particle')
    budget = WP5Budget(WP5Limits(candidate_limit=1))
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids, budget=budget).attribute(cell, origin)
    assert caught.value.code == 'WP5_RESOURCE_LIMIT'


def test_incompatible_native_int_profile_is_a_distinct_refusal():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    packet, points, ids = _observe([[1.0, 1.0, 1.0]])
    packet['build']['int_bits'] = 64
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids)
    assert caught.value.code == 'WP5_UNSUPPORTED_FP_PROFILE'


@pytest.mark.parametrize('power', (False, True))
def test_every_dyadic_native_origin_retains_its_independent_exact_image(power):
    points = [[0.75, 1.25, 1.75], [2.25, 2.75, 3.25], [3.5, 0.5, 2.5]]
    radii = np.array([0.25, 0.5, 0.75]) if power else None
    packet, points, ids = _observe(
        points, params=(4.0, 1.0, 4.0, 0.5, 1.5, 4.0), radii=radii,
    )
    producer = _producer(packet, points, ids, radii)
    route_names = set()
    for cell in packet['cells']:
        for origin in cell['origins']:
            if origin['kind'] == 'construction_bound':
                continue
            normal = tuple(map(Fraction, origin['normal']))
            if origin['kind'] == 'particle':
                delta = points[origin['owner']] - points[cell['id']]
                displacement = tuple(n-Fraction(d) for n, d in zip(normal, delta))
            else:
                displacement = normal
            k = displacement[2]/4
            j = (displacement[1]-Fraction(3, 2)*k)/4
            i = (displacement[0]-j-Fraction(1, 2)*k)/4
            assert all(v.denominator == 1 for v in (i, j, k))
            attribution = producer.attribute(cell, origin)
            assert attribution.shift == (int(i), int(j), int(k))
            route_names.update(route.name for route in attribution.routes)
    assert route_names >= {
        'primary-worklist', 'side-left', 'side-right',
        'vertical-up-left', 'vertical-up-right',
        'vertical-down-left', 'vertical-down-right', 'triclinic-seed',
    }


def test_direct_source_signed_zero_is_not_a_worklist_zero_with_the_same_value():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    packet, points, ids = _observe(
        [[1.0, 0.0, 1.0], [2.0, -0.0, 1.0]], blocks=(1, 1, 1),
        params=(4.0, 0.0, 4.0, 0.0, 0.0, 4.0),
    )
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    origin = deepcopy(next(row for row in cell['origins']
                           if row['kind'] == 'particle' and row['owner'] == 1))
    assert math.copysign(1.0, origin['normal'][1]) == -1.0
    producer = _producer(packet, points, ids)
    assert producer.attribute(cell, origin).shift == (0, 0, 0)
    origin['normal'][1] = 0.0
    with pytest.raises(WP5Failure) as caught:
        producer.attribute(cell, origin)
    assert caught.value.code == 'WP5_IMAGE_INCONSISTENT'


def test_primary_own_particle_is_excluded_even_for_a_matching_zero_plane():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0]], params=(4.0, 1.0, 4.0, 0.5, 1.5, 4.0),
    )
    cell = packet['cells'][0]
    origin = dict(token=100, kind='particle', owner=0, normal=[0.0]*3, offset=0.0)
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids).attribute(cell, origin)
    assert caught.value.code == 'WP5_IMAGE_INCONSISTENT'


def test_direct_power_source_does_not_admit_the_checked_offset_tree():
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    radii = np.array([2.0**27 + 1, 2.0**27 + 1])
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], blocks=(1, 1, 1), radii=radii,
    )
    cell = packet['cells'][0]
    origin = deepcopy(next(row for row in cell['origins']
                           if row['kind'] == 'particle' and row['owner'] == 1))
    origin['offset'] = 1.0
    with pytest.raises(WP5Failure) as caught:
        _producer(packet, points, ids, radii).attribute(cell, origin)
    assert caught.value.code == 'WP5_IMAGE_INCONSISTENT'


def test_worklist_power_retains_both_indistinguishable_offset_histories():
    radii = np.array([0.25, 0.25])
    packet, points, ids = _observe(
        [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]], radii=radii,
    )
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    origin = next(row for row in cell['origins']
                  if row['kind'] == 'particle' and row['owner'] == 1
                  and tuple(row['normal']) == (2.0, 0.0, 0.0))
    attribution = _producer(packet, points, ids, radii).attribute(cell, origin)
    assert attribution.shift == (0, 0, 0)
    assert {route.offset_branch for route in attribution.routes} == {
        'r_scale', 'r_scale_check',
    }
