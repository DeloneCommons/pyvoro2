"""Portable ideal expectations from the independently reviewed G0 evidence.

No test fixes the current native vertex count, cycle, face count or token.
The archived rank-three cycle is a separate pure observation-audit fixture.
"""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, compute
from pyvoro2.diagnostics import TessellationError
from pyvoro2._internal.spatial.wp5_cycle import audit_cycle
from pyvoro2._internal.spatial.wp5_ideal import ExactIdeal
from pyvoro2._internal.spatial.wp5_producer import Producer


_FIXTURE_PATH = Path(__file__).parents[2] / 'fixtures' / 'wp5_characterization.json'
_DATA = json.loads(_FIXTURE_PATH.read_text())
_CASES = _DATA['cases']
_BY_ID = {case['id']: case for case in _CASES}
_BOUNDS = tuple(tuple(axis) for axis in _DATA['bounds'])
_BLOCKS = tuple(_DATA['blocks'])
_IDS = np.arange(4, dtype=np.int32)
_ZERO = (0, 0, 0)


def _inputs(case):
    xy = float.fromhex(case['p3_xy_hex'])
    points = np.array(_DATA['first_three_sites'] + [[xy, xy, 1.0]])
    radii = (None if case['radius_hex'] is None else
             np.full(4, float.fromhex(case['radius_hex'])))
    periodic = (False, False, case['periodic_z'])
    return points, radii, periodic


def _ideal(case):
    points, radii, periodic = _inputs(case)
    # Exact squares of supplied binary64 operands, before any native r*r.
    weights = [F(0)] * 4 if radii is None else [F(float(r))**2 for r in radii]
    return ExactIdeal(points, _DATA['lattice'], weights, periodic, bounds=_BOUNDS)


def _observed_primary(case):
    from pyvoro2 import _core

    points, radii, periodic = _inputs(case)
    packet = _core._observe_box(points, _IDS, _BOUNDS, _BLOCKS, periodic,
                                8, radii=radii)
    cell = next(row for row in packet['cells'] if row['id'] == 0)
    # Match trusted origin provenance, never a final polygon or a token number.
    origins = [origin for origin in cell['origins']
               if origin['kind'] == 'particle' and origin['owner'] == 3
               and origin['normal'][2] == 0.0]
    assert origins, 'the source must attempt the same-block primary p3 cut'
    return packet, cell, origins, points, radii


@pytest.mark.parametrize('case', _CASES, ids=lambda case: case['id'])
def test_full_ideal_matches_independent_exact_transition_reference(case):
    cell = _ideal(case).cell(0)
    contact = cell.contact(3, _ZERO)

    assert cell.dimension == 3
    assert contact.status == case['status']
    assert contact.dimension == case['dimension']
    assert contact.area_squared == F(case['area_squared'])
    if case['status'] == 'zero':
        assert set(contact.vertices) == {(F(3, 2), F(3, 2), -2),
                                         (F(3, 2), F(3, 2), 2)}
    elif case['status'] == 'absent':
        assert contact.vertices == ()


@pytest.mark.parametrize('case', _CASES, ids=lambda case: case['id'])
def test_primary_source_attribution_keeps_zero_and_absent_ideal_contacts(case):
    packet, cell, origins, points, radii = _observed_primary(case)
    producer = Producer(packet, points, _IDS, radii)

    # Every attempted actual origin remains attributable. In particular this
    # must work for the independently known zero/absent contacts and cuts
    # leaving no final native face: positivity cannot filter the source set.
    for origin in origins:
        label = producer.attribute(cell, origin)
        assert label.kind == 'particle'
        assert label.owner == 3
        assert label.shift == _ZERO
        assert label.routes


@pytest.mark.parametrize('name,native_offset', [
    ('std_zero', 4.5), ('p1_zero', 4.5), ('p2pow27_zero', 4.0),
    ('p2pow27plus1_zero', 4.0), ('p2pow27plus2_zero', 4.0),
])
def test_common_radius_gauge_does_not_replace_exact_ideal_with_native_offset(
        name, native_offset):
    case = _BY_ID[name]
    _, _, origins, _, _ = _observed_primary(case)
    assert all(tuple(origin['normal']) == (1.5, 1.5, 0.0)
               and origin['offset'] == native_offset for origin in origins)

    # The exact p3 support is X+Y=3/2 in every common-radius gauge. Its cut
    # touches only the upper x/y corner of this hand-derived full prism.
    cell = _ideal(case).cell(0)
    expected_vertices = product((-2, F(3, 2)), (-2, F(3, 2)), (-2, 2))
    assert set(cell.vertices) == set(expected_vertices)
    assert cell.contact(3, _ZERO).status == 'zero'
    assert cell.contact(3, _ZERO).area_squared == 0


@pytest.mark.parametrize('name', [
    'periodic_z_std_zero', 'periodic_z_p2pow27plus1_zero',
    'p2pow27plus1_plus26', 'p2pow27plus1_plus20',
])
def test_periodic_public_attribution_keeps_semantic_diagnostics_separate(name):
    # Extend the two positive-delta box controls periodically in z as well.
    # The independent same-z domination proof in the fixture applies unchanged.
    case = dict(_BY_ID[name], periodic_z=True)
    packet, native_cell, origins, points, radii = _observed_primary(case)
    producer = Producer(packet, points, _IDS, radii)
    by_token = {origin['token']: origin for origin in native_cell['origins']}
    labels = [producer.attribute(native_cell, by_token[face['token']])
              for face in native_cell['faces']]
    expected_labels = [(label.owner, label.shift) for label in labels]
    observed_target = (3, _ZERO) in expected_labels
    options = {} if radii is None else {'mode': 'power', 'radii': radii}
    kwargs = dict(
        domain=OrthorhombicCell(_BOUNDS, periodic=(False, False, True)),
        blocks=_BLOCKS, return_face_shifts=True,
        return_diagnostics=True,
        return_vertices=False, return_adjacency=False, **options,
    )

    # The default action returns every uniquely attributed native occurrence,
    # including zero/absent ideal contacts. It does not silently drop them or
    # claim they have positive exact area. Qualified native polygons may vary.
    result = compute(points, **kwargs)
    assert result.has_periodic_shifts
    cell = next(cell for cell in result.cells if cell['id'] == 0)
    assert [(face['adjacent_cell'], face.get('adjacent_shift'))
            for face in cell['faces']] == expected_labels
    assert 'vertices' not in cell and 'adjacency' not in cell

    diagnostics = result.require_tessellation_diagnostics()
    assert diagnostics.face_shift_available
    if observed_target:
        expected_code = ('WP5_EXACT_ZERO' if case['status'] == 'zero'
                         else 'WP5_IDEAL_ABSENT')
        matching = [issue for issue in diagnostics.issues
                    if issue.code == expected_code
                    and any(context.get('source_id') == 0
                            and context.get('owner') == 3
                            and context.get('public_shift') == _ZERO
                            and context.get('semantic') == case['status']
                            for context in issue.examples)]
        assert matching and all(issue.severity == 'error' for issue in matching)
        assert not diagnostics.ok

    # Strictness is the caller's action policy; it does not alter attribution.
    if not diagnostics.ok:
        with pytest.raises(TessellationError) as caught:
            compute(points, tessellation_check='raise', **kwargs)
        assert caught.value.diagnostics.face_shift_available
        assert caught.value.diagnostics.issues == diagnostics.issues
    else:
        strict = compute(points, tessellation_check='raise', **kwargs)
        assert strict.has_periodic_shifts


def test_historical_nonplanar_cycle_passes_the_exact_projected_observation_audit():
    sample = _DATA['historical_projected_cycle']
    vertices = tuple(tuple(F(float.fromhex(value)) for value in row)
                     for row in sample['vertices_doubled_hex'])
    a, b, c = [tuple(point[k] - vertices[0][k] for k in range(3))
               for point in vertices[1:4]]
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    assert determinant == F(sample['first_four_affine_determinant']) != 0

    normal = tuple(float.fromhex(value) for value in sample['normal_hex'])
    offset = float.fromhex(sample['offset_hex'])
    assert audit_cycle(vertices, normal, offset) == tuple(
        tuple(F(value) for value in row) for row in sample['projected_vertices']
    )
    # Passing the observation audit does not make its independent S ideal
    # positive. This characterized occurrence has an exact line contact.
    assert _ideal(_BY_ID['periodic_z_p2pow27plus1_zero']).cell(0).contact(
        3, _ZERO,
    ).status == 'zero'
