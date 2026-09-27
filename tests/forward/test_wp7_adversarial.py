"""Public WP7 batch, representation and refusal adversaries.

The mathematical checks use wp7_rational_oracle. Constructed witness/profile
mutations are explicitly labeled constructions, not observed native failures.
"""

import copy
from fractions import Fraction as F
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import pyvoro2
import pyvoro2.planar as planar

from wp7_rational_oracle import ghost_ideal


def _api(dim):
    return planar if dim == 2 else pyvoro2


def _boundary(dim):
    return 'edges' if dim == 2 else 'faces'


def _signature(cell, dim):
    """Semantic set, not native face/edge count or serialization order."""
    refs = set()
    for boundary in cell[_boundary(dim)]:
        ref = boundary['boundary_reference']
        if ref is not None:
            refs.add((ref['kind'], ref['generator_id'],
                      tuple(ref['shift']) if ref['shift'] is not None else None,
                      ref['wall_id']))
    return float(cell['area' if dim == 2 else 'volume']), frozenset(refs)


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('power', (False, True))
def test_public_repeated_reordered_queries_and_small_capacity(dim, power):
    suffix = (.25,) * (dim - 2)
    points = [(.2, .2) + suffix, (.8, .2) + suffix,
              (.2, .8) + suffix, (.8, .8) + suffix]
    queries = [(.5, .5) + (.5,) * (dim - 2),
               (.5, .625) + (.5,) * (dim - 2)]
    radii = {'radii': [.125] * 4, 'ghost_radii': [.25, .125, .25]}
    common = dict(domain=_api(dim).Box(bounds=((0, 1),) * dim),
                  mode='power' if power else 'standard',
                  ids=[11, 29, 37, 53], init_mem=1, blocks=(1,) * dim,
                  return_vertices=False, return_adjacency=False)
    order = (0, 1, 0)
    for perm in (order, (1, 0, 0)):
        selected = [queries[i] for i in perm]
        extras = ({'radii': radii['radii'],
                   'ghost_radii': [radii['ghost_radii'][i] for i in perm]}
                  if power else {})
        batch = _api(dim).ghost_cells(points, selected, **common, **extras)
        assert [c['query_index'] for c in batch] == list(range(3))
        assert _signature(batch[perm.index(0)], dim) == _signature(batch[2], dim)
        for qi, cell in enumerate(batch):
            single_power = ({'radii': radii['radii'],
                             'ghost_radii': extras['ghost_radii'][qi]}
                            if power else {})
            single, = _api(dim).ghost_cells(
                points, [selected[qi]], **common, **single_power,
            )
            assert _signature(cell, dim) == _signature(single, dim)
            ideal = ghost_ideal(points, cell['site'], ((0, 1),) * dim,
                                (1,) * dim, (False,) * dim,
                                [np.float64(.125) ** 2] * 4 if power else None,
                                np.float64(extras['ghost_radii'][qi]) ** 2
                                if power else 0)
            assert ideal.dimension == dim
            assert abs(_signature(cell, dim)[0] - float(ideal.measure)) < 1e-8


@pytest.mark.parametrize('dim', (2, 3))
def test_public_weight_batch_matches_fixed_resolved_radii_semantic_classes(dim):
    suffix = (.5,) * (dim - 2)
    points = [(.25, .25) + suffix, (.75, .25) + suffix]
    queries = [(.5, .625) + suffix, (.5, .75) + suffix]
    weights = np.array([-1 / 16, 0.])
    ghost_weights = np.array([1 / 16, 3 / 16])
    complete = np.r_[weights, ghost_weights]
    fixed = np.sqrt(complete - complete.min())
    common = dict(domain=_api(dim).Box(bounds=((0, 1),) * dim),
                  mode='power', return_vertices=False,
                  return_adjacency=False)
    batch = _api(dim).ghost_cells(
        points, queries, **common, weights=weights,
        ghost_weights=ghost_weights,
    )
    for qi, cell in enumerate(batch):
        independent = ghost_ideal(points, cell['site'], ((0, 1),) * dim,
                                  (1,) * dim, (False,) * dim,
                                  weights, ghost_weights[qi])
        assert independent.dimension == dim
        assert abs(_signature(cell, dim)[0] - float(independent.measure)) < 1e-8
        explicit, = _api(dim).ghost_cells(
            points, [queries[qi]], **common, radii=fixed[:len(points)],
            ghost_radii=fixed[len(points) + qi],
        )
        # Both routes pass identical binary64 radii to the same selected
        # kernel. The explicit-radius S ideal has slightly different exact
        # squares after sqrt rounding, so assert robust classes, not equal S
        # rational vertex coordinates or raw fragment counts.
        assert _signature(cell, dim) == _signature(explicit, dim)


@pytest.mark.parametrize('dim', (2, 3))
def test_public_unused_huge_query_translation_and_required_shift_limit(dim):
    api = _api(dim)
    mask = (True,) + (False,) * (dim - 1)
    domain = (api.RectangularCell(bounds=((0, 1),) * dim, periodic=mask)
              if dim == 2 else api.OrthorhombicCell(
                  bounds=((0, 1),) * dim, periodic=mask))
    huge = float(2**70)
    empty = np.empty((0, dim))
    query = [(huge,) + (.5,) * (dim - 1)]
    # WP8 now materializes the retained public query shift. The private WP7
    # computation still succeeds, but this new view cannot represent 2**70.
    with pytest.raises(ValueError) as caught:
        api.ghost_cells(empty, query, domain=domain,
                        return_vertices=False, return_adjacency=False)
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'
    assert caught.value.details['field'] == 'query_shift'
    output, = api.ghost_cells(empty, [(0.,) + (.5,) * (dim - 1)], domain=domain,
                              return_vertices=False, return_adjacency=False)
    ideal = ghost_ideal([], output['site'], ((0, 1),) * dim,
                        (1,) * dim, mask)
    assert tuple(output['site']) == (0,) + (.5,) * (dim - 1)
    assert ideal.measure == 1
    assert len(output[_boundary(dim)]) == 2 * dim
    assert all(b['boundary_reference'] is not None
               for b in output[_boundary(dim)])

    # Here that same private preparation coefficient becomes a required
    # public persistent-image shift. No wrap or int64 truncation is allowed.
    points = [(huge,) + (.25,) * (dim - 1)]
    q = [(.5,) * dim]
    with pytest.raises(ValueError) as caught:
        api.ghost_cells(points, q, domain=domain,
                        return_vertices=False, return_adjacency=False)
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'
    assert caught.value.stage == 'materialization'
    assert caught.value.query_index == 0
    # Geometry-only did not request a public shift and must still be usable.
    geometry_only, = api.ghost_cells(
        points, q, domain=domain, return_vertices=False,
        return_adjacency=False,
        **({'return_edges': False} if dim == 2 else {'return_faces': False}),
    )
    assert not geometry_only['empty']
    assert _boundary(dim) not in geometry_only


@pytest.mark.parametrize('dim', (2, 3))
def test_constructed_second_query_profile_refusal_is_atomic(monkeypatch, dim):
    """Mutate a copied packet after actual native execution, not N itself."""
    api = _api(dim)
    module = __import__(f'pyvoro2{".planar" if dim == 2 else ""}.api',
                        fromlist=['_require_core'])
    real = module._require_core2d() if dim == 2 else module._require_core()
    if dim == 2:
        def tampered(*args):
            cells, packets = real._ghost_box_standard_witness(*args)
            packets = copy.deepcopy(packets)
            packets[1]['profile']['source_sha256'] = '0' * 64
            return cells, packets
        monkeypatch.setattr(module, '_require_core2d', lambda: SimpleNamespace(
            _ghost_box_standard_witness=tampered))
    else:
        def tampered(*args, **kwargs):
            packets = copy.deepcopy(real._observe_ghost_box(*args, **kwargs))
            packets[1]['build']['ghost_source_sha256'] = '0' * 64
            return packets
        monkeypatch.setattr(module, '_require_core', lambda: SimpleNamespace(
            _observe_ghost_box=tampered))
    with pytest.raises(ValueError) as caught:
        api.ghost_cells(np.empty((0, dim)),
                        [(.5,) * dim, (.625,) * dim],
                        domain=api.Box(bounds=((0, 1),) * dim),
                        return_vertices=False, return_adjacency=False)
    assert caught.value.code == 'GHOST_NATIVE_UNSUPPORTED'
    assert caught.value.stage == 'native'
    assert caught.value.query_index == 1
    assert len(repr(caught.value.details)) < 4096


@pytest.mark.parametrize('dim', (2, 3))
def test_constructed_public_resource_refuses_complete_batch(monkeypatch, dim):
    if dim == 2:
        from pyvoro2._internal.planar import ghost_certificate as cert
        from pyvoro2._internal.planar.wp6_ideal import ExactAuditBudget

        monkeypatch.setattr(cert, 'ExactAuditBudget',
                            lambda: ExactAuditBudget(max_candidates=0))
    else:
        from pyvoro2._internal.spatial import ghost_certificate as cert
        from pyvoro2._internal.spatial.wp5_common import WP5Budget, WP5Limits

        monkeypatch.setattr(cert, 'WP5Budget',
                            lambda: WP5Budget(WP5Limits(work_limit=0)))
    with pytest.raises(ValueError) as caught:
        _api(dim).ghost_cells(
            [(.25,) + (.5,) * (dim - 1)],
            [(.5,) * dim, (.625,) + (.5,) * (dim - 1)],
            domain=_api(dim).Box(bounds=((0, 1),) * dim),
            return_vertices=False, return_adjacency=False,
        )
    assert caught.value.code == 'GHOST_CERTIFICATION_RESOURCE'
    assert caught.value.stage in ('semantic', 'provenance')
    assert caught.value.query_index == 0


@pytest.mark.parametrize('dim', (2, 3))
def test_zero_queries_validate_inputs_without_computation_or_profile(
        monkeypatch, dim):
    api = _api(dim)
    domain = api.Box(bounds=((0, 1),) * dim)
    empty = np.empty((0, dim))

    def unexpected_certificate(*args, **kwargs):
        raise AssertionError('zero queries must not fabricate a ghost certificate')

    dimension = 'planar' if dim == 2 else 'spatial'
    module = __import__(f'pyvoro2._internal.{dimension}.ghost_certificate',
                        fromlist=['certify_semantics'])
    monkeypatch.setattr(module, 'certify_semantics', unexpected_certificate)
    assert api.ghost_cells(empty, empty, domain=domain) == []
    with pytest.raises(ValueError):
        api.ghost_cells(empty, empty, domain=domain, mode='power',
                        weights=[], ghost_radii=[])
    with pytest.raises(ValueError, match='block product'):
        api.ghost_cells(empty, empty, domain=domain, blocks=(100000,) * dim)


def test_observed_planar_public_rounding_collapse_differs_from_internal(
        monkeypatch):
    """One selected execution supplies both public and doubled-local views."""
    import pyvoro2.planar.api as module
    from pyvoro2 import _core2d

    captured = []

    def observe(*args):
        cells, packets = _core2d._ghost_box_standard_witness(*args)
        captured.extend(packets)
        return cells, packets

    monkeypatch.setattr(module, '_require_core2d', lambda: SimpleNamespace(
        _ghost_box_standard_witness=observe))
    origin = float(2**53)
    sites = np.array([[12, 4], [14, 4], [12, 0], [12, 14],
                      [6, 4], [10, 6]], dtype=float) + origin
    domain = planar.RectangularCell(((origin, origin + 16),) * 2)
    cell, = planar.ghost_cells(np.delete(sites, 3, axis=0), sites[3:4],
                               domain=domain, return_vertices=True,
                               return_adjacency=False)
    source, = captured[0]['sources']
    rounded_only = []
    internally_collapsed = []
    for edge in cell['edges']:
        a, b = edge['vertices']
        public_equal = cell['vertices'][a] == cell['vertices'][b]
        local_equal = source['local2'][a] == source['local2'][b]
        if public_equal and not local_equal:
            rounded_only.append(edge)
            assert edge['boundary_reference'] is not None
        if local_equal:
            internally_collapsed.append(edge)
            assert edge['boundary_reference'] is None
    assert rounded_only and internally_collapsed


def test_observed_spatial_public_vertices_all_round_to_one_positive_site(
        monkeypatch):
    """A public point cloud may collapse while all six private facets survive."""
    import pyvoro2.api as module
    from pyvoro2 import _core

    origin = float(2**53)
    q = (origin + 4,) * 3
    points = [tuple(q[j] + 2 * (axis == j) for j in range(3))
              for axis in range(3)] + [
                  tuple(q[j] - 2 * (axis == j) for j in range(3))
                  for axis in range(3)]
    bounds = ((origin, origin + 8),) * 3
    ideal = ghost_ideal(points, q, bounds, (8,) * 3, (False,) * 3,
                        weights=[3] * 6, ghost_weight=0)
    assert ideal.dimension == 3 and ideal.measure == F(1, 8)
    assert ideal.vertices == tuple(sorted(
        (F(x, 4), F(y, 4), F(z, 4))
        for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)))
    assert set(ideal.positive) == {
        ('generator', i, (0, 0, 0)) for i in range(6)
    }

    packets = []

    def observe(*args, **kwargs):
        result = _core._observe_ghost_box(*args, **kwargs)
        packets.extend(result)
        return result

    monkeypatch.setattr(module, '_require_core', lambda: SimpleNamespace(
        _observe_ghost_box=observe))
    cell, = pyvoro2.ghost_cells(
        points, [q], domain=pyvoro2.Box(bounds),
        mode='power', weights=[3] * 6, ghost_weights=0,
        blocks=(1, 1, 1), init_mem=1,
        return_vertices=True, return_adjacency=False,
    )
    assert tuple(cell['site']) == q
    assert cell['volume'] == pytest.approx(float(ideal.measure), abs=1e-12)
    private = [tuple(map(F, vertex)) for vertex in
               packets[0]['cells'][0]['vertices_doubled']]
    assert len(private) == 8 and len(set(private)) == 8
    assert len(cell['vertices']) == 8
    assert {tuple(v) for v in cell['vertices']} == {q}

    supported = set()
    for face in cell['faces']:
        indices = face['vertices']
        assert len(set(indices)) >= 3
        a, b, c = (private[index] for index in indices[:3])
        u = tuple(b[j] - a[j] for j in range(3))
        v = tuple(c[j] - a[j] for j in range(3))
        assert any(u[(j + 1) % 3] * v[(j + 2) % 3]
                   - u[(j + 2) % 3] * v[(j + 1) % 3]
                   for j in range(3))
        ref = face['boundary_reference']
        assert ref is not None and ref['kind'] == 'generator'
        assert ref['shift'] is ref['wall_id'] is None
        assert ref['generator_id'] in range(6)
        assert face['adjacent_cell'] == ref['generator_id']
        supported.add(ref['generator_id'])
    assert supported == set(range(6))


_SUBPROCESS_SCRIPT = textwrap.dedent('''
    import json
    import pyvoro2
    import pyvoro2.planar as planar

    def record(cell, dim):
        refs = []
        for edge in cell['edges' if dim == 2 else 'faces']:
            ref = edge['boundary_reference']
            if ref is not None:
                refs.append((ref['kind'], ref['generator_id'],
                             ref['shift'], ref['wall_id']))
        return (cell['site'], cell['area' if dim == 2 else 'volume'],
                sorted(set(map(str, refs))))

    result = {}
    for dim, api in ((2, planar), (3, pyvoro2)):
        suffix = (.25,) * (dim - 2)
        points = [(.2, .2) + suffix, (.8, .2) + suffix,
                  (.2, .8) + suffix, (.8, .8) + suffix]
        queries = [(.5, .5) + (.5,) * (dim - 2),
                   (.5, .625) + (.5,) * (dim - 2)]
        domain = api.Box(bounds=((0, 1),) * dim)
        records = []
        for order in ((0, 1, 0), (1, 0, 0)):
            cells = api.ghost_cells(
                points, [queries[i] for i in order], domain=domain,
                init_mem=1, blocks=(1,) * dim, return_vertices=False,
                return_adjacency=False,
            )
            records.append([(order[i], record(cell, dim))
                            for i, cell in enumerate(cells)])
        result[str(dim)] = records
    print(json.dumps(result, sort_keys=True))
''')


def test_fresh_subprocess_allocator_churn_preserves_public_semantic_classes():
    """MALLOC_PERTURB_ is churn evidence, not a poisoned native-slot proof."""
    runs = []
    for value in ('17', '151'):
        env = os.environ.copy()
        env['MALLOC_PERTURB_'] = value
        env['PYTHONMALLOC'] = 'debug'
        finished = subprocess.run(
            [sys.executable, '-c', _SUBPROCESS_SCRIPT], env=env,
            capture_output=True, text=True, check=True, timeout=30,
        )
        runs.append(json.loads(finished.stdout))
    assert runs[0] == runs[1]
    for dim in ('2', '3'):
        by_query = {}
        for order in runs[0][dim]:
            for query, signature in order:
                if query in by_query:
                    assert signature == by_query[query]
                else:
                    by_query[query] = signature


@pytest.mark.parametrize(('basis', 'query', 'expected_n'), (
    (
        ((1., 1., 0.), (0., 1., 0.), (0., 0., 1.)),
        (.875, .125, .125), {(-1, 1, 0), (0, 0, 0)},
    ),
    (
        ((-1., -1., 0.), (0., 1., 0.), (0., 0., 1.)),
        (.375, .125, .125), {(0, 0, 0), (1, 1, 0)},
    ),
), ids=('right-handed-roundtrip', 'left-handed-roundtrip'))
def test_triclinic_stored_roundtrip_has_exact_microfacets_missing_from_n(
        monkeypatch, basis, query, expected_n):
    """S uses actual stored g; native six-face geometry cannot cover S eight.

    The 2D rational interval oracle independently cross-checks the 3D plane
    intersection's Cartesian XY supports. The N labels are observed from this
    selected public invocation; they are never oracle input or tie breakers.
    BLAS implementations can change the last bits of the Cartesian roundtrip,
    so S must be checked at this invocation's actual stored-site operand.
    """
    import pyvoro2._internal.spatial.ghost_certificate as cert

    domain = pyvoro2.PeriodicCell(basis)
    point = (.125, .125, .125)
    common = dict(domain=domain, blocks=(1, 1, 1),
                  return_vertices=False, return_adjacency=False)
    geometry, = pyvoro2.ghost_cells([point], [query], return_faces=False,
                                    **common)
    assert geometry['volume'] == pytest.approx(.5, abs=1e-12, rel=0)

    observed = {}
    original = cert.certify_semantics

    def capture(**kwargs):
        observed.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cert, 'certify_semantics', capture)
    with pytest.raises(ValueError) as caught:
        pyvoro2.ghost_cells([point], [query], return_faces=True, **common)
    g = tuple(map(F, observed['ghost_site']))
    assert g == tuple(map(F, geometry['site']))
    micro_span = abs(g[1] - F(point[1]))
    assert g[2] == F(1, 8) and 0 < micro_span < F(1, 2**40)

    ideal3 = ghost_ideal([point], g, None, None, (True,) * 3,
                         lattice=basis)
    generator_s = {key[2] for key in ideal3.positive
                   if key[0] == 'generator'}
    assert ideal3.dimension == 3 and ideal3.measure == F(1, 2)
    assert len(ideal3.facets) == 8 and len(generator_s) == 4

    ideal2 = ghost_ideal([point[:2]], g[:2], ((0, 1), (0, 1)),
                         (1, 1), (True, True))
    physical3 = {tuple(sum(s[i] * F(basis[i][j]) for i in range(3))
                       for j in range(2)) for s in generator_s}
    physical2 = {key[2] for key in ideal2.positive
                 if key[0] == 'generator'}
    assert physical3 == physical2 and len(physical2) == 4

    native_s = {o.shift for o in observed['occurrences']
                if o.owner == 0 and not o.collapsed}
    assert native_s == expected_n
    missing = generator_s - native_s
    assert len(missing) == 2
    for shift in missing:
        contact = ideal3.contact('generator', 0, shift)
        xy = {vertex[:2] for vertex in contact.vertices}
        ys = {vertex[1] for vertex in xy}
        assert contact.status == 'positive' and len(xy) == 2
        assert max(ys) - min(ys) == micro_span

    assert caught.value.code == 'GHOST_SEMANTIC_INCONSISTENT'
    assert caught.value.stage == 'semantic' and caught.value.query_index == 0
    assert caught.value.details['invariant'] == 'positive_facet_coverage'
    assert caught.value.details['required_count'] == 8
    assert caught.value.details['covered_count'] == 6
    assert caught.value.details['missing_count'] == 2
