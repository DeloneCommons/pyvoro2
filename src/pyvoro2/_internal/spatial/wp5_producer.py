"""Qualified source replay for native 3D origin occurrences (ADR 0021).

This module has no ideal-cell or face-area input. Every image is retained by
the producer's ordered binary64 arithmetic and source block constraints alone.
The route set is a necessary-condition superset, not a claim that every
compatible history actually ran. Only complete sets may be classified.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math

from .wp5_binary64 import (
    bits_equal, div, integer_interval, interval_add, interval_div, interval_sub,
    mod, rounding_bin, rounding_preimage, step,
)
from .wp5_common import WP5Budget, WP5Failure


_IMIN = -(1 << 31)
_IMAX = (1 << 31) - 1
_ZERO = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class Route:
    """One replayed history, retained even when another has the same image."""

    name: str
    coefficient: tuple[int, int, int]
    offset_branch: str
    image_block: tuple[int, int, int] | None = None
    displacement: tuple[float, float, float] | None = None
    construction_block: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class Attribution:
    owner: int
    shift: tuple[int, int, int] | None
    kind: str
    routes: tuple[Route, ...]


class _InvalidRoute(Exception):
    """An integer/float operation cannot occur in the qualified producer."""


def _i(value):
    if not _IMIN <= value <= _IMAX:
        raise _InvalidRoute('signed integer operation is outside int32')
    return value


def _add(a, b):
    return _i(a + b)


def _sub(a, b):
    return _i(a - b)


def _mul(a, b):
    return _i(a * b)


def _f(value):
    if not math.isfinite(value):
        raise _InvalidRoute('nonfinite source intermediate')
    return value


def _step(value):
    _f(value)
    _i(int(value))
    return _i(step(value))


def _mod(value, divisor):
    # The native negative branch contains divisor-1-value, not just a%b.
    if value < 0:
        _sub(_sub(divisor, 1), value)
    return mod(value, divisor)


def _same(left, right):
    return len(left) == len(right) and all(
        bits_equal(a, b) for a, b in zip(left, right)
    )


def _point(value):
    exact = Fraction(value)
    return exact, exact


def _count(bounds):
    return max(0, bounds[1] - bounds[0] + 1)


def _integers(bounds):
    return range(bounds[0], bounds[1] + 1)


class Producer:
    """Validate insertion operands, then attribute one occurrence at a time.

    ``removals`` is indexed by dense persistent ID, in exact Python integers.
    It supplements the preparation chart removals already owned by the caller.
    """

    def __init__(self, packet, prepared_native_points, ids, radii=None,
                 budget=None):
        self.packet = packet
        self.budget = budget if budget is not None else WP5Budget()
        try:
            self._initialize(prepared_native_points, ids, radii)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError,
                _InvalidRoute) as exc:
            raise WP5Failure(
                'WP5_SOURCE_PROFILE_MISMATCH',
                f'Native source context or insertion disagrees: {exc}',
            ) from exc

    def _initialize(self, prepared_native_points, ids, radii):
        build = self.packet['build']
        if (build.get('native_fp_policy') != 'binary64-noncontracting-v1'
                or build.get('int_bits') != 32
                or build.get('int_min') != _IMIN or build.get('int_max') != _IMAX
                or not all(build.get(key) is True for key in (
                    'binary64', 'round_to_nearest', 'gradual_underflow',
                    'source_coupled_seed_replay', 'source_coupled_compute'))
                or build.get('float_eval_method') != 0
                or build.get('fast_math') is not False
                or build.get('fp_contract') != 'off'
                or build.get('ipo') is not False):
            raise WP5Failure('WP5_UNSUPPORTED_FP_PROFILE',
                             'Native source arithmetic profile is unqualified')
        self.context = context = self.packet['context']
        self.periodic = tuple(context['periodic'])
        self.blocks = tuple(_i(int(v)) for v in context['blocks'])
        self.widths = tuple(float(v) for v in context['block_widths'])
        self.reciprocals = tuple(float(v)
                                 for v in context['block_reciprocals'])
        self.mask = tuple(_i(int(v)) for v in context['mask_shape'])
        self.power = bool(context['power'])
        self.kind = context['kind']
        if (self.kind not in ('box', 'periodic')
                or len(self.blocks) != 3 or any(v <= 0 for v in self.blocks)
                or len(self.periodic) != 3
                or len(self.widths) != 3 or len(self.reciprocals) != 3
                or self.power != (radii is not None)):
            raise ValueError('invalid native construction controls')
        if self.kind == 'box':
            self.bounds = tuple(tuple(float(v) for v in axis)
                                for axis in context['bounds'])
            periods = tuple(_f(hi - lo) for lo, hi in self.bounds)
            expected_mask = tuple(2*n+1 if periodic else n
                                  for n, periodic in zip(
                                      self.blocks, self.periodic))
        else:
            self.params = tuple(float(v) for v in context['cell_params'])
            bx, bxy, by, bxz, byz, bz = self.params
            if not all(math.isfinite(v) for v in self.params):
                raise ValueError('nonfinite native lattice')
            periods = bx, by, bz
            self.grid = {key: _i(int(context['image_grid'][key]))
                         for key in ('ey', 'ez', 'wy', 'wz', 'oy', 'oz', 'oxyz')}
            nx, ny, nz = self.blocks
            ey, ez = self.grid['ey'], self.grid['ez']
            if (not all(self.periodic) or min(ey, ez) < 1
                    or self.grid['wy'] != _add(ny, ey)
                    or self.grid['wz'] != _add(nz, ez)
                    or self.grid['oy'] != _add(ny, _mul(2, ey))
                    or self.grid['oz'] != _add(nz, _mul(2, ez))
                    or self.grid['oxyz'] != _mul(
                        _mul(nx, self.grid['oy']), self.grid['oz'])):
                raise ValueError('inconsistent native image grid')
            expected_mask = _add(_mul(2, nx), 1), 2*ey+1, 2*ez+1
        if any(v <= 0 for v in periods) or self.mask != expected_mask:
            raise ValueError('inconsistent native mask or lattice')
        self.periods = periods
        _mul(_mul(self.blocks[0], self.blocks[1]), self.blocks[2])
        _mul(_mul(self.mask[0], self.mask[1]), self.mask[2])
        for axis in range(3):
            width = _f(periods[axis] / self.blocks[axis])
            if (width <= 0 or not bits_equal(width, self.widths[axis])
                    or not bits_equal(_f(1.0 / width), self.reciprocals[axis])):
                raise ValueError('native block operand bits differ')

        self.sites = tuple(self.packet['sites'])
        n = len(self.sites)
        if len(prepared_native_points) != n or len(ids) != n:
            raise ValueError('insertion row count differs')
        if sorted(map(int, ids)) != list(range(n)):
            raise ValueError('persistent IDs are not a dense permutation')
        if radii is not None and len(radii) != n:
            raise ValueError('radius row count differs')
        removals = [None] * n
        slots = {}
        for order, (point, identifier) in enumerate(zip(prepared_native_points, ids)):
            owner = int(identifier)
            row = self.sites[owner]
            point = tuple(_f(float(v)) for v in point)
            if len(point) != 3 or row['id'] != owner:
                raise ValueError('persistent site identity differs')
            position, block, removal = self._insert(point)
            index = self._block_index(block)
            slot = slots.get(index, 0)
            slots[index] = _add(slot, 1)
            if (not _same(position, row['site'])
                    or tuple(row['block']) != block
                    or row['block_index'] != index or row['block_slot'] != slot):
                raise ValueError(f'actual native storage differs for ID {owner}')
            radius = None if radii is None else _f(float(radii[order]))
            if ((radius is None) != (row['radius'] is None)
                    or radius is not None and not bits_equal(radius, row['radius'])):
                raise ValueError(f'actual native radius differs for ID {owner}')
            removals[owner] = removal
        self.removals = tuple(removals)
        cell_ids = []
        for cell in self.packet['cells']:
            owner = cell['id']
            cell_ids.append(owner)
            if not _same(cell['site'], self.sites[owner]['site']):
                raise ValueError('cell storage differs from persistent site')
            parity = cell['noninterference']
            if parity['computed'] is not True or cell['computed'] and any(
                    parity[key] is not True
                    for key in ('geometry', 'topology', 'owners', 'volume')):
                raise ValueError('native ordinary/observed parity differs')
        if sorted(cell_ids) != list(range(n)):
            raise ValueError('native cell records do not cover persistent IDs')

    def _block_index(self, block):
        nx, ny, _ = self.blocks
        stride_y = ny if self.kind == 'box' else self.grid['oy']
        return _add(block[0], _mul(nx, _add(block[1], _mul(stride_y, block[2]))))

    def _insert(self, point):
        coordinates = list(point)
        block, removed = [0] * 3, [0] * 3
        if self.kind == 'box':
            for axis, (lower, _) in enumerate(self.bounds):
                c = _step(_f(_f(coordinates[axis] - lower)
                             * self.reciprocals[axis]))
                n = self.blocks[axis]
                if self.periodic[axis]:
                    target = _mod(c, n)
                    difference = _sub(target, c)
                    coordinates[axis] = _f(coordinates[axis] + _f(
                        self.widths[axis] * difference))
                    removed[axis] = -difference // n
                    c = target
                elif not 0 <= c < n:
                    raise _InvalidRoute('nonperiodic insertion outside blocks')
                block[axis] = c
        else:
            bx, bxy, by, bxz, byz, bz = self.params
            rows = ((bx, 0., 0.), (bxy, by, 0.), (bxz, byz, bz))
            for axis in (2, 1, 0):
                c = _step(_f(coordinates[axis] * self.reciprocals[axis]))
                n = self.blocks[axis]
                if not 0 <= c < n:
                    shift = _i(div(c, n))
                    removed[axis] = shift
                    for component in range(axis, -1, -1):
                        coordinates[component] = _f(
                            coordinates[component] - _f(shift * rows[axis][component]))
                    c = _sub(c, _mul(shift, n))
                block[axis] = c
            block[1] = _add(block[1], self.grid['ey'])
            block[2] = _add(block[2], self.grid['ez'])
        return tuple(coordinates), tuple(block), tuple(removed)

    def attribute(self, cell, origin):
        """Return the sole source-compatible class, or a structured failure."""
        source = cell.get('id')
        try:
            if not isinstance(source, int) or not 0 <= source < len(self.sites):
                raise ValueError('cell source ID is not persistent')
            owner = origin['owner']
            normal = tuple(_f(float(v)) for v in origin['normal'])
            offset = _f(float(origin['offset']))
            if len(normal) != 3 or origin['token'] <= 0:
                raise ValueError('malformed native origin')
            if origin['kind'] == 'orthogonal_seed':
                return self._orthogonal_seed(source, origin)
            if origin['kind'] == 'triclinic_seed':
                histories = self._seed(source, normal, offset)
                if owner != source:
                    raise ValueError('triclinic seed owner differs')
            elif origin['kind'] == 'particle':
                if not 0 <= owner < len(self.sites):
                    raise ValueError('particle owner is not persistent')
                histories = (self._box_particle(source, owner, normal, offset)
                             if self.kind == 'box' else
                             self._periodic_particle(source, owner, normal, offset))
            else:
                raise ValueError('unknown or construction-only origin')
        except (KeyError, IndexError, TypeError, ValueError, _InvalidRoute) as exc:
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH', str(exc),
                             source_id=source, token=origin.get('token')) from exc
        classes = {}
        for route in histories:
            classes.setdefault(route.coefficient, []).append(route)
        if not classes:
            raise WP5Failure('WP5_IMAGE_INCONSISTENT',
                             'Complete producer set has no compatible image',
                             source_id=source, token=origin['token'])
        if len(classes) != 1:
            raise WP5Failure('WP5_IMAGE_UNRESOLVED',
                             'Several producer-compatible image classes remain',
                             source_id=source, token=origin['token'],
                             candidates=tuple(sorted(classes)))
        coefficient, routes = next(iter(classes.items()))
        return Attribution(owner, coefficient, origin['kind'], tuple(routes))

    def _offsets(self, source, owner, normal, offset, *, direct=False):
        x, y, z = normal
        distance = _f(_f(_f(x*x) + _f(y*y)) + _f(z*z))
        if not self.power:
            return ('standard',) if bits_equal(distance, offset) else ()
        ri, rj = self.sites[source]['radius'], self.sites[owner]['radius']
        si, sj = _f(ri*ri), _f(rj*rj)
        matching = []
        for name in ('r_scale',) if direct else ('r_scale', 'r_scale_check'):
            try:
                value = (_f(_f(distance + si) - sj) if name == 'r_scale'
                         else _f(distance + _f(si - sj)))
                if bits_equal(value, offset):
                    matching.append(name)
            except _InvalidRoute:
                continue
        return tuple(matching)

    def _orthogonal_seed(self, source, origin):
        if self.kind != 'box':
            raise ValueError('orthogonal support outside a box container')
        axis, sense = origin['axis'], origin['sense']
        if axis not in (0, 1, 2) or sense not in (-1, 1):
            raise ValueError('invalid initialized side')
        side = -(2*axis + (1 if sense < 0 else 2))
        periodic = self.periodic[axis]
        expected_owner = source if periodic else side
        if (origin['side'] != side or origin['legacy_owner'] != side
                or origin['periodic'] != periodic
                or origin['owner'] != expected_owner):
            raise ValueError('initialized side provenance differs')
        normal = [0., 0., 0.]
        normal[axis] = float(sense)
        if periodic:
            upper = _f(0.5*self.periods[axis])
            limit = -upper if sense < 0 else upper
        else:
            bound = self.bounds[axis][0 if sense < 0 else 1]
            limit = _f(bound-self.sites[source]['site'][axis])
        doubled = _f(limit*2.)
        offset = -doubled if sense < 0 else doubled
        if (not _same(normal, origin['normal'])
                or not bits_equal(offset, origin['offset'])):
            raise ValueError('initialized side support bits differ')
        coefficient = tuple(sense if component == axis else 0 for component in range(3))
        return Attribution(expected_owner, coefficient if periodic else None,
                           'orthogonal_seed',
                           (Route('orthogonal-seed' if periodic else 'wall',
                                  coefficient if periodic else _ZERO, 'seed'),))

    def _seed(self, source, normal, offset):
        if self.kind != 'periodic':
            raise ValueError('triclinic support outside a periodic container')
        bx, bxy, by, bxz, byz, bz = self.params
        limit = _sub(_mul(2, int(self.packet['build']['max_unit_voro_shells'])), 1)
        plans = []
        for sign in (-1, 1):
            m = tuple(sign*component for component in normal)
            kb = self._bounds(self._preimage(rounding_bin(m[2])), bz,
                              minimum=-limit, maximum=limit)
            self.budget.candidates(_count(kb), stage='seed z coefficients')
            for k in _integers(kb):
                ky, kx = _f(k*byz), _f(k*bxz)
                jb = self._bounds(self._preimage(self._subtract(
                    self._preimage(rounding_bin(m[1])), ky)), by,
                    minimum=-limit, maximum=limit)
                self.budget.candidates(_count(jb), stage='seed y coefficients')
                for j in _integers(jb):
                    self.budget.charge(stage='seed coefficient bounds')
                    ib = self._bounds(self._preimage(self._subtract(self._preimage(
                        self._subtract(self._preimage(rounding_bin(m[0])), kx)),
                        _f(j*bxy))), bx, minimum=-limit, maximum=limit)
                    plans.append((sign, j, k, ib))
        self.budget.candidates(sum(_count(bounds) for _, _, _, bounds in plans),
                               stage='complete seed source family')
        routes = []
        for sign, j, k, bounds in plans:
            for i in _integers(bounds):
                self.budget.charge(stage='seed source replay')
                # unit_voro_apply visits the positive last nonzero coefficient
                # of each shell, then applies both signs of its rounded vector.
                if not (k > 0 or k == 0 and (j > 0 or j == 0 and i > 0)):
                    continue
                vector = (_f(_f(_f(i*bx)+_f(j*bxy))+_f(k*bxz)),
                          _f(_f(j*by)+_f(k*byz)), _f(k*bz))
                n = tuple(component if sign > 0 else -component for component in vector)
                if not _same(n, normal):
                    continue
                x, y, z = n
                h = _f(_f(_f(x*x)+_f(y*y))+_f(z*z))
                if bits_equal(h, offset):
                    routes.append(Route('triclinic-seed', (sign*i, sign*j, sign*k),
                                        'seed'))
        return routes

    def _direct(self, source, owner, normal, offset):
        left, right = self.sites[source], self.sites[owner]
        if source == owner or left['block'] != right['block']:
            return []
        n = tuple(_f(b-a) for a, b in zip(left['site'], right['site']))
        if not _same(n, normal):
            return []
        return [Route('direct', _ZERO, method, tuple(right['block']))
                for method in self._offsets(source, owner, n, offset, direct=True)]

    def _box_particle(self, source, owner, normal, offset):
        routes = self._direct(source, owner, normal, offset)
        left, right = self.sites[source], self.sites[owner]
        choices = [(-1, 0, 1) if p else (0,) for p in self.periodic]
        self.budget.candidates(math.prod(map(len, choices)), stage='box routes')
        center = tuple(n if p else c for n, p, c in zip(
            self.blocks, self.periodic, left['block']))
        for shift in product(*choices):
            self.budget.charge(stage='box source replay')
            axes = zip(left['block'], right['block'], self.blocks,
                       shift, self.periodic)
            indices = tuple(
                b-a+n+s*n if p else b
                for a, b, n, s, p in axes)
            if indices == center or any(not 0 <= e < h
                                        for e, h in zip(indices, self.mask)):
                continue
            try:
                # region_index evaluates ci+ei before either periodic branch.
                for c, e, periodic in zip(left['block'], indices, self.periodic):
                    if periodic:
                        _add(c, e)
                q = tuple((-period if s < 0 else period) if s else 0.
                          for period, s in zip(self.periods, shift))
                n = tuple(_f(b - _f(a-d)) for a, b, d in zip(
                    left['site'], right['site'], q))
                if _same(n, normal):
                    routes.extend(Route('box-worklist', shift, method,
                                        tuple(right['block']), q)
                                  for method in self._offsets(source, owner, n, offset))
            except _InvalidRoute:
                continue
        return routes

    def _worklist(self, source, owner, normal, offset, block, image, coefficient,
                  name, displacement=None, construction=None):
        routes = []
        left = self.sites[source]
        nx, _, _ = self.blocks
        ey, ez = self.grid['ey'], self.grid['ez']
        ci, cj, ck = left['block']
        for vx in (-1, 0, 1):
            ei, ej, ek = block[0]-ci+nx+vx*nx, block[1]-cj+ey, block[2]-ck+ez
            if ((ei, ej, ek) == (nx, ey, ez)
                    or any(not 0 <= e < h for e, h in zip((ei, ej, ek), self.mask))):
                continue
            try:
                qx = _f(vx * self.params[0]) if vx else 0.
                n = (_f(image[0] - _f(left['site'][0]-qx)),
                     _f(image[1] - _f(left['site'][1]-0.)),
                     _f(image[2] - _f(left['site'][2]-0.)))
                if not _same(n, normal):
                    continue
                shift = coefficient[0]+vx, coefficient[1], coefficient[2]
                routes.extend(Route(name, shift, method, block,
                                    displacement, construction)
                              for method in self._offsets(source, owner, n, offset))
            except _InvalidRoute:
                continue
        return routes

    def _bounds(self, interval, positive, *, minimum=_IMIN, maximum=_IMAX):
        if interval is None:
            return 1, 0
        result = interval_div(interval, Fraction(positive))
        for value in result:
            self.budget.fraction(value, stage='source coefficient preimage')
        lower, upper = integer_interval(result)
        return max(lower, minimum), min(upper, maximum)

    @staticmethod
    def _preimage(interval):
        return None if interval is None else rounding_preimage(interval)

    @staticmethod
    def _subtract(interval, value):
        return None if interval is None else interval_sub(interval, _point(value))

    def _displacement_interval(self, component, source, owner, normal):
        shifted = interval_add(rounding_bin(normal[component]),
                               _point(self.sites[source]['site'][component]))
        return self._subtract(self._preimage(shifted),
                              self.sites[owner]['site'][component])

    def _periodic_particle(self, source, owner, normal, offset):
        routes = self._direct(source, owner, normal, offset)
        row = self.sites[owner]
        routes.extend(self._worklist(source, owner, normal, offset,
                                     tuple(row['block']), tuple(row['site']),
                                     _ZERO, 'primary-worklist'))
        jy = self._displacement_interval(1, source, owner, normal)
        jz = self._displacement_interval(2, source, owner, normal)
        by, bz = self.params[2], self.params[5]
        side = self._bounds(self._preimage(jy), by)
        vertical = self._bounds(self._preimage(jz), bz)
        self.budget.candidates(_count(vertical), stage='vertical z family')
        plans = []
        for k in _integers(vertical):
            if k == 0:
                continue
            self.budget.charge(stage='vertical coefficient bounds')
            try:
                t = _f(k*self.params[4])
            except _InvalidRoute:
                continue
            ordinary = self._bounds(self._preimage(self._subtract(
                self._preimage(jy), t)), by)
            upper = self._bounds(self._preimage(self._subtract(self._preimage(
                self._subtract(self._preimage(jy), by)), t)), by)
            # Both are complete closed necessary families. Their union is kept
            # as intervals; source replay decides the upper-wrap branch.
            intervals = sorted((ordinary, upper))
            merged = []
            for low, high in intervals:
                if low > high:
                    continue
                if merged and low <= merged[-1][1]+1:
                    merged[-1] = merged[-1][0], max(high, merged[-1][1])
                else:
                    merged.append((low, high))
            plans.append((k, tuple(merged)))
        total = 7 + self.blocks[0]*(12*_count(side) + 48*sum(
            _count(bounds) for _, intervals in plans for bounds in intervals))
        self.budget.candidates(total, stage='complete triclinic source family')
        for beta in _integers(side):
            if beta:
                for di in range(self.blocks[0]):
                    try:
                        routes.extend(self._side(source, owner, normal, offset,
                                                 beta, di))
                    except _InvalidRoute:
                        continue
        for k, intervals in plans:
            for bounds in intervals:
                for beta in _integers(bounds):
                    for di in range(self.blocks[0]):
                        try:
                            routes.extend(self._vertical(source, owner, normal,
                                                         offset, k, beta, di))
                        except _InvalidRoute:
                            continue
        return routes

    def _image(self, source, owner, normal, offset, block, coefficient,
               displacement, name, construction):
        self.budget.charge(stage='triclinic source image replay')
        if any(not 0 <= v < limit for v, limit in zip(
                block, (self.blocks[0], self.grid['oy'], self.grid['oz']))):
            return []
        self._block_index(block)
        point = tuple(_f(p+d) for p, d in zip(self.sites[owner]['site'], displacement))
        return self._worklist(source, owner, normal, offset, block, point,
                              coefficient, name, displacement, construction)

    def _side(self, source, owner, normal, offset, beta, di):
        """container_prd.cc:583-624, one primary row and constructor block.

        A quadrant may have been populated by an earlier neighboring call.
        Enumerating both source quadrant histories retains that possibility;
        the bit masks suppress duplicate execution, not additional images.
        """
        self.budget.charge(stage='side source route')
        nx, ny, _ = self.blocks
        bx, bxy, by, _, _, _ = self.params
        sx, sy, sz = self.sites[owner]['block']
        px = self.sites[owner]['site'][0]
        dj, dk = _add(sy, _mul(beta, ny)), sz
        if (not 0 <= dj < self.grid['oy']
                or self.grid['ey'] <= dj < self.grid['wy']):
            return []
        if div(_sub(dj, self.grid['ey']), ny) != beta:
            return []
        qua = _add(di, _step(_f(_f(_i(-beta)*bxy) * self.reciprocals[0])))
        alpha = _i(div(qua, nx))
        fi = _sub(qua, _mul(alpha, nx))
        dis = _f(_f(beta*bxy) + _f(alpha*bx))
        switch = _f(_f(_f(di*self.widths[0]) - _f(beta*bxy)) - _f(alpha*bx))
        dy = _f(by*beta)
        routes = []
        construction = di, dj, dk
        self._block_index(construction)
        if sx == fi:
            if px > switch:
                outx, dx, ax = di, dis, alpha
            elif di > 0:
                outx, dx, ax = di-1, dis, alpha
            else:
                outx, dx, ax = nx-1, _f(dis+bx), alpha+1
            routes.extend(self._image(source, owner, normal, offset,
                                      (outx, dj, dk), (ax, beta, 0),
                                      (dx, dy, 0.), 'side-left', construction))
        if fi == nx-1:
            fi = 0
            switch = _f(switch + _f(_sub(1, nx)*self.widths[0]))
            dis = _f(dis+bx)
            alpha += 1
        else:
            fi = _add(fi, 1)
            switch = _f(switch+self.widths[0])
        if sx == fi:
            if px < switch:
                outx, dx, ax = di, dis, alpha
            elif di == nx-1:
                outx, dx, ax = 0, _f(dis-bx), alpha-1
            else:
                outx, dx, ax = di+1, dis, alpha
            routes.extend(self._image(source, owner, normal, offset,
                                      (outx, dj, dk), (ax, beta, 0),
                                      (dx, dy, 0.), 'side-right', construction))
        return routes

    def _vertical(self, source, owner, normal, offset, k, beta, di):
        """container_prd.cc:635-757, including both possible source rows."""
        self.budget.charge(stage='vertical source route')
        nx, ny, nz = self.blocks
        bx, bxy, by, bxz, byz, bz = self.params
        sx, sy, sz = self.sites[owner]['block']
        px, py, _ = self.sites[owner]['site']
        ey, wy, oy = self.grid['ey'], self.grid['wy'], self.grid['oy']
        dk = _add(sz, _mul(k, nz))
        if (not 0 <= dk < self.grid['oz']
                or self.grid['ez'] <= dk < self.grid['wz']):
            return []
        if div(_sub(dk, self.grid['ez']), nz) != k:
            return []
        ystep = _step(_f(_f(_i(-k)*byz) * self.reciprocals[1]))
        # A copied row is either the lower source row or its successor, with
        # the latter wrapped from wy-1 to ey. These are all possible dj's.
        lower_rows = (sy, sy-1 if sy > ey else wy-1)
        starts = {_sub(_add(fj, _mul(beta, ny)), ystep) for fj in lower_rows}
        routes = []
        for dj in sorted(starts):
            if not 0 <= dj < oy:
                continue
            qj = _add(dj, ystep)
            if div(_sub(qj, ey), ny) != beta:
                continue
            fj = _sub(qj, _mul(beta, ny))
            qi = _add(di, _step(_f(_f(_f(_i(-k)*bxz) - _f(beta*bxy))
                                   * self.reciprocals[0])))
            alpha = _i(div(qi, nx))
            fi = _sub(qi, _mul(alpha, nx))
            fijk = self._block_index((fi, fj, sz))
            disy = _f(_f(k*byz) + _f(beta*by))
            switchy = _f(_f(_f(_sub(dj, ey)*self.widths[1]) - _f(k*byz))
                         - _f(beta*by))
            disx = _f(_f(_f(k*bxz) + _f(beta*bxy)) + _f(alpha*bx))
            switchx = _f(_f(di*self.widths[0]) - _f(k*bxz))
            switchx = _f(switchx - _f(beta*bxy))
            switchx = _f(switchx - _f(alpha*bx))
            disxl = _f(disx+bx) if di == 0 else disx
            disxr = _f(disx-bx) if di == nx-1 else disx
            # Coefficient bookkeeping is exact mathematical integer arithmetic.
            # The producer wraps disx as a double; it does not increment qidiv.
            axl = alpha+1 if di == 0 else alpha
            axr = alpha-1 if di == nx-1 else alpha
            left = nx-1 if di == 0 else di-1
            right = 0 if di == nx-1 else di+1
            construction = di, dj, dk
            self._block_index(construction)

            def emit_pair(up, fi, fj, switchx, switchy, disx, disxl, disxr,
                          alpha, axl, axr, image_beta, disy):
                found = []
                if sy != fj:
                    return found
                if up:
                    outy = dj+1 if py > switchy else dj
                else:
                    outy = dj if py > switchy else dj-1
                if not 0 <= outy < oy:
                    return found
                if sx == fi:
                    ox, dx, ax = ((di, disx, alpha) if px > switchx
                                  else (left, disxl, axl))
                    found.extend(self._image(
                        source, owner, normal, offset, (ox, outy, dk),
                        (ax, image_beta, k), (dx, disy, _f(bz*k)),
                        'vertical-up-left' if up else 'vertical-down-left',
                        construction))
                if fi == nx-1:
                    fi2 = 0
                    switch2 = _f(switchx + _f(_sub(1, nx)*self.widths[0]))
                    dx2, dxr2 = _f(disx+bx), _f(disxr+bx)
                    a2, ar2 = alpha+1, axr+1
                else:
                    fi2 = _add(fi, 1)
                    switch2 = _f(switchx+self.widths[0])
                    dx2, dxr2, a2, ar2 = disx, disxr, alpha, axr
                if sx == fi2:
                    ox, dx, ax = ((right, dxr2, ar2) if px > switch2
                                  else (di, dx2, a2))
                    found.extend(self._image(
                        source, owner, normal, offset, (ox, outy, dk),
                        (ax, image_beta, k), (dx, disy, _f(bz*k)),
                        'vertical-up-right' if up else 'vertical-down-right',
                        construction))
                return found

            routes.extend(emit_pair(False, fi, fj, switchx, switchy, disx, disxl,
                                    disxr, alpha, axl, axr, beta, disy))
            image_beta = beta
            if fj == wy-1:
                # These are distinct source operations, not a lattice sum.
                fijk = _sub(_add(fijk, _mul(nx, _sub(1, ny))), fi)
                switchy = _f(switchy + _f(_sub(1, ny)*self.widths[1]))
                disy = _f(disy+by)
                image_beta = _add(beta, 1)
                upper = _f(-_f(_f(k*bxz) + _f(image_beta*bxy)))
                qi = _add(di, _step(_f(upper*self.reciprocals[0])))
                new_alpha = _i(div(qi, nx))
                change = _sub(new_alpha, alpha)
                alpha = _add(alpha, change)
                fi = _sub(qi, _mul(alpha, nx))
                _add(fijk, fi)
                correction = _f(bxy + _f(bx*change))
                disx, disxl, disxr = (_f(disx+correction), _f(disxl+correction),
                                      _f(disxr+correction))
                switchx = _f(switchx-correction)
                axl, axr = axl+change, axr+change
                fj = ey
            else:
                _add(fijk, nx)
                fj = _add(fj, 1)
                switchy = _f(switchy+self.widths[1])
            routes.extend(emit_pair(True, fi, fj, switchx, switchy, disx, disxl,
                                    disxr, alpha, axl, axr, image_beta, disy))
        return routes
