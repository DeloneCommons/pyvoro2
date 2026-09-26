"""Independent exact WP7 ghost ideal; no production geometry/provenance imports.

All input floats are exactified as their individual binary64 rational values.
The rectangle family is complete: self images confine each periodic coordinate
to +/- half a period. Center an owner's displacement into that interval and
retain its centered image and its immediate neighbors in each periodic axis.
For every omitted same-owner image, the retained nearest-side image has a
strictly smaller squared distance throughout that slab; weights cancel.

For a full 3D lattice, +/- each supplied row gives a bounded outer polytope
P0. Let R=max_vertex sum(abs(coordinate)), hence ||x|| <= R on P0. A cut at
displacement d and weight difference delta can meet P0 only if
||d|| <= R+sqrt(R*R+abs(delta)). An outward rational bound T is used below.
Since d=(P-g)+s@A, inverse-column L1 bounds then include every possible
integer s. The complete coefficient box is preflighted before enumeration;
each candidate is retained exactly when its plane can meet/restrict P0.
Equality is retained. This is intentionally unrelated to native producers.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product, combinations
from math import comb, isqrt, prod


def _f(value):
    return value if isinstance(value, F) else F(value)


def _vec(values):
    return tuple(map(_f, values))


def _dot(a, b):
    return sum((u * v for u, v in zip(a, b)), F(0))


def _rank(vectors):
    if not vectors:
        return -1
    rows = [[*map(F, (a - b for a, b in zip(v, vectors[0])))]
            for v in vectors[1:]]
    rank = 0
    for col in range(len(vectors[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        q = rows[rank][col]
        rows[rank] = [x / q for x in rows[rank]]
        for i in range(rank + 1, len(rows)):
            q = rows[i][col]
            rows[i] = [x - q * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def _solve(rows, rhs):
    n = len(rows)
    a = [list(row) + [v] for row, v in zip(rows, rhs)]
    for j in range(n):
        pivot = next((k for k in range(j, n) if a[k][j]), None)
        if pivot is None:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        q = a[j][j]
        a[j] = [v / q for v in a[j]]
        for k in range(n):
            if k != j:
                q = a[k][j]
                a[k] = [v - q * w for v, w in zip(a[k], a[j])]
    return tuple(row[-1] for row in a)


def _vertices(planes, dim):
    result = set()
    for triple in combinations(planes, dim):
        x = _solve([p.normal for p in triple], [p.limit for p in triple])
        if x is not None and all(_dot(p.normal, x) <= p.limit
                                 for p in planes):
            result.add(x)
    return tuple(sorted(result))


def _ceil(q):
    return -(-q.numerator // q.denominator)


def _floor(q):
    return q.numerator // q.denominator


def _ceil_sqrt(q):
    # Smallest integer N with N^2 >= q, using integer arithmetic only.
    n = isqrt(_ceil(q))
    return n if n * n >= q else n + 1


def _det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _order_face(points, normal):
    center = tuple(sum(v[j] for v in points) / len(points) for j in range(3))
    u = tuple(x - y for x, y in zip(min(points), center))
    v = _cross(normal, u)

    def coordinates(p):
        w = tuple(x - y for x, y in zip(p, center))
        return _dot(u, w), _dot(v, w)

    def half(p):
        x, y = coordinates(p)
        return 0 if y > 0 or (y == 0 and x >= 0) else 1

    from functools import cmp_to_key

    def compare(a, b):
        ha, hb = half(a), half(b)
        if ha != hb:
            return -1 if ha < hb else 1
        ax, ay = coordinates(a)
        bx, by = coordinates(b)
        z = ax * by - ay * bx
        return -1 if z > 0 else (1 if z < 0 else 0)
    return sorted(points, key=cmp_to_key(compare))


@dataclass(frozen=True)
class Plane:
    key: tuple
    normal: tuple
    limit: F


@dataclass(frozen=True)
class Contact:
    status: str
    vertices: tuple = ()


@dataclass
class IdealCell:
    dimension: int
    vertices: tuple
    measure: F
    contacts: dict

    def contact(self, kind, owner=None, shift=None, wall_id=None):
        key = ((kind, owner, tuple(shift)) if kind == 'generator'
               else (kind, tuple(shift)) if kind == 'ghost_self'
               else ('wall', wall_id))
        return self.contacts.get(key, Contact('absent'))

    @property
    def positive(self):
        return {key: c for key, c in self.contacts.items()
                if c.status == 'positive'}

    @property
    def facets(self):
        """Group coincident positive classes by exact geometric vertex sets."""
        groups = {}
        for key, contact in self.positive.items():
            groups.setdefault(frozenset(contact.vertices), set()).add(key)
        return groups


def _image(d, s, lattice):
    return tuple(d[j] + sum(s[i] * lattice[i][j] for i in range(len(s)))
                 for j in range(len(d)))


def _short_outer_rows(lattice):
    """Elementary exact integral size reduction for a smaller P0, not N proof.

    Each replacement is unimodular and strictly decreases a squared row norm;
    original user coefficients are tracked explicitly. A finite work guard
    refuses uncomfortably poor bases instead of certifying a partial search.
    """
    dim = len(lattice)
    rows = list(lattice)
    coefficients = [tuple(int(i == j) for j in range(dim))
                    for i in range(dim)]
    for _ in range(1000):
        changed = False
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                n = _floor(_dot(rows[i], rows[j]) / _dot(rows[j], rows[j])
                           + F(1, 2))
                replacement = tuple(a - n * b for a, b in zip(rows[i], rows[j]))
                if _dot(replacement, replacement) < _dot(rows[i], rows[i]):
                    rows[i] = replacement
                    coefficients[i] = tuple(a - n * b for a, b in zip(
                        coefficients[i], coefficients[j]))
                    changed = True
        if not changed:
            return tuple(zip(coefficients, rows))
    raise ValueError('oracle exact outer-basis reduction budget')


def _rect_shifts(d, periods, mask):
    axes = []
    for dx, length, periodic in zip(d, periods, mask):
        if periodic:
            center = _floor(-dx / length + F(1, 2))
            axes.append((center - 1, center, center + 1))
        else:
            axes.append((0,))
    return product(*axes)


def _lattice_shifts(d, delta, lattice, outer_vertices, max_candidates):
    dim = len(d)
    R = max(sum(map(abs, x)) for x in outer_vertices)
    T = R + _ceil_sqrt(R * R + abs(delta))
    inverse = []
    for col in range(dim):
        solution = _solve([list(row) for row in zip(*lattice)],
                          tuple(F(int(k == col)) for k in range(dim)))
        if solution is None:
            raise ValueError('singular exact lattice')
        inverse.append(solution)
    # inverse[k] is a column of A^-1; d @ A^-1 uses one component from each.
    columns = tuple(zip(*inverse))
    ranges = []
    for col in columns:
        q = _dot(d, col)
        width = T * sum(map(abs, col))
        ranges.append(range(_ceil(-q - width), _floor(-q + width) + 1))
    count = prod(map(len, ranges))
    if count > max_candidates:
        raise ValueError(f'oracle candidate budget: {count} > {max_candidates}')
    for s in product(*ranges):
        vector = _image(d, s, lattice)
        limit = _dot(vector, vector) + delta
        if limit <= max(2 * _dot(vector, x) for x in outer_vertices):
            yield s


def _line_contact(plane, planes):
    a, b = plane.normal
    square = a * a + b * b
    if not square:
        return Contact('identical' if plane.limit == 0 else 'absent')
    x0 = (a * plane.limit / square, b * plane.limit / square)
    direction = (-b, a)
    low = high = None
    for other in planes:
        v = _dot(other.normal, direction)
        residual = other.limit - _dot(other.normal, x0)
        if v == 0:
            if residual < 0:
                return Contact('absent')
        elif v > 0:
            bound = residual / v
            high = bound if high is None else min(high, bound)
        else:
            bound = residual / v
            low = bound if low is None else max(low, bound)
    if low is None or high is None or low > high:
        return Contact('absent')
    values = (low,) if low == high else (low, high)
    points = tuple(tuple(x0[i] + t * direction[i] for i in range(2))
                   for t in values)
    return Contact('point' if len(points) == 1 else 'positive',
                   tuple(sorted(points)))


def _area(vertices):
    if len(vertices) < 3:
        return F(0)
    # For a convex polygon, its centroid fan covers it without sorting.
    center = tuple(sum(x[i] for x in vertices) / len(vertices)
                   for i in range(2))
    edges = set()
    for a, b in combinations(vertices, 2):
        if all((b[0] - a[0]) * (c[1] - a[1])
               - (b[1] - a[1]) * (c[0] - a[0]) >= 0 for c in vertices) or all(
                (b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]) <= 0 for c in vertices):
            # Collinear intermediate vertices are excluded from hull edges.
            if any(c not in (a, b) and
                   (b[0] - a[0]) * (c[1] - a[1])
                   == (b[1] - a[1]) * (c[0] - a[0]) and
                   min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                   min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
                   for c in vertices):
                continue
            edges.add((a, b))
    return sum(abs((a[0] - center[0]) * (b[1] - center[1])
                   - (a[1] - center[1]) * (b[0] - center[0])) / 2
               for a, b in edges)


def _volume(vertices, planes):
    center = tuple(sum(x[i] for x in vertices) / len(vertices)
                   for i in range(3))
    faces = {}
    for plane in planes:
        face = frozenset(x for x in vertices
                         if _dot(plane.normal, x) == plane.limit)
        if _rank(tuple(face)) == 2:
            faces.setdefault(face, plane.normal)
    total = F(0)
    for face, normal in faces.items():
        ordered = _order_face(tuple(face), normal)
        a = ordered[0]
        for k in range(1, len(ordered) - 1):
            b, c = ordered[k:k + 2]
            total += abs(_det3(tuple(x - y for x, y in zip(a, center)),
                               tuple(x - y for x, y in zip(b, center)),
                               tuple(x - y for x, y in zip(c, center)))) / 6
    return total


def ghost_ideal(original_points, stored_ghost, bounds, periods, periodic_mask,
                weights=None, ghost_weight=0, lattice=None,
                max_candidates=100_000, max_intersections=200_000):
    """Exact cell and all candidate semantic classes in the stored-ghost chart.

    ``periods`` are the declared binary64 rectangle spans, independently of
    exactified endpoint subtraction. In 3D triclinic mode pass ``lattice`` as
    the *original user row basis*, and all axes periodic; bounds are ignored.
    Weights are mathematical values; a caller using radii passes F(radius)**2.
    """
    g = _vec(stored_ghost)
    dim = len(g)
    assert dim in (2, 3) and len(periodic_mask) == dim
    points = tuple(map(_vec, original_points))
    values = tuple(map(_f, weights if weights is not None
                       else [0] * len(points)))
    wg = _f(ghost_weight)
    mask = tuple(map(bool, periodic_mask))
    if lattice is None:
        spans = _vec(periods)
        A = tuple(tuple(spans[i] if i == j else F(0)
                        for j in range(dim)) for i in range(dim))
    else:
        A = tuple(map(_vec, lattice))
        assert dim == 3 and all(mask)
    planes = []
    outer = []
    if lattice is not None:
        for coeff, row in _short_outer_rows(A):
            for sign in (-1, 1):
                d = tuple(sign * x for x in row)
                outer.append(Plane(('ghost_self', tuple(sign * x for x in coeff)),
                                   tuple(2 * x for x in d), _dot(d, d)))
    for j in range(dim):
        if mask[j] and lattice is None:
            for sign in (-1, 1):
                displacement = tuple(sign * t for t in A[j])
                p = Plane(('ghost_self', tuple(sign if k == j else 0
                                               for k in range(dim))),
                          tuple(2 * t for t in displacement),
                          _dot(displacement, displacement))
                outer.append(p)
        elif not mask[j]:
            lo, hi = map(_f, bounds[j])
            for sign, wall, limit in ((-1, -(2 * j + 1), g[j] - lo),
                                      (1, -(2 * j + 2), hi - g[j])):
                normal = tuple(F(sign if k == j else 0)
                               for k in range(dim))
                outer.append(Plane(('wall', wall), normal, limit))
    outer_vertices = _vertices(outer, dim)
    assert outer_vertices, 'outer ghost-self/wall polytope must be bounded'
    planes.extend(outer)
    shifts = (lambda d, delta: _lattice_shifts(
        d, delta, A, outer_vertices, max_candidates)) if lattice is not None else (
        lambda d, delta: _rect_shifts(d, spans, mask))
    for s in shifts(tuple(F(0) for _ in g), F(0)):
        if not any(s):
            continue
        d = _image(tuple(F(0) for _ in g), s, A)
        planes.append(Plane(('ghost_self', tuple(s)), tuple(2 * x for x in d),
                            _dot(d, d)))
    for owner, (point, weight) in enumerate(zip(points, values)):
        d0 = tuple(p - x for p, x in zip(point, g))
        for s in shifts(d0, wg - weight):
            d = _image(d0, s, A)
            planes.append(Plane(('generator', owner, tuple(s)),
                                tuple(2 * x for x in d), _dot(d, d) + wg - weight))
    # The same axial self constraints appear in both outer and complete sets.
    unique = {p.key: p for p in planes}
    planes = tuple(unique.values())
    intersection_count = comb(len(planes), dim)
    if intersection_count > max_intersections:
        raise ValueError('oracle intersection budget: '
                         f'{intersection_count} > {max_intersections}')
    vertices = _vertices(planes, dim)
    rank = _rank(vertices)
    contacts = {}
    for plane in planes:
        if plane.normal == (F(0),) * dim:
            c = Contact('identical' if plane.limit == 0 else 'absent')
        elif rank < 0:
            c = Contact('absent')
        elif dim == 2:
            c = _line_contact(plane, planes)
        else:
            on = tuple(v for v in vertices if _dot(plane.normal, v) == plane.limit)
            contact_rank = _rank(on)
            c = Contact(('absent', 'point', 'line', 'positive')[contact_rank + 1],
                        on)
        if rank != dim and c.status in ('positive', 'point', 'line'):
            c = Contact('lower-dimensional', c.vertices)
        contacts[plane.key] = c
    measure = (_area(vertices) if dim == 2 else _volume(vertices, planes))\
        if rank == dim else F(0)
    return IdealCell(rank, vertices, measure, contacts)
