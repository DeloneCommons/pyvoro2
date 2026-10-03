"""Small independent Fraction oracle inherited from #96's spatial_oracle.py.

No production geometry is imported. Ordered facet polygons are clipped by
Sutherland--Hodgman; an exact planar hull closes each cap. Self-image slabs
and real walls bound the initial cell. If its radius is at most R, any image
cut meeting it has |d| <= R + sqrt(R**2 + |wi-wj|). Outward rational roots
and inverse-column L1 bounds therefore enclose *all* relevant images.

This test utility returns exact incidence, never native vertex identities.
Only the compact cell/quotient part of the reviewed oracle is retained;
campaign, provenance-graph and serialization machinery is intentionally absent.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import product
from math import ceil, floor, isqrt


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, t):
    return tuple(x * t for x in a)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F())


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def rowmat(v, m):
    return tuple(sum((v[k] * m[k][j] for k in range(3)), F()) for j in range(3))


def inverse(m):
    det = dot(m[0], cross(m[1], m[2]))
    assert det
    cols = [mul(cross(m[1], m[2]), 1 / det),
            mul(cross(m[2], m[0]), 1 / det), mul(cross(m[0], m[1]), 1 / det)]
    return tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))


def dimension(points):
    points = list(points)
    if not points:
        return -1
    ds = [sub(p, points[0]) for p in points[1:] if p != points[0]]
    if not ds:
        return 0
    normal = next((cross(ds[0], d) for d in ds if any(cross(ds[0], d))), None)
    if normal is None:
        return 1
    return 3 if any(dot(normal, d) for d in ds) else 2


def sqrt_upper(x, bits=24):
    a, b = isqrt(x.numerator), isqrt(x.denominator)
    if a * a == x.numerator and b * b == x.denominator:
        return F(a, b)
    n = x.numerator * x.denominator * (1 << (2 * bits))
    r = isqrt(n)
    return F(r + (r * r < n), x.denominator * (1 << bits))


def hull(points, normal):
    points = sorted(set(points))
    if dimension(points) < 2:
        return points
    drop = next(k for k, v in enumerate(normal) if v)
    a, b = [k for k in range(3) if k != drop]
    points.sort(key=lambda p: (p[a], p[b]))

    def turn(o, p, q):
        u, v = sub(p, o), sub(q, o)
        return u[a] * v[b] - u[b] * v[a]

    polygon = []
    for seq in (points, list(reversed(points))):
        half = []
        for p in seq:
            while len(half) > 1 and turn(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        polygon.extend(half[:-1])
    return polygon


def polygon_normal(points):
    return next(cross(sub(a, points[0]), sub(b, points[0]))
                for a in points[1:] for b in points[1:]
                if any(cross(sub(a, points[0]), sub(b, points[0]))))


def vertices(faces):
    return sorted({p for face in faces for p in face})


def clip(faces, n, b):
    gaps = {p: dot(n, p) - b for p in vertices(faces)}
    if all(g <= 0 for g in gaps.values()):
        return faces
    if all(g > 0 for g in gaps.values()):
        return []
    new, cap = [], set()
    for face in faces:
        polygon = []
        for u, v in zip(face, face[1:] + face[:1]):
            gu, gv = gaps[u], gaps[v]
            if gu <= 0:
                polygon.append(u)
                if gu == 0:
                    cap.add(u)
            if (gu < 0 < gv) or (gv < 0 < gu):
                p = add(u, mul(sub(v, u), gu / (gu - gv)))
                polygon.append(p)
                cap.add(p)
        if dimension(polygon) == 2:
            new.append(hull(polygon, polygon_normal(polygon)))
    if dimension(cap) == 2:
        new.append(hull(cap, n))
    return new


def cell(source, sites, lattice, weights, periodic, bounds):
    normals, intervals, cuts = [], [], []
    for k in range(3):
        if periodic[k]:
            a = lattice[k]
            c = dot(a, a) / 2
            normals.append(a)
            intervals.append((-c, c))
            for sign in (-1, 1):
                shift = tuple(sign if j == k else 0 for j in range(3))
                cuts.append(((source, shift), mul(a, 2 * sign), 2 * c))
        else:
            a = tuple(F(j == k) for j in range(3))
            normals.append(a)
            lo, hi = (F(x) - sites[source][k] for x in bounds[k])
            intervals.append((lo, hi))
            cuts.extend([(('wall', -2 * k - 1), mul(a, -1), -lo),
                         (('wall', -2 * k - 2), a, hi)])
    corners = {bits: rowmat(
        tuple(intervals[k][bits[k]] for k in range(3)), tuple(zip(*inverse(normals))))
               for bits in product((0, 1), repeat=3)}
    faces = [hull([p for bits, p in corners.items() if bits[k] == side], normals[k])
             for k in range(3) for side in (0, 1)]
    outer = vertices(faces)
    radius = sqrt_upper(max(dot(p, p) for p in outer))
    inv = inverse(lattice)
    seed_count, labels = len(cuts), {label for label, _, _ in cuts}
    for owner, p in enumerate(sites):
        delta, dw = sub(p, sites[source]), weights[source] - weights[owner]
        distance = radius + sqrt_upper(radius * radius + abs(dw))
        regions = []
        for k in range(3):
            col = tuple(row[k] for row in inv)
            center, bound = -dot(delta, col), distance * sum(abs(x) for x in col)
            lo, hi = (ceil(center - bound), floor(center + bound))
            regions.append(range(lo, hi + 1) if periodic[k] else (0,))
        for shift in product(*regions):
            label = (owner, shift)
            if label in labels or (owner == source and not any(shift)):
                continue
            d = add(delta, rowmat(shift, lattice))
            n, b = mul(d, 2), dot(d, d) + dw
            if b <= max(dot(n, v) for v in outer):
                cuts.append((label, n, b))
    cuts[seed_count:] = sorted(
        cuts[seed_count:], key=lambda c: (dot(c[1], c[1]), repr(c[0])))
    for _, n, b in cuts[seed_count:]:
        faces = clip(faces, n, b)
    verts = vertices(faces)
    assert dimension(verts) == 3
    contacts = {}
    for label, n, b in cuts:
        assert all(dot(n, p) <= b for p in verts)
        ps = [p for p in verts if dot(n, p) == b]
        if ps:
            contacts[label] = tuple(hull(ps, n))
    facets = {frozenset(ps) for ps in contacts.values() if dimension(ps) == 2}
    edge_counts = Counter()
    for facet in facets:
        polygon = hull(facet, polygon_normal(list(facet)))
        edge_counts.update(tuple(sorted((u, v)))
                           for u, v in zip(polygon, polygon[1:] + polygon[:1]))
    assert set(edge_counts.values()) == {2}
    assert len(verts) - len(edge_counts) + len(facets) == 2
    return dict(vertices=verts, contacts=contacts,
                edges=set(edge_counts), facets=facets)


def quotient_key(points, lattice, periodic):
    """Exact lifted geometry modulo one *common* allowed lattice translation."""
    inv = inverse(lattice)
    fractional = [rowmat(p, inv) for p in points]
    forms = []
    for anchor in fractional:
        shift = tuple(floor(x) if flag else 0 for x, flag in zip(anchor, periodic))
        forms.append(tuple(sorted(sub(p, shift) for p in fractional)))
    return min(forms)


def diagram(points, weights=None, lattice=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            periodic=(True, True, True), bounds=((0, 1),) * 3):
    sites = [tuple(F(x) for x in p) for p in points]
    lattice = [tuple(F(x) for x in row) for row in lattice]
    weights = [F(w) for w in (weights if weights is not None else [0] * len(sites))]
    cells = [cell(i, sites, lattice, weights, periodic, bounds)
             for i in range(len(sites))]
    vs, es, fs = set(), set(), set()
    for site, c in zip(sites, cells):
        vs.update(quotient_key([add(site, v)], lattice, periodic)[0]
                  for v in c['vertices'])
        es.update(quotient_key([add(site, p) for p in e], lattice, periodic)
                  for e in c['edges'])
        fs.update(quotient_key([add(site, p) for p in f], lattice, periodic)
                  for f in c['facets'])
    counts = (len(vs), len(es), len(fs), len(cells))
    assert counts[0] - counts[1] + counts[2] - counts[3] == 0
    return dict(cells=cells, counts=counts, quotient_vertices=vs)
