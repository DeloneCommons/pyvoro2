"""Independent test-only exact line-parameter feasibility oracle.

No pyvoro2 geometry is imported. Each supporting line is reduced to a rational
one-dimensional interval. The image family uses floor-centered FOUR images
per periodic axis, containing the nearest-centered three-image family. Every
extreme point of a nonempty bounded cell occurs as an interval endpoint.
"""

from fractions import Fraction as F
from itertools import product


def dimension(points):
    points = sorted(set(points))
    if len(points) < 2:
        return len(points) - 1
    a, b = points[:2]
    return 2 if any(
        (b[0] - a[0]) * (c[1] - a[1])
        != (b[1] - a[1]) * (c[0] - a[0]) for c in points[2:]
    ) else 1


def line_contact(normal, offset, rows):
    if not any(normal):
        raise ValueError('line oracle requires a nonzero normal')
    x, y = normal
    base = (offset / x, F(0)) if x else (F(0), offset / y)
    tangent = (-y, x)
    low = high = None
    for (a, b), bound in rows:
        slope = a * tangent[0] + b * tangent[1]
        slack = bound - a * base[0] - b * base[1]
        if slope == 0:
            if slack < 0:
                return ()
        elif slope > 0:
            candidate = slack / slope
            high = candidate if high is None else min(high, candidate)
        else:
            candidate = slack / slope
            low = candidate if low is None else max(low, candidate)
        if low is not None and high is not None and low > high:
            return ()
    assert low is not None and high is not None
    return tuple(sorted({
        (base[0] + t * tangent[0], base[1] + t * tangent[1])
        for t in (low, high)
    }))


def oracle_cell(points, weights, bounds, periods, periodic, source):
    points = [tuple(map(F, p)) for p in points]
    weights = list(map(F, weights))
    bounds = [tuple(map(F, b)) for b in bounds]
    periods = tuple(map(F, periods))
    point = points[source]
    rows, labels = [], {}
    for axis in range(2):
        low, high = ((-periods[axis] / 2, periods[axis] / 2)
                     if periodic[axis]
                     else tuple(v - point[axis] for v in bounds[axis]))
        for side, (sign, value) in enumerate(((-1, -low), (1, high))):
            normal = (F(sign), F(0)) if axis == 0 else (F(0), F(sign))
            row = (normal, value)
            rows.append(row)
            if not periodic[axis]:
                labels[-(2 * axis + side + 1)] = row
    for owner, other in enumerate(points):
        ranges = []
        for axis in range(2):
            center = (point[axis] - other[axis]) // periods[axis]
            ranges.append(range(center - 1, center + 3)
                          if periodic[axis] else (0,))
        for shift in product(*ranges):
            d = tuple(other[k] + shift[k] * periods[k] - point[k]
                      for k in range(2))
            normal = (2 * d[0], 2 * d[1])
            offset = d[0] ** 2 + d[1] ** 2 + weights[source] - weights[owner]
            row = normal, offset
            rows.append(row)
            if (owner, shift) != (source, (0, 0)):
                labels[owner, shift] = row
    vertices = set()
    for normal, offset in rows:
        if any(normal):
            vertices.update(line_contact(normal, offset, rows))
    rank = dimension(vertices)
    contacts = {}
    for label, (normal, offset) in labels.items():
        if not any(normal):
            status = 'identical' if offset == 0 else 'absent'
            endpoints = tuple(sorted(vertices)) if offset == 0 else ()
        else:
            endpoints = line_contact(normal, offset, rows)
            contact_rank = dimension(endpoints)
            status = ('absent' if contact_rank < 0 else 'lower-dimensional'
                      if rank < 2 else 'point' if contact_rank == 0 else 'positive')
        contacts[label] = status, endpoints
    return rank, vertices, contacts
