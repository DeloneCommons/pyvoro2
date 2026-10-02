"""Small value-binding and integer graph utilities for proved identities.

Geometry belongs to the producing adapter. Numerical coincidence and boundary
incidence cannot enter this graph. No public or serialized certification token
is provided by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import struct

import numpy as np


class ProofFailure(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def value_digest(value):
    """Canonical typed values, including binary64 bits and arbitrary integers."""
    digest = hashlib.sha256()

    def feed(item):
        if isinstance(item, np.ndarray):
            feed(('array', item.dtype.str, item.shape, item.tolist()))
        elif isinstance(item, np.generic):
            feed(item.item())
        elif item is None:
            digest.update(b'n')
        elif type(item) is bool:
            digest.update(b't' if item else b'f')
        elif type(item) is int:
            data = str(item).encode('ascii')
            digest.update(b'i' + str(len(data)).encode('ascii') + b':' + data)
        elif isinstance(item, Fraction):
            digest.update(b'r')
            feed((item.numerator, item.denominator))
        elif type(item) is float:
            digest.update(b'd' + struct.pack('>d', item))
        elif isinstance(item, (bytes, str)):
            data = item.encode('utf8') if isinstance(item, str) else item
            digest.update((b's' if isinstance(item, str) else b'b')
                          + str(len(data)).encode('ascii') + b':' + data)
        elif isinstance(item, dict):
            digest.update(b'm')
            # Key digests give a stable ordering even for heterogeneous keys.
            feed(tuple(sorted(item.items(), key=lambda pair: value_digest(pair[0]))))
        elif isinstance(item, (tuple, list)):
            digest.update((b'(' if isinstance(item, tuple) else b'[')
                          + str(len(item)).encode('ascii') + b':')
            for member in item:
                feed(member)
        else:
            raise TypeError(f'unsupported proof operand: {type(item).__name__}')

    feed(value)
    return digest.digest()


@dataclass(frozen=True)
class VertexIdentity:
    domain: bytes
    dimension: int
    first: tuple[int, int]
    second: tuple[int, int]
    shift: tuple[int, ...]
    kind: str
    sources: tuple[tuple[int, int], ...]

    def operands(self):
        return (self.domain, self.dimension, self.first, self.second,
                self.shift, self.kind, self.sources)


def identity_closure(nodes, identities, *, domain, dimension, anchors=()):
    """Return (component, potential) with X(u)=X(root)+potential(u)@A.

    Anchors are adapter-proved (quotient point, integer lift) pairs. Every
    alternative path is checked, including closed cycles; the integers have
    no public int64 restriction here. Incidence loops must stay outside.
    """
    graph = {node: [] for node in nodes}

    def conflict(message):
        raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT', message)

    for edge in identities:
        if (not isinstance(edge, VertexIdentity) or edge.domain != domain
                or edge.dimension != dimension
                or edge.kind not in ('point-alias', 'reciprocal-endpoint')
                or not edge.sources or len(edge.shift) != dimension
                or any(type(v) is not int for v in edge.shift)
                or edge.first not in graph or edge.second not in graph):
            conflict('Relation is not a certified identity in this proof domain')
        graph[edge.first].append((edge.second, tuple(-s for s in edge.shift)))
        graph[edge.second].append((edge.first, edge.shift))

    result = {}
    anchors = dict(anchors)
    for root in graph:
        if root in result:
            continue
        result[root] = root, (0,) * dimension
        todo = [root]
        protected = None
        while todo:
            node = todo.pop()
            _, potential = result[node]
            if node in anchors:
                point, lift = anchors[node]
                anchor = (point, tuple(lift[a] - potential[a]
                                       for a in range(dimension)))
                if protected is not None and protected != anchor:
                    conflict('Identity path merges distinct proved S anchors '
                             'or lifts')
                protected = anchor
            for peer, delta in graph[node]:
                target = tuple(potential[a] + delta[a] for a in range(dimension))
                if peer in result:
                    if result[peer] != (root, target):
                        conflict('Alternative certified identity paths '
                                 'disagree in lift')
                else:
                    result[peer] = root, target
                    todo.append(peer)
    return result


class NumericalCopy:
    """Copy/pickle the public dataclass fields, stripping execution authority."""

    def __getstate__(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def __setstate__(self, state):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, state[name])


def normalized_operands(normalized):
    return (normalized.global_vertices, normalized.cells,
            getattr(normalized, 'global_edges', None))
