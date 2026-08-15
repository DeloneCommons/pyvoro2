"""Private separator observation and geometry-source identity machinery.

Every valid observation set has a source-independent identity.  A separate
source binding is established only when the original points and domain are
known.  The two layers deliberately remain distinct: binding a source never
changes row IDs or the ordered observation-set fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from threading import Lock
from typing import Any, TYPE_CHECKING

import numpy as np

from ..._internal.inputs import coerce_point_array
from ...domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ...planar.domains import Box as Box2D, RectangularCell

if TYPE_CHECKING:  # pragma: no cover - imports used only by static tooling
    from .constraints import DomainAny, SeparatorObservations


_NAMESPACE_TYPE = 'pyvoro2-separator-observation-namespace'
_ROW_TYPE = 'pyvoro2-separator-row'
_OBSERVATION_SET_TYPE = 'pyvoro2-separator-observation-set'
_SOURCE_TYPE = 'pyvoro2-separator-source'
_IDENTITY_VERSION = 1
_FINGERPRINT_PREFIX = 'sha256:'
_SOURCE_BINDING_LOCK = Lock()
_VERIFICATION_IMAGE_SEARCH = 1
_VERIFICATION_FALLBACK_IMAGE_SEARCH = sys.maxsize
_VERIFICATION_RESOURCE_FAILURE_STAGES = frozenset({
    'pair_candidate_budget',
    'batch_candidate_budget',
})


def _canonical_float(value: float) -> float:
    """Return a built-in finite float with signed zero normalized."""

    result = float(value)
    if not np.isfinite(result):
        raise ValueError('canonical identity values must be finite')
    return 0.0 if result == 0.0 else result


def _canonical_json_value(value: Any) -> Any:
    """Return the frozen JSON encoding value used for identity hashing."""

    if isinstance(value, np.ndarray):
        return _canonical_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_json_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if type(value) is float:
        return _canonical_float(value).hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError(
        'canonical identity payloads require JSON primitives, arrays, and '
        f'mappings; received {type(value).__name__}'
    )


def _fingerprint_payload(payload: dict[str, object]) -> str:
    """Return the frozen SHA-256 fingerprint for one canonical payload."""

    encoded = json.dumps(
        _canonical_json_value(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return _FINGERPRINT_PREFIX + hashlib.sha256(encoded).hexdigest()


def _ids_tuple(ids: np.ndarray | None) -> tuple[int, ...] | None:
    if ids is None:
        return None
    return tuple(int(value) for value in ids.tolist())


@dataclass(frozen=True, slots=True)
class _ObservationNamespace:
    dimension: int
    n_points: int
    measurement: str
    ids: tuple[int, ...] | None

    def payload(self) -> dict[str, object]:
        return {
            'type': _NAMESPACE_TYPE,
            'version': _IDENTITY_VERSION,
            'dimension': int(self.dimension),
            'n_points': int(self.n_points),
            'measurement': self.measurement,
            'ids': None if self.ids is None else list(self.ids),
        }


@dataclass(frozen=True, slots=True)
class _ObservationRow:
    i: int
    j: int
    shift: tuple[int, ...]
    measurement: str
    target: float
    confidence: float
    distance: float
    distance2: float
    delta: tuple[float, ...]
    target_fraction: float
    target_position: float
    explicit_shift: bool
    input_index: int

    def payload(self) -> dict[str, object]:
        return {
            'type': _ROW_TYPE,
            'version': _IDENTITY_VERSION,
            'i': int(self.i),
            'j': int(self.j),
            'shift': list(self.shift),
            'measurement': self.measurement,
            'target': float(self.target),
            'confidence': float(self.confidence),
            'distance': float(self.distance),
            'distance2': float(self.distance2),
            'delta': list(self.delta),
            'target_fraction': float(self.target_fraction),
            'target_position': float(self.target_position),
            'explicit_shift': bool(self.explicit_shift),
        }


@dataclass(frozen=True, slots=True)
class _ObservationIdentity:
    namespace: _ObservationNamespace
    namespace_fingerprint: str
    rows: tuple[_ObservationRow, ...]
    row_fingerprints: tuple[str, ...]
    row_ids: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _CanonicalDomain:
    kind: str
    bounds: tuple[tuple[float, float], ...] | None = None
    periodic: tuple[bool, ...] | None = None
    vectors: tuple[tuple[float, float, float], ...] | None = None
    origin: tuple[float, float, float] | None = None

    def payload(self) -> dict[str, object]:
        result: dict[str, object] = {'kind': self.kind}
        if self.bounds is not None:
            result['bounds'] = [list(pair) for pair in self.bounds]
        if self.periodic is not None:
            result['periodic'] = list(self.periodic)
        if self.vectors is not None:
            result['vectors'] = [list(row) for row in self.vectors]
        if self.origin is not None:
            result['origin'] = list(self.origin)
        return result


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    dimension: int
    n_points: int
    points: tuple[tuple[float, ...], ...]
    domain: _CanonicalDomain
    ids: tuple[int, ...] | None
    fingerprint: str

    def payload(self) -> dict[str, object]:
        return {
            'type': _SOURCE_TYPE,
            'version': _IDENTITY_VERSION,
            'dimension': int(self.dimension),
            'n_points': int(self.n_points),
            'points': [list(row) for row in self.points],
            'domain': self.domain.payload(),
            'ids': None if self.ids is None else list(self.ids),
        }


class _ObservationIdentityStorage:
    """Private slots added without changing public dataclass fields."""

    __slots__ = ('_observation_identity', '_source_identity')


class _SourceBindingInit:
    """Carry a private source binding through dataclass replacement."""

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> _SourceIdentity | None:
        if instance is None:
            return None
        return getattr(instance, '_source_identity', None)


class _ObservationBoundResult:
    """Private origin storage for row-aligned result objects."""

    __slots__ = ('_originating_observations',)


class _ObservationBindingInit:
    """Carry a private observation origin through dataclass replacement."""

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> SeparatorObservations | None:
        if instance is None:
            return None
        return getattr(instance, '_originating_observations', None)


def _build_observation_identity(
    observations: SeparatorObservations,
) -> _ObservationIdentity:
    namespace = _ObservationNamespace(
        dimension=int(observations.dim),
        n_points=int(observations.n_points),
        measurement=observations.measurement,
        ids=_ids_tuple(observations.ids),
    )
    namespace_fingerprint = _fingerprint_payload(namespace.payload())
    namespace_hex = namespace_fingerprint.removeprefix(_FINGERPRINT_PREFIX)

    rows: list[_ObservationRow] = []
    row_fingerprints: list[str] = []
    row_ids: list[str] = []
    for index in range(int(observations.n_constraints)):
        row = _ObservationRow(
            i=int(observations.i[index]),
            j=int(observations.j[index]),
            shift=tuple(int(value) for value in observations.shifts[index]),
            measurement=observations.measurement,
            target=_canonical_float(observations.target[index]),
            confidence=_canonical_float(observations.confidence[index]),
            distance=_canonical_float(observations.distance[index]),
            distance2=_canonical_float(observations.distance2[index]),
            delta=tuple(
                _canonical_float(value) for value in observations.delta[index]
            ),
            target_fraction=_canonical_float(
                observations.target_fraction[index]
            ),
            target_position=_canonical_float(
                observations.target_position[index]
            ),
            explicit_shift=bool(observations.explicit_shift[index]),
            input_index=int(observations.input_index[index]),
        )
        row_fingerprint = _fingerprint_payload(row.payload())
        row_hex = row_fingerprint.removeprefix(_FINGERPRINT_PREFIX)
        row_id = (
            f'pyvoro2-separator-row-v1:{namespace_hex}:'
            f'{row.input_index}:{row_hex}'
        )
        rows.append(row)
        row_fingerprints.append(row_fingerprint)
        row_ids.append(row_id)

    set_payload = {
        'type': _OBSERVATION_SET_TYPE,
        'version': _IDENTITY_VERSION,
        'namespace_fingerprint': namespace_fingerprint,
        'row_ids': row_ids,
    }
    return _ObservationIdentity(
        namespace=namespace,
        namespace_fingerprint=namespace_fingerprint,
        rows=tuple(rows),
        row_fingerprints=tuple(row_fingerprints),
        row_ids=tuple(row_ids),
        fingerprint=_fingerprint_payload(set_payload),
    )


def _canonical_domain(
    domain: DomainAny | None,
    *,
    dimension: int,
) -> _CanonicalDomain:
    if domain is None:
        return _CanonicalDomain(kind='none')
    if dimension == 2 and isinstance(domain, Box2D):
        return _CanonicalDomain(
            kind='planar_box',
            bounds=tuple(
                tuple(_canonical_float(value) for value in pair)
                for pair in domain.bounds
            ),
        )
    if dimension == 2 and isinstance(domain, RectangularCell):
        return _CanonicalDomain(
            kind='planar_rectangular_cell',
            bounds=tuple(
                tuple(_canonical_float(value) for value in pair)
                for pair in domain.bounds
            ),
            periodic=tuple(bool(value) for value in domain.periodic),
        )
    if dimension == 3 and isinstance(domain, Box3D):
        return _CanonicalDomain(
            kind='spatial_box',
            bounds=tuple(
                tuple(_canonical_float(value) for value in pair)
                for pair in domain.bounds
            ),
        )
    if dimension == 3 and isinstance(domain, OrthorhombicCell):
        return _CanonicalDomain(
            kind='spatial_orthorhombic_cell',
            bounds=tuple(
                tuple(_canonical_float(value) for value in pair)
                for pair in domain.bounds
            ),
            periodic=tuple(bool(value) for value in domain.periodic),
        )
    if dimension == 3 and isinstance(domain, PeriodicCell):
        return _CanonicalDomain(
            kind='spatial_periodic_cell',
            vectors=tuple(
                tuple(_canonical_float(value) for value in row)
                for row in domain.vectors
            ),
            origin=tuple(_canonical_float(value) for value in domain.origin),
        )
    if dimension == 2:
        raise ValueError(
            '2D separator sources require domain=None or a planar Box or '
            'RectangularCell'
        )
    if dimension == 3:
        raise ValueError(
            '3D separator sources require domain=None or a spatial Box, '
            'OrthorhombicCell, or PeriodicCell'
        )
    raise ValueError('separator source dimension must be exactly 2 or 3')


def _domain_object(domain: _CanonicalDomain) -> DomainAny | None:
    if domain.kind == 'none':
        return None
    if domain.kind == 'planar_box':
        return Box2D(domain.bounds)
    if domain.kind == 'planar_rectangular_cell':
        return RectangularCell(domain.bounds, domain.periodic)
    if domain.kind == 'spatial_box':
        return Box3D(domain.bounds)
    if domain.kind == 'spatial_orthorhombic_cell':
        return OrthorhombicCell(domain.bounds, domain.periodic)
    if domain.kind == 'spatial_periodic_cell':
        return PeriodicCell(domain.vectors, domain.origin)
    raise ValueError(f'unsupported canonical separator domain kind {domain.kind!r}')


def _build_source_identity(
    points: np.ndarray,
    *,
    domain: DomainAny | None,
    canonical_domain: _CanonicalDomain | None,
    observations: SeparatorObservations,
) -> _SourceIdentity:
    dimension = int(observations.dim)
    raw = np.asarray(points, dtype=object)
    if raw.ndim != 2 or raw.shape != (observations.n_points, dimension):
        raise ValueError(
            'source points must have shape '
            f'({observations.n_points}, {dimension})'
        )
    array = coerce_point_array(raw, name='points', dim=dimension)
    point_values = tuple(
        tuple(_canonical_float(value) for value in row) for row in array
    )
    domain_value = (
        _canonical_domain(domain, dimension=dimension)
        if canonical_domain is None
        else canonical_domain
    )
    source_without_fingerprint = {
        'type': _SOURCE_TYPE,
        'version': _IDENTITY_VERSION,
        'dimension': dimension,
        'n_points': int(observations.n_points),
        'points': [list(row) for row in point_values],
        'domain': domain_value.payload(),
        'ids': (
            None
            if observations.ids is None
            else [int(value) for value in observations.ids.tolist()]
        ),
    }
    return _SourceIdentity(
        dimension=dimension,
        n_points=int(observations.n_points),
        points=point_values,
        domain=domain_value,
        ids=_ids_tuple(observations.ids),
        fingerprint=_fingerprint_payload(source_without_fingerprint),
    )


def _source_equal(left: _SourceIdentity, right: _SourceIdentity) -> bool:
    """Compare a source fingerprint and then all exact canonical values."""

    return left.fingerprint == right.fingerprint and left == right


def _observation_identity_equal(
    left: _ObservationIdentity,
    right: _ObservationIdentity,
) -> bool:
    """Compare a set fingerprint and then all exact canonical values."""

    return left.fingerprint == right.fingerprint and left == right


def _verify_source_geometry(
    observations: SeparatorObservations,
    source: _SourceIdentity,
) -> None:
    """Recompute every connector from exact source points and domain."""

    from ..._internal.periodic_images import MinimumImageCertificationError
    from .constraints import (
        _derive_connector_geometry,
        _derive_target_representations,
    )

    points = np.asarray(source.points, dtype=np.float64).reshape(
        source.n_points,
        source.dimension,
    )
    domain = _domain_object(source.domain)

    def derive_geometry(image_search: int):
        return _derive_connector_geometry(
            points,
            observations.i,
            observations.j,
            observations.shifts,
            observations.explicit_shift,
            domain=domain,
            image_search=image_search,
        )

    # The public default is a cheap, normally strong incumbent seed and still
    # leads to a complete independent R4 certificate.  Retry only proof-box
    # resource failures with the full capped seed set, whose incumbent is at
    # least as strong as every seed permitted through the public resolver.
    try:
        geometry = derive_geometry(_VERIFICATION_IMAGE_SEARCH)
    except MinimumImageCertificationError as exc:
        if exc.stage not in _VERIFICATION_RESOURCE_FAILURE_STAGES:
            raise
        geometry = derive_geometry(_VERIFICATION_FALLBACK_IMAGE_SEARCH)
    shifts, distance, distance2, delta, _ = geometry
    target_fraction, target_position = _derive_target_representations(
        observations.target,
        distance,
        measurement=observations.measurement,
    )
    comparisons = (
        ('shifts', observations.shifts, shifts),
        ('delta', observations.delta, delta),
        ('distance2', observations.distance2, distance2),
        ('distance', observations.distance, distance),
        ('target_fraction', observations.target_fraction, target_fraction),
        ('target_position', observations.target_position, target_position),
    )
    for name, cached, recomputed in comparisons:
        if not np.array_equal(cached, recomputed):
            raise ValueError(
                'separator observations are inconsistent with the exact '
                f'geometry source ({name})'
            )


def _validate_source_for_observations(
    observations: SeparatorObservations,
    source: _SourceIdentity,
) -> None:
    identity = _observation_identity(observations)
    namespace = identity.namespace
    if source.dimension != namespace.dimension:
        raise ValueError('separator source dimension does not match observations')
    if source.n_points != namespace.n_points:
        raise ValueError('separator source point count does not match observations')
    if source.ids != namespace.ids:
        raise ValueError('separator source IDs do not match observations')
    if source.fingerprint != _fingerprint_payload(source.payload()):
        raise ValueError('separator source fingerprint is inconsistent')
    _verify_source_geometry(observations, source)


def _initialize_observation_identity(
    observations: SeparatorObservations,
    source_init: _SourceIdentity | None,
) -> None:
    """Initialize private identity slots after public-field validation."""

    object.__setattr__(
        observations,
        '_observation_identity',
        _build_observation_identity(observations),
    )
    object.__setattr__(observations, '_source_identity', None)
    if source_init is not None:
        if not isinstance(source_init, _SourceIdentity):
            raise ValueError('invalid private separator source binding')
        _validate_source_for_observations(observations, source_init)
        object.__setattr__(observations, '_source_identity', source_init)


def _observation_identity(
    observations: SeparatorObservations,
) -> _ObservationIdentity:
    identity = getattr(observations, '_observation_identity', None)
    if not isinstance(identity, _ObservationIdentity):
        raise ValueError('separator observations do not have canonical identity')
    return identity


def _source_identity(
    observations: SeparatorObservations,
) -> _SourceIdentity | None:
    source = getattr(observations, '_source_identity', None)
    if source is not None and not isinstance(source, _SourceIdentity):
        raise ValueError('separator observations have an invalid source binding')
    return source


def _establish_or_verify_source(
    observations: SeparatorObservations,
    points: np.ndarray,
    domain: DomainAny | None,
    *,
    omitted_none_domain: bool,
) -> _SourceIdentity:
    """Monotonically bind or exactly verify one geometry source."""

    # Binding is a one-way state transition. Keep read, independent geometry
    # validation, and commit in one critical section so two first users cannot
    # both succeed with different exact sources.
    with _SOURCE_BINDING_LOCK:
        current = _source_identity(observations)
        canonical_domain = (
            current.domain
            if current is not None and domain is None and omitted_none_domain
            else None
        )
        candidate = _build_source_identity(
            points,
            domain=domain,
            canonical_domain=canonical_domain,
            observations=observations,
        )
        if current is None:
            _validate_source_for_observations(observations, candidate)
            object.__setattr__(observations, '_source_identity', candidate)
            return candidate
        if not _source_equal(current, candidate):
            if current.points != candidate.points:
                detail = 'points'
            elif current.domain != candidate.domain:
                detail = 'domain'
            elif current.ids != candidate.ids:
                detail = 'ids'
            else:
                detail = 'canonical source'
            raise ValueError(
                'separator observations are already bound to a different '
                f'exact geometry source ({detail})'
            )
        return current


def _bind_resolver_source(
    observations: SeparatorObservations,
    points: np.ndarray,
    domain: DomainAny | None,
) -> SeparatorObservations:
    _establish_or_verify_source(
        observations,
        points,
        domain,
        omitted_none_domain=False,
    )
    return observations


def _bind_fitting_source(
    observations: SeparatorObservations,
    points: np.ndarray,
    domain: DomainAny | None,
) -> SeparatorObservations:
    _establish_or_verify_source(
        observations,
        points,
        domain,
        omitted_none_domain=True,
    )
    return observations


def _bind_full_source(
    observations: SeparatorObservations,
    points: np.ndarray,
    domain: DomainAny | None,
) -> SeparatorObservations:
    _establish_or_verify_source(
        observations,
        points,
        domain,
        omitted_none_domain=False,
    )
    return observations


_OBSERVATION_ARRAY_FIELDS = (
    'i',
    'j',
    'shifts',
    'target',
    'confidence',
    'distance',
    'distance2',
    'delta',
    'target_fraction',
    'target_position',
    'input_index',
    'explicit_shift',
    'ids',
)


def _observation_mismatch(
    originating: SeparatorObservations,
    supplied: SeparatorObservations,
) -> str | None:
    """Return the first exact model/source association mismatch."""

    if originating is supplied:
        return None
    left_identity = _observation_identity(originating)
    right_identity = _observation_identity(supplied)
    if not _observation_identity_equal(left_identity, right_identity):
        for name in ('n_points', 'measurement'):
            if getattr(originating, name) != getattr(supplied, name):
                return name
        if originating.dim != supplied.dim:
            return 'dimension'
        for name in _OBSERVATION_ARRAY_FIELDS:
            expected = getattr(originating, name)
            actual = getattr(supplied, name)
            if expected is None or actual is None:
                if expected is not actual:
                    return name
            elif not np.array_equal(expected, actual):
                return name
        return 'observation identity'

    left_source = _source_identity(originating)
    right_source = _source_identity(supplied)
    if (left_source is None) != (right_source is None):
        return 'source binding'
    if left_source is not None and right_source is not None:
        if not _source_equal(left_source, right_source):
            if left_source.points != right_source.points:
                return 'source points'
            if left_source.domain != right_source.domain:
                return 'source domain'
            if left_source.ids != right_source.ids:
                return 'source ids'
            return 'source'
    return None


def _require_observation_association(
    originating: SeparatorObservations,
    supplied: SeparatorObservations,
    *,
    context: str,
) -> None:
    mismatch = _observation_mismatch(originating, supplied)
    if mismatch is not None:
        raise ValueError(
            f'{context} observations do not match the authoritative '
            f'observation origin ({mismatch})'
        )


def _require_observation_row_data(
    observations: SeparatorObservations,
    *,
    i: np.ndarray,
    j: np.ndarray,
    shifts: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    context: str,
) -> None:
    """Require duplicated row data to match one authoritative origin."""

    comparisons = (
        ('site_i', observations.i, i),
        ('site_j', observations.j, j),
        ('shift', observations.shifts, shifts),
        ('target', observations.target, target),
        ('confidence', observations.confidence, confidence),
    )
    for name, expected, actual in comparisons:
        actual_array = np.asarray(actual)
        if (
            actual_array.shape != expected.shape
            or not np.array_equal(expected, actual_array)
        ):
            raise ValueError(
                f'{context} row data do not match the authoritative '
                f'observation origin ({name})'
            )


def _bind_originating_observations(
    result: Any,
    observations: SeparatorObservations,
) -> Any:
    """Monotonically retain an authoritative observation origin."""

    existing = getattr(result, '_originating_observations', None)
    if existing is not None:
        _require_observation_association(
            existing,
            observations,
            context=type(result).__name__,
        )
        return result
    object.__setattr__(result, '_originating_observations', observations)
    return result


def _originating_observations(
    result: Any,
    *,
    context: str,
) -> SeparatorObservations:
    observations = getattr(result, '_originating_observations', None)
    if observations is None:
        raise ValueError(
            f'{context} is not bound to authoritative originating observations'
        )
    _observation_identity(observations)
    return observations


def _row_ids(observations: SeparatorObservations) -> tuple[str, ...]:
    return _observation_identity(observations).row_ids


def _observation_set_report(
    observations: SeparatorObservations,
) -> dict[str, object]:
    identity = _observation_identity(observations)
    return {
        'fingerprint': identity.fingerprint,
        'measurement': identity.namespace.measurement,
        'n_rows': len(identity.rows),
        'row_ids': list(identity.row_ids),
    }


def _source_report(observations: SeparatorObservations) -> dict[str, object]:
    source = _source_identity(observations)
    if source is None:
        identity = _observation_identity(observations)
        return {
            'binding': 'unbound',
            'fingerprint': None,
            'dimension': int(identity.namespace.dimension),
            'n_points': int(identity.namespace.n_points),
            'points': None,
            'domain': None,
            'ids': None,
        }
    return {
        'binding': 'bound',
        'fingerprint': source.fingerprint,
        'dimension': int(source.dimension),
        'n_points': int(source.n_points),
        'points': [list(row) for row in source.points],
        'domain': source.domain.payload(),
        'ids': None if source.ids is None else list(source.ids),
    }


__all__ = []
