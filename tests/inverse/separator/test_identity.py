"""Independent regression oracles for separator observation/source identity.

The encoder in this module intentionally does not import the production
``_identity`` encoder.  Hard-coded vectors therefore detect coordinated drift
in both the payload construction and its canonical JSON representation.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
import hashlib
import inspect
import json
import math
import pickle
import sys
from threading import Event
from typing import Any

import numpy as np
import pytest

import pyvoro2
import pyvoro2.planar as planar
from pyvoro2.inverse.separator import (
    ActiveSetOptions,
    FixedValue,
    FitModel,
    SeparatorObservations,
    build_active_set_report,
    build_fit_report,
    build_power_fit_problem,
    build_power_fit_result,
    build_realized_report,
    dumps_report_json,
    fit_weights_from_separators,
    match_realized_pairs,
    resolve_separator_observations,
    solve_self_consistent_power_weights,
)


_PUBLIC_OBSERVATION_FIELDS = (
    'n_points',
    'i',
    'j',
    'shifts',
    'target',
    'confidence',
    'measurement',
    'distance',
    'distance2',
    'delta',
    'target_fraction',
    'target_position',
    'input_index',
    'explicit_shift',
    'ids',
    'warnings',
)

_NAMESPACE_FINGERPRINT = (
    'sha256:9fbb685fe9c86151d37a493a150cffb12a4d0d947817ca09db77fb578762b290'
)
_ROW_FINGERPRINT = (
    'sha256:046f94d69a9288ec2d0d00dda7b95a1bb7b9a4e984cbe52d0fa33ea7fa1e45be'
)
_ROW_ID = (
    'pyvoro2-separator-row-v1:'
    '9fbb685fe9c86151d37a493a150cffb12a4d0d947817ca09db77fb578762b290:'
    '7:'
    '046f94d69a9288ec2d0d00dda7b95a1bb7b9a4e984cbe52d0fa33ea7fa1e45be'
)
_SET_FINGERPRINT = (
    'sha256:288bd2bf65da83d6516f07339cae52a1f0dcc4bdf2b54cab2ea4440a532c8169'
)
_SOURCE_FINGERPRINT = (
    'sha256:bac9a9bb81e8be1ed02389d98354f580f17b3040cc6dd674ff28ce99ec0742b5'
)

_DOMAIN_SOURCE_FINGERPRINTS = {
    'none': (
        'sha256:b7a5e30bdedb7068a00b92b0f28eec232f3e1c6a40378cd3f07b328561738d64'
    ),
    'planar_box': (
        'sha256:4256ee992124f564abab5c910af799ecf437e8bc64392b99ff130179efef3680'
    ),
    'planar_rectangular_cell': (
        'sha256:c953bec574f6c4840512771a9ff4fc4e2bed76fcac912bf1279840bbe844f1ad'
    ),
    'spatial_box': (
        'sha256:2e9c83b9f5249e8baad32e1c8aba9dfa420933c0aa57584b269472162ff69473'
    ),
    'spatial_orthorhombic_cell': (
        'sha256:223946f929f665c767b4bd698967cba4507853d0c1c418669abbcaa321a0689c'
    ),
    'spatial_periodic_cell': (
        'sha256:6f8857afeb0813d4d76a86a604cdf4b4e59241e060f6eff6042b693c20457506'
    ),
}


def _canonical_value(value: Any) -> Any:
    """Test-local implementation of the frozen canonical JSON vocabulary."""

    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('identity test payloads must be finite')
        return (0.0 if value == 0.0 else value).hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError(type(value).__name__)


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return 'sha256:' + hashlib.sha256(encoded).hexdigest()


def _namespace_payload(observations: SeparatorObservations) -> dict[str, object]:
    return {
        'type': 'pyvoro2-separator-observation-namespace',
        'version': 1,
        'dimension': observations.dim,
        'n_points': observations.n_points,
        'measurement': observations.measurement,
        'ids': (
            None
            if observations.ids is None
            else [int(value) for value in observations.ids]
        ),
    }


def _row_payload(
    observations: SeparatorObservations,
    index: int,
) -> dict[str, object]:
    return {
        'type': 'pyvoro2-separator-row',
        'version': 1,
        'i': int(observations.i[index]),
        'j': int(observations.j[index]),
        'shift': [int(value) for value in observations.shifts[index]],
        'measurement': observations.measurement,
        'target': float(observations.target[index]),
        'confidence': float(observations.confidence[index]),
        'distance': float(observations.distance[index]),
        'distance2': float(observations.distance2[index]),
        'delta': [float(value) for value in observations.delta[index]],
        'target_fraction': float(observations.target_fraction[index]),
        'target_position': float(observations.target_position[index]),
        'explicit_shift': bool(observations.explicit_shift[index]),
    }


def _independent_row_id(
    observations: SeparatorObservations,
    index: int,
) -> str:
    namespace_hex = _fingerprint(_namespace_payload(observations))[7:]
    row_hex = _fingerprint(_row_payload(observations, index))[7:]
    return (
        f'pyvoro2-separator-row-v1:{namespace_hex}:'
        f'{int(observations.input_index[index])}:{row_hex}'
    )


def _independent_set_fingerprint(
    observations: SeparatorObservations,
) -> str:
    namespace = _fingerprint(_namespace_payload(observations))
    return _fingerprint(
        {
            'type': 'pyvoro2-separator-observation-set',
            'version': 1,
            'namespace_fingerprint': namespace,
            'row_ids': [
                _independent_row_id(observations, index)
                for index in range(observations.n_constraints)
            ],
        }
    )


def _source_payload(
    observations: SeparatorObservations,
    points: np.ndarray,
    domain: dict[str, object],
) -> dict[str, object]:
    return {
        'type': 'pyvoro2-separator-source',
        'version': 1,
        'dimension': observations.dim,
        'n_points': observations.n_points,
        'points': np.asarray(points, dtype=np.float64).tolist(),
        'domain': domain,
        'ids': (
            None
            if observations.ids is None
            else [int(value) for value in observations.ids]
        ),
    }


def _one_observation(**changes: object) -> SeparatorObservations:
    values: dict[str, object] = {
        'n_points': 3,
        'i': np.array([0], dtype=np.int64),
        'j': np.array([1], dtype=np.int64),
        'shifts': np.array([[0, 0]], dtype=np.int64),
        'target': np.array([0.25], dtype=np.float64),
        'confidence': np.array([0.75], dtype=np.float64),
        'measurement': 'fraction',
        'distance': np.array([2.0], dtype=np.float64),
        'distance2': np.array([4.0], dtype=np.float64),
        'delta': np.array([[2.0, -0.0]], dtype=np.float64),
        'target_fraction': np.array([0.25], dtype=np.float64),
        'target_position': np.array([0.5], dtype=np.float64),
        'input_index': np.array([7], dtype=np.int64),
        'explicit_shift': np.array([True], dtype=bool),
        'ids': np.array([10, 20, 30], dtype=np.int64),
        'warnings': ('direct-construction warning',),
    }
    values.update(changes)
    return SeparatorObservations(**values)


def _duplicate_observations() -> SeparatorObservations:
    return SeparatorObservations(
        n_points=3,
        i=np.array([0, 1, 0]),
        j=np.array([1, 2, 1]),
        shifts=np.zeros((3, 2), dtype=np.int64),
        target=np.array([0.25, 0.75, 0.25]),
        confidence=np.array([0.75, 0.5, 0.75]),
        measurement='fraction',
        distance=np.array([2.0, 2.0, 2.0]),
        distance2=np.array([4.0, 4.0, 4.0]),
        delta=np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        target_fraction=np.array([0.25, 0.75, 0.25]),
        target_position=np.array([0.5, 1.5, 0.5]),
        input_index=np.array([7, 9, 11]),
        explicit_shift=np.ones(3, dtype=bool),
        ids=np.array([10, 20, 30]),
        warnings=(),
    )


def _row_only_result(observations: SeparatorObservations):
    problem = build_power_fit_problem(observations)
    return build_power_fit_result(
        problem,
        np.zeros(observations.n_points, dtype=np.float64),
    )


def _row_only_report(observations: SeparatorObservations) -> dict[str, object]:
    return build_fit_report(_row_only_result(observations), observations)


def _reorder(
    observations: SeparatorObservations,
    order: np.ndarray,
) -> SeparatorObservations:
    row_fields = (
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
    )
    return replace(
        observations,
        **{
            name: np.asarray(getattr(observations, name))[order].copy()
            for name in row_fields
        },
    )


def _assert_owned_observation_arrays(observations: SeparatorObservations) -> None:
    for name in _PUBLIC_OBSERVATION_FIELDS:
        value = getattr(observations, name)
        if not isinstance(value, np.ndarray):
            continue
        assert value.flags.owndata, name
        assert value.flags.c_contiguous, name
        assert not value.flags.writeable, name


def _assert_json_roundtrip(report: dict[str, object]) -> None:
    assert json.loads(dumps_report_json(report, sort_keys=True)) == report
    json.dumps(report, allow_nan=False)


def test_public_constructor_shape_and_independent_reference_vectors() -> None:
    observations = _one_observation()
    assert tuple(field.name for field in fields(SeparatorObservations)) == (
        _PUBLIC_OBSERVATION_FIELDS
    )
    assert tuple(inspect.signature(SeparatorObservations).parameters) == (
        _PUBLIC_OBSERVATION_FIELDS
    )

    namespace = _namespace_payload(observations)
    row = _row_payload(observations, 0)
    assert namespace['type'] == 'pyvoro2-separator-observation-namespace'
    assert row['type'] == 'pyvoro2-separator-row'
    assert namespace['version'] == row['version'] == 1
    assert _fingerprint(namespace) == _NAMESPACE_FINGERPRINT
    assert _fingerprint(row) == _ROW_FINGERPRINT
    assert _independent_row_id(observations, 0) == _ROW_ID
    assert _independent_set_fingerprint(observations) == _SET_FINGERPRINT

    report = _row_only_report(observations)
    assert observations.to_records()[0]['row_id'] == _ROW_ID
    assert report['observation_set'] == {
        'fingerprint': _SET_FINGERPRINT,
        'measurement': 'fraction',
        'n_rows': 1,
        'row_ids': [_ROW_ID],
    }

    source_points = np.array(
        [[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]],
        dtype=np.float64,
    )
    fit_weights_from_separators(source_points, observations, ids=[999, 998, 997])
    bound = _row_only_report(observations)
    source_payload = _source_payload(
        observations,
        source_points,
        {'kind': 'none'},
    )
    assert source_payload['type'] == 'pyvoro2-separator-source'
    assert source_payload['version'] == 1
    assert _fingerprint(source_payload) == _SOURCE_FINGERPRINT
    assert bound['source']['fingerprint'] == _SOURCE_FINGERPRINT
    assert bound['source']['ids'] == [10, 20, 30]
    assert bound['observation_set'] == report['observation_set']


@pytest.mark.parametrize(
    'changes',
    [
        {'n_points': True},
        {'n_points': -1},
        {'i': np.array([0.0], dtype=object)},
        {'j': np.array([True], dtype=object)},
        {'i': np.array([-1])},
        {'j': np.array([3])},
        {'j': np.array([0])},
        {'shifts': np.array([[0.0, 0.0]], dtype=object)},
        {
            'shifts': np.array([[0]], dtype=np.int64),
            'delta': np.array([[2.0]]),
        },
        {'target': np.array([np.nan])},
        {'confidence': np.array([-1.0])},
        {'confidence': np.array([np.inf])},
        {'distance': np.array([3.0])},
        {'distance2': np.array([3.0])},
        {'delta': np.array([[0.0, 0.0]]), 'distance': np.array([0.0]),
         'distance2': np.array([0.0])},
        {'target_fraction': np.array([0.5])},
        {'target_position': np.array([0.75])},
        {'input_index': np.array([-1])},
        {'input_index': np.array([1.0], dtype=object)},
        {'explicit_shift': np.array([1], dtype=np.int64)},
        {'ids': np.array([10, 10, 30])},
        {'ids': np.array([10, 20])},
        {'warnings': ('ok', 1)},
        {'measurement': 'distance'},
    ],
)
def test_direct_constructor_rejects_invalid_invariant_matrix(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _one_observation(**changes)


def test_direct_constructor_rejects_shapes_and_duplicate_input_indices() -> None:
    with pytest.raises(ValueError, match='input_index.*unique'):
        replace(
            _duplicate_observations(),
            input_index=np.array([7, 7, 11]),
        )
    with pytest.raises(ValueError, match=r'j must have shape'):
        _one_observation(j=np.array([1, 2]))
    with pytest.raises(ValueError, match=r'delta'):
        _one_observation(delta=np.array([[2.0, 0.0, 0.0]]))


def test_redundant_values_use_frozen_tolerance_then_are_canonicalized() -> None:
    observations = _one_observation()
    near = replace(
        observations,
        distance=np.nextafter(observations.distance, np.inf),
        distance2=np.nextafter(observations.distance2, np.inf),
        target_fraction=np.nextafter(observations.target_fraction, np.inf),
        target_position=np.nextafter(observations.target_position, np.inf),
    )
    for name in ('distance', 'distance2', 'target_fraction', 'target_position'):
        np.testing.assert_array_equal(
            getattr(near, name),
            getattr(observations, name),
        )
    _assert_owned_observation_arrays(near)

    scale = 1.0 + 32.0 * np.finfo(np.float64).eps
    for name in ('distance', 'distance2', 'target_fraction', 'target_position'):
        with pytest.raises(ValueError, match=f'{name}.*inconsistent'):
            replace(
                observations,
                **{name: np.asarray(getattr(observations, name)) * scale},
            )


def test_direct_unbound_row_only_chain_and_report_source() -> None:
    observations = _one_observation()
    problem = build_power_fit_problem(observations)
    result = build_power_fit_result(problem, np.zeros(3))
    report = build_fit_report(result, observations)

    assert report['kind'] == 'power_weight_fit'
    assert report['source'] == {
        'binding': 'unbound',
        'fingerprint': None,
        'dimension': 2,
        'n_points': 3,
        'points': None,
        'domain': None,
        'ids': None,
    }
    assert report['constraints'][0]['row_id'] == _ROW_ID
    assert report['fit_records'][0]['row_id'] == _ROW_ID
    _assert_json_roundtrip(report)


def test_row_set_stability_subset_duplicates_reorder_and_late_binding() -> None:
    observations = _duplicate_observations()
    initial = _row_only_report(observations)['observation_set']
    row_ids = initial['row_ids']
    assert row_ids == [
        _independent_row_id(observations, index)
        for index in range(3)
    ]
    assert initial['fingerprint'] == _independent_set_fingerprint(observations)

    # Duplicate row payloads remain distinct because the stable input index is
    # part of the row ID, but their row-payload hashes are equal.
    first_parts = row_ids[0].split(':')
    duplicate_parts = row_ids[2].split(':')
    assert first_parts[1] == duplicate_parts[1]
    assert first_parts[2] != duplicate_parts[2]
    assert first_parts[3] == duplicate_parts[3]

    subset = observations.subset(np.array([True, False, True]))
    subset_identity = _row_only_report(subset)['observation_set']
    assert subset_identity['row_ids'] == [row_ids[0], row_ids[2]]
    assert subset_identity['fingerprint'] == _independent_set_fingerprint(subset)

    reordered = _reorder(observations, np.array([2, 0, 1]))
    reordered_identity = _row_only_report(reordered)['observation_set']
    assert reordered_identity['row_ids'] == [row_ids[2], row_ids[0], row_ids[1]]
    assert reordered_identity['fingerprint'] != initial['fingerprint']

    warning_only = replace(observations, warnings=('warnings are not identity',))
    assert _row_only_report(warning_only)['observation_set'] == initial

    points = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    fit_weights_from_separators(points, observations)
    assert _row_only_report(observations)['observation_set'] == initial
    assert _row_only_report(observations)['source']['binding'] == 'bound'
    assert _row_only_report(observations.subset(np.array([True, False, True])))[
        'source'
    ]['fingerprint'] == _row_only_report(observations)['source']['fingerprint']


def test_periodic_parallel_shifts_have_distinct_independent_row_identity() -> None:
    points = (
        np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64),
        np.array([[3.5, 0.5], [0.5, 0.5]], dtype=np.float64),
    )
    domain = planar.RectangularCell(
        ((0.0, 4.0), (0.0, 2.0)),
        periodic=(True, False),
    )
    shifts = ((0, 0), (1, 0))
    lattice_lengths = np.array([4.0, 2.0], dtype=np.float64)
    expected_deltas = tuple(
        source[1] + np.asarray(shift) * lattice_lengths - source[0]
        for source, shift in zip(points, shifts)
    )
    np.testing.assert_array_equal(expected_deltas[0], [1.0, 0.0])
    np.testing.assert_array_equal(expected_deltas[1], [1.0, 0.0])

    singles = tuple(
        resolve_separator_observations(
            source,
            [(0, 1, 0.25, shift)],
            domain=domain,
            ids=[11, 22],
            confidence=[0.75],
            image='given_only',
        )
        for source, shift in zip(points, shifts)
    )
    for observations, expected_delta in zip(singles, expected_deltas):
        assert observations.input_index.tolist() == [0]
        np.testing.assert_array_equal(observations.delta[0], expected_delta)
        assert float(observations.distance2[0]) == float(
            np.dot(expected_delta, expected_delta)
        )

    namespace_fingerprints = tuple(
        _fingerprint(_namespace_payload(observations))
        for observations in singles
    )
    assert namespace_fingerprints[0] == namespace_fingerprints[1]
    assert namespace_fingerprints[0] == (
        'sha256:3b7ece49bc1cba8dcb8c43404f8b27ac051bd6bc6284800c8facb4a1781df2f4'
    )

    row_payloads = tuple(
        _row_payload(observations, 0) for observations in singles
    )
    assert row_payloads[0]['shift'] == [0, 0]
    assert row_payloads[1]['shift'] == [1, 0]
    assert {
        key: value for key, value in row_payloads[0].items() if key != 'shift'
    } == {
        key: value for key, value in row_payloads[1].items() if key != 'shift'
    }
    expected_row_fingerprints = (
        'sha256:3cc46d048ccaca7206f45691796656c2a2614c53714d1241fe2b0071295a7ec0',
        'sha256:b03c10bdd70ca0e5cb57905e18d5844821700550c29bb9134328e6d1c7a8190d',
    )
    assert tuple(_fingerprint(payload) for payload in row_payloads) == (
        expected_row_fingerprints
    )

    single_row_ids = tuple(
        observations.to_records()[0]['row_id']
        for observations in singles
    )
    single_row_parts = tuple(row_id.split(':') for row_id in single_row_ids)
    assert single_row_parts[0][1] == single_row_parts[1][1]
    assert single_row_parts[0][1] == namespace_fingerprints[0][7:]
    assert single_row_parts[0][2] == single_row_parts[1][2] == '0'
    assert single_row_parts[0][3] != single_row_parts[1][3]
    assert single_row_ids[0] != single_row_ids[1]
    expected_row_ids = tuple(
        'pyvoro2-separator-row-v1:'
        f'{namespace_fingerprints[0][7:]}:0:{fingerprint[7:]}'
        for fingerprint in expected_row_fingerprints
    )
    assert single_row_ids == expected_row_ids

    expected_single_set_fingerprints = (
        'sha256:60572ae78f893227be1803e5eae4854869528d4f9be0760e9aeb3aa79e32079a',
        'sha256:8f19faa8fee893b0c56610d054e10407a6e37cc1de7fc0c5c00185fad4883488',
    )

    for observations, row_id, row_parts, expected_set in zip(
        singles,
        single_row_ids,
        single_row_parts,
        expected_single_set_fingerprints,
    ):
        assert row_id == _independent_row_id(observations, 0)
        assert row_parts[3] == _fingerprint(_row_payload(observations, 0))[7:]
        report_identity = _row_only_report(observations)['observation_set']
        assert report_identity['row_ids'] == [row_id]
        assert report_identity['fingerprint'] == expected_set
        assert report_identity['fingerprint'] == (
            _independent_set_fingerprint(observations)
        )

    parallel = resolve_separator_observations(
        points[0],
        [
            (0, 1, 0.25, shifts[0]),
            (0, 1, 0.25, shifts[1]),
        ],
        domain=domain,
        ids=[11, 22],
        confidence=[0.75, 0.75],
        image='given_only',
    )
    expected_parallel_deltas = np.array([[1.0, 0.0], [5.0, 0.0]])
    np.testing.assert_array_equal(parallel.delta, expected_parallel_deltas)
    assert parallel.input_index.tolist() == [0, 1]
    parallel_identity = _row_only_report(parallel)['observation_set']
    assert parallel_identity['row_ids'] == [
        _independent_row_id(parallel, 0),
        _independent_row_id(parallel, 1),
    ]
    assert parallel_identity['row_ids'][0] != parallel_identity['row_ids'][1]
    assert parallel_identity['row_ids'][0] == expected_row_ids[0]
    assert parallel_identity['row_ids'][1].split(':')[3] == (
        '9fa16227c162cc0d6420183508bd4a111d99d7e20f986e5021140b009d5421b1'
    )
    assert parallel_identity['fingerprint'] == (
        'sha256:730891b218c5a389fa4ea0bb2ffebf7990399b7624681c8e677bdd2afc02ac6d'
    )
    assert parallel_identity['fingerprint'] == (
        _independent_set_fingerprint(parallel)
    )

    retained = parallel.subset(np.array([False, True]))
    retained_identity = _row_only_report(retained)['observation_set']
    assert retained.input_index.tolist() == [1]
    assert retained_identity['row_ids'] == [parallel_identity['row_ids'][1]]
    assert retained_identity['fingerprint'] == (
        'sha256:dcf7f537decbbb5c87de025407f05814f6c1e6bfc4f24b559d4dd909be1569e9'
    )
    assert retained_identity['fingerprint'] == (
        _independent_set_fingerprint(retained)
    )


def test_identity_generation_is_deterministic_under_parallel_construction() -> None:
    def build(_: int) -> tuple[str, str]:
        observations = _duplicate_observations()
        report = _row_only_report(observations)
        return (
            report['observation_set']['fingerprint'],
            report['observation_set']['row_ids'][0],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(build, range(16)))
    assert len(set(identities)) == 1


def test_first_source_binding_is_atomic_under_parallel_use(monkeypatch) -> None:
    from pyvoro2.inverse.separator import _identity

    observations = _one_observation()
    first_points = np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    other_points = first_points + np.array([10.0, 5.0])
    first_entered = Event()
    other_entered = Event()
    release_first = Event()
    original_build = _identity._build_source_identity

    def guarded_build(points, **kwargs):
        if float(np.asarray(points)[0, 0]) == 0.0:
            first_entered.set()
            if not release_first.wait(5.0):
                raise AssertionError('timed out coordinating first source bind')
        else:
            other_entered.set()
        return original_build(points, **kwargs)

    monkeypatch.setattr(_identity, '_build_source_identity', guarded_build)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            fit_weights_from_separators,
            first_points,
            observations,
        )
        assert first_entered.wait(5.0)
        other = executor.submit(
            fit_weights_from_separators,
            other_points,
            observations,
        )
        try:
            assert not other_entered.wait(0.2)
        finally:
            release_first.set()
        first.result(timeout=5.0)
        with pytest.raises(ValueError, match='already bound.*points'):
            other.result(timeout=5.0)

    source = _row_only_report(observations)['source']
    assert source['points'] == first_points.tolist()


def test_failed_first_source_geometry_check_leaves_observation_unbound() -> None:
    observations = _one_observation()
    before = _row_only_report(observations)
    source_points = np.array(
        [[0.0, 0.0], [20.0, 0.0], [40.0, 1.0]],
        dtype=np.float64,
    )
    expected_connector = source_points[1] - source_points[0]
    np.testing.assert_array_equal(expected_connector, [20.0, 0.0])
    assert float(np.dot(expected_connector, expected_connector)) == 400.0
    np.testing.assert_array_equal(observations.delta[0], [2.0, -0.0])
    assert float(observations.distance[0]) == 2.0

    with pytest.raises(
        ValueError,
        match=r'exact geometry source \(delta\)',
    ):
        fit_weights_from_separators(source_points, observations)

    report = _row_only_report(observations)
    assert report['kind'] == 'power_weight_fit'
    assert report['source'] == before['source']
    assert report['source']['binding'] == 'unbound'
    assert report['observation_set'] == before['observation_set']


def _assert_fit_association(result: object, candidate: SeparatorObservations) -> None:
    result.observation_view(candidate)
    result.to_records(candidate)
    build_fit_report(result, candidate)


def test_association_policy_matrix_and_authoritative_report_source() -> None:
    points = np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])

    left = _one_observation()
    right = copy.deepcopy(left)
    result = _row_only_result(left)
    _assert_fit_association(result, right)  # both unbound, exact same model

    fit_weights_from_separators(points, left)
    with pytest.raises(ValueError, match='source binding'):
        _assert_fit_association(result, right)

    same_source = copy.deepcopy(left)
    _assert_fit_association(result, same_source)

    translated = _one_observation()
    translated_points = points + np.array([10.0, 5.0])
    fit_weights_from_separators(translated_points, translated)
    with pytest.raises(ValueError, match='source points'):
        _assert_fit_association(result, translated)

    changed_row = replace(same_source, confidence=np.array([0.5]))
    with pytest.raises(ValueError, match='confidence'):
        _assert_fit_association(result, changed_row)

    # Reports must use the authoritative origin, including its source.  A
    # warnings-only replacement is associable but cannot inject provenance.
    warning_only = replace(left, warnings=('different warning',))
    report = build_fit_report(result, warning_only)
    assert report['source']['fingerprint'] == _SOURCE_FINGERPRINT


def test_hash_collision_defense_compares_exact_private_source_values() -> None:
    # This is the one intentional private reach: public state cannot forge a
    # colliding fingerprint, so collision defense itself requires private
    # identity values.  Runtime association must compare exact values after a
    # fingerprint agreement rather than treating a hash as equality.
    from pyvoro2.inverse.separator import _identity

    points = np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    left = _one_observation()
    right = _one_observation()
    fit_weights_from_separators(points, left)
    fit_weights_from_separators(points + 10.0, right)
    left_source = _identity._source_identity(left)
    right_source = _identity._source_identity(right)
    forged = replace(right_source, fingerprint=left_source.fingerprint)
    assert forged.fingerprint == left_source.fingerprint
    assert not _identity._source_equal(left_source, forged)


def test_omitted_domain_preserves_binding_and_none_binding_is_distinct() -> None:
    points = np.array([[0.1, 0.25], [1.9, 0.75]])
    domain = planar.RectangularCell(
        ((0.0, 2.0), (0.0, 1.0)),
        periodic=(True, False),
    )
    observations = resolve_separator_observations(
        points,
        [(0, 1, 0.25, (-1, 0))],
        domain=domain,
        ids=[101, 202],
        confidence=[0.75],
        image='given_only',
    )
    original_source = _row_only_report(observations)['source']
    fit_weights_from_separators(points, observations)  # domain omitted
    fit_weights_from_separators(points, observations, domain=domain)
    assert _row_only_report(observations)['source'] == original_source

    with pytest.raises(ValueError, match='different exact.*domain'):
        fit_weights_from_separators(
            points,
            observations,
            domain=planar.RectangularCell(
                ((0.0, 2.0), (0.0, 1.0)),
                periodic=(True, True),
            ),
        )
    with pytest.raises(ValueError, match='different exact.*points'):
        fit_weights_from_separators(points + 1.0, observations)

    unbound = _one_observation()
    unbound_source = _row_only_report(unbound)['source']
    assert unbound_source['binding'] == 'unbound'
    assert unbound_source['domain'] is None
    fit_weights_from_separators(
        np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]]),
        unbound,
    )
    none_bound = _row_only_report(unbound)['source']
    assert none_bound['binding'] == 'bound'
    assert none_bound['domain'] == {'kind': 'none'}
    assert none_bound['fingerprint'] is not None


def test_invalid_realization_domains_do_not_poison_first_source_binding() -> None:
    points = np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    domain = planar.Box(((-1.0, 5.0), (-1.0, 2.0)))

    realized_observations = _one_observation()
    with pytest.raises(ValueError, match='2D points require a planar domain'):
        match_realized_pairs(
            points,
            domain=None,
            constraints=realized_observations,
            weights=np.zeros(3),
        )
    assert _row_only_report(realized_observations)['source']['binding'] == (
        'unbound'
    )
    match_realized_pairs(
        points,
        domain=domain,
        constraints=realized_observations,
        weights=np.zeros(3),
        unaccounted_pair_check='none',
    )
    assert _row_only_report(realized_observations)['source']['domain'] == {
        'kind': 'planar_box',
        'bounds': [[-1.0, 5.0], [-1.0, 2.0]],
    }

    active_observations = _one_observation()
    with pytest.raises(ValueError, match='2D points require a planar domain'):
        solve_self_consistent_power_weights(
            points,
            active_observations,
            domain=None,
        )
    assert _row_only_report(active_observations)['source']['binding'] == 'unbound'
    solve_self_consistent_power_weights(
        points,
        active_observations,
        domain=domain,
        options=ActiveSetOptions(max_iter=1),
        unaccounted_pair_check='none',
    )
    assert _row_only_report(active_observations)['source']['domain'] == {
        'kind': 'planar_box',
        'bounds': [[-1.0, 5.0], [-1.0, 2.0]],
    }


def _domain_cases() -> list[tuple[Any, ...]]:
    return [
        (
            'none',
            np.array([[-0.0, 0.25], [1.5, 0.75]]),
            None,
            {'kind': 'none'},
            (0, 0),
            np.array([1.5, 0.5]),
        ),
        (
            'planar_box',
            np.array([[-0.0, 0.25], [1.5, 0.75]]),
            planar.Box(((-1.0, 2.0), (-1.0, 2.0))),
            {
                'kind': 'planar_box',
                'bounds': [[-1.0, 2.0], [-1.0, 2.0]],
            },
            (0, 0),
            np.array([1.5, 0.5]),
        ),
        (
            'planar_rectangular_cell',
            np.array([[0.1, 0.25], [1.9, 0.75]]),
            planar.RectangularCell(
                ((0.0, 2.0), (0.0, 1.0)),
                periodic=(True, False),
            ),
            {
                'kind': 'planar_rectangular_cell',
                'bounds': [[0.0, 2.0], [0.0, 1.0]],
                'periodic': [True, False],
            },
            (-1, 0),
            np.array([-0.2, 0.5]),
        ),
        (
            'spatial_box',
            np.array([[0.1, 0.2, 0.3], [1.5, 0.8, 1.2]]),
            pyvoro2.Box(((-1.0, 2.0), (-1.0, 2.0), (-1.0, 2.0))),
            {
                'kind': 'spatial_box',
                'bounds': [[-1.0, 2.0], [-1.0, 2.0], [-1.0, 2.0]],
            },
            (0, 0, 0),
            np.array([1.4, 0.6, 0.9]),
        ),
        (
            'spatial_orthorhombic_cell',
            np.array([[0.1, 0.2, 0.3], [1.9, 0.8, 1.2]]),
            pyvoro2.OrthorhombicCell(
                ((0.0, 2.0), (0.0, 1.0), (0.0, 1.5)),
                periodic=(True, False, True),
            ),
            {
                'kind': 'spatial_orthorhombic_cell',
                'bounds': [[0.0, 2.0], [0.0, 1.0], [0.0, 1.5]],
                'periodic': [True, False, True],
            },
            (-1, 0, 0),
            np.array([-0.2, 0.6, 0.9]),
        ),
        (
            'spatial_periodic_cell',
            np.array([[0.1, 0.2, 0.3], [1.9, 0.8, 1.2]]),
            pyvoro2.PeriodicCell(
                (
                    (2.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.5),
                ),
                origin=(-0.0, 0.0, 0.0),
            ),
            {
                'kind': 'spatial_periodic_cell',
                'vectors': [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.5],
                ],
                'origin': [0.0, 0.0, 0.0],
            },
            (-1, 0, 0),
            np.array([-0.2, 0.6, 0.9]),
        ),
    ]


@pytest.mark.parametrize(
    'kind,points,domain,domain_payload,shift,expected_delta',
    _domain_cases(),
    ids=[case[0] for case in _domain_cases()],
)
def test_six_source_domain_kinds_have_independent_geometry_and_vectors(
    kind: str,
    points: np.ndarray,
    domain: object,
    domain_payload: dict[str, object],
    shift: tuple[int, ...],
    expected_delta: np.ndarray,
) -> None:
    observations = resolve_separator_observations(
        points,
        [(0, 1, 0.25, shift)],
        domain=domain,
        ids=[101, 202],
        confidence=[0.75],
        image='given_only',
    )
    # Connector arithmetic is an independent test oracle, not a call back into
    # production identity/geometry helpers.
    np.testing.assert_allclose(
        observations.delta[0],
        expected_delta,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )
    expected_distance2 = float(np.dot(expected_delta, expected_delta))
    np.testing.assert_allclose(
        observations.distance2[0],
        expected_distance2,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )

    report_source = _row_only_report(observations)['source']
    expected_fingerprint = _fingerprint(
        _source_payload(observations, points, domain_payload)
    )
    assert expected_fingerprint == _DOMAIN_SOURCE_FINGERPRINTS[kind]
    assert report_source == {
        'binding': 'bound',
        'fingerprint': expected_fingerprint,
        'dimension': observations.dim,
        'n_points': 2,
        'points': points.tolist(),
        'domain': domain_payload,
        'ids': [101, 202],
    }


def _copy_variants(observations: SeparatorObservations) -> list[object]:
    variants: list[object] = [
        copy.copy(observations),
        copy.deepcopy(observations),
        replace(observations),
        pickle.loads(pickle.dumps(observations)),
    ]
    if hasattr(copy, 'replace'):
        variants.append(copy.replace(observations))
    return variants


@pytest.mark.parametrize('bound', [False, True], ids=['unbound', 'bound'])
def test_copy_replace_pickle_preserve_identity_binding_and_ownership(
    bound: bool,
) -> None:
    observations = _one_observation()
    if bound:
        fit_weights_from_separators(
            np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]]),
            observations,
        )
    result = _row_only_result(observations)
    expected = _row_only_report(observations)

    for variant in _copy_variants(observations):
        _assert_owned_observation_arrays(variant)
        assert _row_only_report(variant)['observation_set'] == (
            expected['observation_set']
        )
        assert _row_only_report(variant)['source'] == expected['source']
        _assert_fit_association(result, variant)


def test_source_inconsistent_replace_and_rebinding_are_rejected() -> None:
    points = np.array([[-0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    observations = _one_observation()
    fit_weights_from_separators(points, observations)

    with pytest.raises(ValueError, match='exact geometry source.*delta'):
        replace(
            observations,
            delta=np.array([[3.0, 0.0]]),
            distance=np.array([3.0]),
            distance2=np.array([9.0]),
            target_position=np.array([0.75]),
        )
    with pytest.raises(ValueError, match='source IDs'):
        replace(observations, ids=np.array([11, 20, 30]))
    with pytest.raises(ValueError, match='already bound.*points'):
        fit_weights_from_separators(points + 5.0, observations)

    changed_target = replace(
        observations,
        target=np.array([0.5]),
        target_fraction=np.array([0.5]),
        target_position=np.array([1.0]),
    )
    assert _row_only_report(changed_target)['source']['fingerprint'] == (
        _SOURCE_FINGERPRINT
    )
    assert _row_only_report(changed_target)['observation_set']['fingerprint'] != (
        _SET_FINGERPRINT
    )


def test_bound_inferred_triclinic_replace_reuses_minimum_image_geometry() -> None:
    cell = pyvoro2.PeriodicCell(
        vectors=(
            (2.0, 0.0, 0.0),
            (0.25, 1.0, 0.0),
            (0.1, 0.2, 1.5),
        )
    )
    points = np.array(
        [
            [0.145, 0.14, 0.3],
            [2.18, 0.96, 1.2],
        ],
        dtype=np.float64,
    )
    observations = resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        domain=cell,
    )

    assert tuple(int(value) for value in observations.shifts[0]) == (0, -1, -1)
    assert not bool(observations.explicit_shift[0])
    assert tuple(float(value).hex() for value in observations.delta[0]) == (
        '-0x1.428f5c28f5c26p-2',
        '-0x1.851eb851eb853p-2',
        '-0x1.3333333333334p-1',
    )

    # Revalidating an inferred row must rerun certified minimum-image
    # geometry. Treating the retained shift as explicit changes y by one ULP.
    rebuilt = replace(observations)
    np.testing.assert_array_equal(rebuilt.delta, observations.delta)
    assert _row_only_report(rebuilt)['source'] == (
        _row_only_report(observations)['source']
    )


def test_bound_inferred_skew_cell_preserves_public_certified_resolution() -> None:
    cell = pyvoro2.PeriodicCell(
        (
            (1.0, 0.0, 0.0),
            (10.0, 1.0, 0.0),
            (-10.0, 5.0, 1.0),
        )
    )
    points = np.array(
        [
            [
                0.5267784068878942,
                0.6555917078713585,
                0.37866621210455154,
            ],
            [
                0.7746884340006319,
                0.01514600791570253,
                0.9813692271192944,
            ],
        ],
        dtype=np.float64,
    )
    constraints = [(0, 1, 0.5)]
    observations = resolve_separator_observations(
        points,
        constraints,
        domain=cell,
        image_search=1,
    )

    assert observations.shifts.tolist() == [[-70, 6, -1]]
    assert observations.delta.tolist() == [[
        0.24791002711273769,
        0.3595543000443441,
        -0.3972969849852571,
    ]]
    assert observations.distance2.tolist() == [0.3485835705017922]
    assert observations.distance.tolist() == [0.5904096632862577]

    report = _row_only_report(observations)
    row_id = _independent_row_id(observations, 0)
    set_fingerprint = _independent_set_fingerprint(observations)
    assert observations.to_records()[0]['row_id'] == row_id
    assert report['observation_set'] == {
        'fingerprint': set_fingerprint,
        'measurement': 'fraction',
        'n_rows': 1,
        'row_ids': [row_id],
    }
    assert report['source']['binding'] == 'bound'

    repeated = resolve_separator_observations(
        points.copy(),
        constraints,
        domain=cell,
        image_search=1,
    )
    assert repeated.to_records()[0]['row_id'] == row_id
    assert _row_only_report(repeated)['observation_set'] == (
        report['observation_set']
    )

    raw_fit = fit_weights_from_separators(
        points,
        constraints,
        domain=cell,
        image_search=1,
    )
    assert raw_fit.weights.shape == (2,)

    for variant in _copy_variants(observations):
        variant_fit = fit_weights_from_separators(
            points,
            variant,
            domain=cell,
        )
        assert variant_fit.weights.shape == (2,)
        assert _row_only_report(variant)['observation_set'] == (
            report['observation_set']
        )
        assert _row_only_report(variant)['source']['binding'] == 'bound'

    different_points = points.copy()
    different_points[0, 0] = np.nextafter(different_points[0, 0], np.inf)
    with pytest.raises(ValueError, match='already bound.*points'):
        fit_weights_from_separators(
            different_points,
            observations,
            domain=cell,
        )


def _ordinary_bound_triclinic_observations() -> SeparatorObservations:
    cell = pyvoro2.PeriodicCell(
        (
            (1.0, 0.0, 0.0),
            (0.3, 1.0, 0.0),
            (0.2, 0.1, 1.0),
        )
    )
    points = np.array(
        [
            [0.15, 0.25, 0.35],
            [0.85, 0.75, 0.65],
        ],
        dtype=np.float64,
    )
    return resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        domain=cell,
        image_search=1,
    )


def test_source_verification_uses_small_seed_without_maximum_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyvoro2.inverse.separator import constraints as constraints_module

    observations = _ordinary_bound_triclinic_observations()
    original = constraints_module._derive_connector_geometry
    image_search_calls: list[int] = []

    def traced(*args, **kwargs):
        image_search_calls.append(kwargs['image_search'])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        constraints_module,
        '_derive_connector_geometry',
        traced,
    )
    rebuilt = replace(observations)

    assert image_search_calls == [1]
    np.testing.assert_array_equal(rebuilt.shifts, observations.shifts)
    np.testing.assert_array_equal(rebuilt.delta, observations.delta)
    np.testing.assert_array_equal(rebuilt.distance2, observations.distance2)
    np.testing.assert_array_equal(rebuilt.distance, observations.distance)


@pytest.mark.parametrize(
    'failure_stage',
    ['pair_candidate_budget', 'batch_candidate_budget'],
)
def test_source_verification_retries_resource_failure_with_maximum_seed(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    from pyvoro2._internal.periodic_images import (
        MinimumImageCertificationError,
    )
    from pyvoro2.inverse.separator import constraints as constraints_module

    observations = _ordinary_bound_triclinic_observations()
    original = constraints_module._derive_connector_geometry
    image_search_calls: list[int] = []

    def fail_then_certify(*args, **kwargs):
        image_search = kwargs['image_search']
        image_search_calls.append(image_search)
        if len(image_search_calls) == 1:
            raise MinimumImageCertificationError(
                'injected verification resource failure',
                stage=failure_stage,
                method='triclinic-finite-box',
                pair_index=0,
                basis_summary={},
                candidate_bound=1_000_001,
                configured_limit=1_000_000,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        constraints_module,
        '_derive_connector_geometry',
        fail_then_certify,
    )
    rebuilt = replace(observations)

    assert image_search_calls == [1, sys.maxsize]
    np.testing.assert_array_equal(rebuilt.shifts, observations.shifts)
    np.testing.assert_array_equal(rebuilt.delta, observations.delta)
    np.testing.assert_array_equal(rebuilt.distance2, observations.distance2)
    np.testing.assert_array_equal(rebuilt.distance, observations.distance)


def test_source_verification_propagates_nonresource_certification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyvoro2._internal.periodic_images import (
        MinimumImageCertificationError,
    )
    from pyvoro2.inverse.separator import constraints as constraints_module

    observations = _ordinary_bound_triclinic_observations()
    image_search_calls: list[int] = []

    def fail_without_retry(*args, **kwargs):
        image_search_calls.append(kwargs['image_search'])
        raise MinimumImageCertificationError(
            'injected non-resource verification failure',
            stage='basis_preparation',
            method='unresolved',
            pair_index=None,
            basis_summary={},
        )

    monkeypatch.setattr(
        constraints_module,
        '_derive_connector_geometry',
        fail_without_retry,
    )
    with pytest.raises(
        MinimumImageCertificationError,
        match='injected non-resource verification failure',
    ):
        replace(observations)

    assert image_search_calls == [1]


def test_fit_and_realized_reports_exact_json_roundtrip() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0]])
    domain = planar.Box(((-5.0, 5.0), (-5.0, 5.0)))
    observations = resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        domain=domain,
    )
    fit = fit_weights_from_separators(points, observations, domain=domain)
    fit_report = build_fit_report(fit, observations)
    assert fit_report['kind'] == 'power_weight_fit'
    assert fit_report['fit_records'][0]['row_id'] == (
        observations.to_records()[0]['row_id']
    )
    _assert_json_roundtrip(fit_report)

    realized = match_realized_pairs(
        points,
        domain=domain,
        constraints=observations,
        weights=fit.weights,
    )
    realized_report = build_realized_report(realized, observations)
    assert realized_report['kind'] == 'realized_pair_diagnostics'
    assert realized_report['records'][0]['row_id'] == (
        observations.to_records()[0]['row_id']
    )
    _assert_json_roundtrip(realized_report)


def test_finite_active_report_roundtrips_and_nonfinite_failure_fails_closed() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0]])
    domain = planar.Box(((-5.0, 5.0), (-5.0, 5.0)))
    finite = solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        domain=domain,
        options=ActiveSetOptions(max_iter=3),
        return_history=True,
    )
    finite_report = build_active_set_report(finite)
    assert finite_report['kind'] == 'self_consistent_power_fit'
    assert finite_report['diagnostics'][0]['row_id'] == (
        finite.constraints.to_records()[0]['row_id']
    )
    _assert_json_roundtrip(finite_report)

    replacements = {
        'site_i': np.array([1]),
        'site_j': np.array([0]),
        'shift': np.array([[9, 9]]),
        'target': np.array([0.125]),
        'confidence': np.array([0.25]),
    }
    for name, value in replacements.items():
        with pytest.raises(ValueError, match=rf'origin \({name}\)'):
            replace(finite.diagnostics, **{name: value})

    tampered = copy.deepcopy(finite)
    tampered.diagnostics.site_i[0] = 1
    with pytest.raises(ValueError, match=r'origin \(site_i\)'):
        build_active_set_report(tampered)

    failure_points = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    failure_domain = planar.Box(((-5.0, 10.0), (-5.0, 5.0)))
    failure = solve_self_consistent_power_weights(
        failure_points,
        [(0, 1, 0.0), (1, 2, 0.0), (0, 2, 0.0)],
        measurement='position',
        domain=failure_domain,
        model=FitModel(feasible=FixedValue(0.0)),
        fit_solver='admm',
        options=ActiveSetOptions(max_iter=2),
    )
    assert failure.termination == 'infeasible_active_set'
    assert not np.isfinite(failure.rms_residual_all)
    with pytest.raises(ValueError, match='NaN or infinite'):
        build_active_set_report(failure)
