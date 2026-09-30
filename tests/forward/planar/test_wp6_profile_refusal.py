"""External artifact admission and malformed planar consistency records."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from pyvoro2 import _core2d
import pyvoro2.planar as planar
from pyvoro2._internal.planar.wp6_profile import (
    SCHEMA, _validate_metadata, validate_profile,
)
from pyvoro2._internal.native_admission import require_component
from pyvoro2._internal.native_qualification import NativeQualificationError


BOUNDS = ((0.0, 1.0), (0.0, 1.0))


def test_actual_native_profile_matches_installed_python_source_schema():
    profile = _core2d._planar_witness_profile()
    print('WP6 native profile: ' + json.dumps(profile, sort_keys=True))
    assert profile['schema'] == SCHEMA
    assert not {'qualified', 'cohort_supported', 'source_supported'} & profile.keys()
    assert len(profile['source_sha256']) == 64
    try:
        require_component('wp6-planar')
    except NativeQualificationError as exc:
        with pytest.raises(RuntimeError, match=f'profile:{exc.reason}:'):
            validate_profile(profile)
    else:
        validate_profile(profile)
        assert profile['runtime_compatible'] is True


@pytest.mark.parametrize('periodic', tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize('mode', ['standard', 'radii', 'weights'])
def test_all_masks_and_modes_qualify_or_refuse_structurally(periodic, mode):
    points = np.array([[0.25, 0.5]])
    args = [points, np.array([0], dtype=np.int32)]
    if mode == 'standard':
        native = _core2d._compute_box_standard_witness
        representation = {}
    else:
        native = _core2d._compute_box_power_witness
        args.append(np.array([0.5]))
        representation = {'mode': 'power', mode: [0.25]}
    args.extend([BOUNDS, (1, 1), periodic, 1, (True, True, True)])
    options = dict(domain=planar.RectangularCell(BOUNDS, periodic=periodic),
                   return_edges=True, return_edge_shifts=any(periodic),
                   tessellation_check='none', **representation)
    try:
        require_component('wp6-planar')
    except NativeQualificationError:
        with pytest.raises((RuntimeError, ValueError)):
            native(*args)
        with pytest.raises(planar.TessellationError) as caught:
            planar.compute(points, **options)
        diagnostic = caught.value.diagnostics
        assert diagnostic.n_cells_returned == 0
        assert {issue.code for issue in diagnostic.issues} == {
            'WP6_PROFILE_UNSUPPORTED',
        }
    else:
        cells, packet = native(*args)
        assert len(cells) == 1
        validate_profile(packet['profile'])
        assert len(planar.compute(points, **options).cells) == 1


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('schema', 'invented', 'schema'),
    ('source_sha256', '0' * 64, 'source'),
    ('compiler', 'invented', 'build'),
    ('runtime_compatible', False, 'evaluation'),
    ('round_to_nearest', False, 'evaluation'),
    ('gradual_underflow', False, 'evaluation'),
    ('flt_eval_method', 2, 'evaluation'),
    ('fp_contract', 'fast', 'evaluation'),
    ('fast_math', True, 'evaluation'),
    ('lto', True, 'evaluation'),
    ('build_sha256', 'not-a-digest', 'build'),
])
def test_constructed_malformed_profile_is_never_accepted(field, value, reason):
    # Test consistency separately: this helper grants no artifact admission.
    record = {
        'schema': SCHEMA, 'source_sha256': '2' * 64,
        'compiler': 'example-compiler', 'compiler_version': 'example-version',
        'runtime_compatible': True,
        'flt_eval_method': 0, 'int_bits': 32, 'uint_bits': 32,
        'double_digits': 53, 'round_to_nearest': True,
        'gradual_underflow': True, 'fp_contract': 'off',
        'fast_math': False, 'lto': False, 'build_sha256': '1' * 64,
    }
    changed = dict(record, **{field: value})
    with pytest.raises(RuntimeError,
                       match=f'planar_certification:profile:{reason}:'):
        _validate_metadata(changed, record)
