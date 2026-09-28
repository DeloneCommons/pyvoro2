"""External planar admission and producer/consumer metadata consistency.

Compiler labels and measured source digests do not grant qualification. The
external record first admits the actual loaded artifact; its profile then
checks packet identity and the unchanged binary64 witness predicates.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

from ..native_admission import require_component, require_environment
from ..native_qualification import NativeQualificationError


SCHEMA = 'pyvoro2.planar.occurrences.v1'
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def _refuse(reason, detail):
    raise RuntimeError(f'planar_certification:profile:{reason}: {detail}')


def require_planar(*, component='wp6-planar', artifact=True):
    """Raw controls precede metadata, replay, and retained certificate views."""
    try:
        if artifact:
            return require_component(component)
        require_environment()
    except NativeQualificationError as exc:
        _refuse(exc.reason, exc.detail)


def _validate_metadata(profile, expected_profile):
    """Consistency only; calling this helper alone never admits an artifact."""
    if not isinstance(profile, Mapping):
        _refuse('schema', 'native witness profile is not a mapping')
    if profile.get('schema') != SCHEMA:
        _refuse('schema', 'Python/native occurrence schema differs')
    for key, reason in (('source_sha256', 'source'), ('build_sha256', 'build')):
        value = profile.get(key)
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            _refuse(reason, f'native {key} is absent or malformed')
        if value != expected_profile.get(key):
            _refuse(reason, f'packet {key} differs from the admitted artifact')
    for key in ('compiler', 'compiler_version'):
        if profile.get(key) != expected_profile.get(key):
            _refuse('build', f'packet {key} differs from the admitted artifact')
    expected = {'flt_eval_method': 0, 'int_bits': 32, 'uint_bits': 32,
                'double_digits': 53, 'round_to_nearest': True,
                'gradual_underflow': True, 'fp_contract': 'off',
                'fast_math': False, 'lto': False, 'runtime_compatible': True}
    if any(profile.get(key) != value for key, value in expected.items()):
        _refuse('evaluation', 'native arithmetic/build predicate is not satisfied')


def require_supported_profile(profile, *, component='wp6-planar') -> None:
    """Refuse an unadmitted artifact before consuming any packet provenance."""
    module = require_planar(component=component)
    try:
        expected = module._planar_witness_profile()
    finally:
        require_planar(artifact=False)
    _validate_metadata(profile, expected)
    require_planar(artifact=False)


validate_profile = require_supported_profile
