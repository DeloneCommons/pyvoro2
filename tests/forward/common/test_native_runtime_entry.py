"""Raw runtime admission must precede every native floating argument caster."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


def test_native_runtime_inspection_is_available_on_all_modules():
    from pyvoro2 import _core, _core2d, _fpguard

    for module in (_core, _core2d, _fpguard):
        state = module._runtime_fp_state()
        assert state['supported'] is True
        assert state['compatible'] is True
        module._require_runtime_environment()
        assert module._runtime_fp_state() == state


@pytest.fixture(scope='module')
def x86_controls(tmp_path_factory):
    if sys.platform != 'linux' or platform.machine().lower() != 'x86_64':
        pytest.skip('GNU/Linux x86-64 raw register controls')
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('an external C++ compiler is needed for the test controller')
    directory = tmp_path_factory.mktemp('native-runtime-controls')
    source = directory / 'controls.cpp'
    source.write_text(textwrap.dedent(r'''
        #include <cstdint>
        extern "C" void controls(std::uint16_t cw, std::uint32_t mxcsr) {
          asm volatile("fldcw %0" : : "m"(cw) : "memory");
          asm volatile("ldmxcsr %0" : : "m"(mxcsr) : "memory");
        }
        extern "C" void sticky() {
          alignas(16) unsigned char environment[32]{};
          asm volatile("fnstenv %0" : "=m"(environment) : : "memory");
          environment[4] |= 0x3f;
          asm volatile("fldenv %0" : : "m"(environment) : "memory");
          std::uint32_t mxcsr;
          asm volatile("stmxcsr %0" : "=m"(mxcsr) : : "memory");
          mxcsr |= 0x3f;
          asm volatile("ldmxcsr %0" : : "m"(mxcsr) : "memory");
        }
        extern "C" void restore_x87_status(std::uint16_t status) {
          alignas(16) unsigned char environment[32]{};
          asm volatile("fnstenv %0" : "=m"(environment) : : "memory");
          environment[4] = static_cast<unsigned char>(status);
          environment[5] = static_cast<unsigned char>(status >> 8);
          asm volatile("fldenv %0" : : "m"(environment) : "memory");
        }
    '''))
    output = directory / 'controls.so'
    subprocess.run([compiler, '-shared', '-fPIC', '-O2', '-fno-fast-math',
                    '-ffp-contract=off', str(source), '-o', str(output)],
                   check=True, capture_output=True, text=True)
    return output


_HOSTILE_STATES = (
    ('x87_up', 0x0c00, 0x0800, 0, 0),
    ('sse_up', 0, 0, 0x6000, 0x4000),
    ('joint_up', 0x0c00, 0x0800, 0x6000, 0x4000),
    ('joint_down', 0x0c00, 0x0400, 0x6000, 0x2000),
    ('joint_zero', 0x0c00, 0x0c00, 0x6000, 0x6000),
    ('ftz', 0, 0, 0, 0x8000),
    ('daz', 0, 0, 0, 0x0040),
    ('ftz_daz', 0, 0, 0, 0x8040),
    ('pc24', 0x0300, 0, 0, 0),
    ('pc53', 0x0300, 0x0200, 0, 0),
) + tuple((f'x87_exception_{bit}', 1 << bit, 0, 0, 0)
          for bit in range(6)) + tuple(
    (f'sse_exception_{bit}', 0, 0, 1 << (bit + 7), 0)
    for bit in range(6)
) + (('combined_masks', 0x0025, 0, 0x1280, 0),)


def _child(controller: Path, body: str):
    script = textwrap.dedent(r'''
        import ctypes
        import json
        import numpy as np
        from pyvoro2 import _core, _core2d, _fpguard
        controller = ctypes.CDLL(CONTROLLER)
        controller.controls.argtypes = [ctypes.c_uint16, ctypes.c_uint32]
        controller.controls.restype = None
        controller.sticky.restype = None
        controller.restore_x87_status.argtypes = [ctypes.c_uint16]
        controller.restore_x87_status.restype = None
        original = _core._runtime_fp_state()
    ''').replace('CONTROLLER', repr(str(controller)))
    result = subprocess.run([sys.executable, '-c', script + '\n' + body],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    return json.loads(result.stdout)


@pytest.mark.parametrize('case', _HOSTILE_STATES, ids=lambda case: case[0])
def test_hostile_control_refusal_precedes_native_casters(x86_controls, case):
    body = textwrap.dedent(r'''
        class Uncoercible:
            def __array__(self, *args, **kwargs):
                raise AssertionError('floating array caster ran before refusal')
            def __float__(self):
                raise AssertionError('floating scalar caster ran before refusal')
        p = Uncoercible()
        ids = np.array([0], dtype=np.int32)
        bounds3 = ((0., 1.),) * 3
        bounds2 = ((0., 1.),) * 2
        opts = (False, False, False)
        calls = [
            lambda: _core.compute_box_standard(p, ids, bounds3, (1, 1, 1),
                                               (False,) * 3, 1, opts),
            lambda: _core.compute_periodic_standard(p, ids, (1., 0., 1., 0., 0., 1.),
                                                    (1, 1, 1), 1, opts),
            lambda: _core._observe_box(p, ids, bounds3, (1, 1, 1), (False,) * 3, 1),
            lambda: _core._observe_ghost_box(p, ids, bounds3, (1, 1, 1),
                                             (False,) * 3, 1, p),
            lambda: _core.locate_box_standard(p, ids, bounds3, (1, 1, 1),
                                              (False,) * 3, 1, p, False),
            lambda: _core.ghost_box_standard(p, ids, bounds3, (1, 1, 1),
                                             (False,) * 3, 1, opts, p),
            lambda: _core._test_power_offset_order(p, p, p),
            lambda: _core2d.compute_box_standard(p, ids, bounds2, (1, 1),
                                                 (False,) * 2, 1, opts),
            lambda: _core2d._compute_box_standard_witness(p, ids, bounds2, (1, 1),
                                                          (False,) * 2, 1, opts),
            lambda: _core2d.locate_box_standard(p, ids, bounds2, (1, 1),
                                                (False,) * 2, 1, p, False),
            lambda: _core2d.ghost_box_standard(p, ids, bounds2, (1, 1),
                                               (False,) * 2, 1, opts, p),
            lambda: _core2d._ghost_box_standard_witness(p, ids, bounds2, (1, 1),
                                                        (False,) * 2, 1, opts, p),
        ]
        _, clear_cw, set_cw, clear_mx, set_mx = CASE
        cw = (original['x87_control'] & ~clear_cw) | set_cw
        mxcsr = (original['mxcsr'] & ~clear_mx) | set_mx
        try:
            controller.controls(cw, mxcsr)
            incoming = _core._runtime_fp_state()
            assert incoming['compatible'] is False
            for module in (_core, _core2d, _fpguard):
                try:
                    module._require_runtime_environment()
                except ValueError as error:
                    assert 'native runtime FP profile' in str(error)
                else:
                    raise AssertionError('hostile environment admitted')
                assert module._runtime_fp_state() == incoming
            for call in calls:
                try:
                    call()
                except (RuntimeError, ValueError) as error:
                    assert 'runtime FP profile' in str(error), str(error)
                else:
                    raise AssertionError('native floating entry admitted')
                assert _core._runtime_fp_state() == incoming
            profile = _core2d._planar_witness_profile()
            assert profile['runtime_compatible'] is False
            assert _core._spatial_witness_profile()['runtime_compatible'] is False
            assert _core._runtime_fp_state() == incoming
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
        assert _core._runtime_fp_state() == original
        print(json.dumps({'refused': len(calls), 'restored': True}))
    ''').replace('CASE', repr(case))
    assert _child(x86_controls, body) == {'refused': 12, 'restored': True}


def test_masked_sticky_flags_are_accepted_and_unchanged(x86_controls):
    body = textwrap.dedent(r'''
        try:
            controller.sticky()
            incoming = _core._runtime_fp_state()
            assert incoming['x87_status'] & 0x3f == 0x3f
            assert incoming['mxcsr'] & 0x3f == 0x3f
            for module in (_core, _core2d, _fpguard):
                assert module._runtime_fp_state()['compatible'] is True
                module._require_runtime_environment()
                assert module._runtime_fp_state() == incoming
            _core2d._planar_witness_profile()
            _core._spatial_witness_profile()
            assert _core._runtime_fp_state() == incoming
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
            controller.restore_x87_status(original['x87_status'])
        assert _core._runtime_fp_state() == original
        print(json.dumps({'accepted': True}))
    ''')
    assert _child(x86_controls, body) == {'accepted': True}


def test_guard_rechecks_each_call_on_the_executing_thread(x86_controls):
    body = textwrap.dedent(r'''
        from concurrent.futures import ThreadPoolExecutor
        def worker():
            saved = _core._runtime_fp_state()
            try:
                _core._require_runtime_environment()
                controller.controls(saved['x87_control'], saved['mxcsr'] | 0x4000)
                for module in (_core, _core2d, _fpguard):
                    try:
                        module._require_runtime_environment()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('cached another call or thread state')
                controller.controls(saved['x87_control'], saved['mxcsr'])
                _core._require_runtime_environment()
                return _core._runtime_fp_state() == saved
            finally:
                controller.controls(saved['x87_control'], saved['mxcsr'])
        _core._require_runtime_environment()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(worker).result()
        assert _core._runtime_fp_state() == original
        print(json.dumps({'worker_restored': result, 'caller_unchanged': True}))
    ''')
    assert _child(x86_controls, body) == {
        'worker_restored': True, 'caller_unchanged': True,
    }


@pytest.mark.parametrize('callback', ('float', 'array', 'index', 'iteration'))
def test_native_dispatch_rechecks_foreign_conversion_returns(x86_controls, callback):
    body = textwrap.dedent(r'''
        from pyvoro2._internal import native_runtime  # Load under safe controls.
        bounds = ((0., 1.),) * 3
        ids = np.array([0], dtype=np.int32)
        points = np.array([[0.25, 0.5, 0.75]])
        signaling = np.array([[0x7f800001, 0x3f000000, 0x3f400000]],
                             dtype=np.uint32).view(np.float32)
        calls = []
        def hostile():
            # Unmask invalid in both execution domains. The array case would
            # trap during NumPy's float32-to-object promotion without the
            # check immediately following the __array__ return.
            controller.controls(original['x87_control'] & ~1,
                                original['mxcsr'] & ~(1 << 7))
        class FloatChanger(float):
            def __float__(self):
                calls.append('float')
                hostile()
                return 0.25
        class ArrayChanger:
            def __array__(self, dtype=None, copy=None):
                calls.append('array')
                hostile()
                return signaling
        class IndexChanger:
            def __index__(self):
                calls.append('index')
                hostile()
                return 1
        class SequenceChanger:
            def __iter__(self):
                calls.append('iteration')
                hostile()
                return iter((1, 1, 1))
        kind = CALLBACK
        if kind == 'float':
            call = lambda: _core._test_power_offset_order(FloatChanger(.25), .5, .75)
        else:
            p = ArrayChanger() if kind == 'array' else points
            blocks = SequenceChanger() if kind == 'iteration' else (1, 1, 1)
            memory = IndexChanger() if kind == 'index' else 1
            call = lambda: _core.compute_box_standard(
                p, ids, bounds, blocks, (False,) * 3, memory, (False,) * 3)
        try:
            try:
                call()
            except (ValueError, RuntimeError) as error:
                assert 'runtime FP profile' in str(error), str(error)
            else:
                raise AssertionError('foreign state mutation was admitted')
            assert calls == [kind]
            assert _core._runtime_fp_state()['compatible'] is False
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
            controller.restore_x87_status(original['x87_status'])
        assert _core._runtime_fp_state() == original
        print(json.dumps({'refused': kind, 'restored': True}))
    ''').replace('CALLBACK', repr(callback))
    assert _child(x86_controls, body) == {'refused': callback, 'restored': True}


@pytest.mark.parametrize('trap_bit', [0, 5], ids=['invalid', 'inexact'])
@pytest.mark.parametrize('dimension', [2, 3])
def test_malformed_native_calls_refuse_before_pybind_error_repr(
        x86_controls, trap_bit, dimension):
    body = textwrap.dedent(r'''
        module = _core if DIMENSION == 3 else _core2d
        target = module.compute_box_standard
        value = (np.array([0x7f800001], dtype=np.uint32).view(np.float32)[0]
                 if TRAP_BIT == 0 else [0.1])
        calls = [
            lambda: target(value),
            lambda: target(*([value] * 8)),
            lambda: target(unexpected=value),
            lambda: target(value, points=value),
        ]
        try:
            controller.controls(original['x87_control'] & ~(1 << TRAP_BIT),
                                original['mxcsr'] & ~(1 << (TRAP_BIT + 7)))
            incoming = _core._runtime_fp_state()
            for call in calls:
                try:
                    call()
                except (RuntimeError, ValueError) as error:
                    assert 'runtime FP profile' in str(error)
                else:
                    raise AssertionError('malformed call escaped raw guard')
                assert _core._runtime_fp_state() == incoming
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
            controller.restore_x87_status(original['x87_status'])
        print(json.dumps({'refused': len(calls)}))
    ''').replace('TRAP_BIT', str(trap_bit)).replace('DIMENSION', str(dimension))
    assert _child(x86_controls, body) == {'refused': 4}


def test_native_binding_errors_never_format_foreign_values(x86_controls):
    body = textwrap.dedent(r'''
        calls = []
        class HostileRepr:
            def __repr__(self):
                calls.append('repr')
                controller.controls(original['x87_control'] & ~1,
                                    original['mxcsr'] & ~(1 << 7))
                return 'hostile'
        bad = HostileRepr()
        targets = [_core.compute_box_standard, _core2d.compute_box_standard,
                   _core._test_power_offset_order]
        try:
            for target in targets:
                for invoke in (lambda: target(bad),
                               lambda: target(*([bad] * 12)),
                               lambda: target(unexpected=bad)):
                    try:
                        invoke()
                    except TypeError:
                        pass
                    else:
                        raise AssertionError('invalid binding did not fail')
                    assert calls == [], 'argument repr ran during native binding'
                    assert _core._runtime_fp_state() == original
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
            controller.restore_x87_status(original['x87_status'])
        print(json.dumps({'formatted': calls}))
    ''')
    assert _child(x86_controls, body) == {'formatted': []}


@pytest.mark.parametrize('trap_bit', [0, 5], ids=['invalid', 'inexact'])
def test_metadata_binding_errors_are_safe_under_hostile_controls(
        x86_controls, trap_bit):
    body = textwrap.dedent(r'''
        value = (np.array([0x7f800001], dtype=np.uint32).view(np.float32)[0]
                 if TRAP_BIT == 0 else [0.1])
        methods = [module._runtime_fp_state for module in (_core, _core2d, _fpguard)]
        methods += [module._qualification_identity
                    for module in (_core, _core2d, _fpguard)]
        methods += [_core._spatial_witness_profile, _core2d._planar_witness_profile]
        try:
            controller.controls(original['x87_control'] & ~(1 << TRAP_BIT),
                                original['mxcsr'] & ~(1 << (TRAP_BIT + 7)))
            incoming = _core._runtime_fp_state()
            for target in methods:
                for invoke in (lambda: target(value),
                               lambda: target(unexpected=value)):
                    try:
                        invoke()
                    except TypeError:
                        pass
                    else:
                        raise AssertionError('invalid metadata binding admitted')
                    assert _core._runtime_fp_state() == incoming
        finally:
            controller.controls(original['x87_control'], original['mxcsr'])
            controller.restore_x87_status(original['x87_status'])
        print(json.dumps({'refused': len(methods) * 2}))
    ''').replace('TRAP_BIT', str(trap_bit))
    assert _child(x86_controls, body) == {'refused': 16}


def test_guarded_native_binding_preserves_keywords_and_defaults():
    from pyvoro2 import _core

    positional = _core._test_power_offset_order(2., 3., 4.)
    assert _core._test_power_offset_order(
        neighbor_radius=4., owner_radius=3., distance_squared=2.) == positional
    assert _core._test_power_offset_order(
        2., neighbor_radius=4., owner_radius=3., max_radius_squared=1e100,
    ) == positional
    with pytest.raises(TypeError):
        _core._test_power_offset_order(2., 3., 4., owner_radius=3.)


def test_native_keyword_names_compare_unicode_without_foreign_callbacks():
    from pyvoro2 import _core

    class Keyword(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            raise AssertionError('native binding called foreign keyword equality')

        def __str__(self):
            raise AssertionError('native binding formatted a foreign keyword')

        def __repr__(self):
            raise AssertionError('native binding formatted a foreign keyword')

    keywords = {Keyword('distance_squared'): 2., Keyword('owner_radius'): 3.,
                Keyword('neighbor_radius'): 4.}
    assert _core._test_power_offset_order(**keywords) == (
        _core._test_power_offset_order(2., 3., 4.))
