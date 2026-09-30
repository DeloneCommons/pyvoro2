"""Domain overrides cannot change FP controls before package arithmetic."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_script(script: str, **values: object) -> None:
    parameters = '\n'.join(f'{key} = {value!r}' for key, value in values.items())
    result = subprocess.run(
        [sys.executable, '-X', 'faulthandler', '-c',
         parameters + '\n' + textwrap.dedent(script)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
@pytest.mark.parametrize(('read_number', 'diagnostics'), [
    (1, False), (2, False), (3, False), (5, True),
])
def test_public_box_bounds_override_is_checked(read_number, diagnostics):
    _run_script(r'''
        import ctypes
        import pyvoro2

        libc = ctypes.CDLL(None)
        enable = libc.feenableexcept
        restore = libc.fesetenv
        raw_state = pyvoro2._internal.native_runtime._early_guard._runtime_fp_state
        class PoisonBox(pyvoro2.Box):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                if name == 'bounds' and active:
                    reads.append(True)
                    if len(reads) == READ_NUMBER:
                        enable(32)
                return value

        active = False
        reads = []
        domain = PoisonBox(((.1, 1.3),) * 3)
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        active = True
        error = None
        try:
            try:
                pyvoro2.compute([[.2] * 3], domain=domain, output='cells',
                                return_diagnostics=DIAGNOSTICS)
            except (ValueError, RuntimeError) as caught:
                error = caught
            trapped = not raw_state()['masked']
        finally:
            assert restore(saved) == 0
        assert len(reads) == READ_NUMBER, 'poisoned getter was not reached'
        assert trapped, 'caller controls were normalized'
        assert error is not None and 'UNSUPPORTED' in str(error), repr(error)
    ''', READ_NUMBER=read_number, DIAGNOSTICS=diagnostics)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('boundary', [
    'bounds', 'periodic', 'remap_lookup', 'remap_return', 'lattice_bounds',
    'lattice_vectors',
])
def test_rectangular_domain_callback_is_checked(dimension, boundary):
    _run_script(r'''
        import ctypes
        import numpy as np
        import pyvoro2
        import pyvoro2.planar
        from pyvoro2 import _core
        from pyvoro2._internal.native_runtime import RuntimeFPError
        from pyvoro2._internal.planar.domain_geometry import geometry2d
        from pyvoro2._internal.spatial.domain_geometry import geometry3d

        libc = ctypes.CDLL(None)
        enable = libc.feenableexcept
        restore = libc.fesetenv
        raw_state = pyvoro2._internal.native_runtime._early_guard._runtime_fp_state
        base = (pyvoro2.planar.RectangularCell if DIM == 2
                else pyvoro2.OrthorhombicCell)
        attribute = {'bounds': 'bounds', 'periodic': 'periodic',
                     'remap_lookup': 'remap_cart',
                     'lattice_bounds': 'bounds',
                     'lattice_vectors': 'lattice_vectors'}.get(BOUNDARY)
        calls = []
        def poison():
            calls.append(True)
            enable(32)

        class PoisonCell(base):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                if active and name == attribute:
                    poison()
                return value

            def remap_cart(self, *args, **kwargs):
                result = super().remap_cart(*args, **kwargs)
                if active and BOUNDARY == 'remap_return':
                    poison()
                return result

        active = False
        domain = PoisonCell(((.1, 1.3),) * DIM)
        geometry = (geometry2d if DIM == 2 else geometry3d)(domain)
        points = np.array([[.2] * DIM])
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        active = True
        error = None
        try:
            try:
                if BOUNDARY == 'lattice_vectors' and DIM == 3:
                    from pyvoro2._internal.spatial.domain_utils import (
                        domain_lattice_vectors)
                    domain_lattice_vectors(domain)
                elif BOUNDARY in ('lattice_vectors', 'lattice_bounds'):
                    geometry.lattice_vectors_cart
                else:
                    geometry.remap_cart(points)
            except RuntimeFPError as caught:
                error = caught
            trapped = not raw_state()['masked']
        finally:
            assert restore(saved) == 0
        assert calls, 'domain callback was not exercised'
        assert trapped, 'caller controls were normalized'
        assert error is not None, 'domain callback escaped runtime refusal'
    ''', DIM=dimension, BOUNDARY=boundary)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
@pytest.mark.parametrize('boundary', [
    'vectors', 'origin', 'to_internal_params', 'cart_to_internal',
    'remap_internal', '_backend_frame',
])
@pytest.mark.parametrize('lookup', [False, True])
def test_triclinic_domain_callback_is_checked(boundary, lookup):
    _run_script(r'''
        import ctypes
        import pyvoro2
        import pyvoro2._core
        from pyvoro2._internal.native_runtime import RuntimeFPError

        libc = ctypes.CDLL(None)
        enable = libc.feenableexcept
        restore = libc.fesetenv
        raw_state = pyvoro2._internal.native_runtime._early_guard._runtime_fp_state
        active = False
        calls = []
        def poison():
            calls.append(True)
            enable(32)

        class PoisonCell(pyvoro2.PeriodicCell):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                if active and name == BOUNDARY:
                    if LOOKUP or name in ('vectors', 'origin'):
                        poison()
                    else:
                        def method(*args, **kwargs):
                            result = value(*args, **kwargs)
                            poison()
                            return result
                        return method
                return value

        domain = PoisonCell(((1.2, 0., 0.), (.1, 1.1, 0.), (.2, .1, 1.3)))
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        active = True
        error = None
        try:
            try:
                domain.remap_cart([[.2] * 3])
            except RuntimeFPError as caught:
                error = caught
            trapped = not raw_state()['masked']
        finally:
            assert restore(saved) == 0
        assert calls, 'domain callback was not exercised'
        assert trapped, 'caller controls were normalized'
        assert error is not None, 'domain callback escaped runtime refusal'
    ''', BOUNDARY=boundary, LOOKUP=lookup)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('return_number', [1, 2])
def test_normalization_domain_method_return_is_checked(dimension, return_number):
    _run_script(r'''
        import ctypes
        import pyvoro2
        import pyvoro2.planar
        import pyvoro2._core
        from pyvoro2._internal.native_runtime import RuntimeFPError
        from pyvoro2.normalize import normalize_vertices as normalize3d
        from pyvoro2.planar.normalize import normalize_vertices as normalize2d

        libc = ctypes.CDLL(None)
        enable = libc.feenableexcept
        restore = libc.fesetenv
        raw_state = pyvoro2._internal.native_runtime._early_guard._runtime_fp_state
        base = (pyvoro2.planar.RectangularCell if DIM == 2
                else pyvoro2.OrthorhombicCell)
        calls = []
        class PoisonCell(base):
            def remap_cart(self, *args, **kwargs):
                result = super().remap_cart(*args, **kwargs)
                calls.append(True)
                if len(calls) == RETURN_NUMBER:
                    enable(32)
                return result

        domain = PoisonCell(((.1, 1.3),) * DIM)
        cells = [{'id': 7, 'vertices': [[.2] * DIM],
                  'edges' if DIM == 2 else 'faces': []}]
        normalize = normalize2d if DIM == 2 else normalize3d
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        error = None
        try:
            try:
                normalize(cells, domain=domain, tol=1e-8)
            except RuntimeFPError as caught:
                error = caught
            trapped = not raw_state()['masked']
        finally:
            assert restore(saved) == 0
        assert len(calls) == RETURN_NUMBER, 'method return was not exercised'
        assert trapped, 'caller controls were normalized'
        assert error is not None, 'method return escaped runtime refusal'
    ''', DIM=dimension, RETURN_NUMBER=return_number)


@pytest.mark.parametrize('dimension', [2, 3])
def test_domain_only_subclass_behavior_preserved_without_certificate(dimension):
    _run_script(r'''
        import sys
        import numpy as np
        import pyvoro2
        import pyvoro2.planar
        from pyvoro2._internal import native_qualification
        from pyvoro2._internal.planar.domain_geometry import geometry2d
        from pyvoro2._internal.spatial.domain_geometry import geometry3d

        def unexpected(*args, **kwargs):
            raise AssertionError('domain-only work requested an artifact')
        native_qualification.require_native = unexpected
        native_qualification.register_native = unexpected
        base = (pyvoro2.planar.RectangularCell if DIM == 2
                else pyvoro2.OrthorhombicCell)
        calls = []
        class CustomCell(base):
            def remap_cart(self, *args, **kwargs):
                calls.append('remap')
                result = super().remap_cart(*args, **kwargs)
                return result + .125

        domain = CustomCell(((.1, 1.3),) * DIM)
        geometry = (geometry2d if DIM == 2 else geometry3d)(domain)
        result = geometry.remap_cart([[1.4] * DIM])
        np.testing.assert_allclose(result, [[.325] * DIM])
        assert calls == ['remap'], 'domain override was bypassed'
        assert 'pyvoro2._core' not in sys.modules
        assert 'pyvoro2._core2d' not in sys.modules
    ''', DIM=dimension)
