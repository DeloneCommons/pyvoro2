"""Retained certificate views keep the owning structured refusal family."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
def test_retained_spatial_certificate_refuses_without_numeric_diagnostics():
    script = r'''
        import ctypes
        from types import SimpleNamespace
        import numpy as np
        from pyvoro2 import OrthorhombicCell, TessellationError, _core
        from pyvoro2._internal.spatial.wp5_certificate import FaceCertificate
        cert = FaceCertificate(
            packet={'cells': [{'id': 0, 'volume': 1., 'computed': True}]},
            prepared=SimpleNamespace(external_ids=np.array([0], dtype=np.int64)),
            domain=OrthorhombicCell(((0., 1.1),) * 3), snapshot=None,
            mode='standard', shifts=(), lattice=(), labels={}, semantic=None,
            effective=None, epsilon=(), lattice_defect=(), frame=(), charts=(),
            work=0,
        )
        libc = ctypes.CDLL(None)
        enable = libc.feenableexcept
        restore = libc.fesetenv
        raw_state = _core._runtime_fp_state
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        refused = False
        try:
            enable(32)
            incoming = raw_state()
            try:
                cert.boundary_measures()
            except TessellationError as error:
                refused = 'WP5_UNSUPPORTED_FP_PROFILE' in str(error)
            # Use the non-waiting inspector while x87 exceptions are pending.
            assert raw_state() == incoming
            assert incoming['x87_control'] & 0x20 == 0
            assert incoming['mxcsr'] & 0x1000 == 0
        finally:
            assert restore(saved) == 0
        assert refused
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
