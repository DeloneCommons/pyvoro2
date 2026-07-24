from __future__ import annotations

import numpy as np
import pytest

import pyvoro2.planar as pv2
import pyvoro2.planar.api as api2d


def test_planar_compute_raises_helpful_error_when_core_missing(monkeypatch) -> None:
    monkeypatch.setattr(api2d, '_core2d', None, raising=False)
    monkeypatch.setattr(
        api2d,
        '_CORE2D_IMPORT_ERROR',
        ImportError('dummy'),
        raising=False,
    )

    with pytest.raises(ImportError) as exc:
        pv2.compute(np.zeros((1, 2)), domain=pv2.Box(((0, 1), (0, 1))))

    msg = str(exc.value)
    assert '_core2d' in msg
    assert 'planar support' in msg or 'build from source' in msg
