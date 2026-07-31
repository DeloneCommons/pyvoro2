"""Cross-platform regressions for private separator numeric kernels."""

from __future__ import annotations

import numpy as np

import pyvoro2.inverse.separator as separator
import pyvoro2.inverse.separator._numerics as separator_numerics
import pyvoro2.inverse.separator._quadratic as separator_quadratic


def test_vectorized_ldexp_paths_use_c_int_exponents(monkeypatch) -> None:
    """Keep NumPy 1.x Windows ``ldexp`` compatibility on every vector path."""

    original_ldexp = np.ldexp
    exponent_dtypes: list[np.dtype] = []

    def windows_numpy_1x_ldexp(
        mantissa: object,
        exponent: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        exponent_dtype = np.asarray(exponent).dtype
        exponent_dtypes.append(exponent_dtype)
        if exponent_dtype == np.dtype(np.int64):
            raise TypeError(
                "ufunc 'ldexp' not supported for float64/int64 inputs"
            )
        return original_ldexp(mantissa, exponent, *args, **kwargs)

    monkeypatch.setattr(np, 'ldexp', windows_numpy_1x_ldexp)

    fit = separator.fit_weights_from_separators(
        np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64),
        [(0, 1, 0.25)],
        solver='direct',
        linear_backend='dense',
        connectivity_check='diagnose',
    )

    extended_mantissa = np.array([0.75], dtype=np.longdouble)
    extended = separator_numerics._ldexp(
        extended_mantissa,
        np.array([2], dtype=np.int64),
    )
    ratio = separator_numerics._stable_ratio_product(
        (np.array([0.5]),),
        (np.array([0.25]),),
    )
    product = separator_numerics._stable_product(
        np.array([1.0e308]),
        np.array([1.0e308]),
        np.array([1.0e-308]),
    )
    normalized = separator_numerics._stable_normalized_ratio(
        np.array([1.0]),
        np.array([2.0]),
        active=np.array([True]),
    )
    split_high, split_low = separator_numerics._split_product_operand(
        np.array([1.1]),
    )
    scaled = separator_numerics._power_scaled_product(
        2,
        np.array([0.5]),
    )
    product_high, product_low = (
        separator_quadratic._power_scaled_product_parts(
            0,
            np.array([1.1]),
            np.array([1.1]),
        )
    )

    assert fit.status == 'optimal'
    assert fit.solver == 'direct'
    assert fit.linear_backend == 'dense'
    assert extended.dtype == extended_mantissa.dtype
    np.testing.assert_array_equal(extended, np.array([3.0], dtype=np.longdouble))
    np.testing.assert_array_equal(ratio, np.array([2.0]))
    np.testing.assert_allclose(product, np.array([1.0e308]), rtol=2e-15)
    np.testing.assert_array_equal(normalized, np.array([0.5]))
    np.testing.assert_array_equal(split_high + split_low, np.array([1.1]))
    np.testing.assert_array_equal(scaled, np.array([2.0]))
    np.testing.assert_allclose(
        product_high + product_low,
        np.array([1.21]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    assert exponent_dtypes
    assert set(exponent_dtypes) == {np.dtype(np.intc)}
