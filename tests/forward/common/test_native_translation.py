"""Independent exact-envelope contracts for private native translation recovery."""

from dataclasses import replace
from fractions import Fraction
import importlib
import importlib.util
from itertools import product
import math
import random
import tracemalloc

import numpy as np
import pytest

from pyvoro2._internal.exact_lattice import (
    DEFAULT_REDUCTION_LIMITS,
    ExactLatticeReductionInvariantError,
    ExactLatticeReductionResourceError,
)
from pyvoro2._internal import periodic_images


def _native():
    # This assertion makes the pre-implementation RED a feature failure rather
    # than a collection/import error. All expected geometry below is independent.
    name = 'pyvoro2._internal.native_translation'
    assert importlib.util.find_spec(name) is not None, (
        'WP4 exact-envelope native translation certification is missing'
    )
    return importlib.import_module(name)


def _source_oracle(lattice, lower, upper):
    """Cofactor inverse plus a separately proved symmetric source search box.

    For every compatible t, |s_j| <= sum_k max(|lo_k|, |hi_k|)
    |(A^-1)_kj| by the triangle inequality. This bound encloses every candidate;
    it does not inspect an observed winner or invoke any production helper.
    """
    rows = tuple(tuple(Fraction(float(x)) for x in row) for row in lattice)
    cofactors = []
    for i in range(3):
        cofactor_row = []
        for j in range(3):
            minor = [[rows[r][c] for c in range(3) if c != j]
                     for r in range(3) if r != i]
            cofactor_row.append((-1)**(i + j) * (
                minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
            ))
        cofactors.append(cofactor_row)
    determinant = sum(rows[0][j] * cofactors[0][j] for j in range(3))
    inverse = tuple(tuple(cofactors[j][i] / determinant for j in range(3))
                    for i in range(3))
    bounds = tuple(math.floor(sum(
        max(abs(lower[k]), abs(upper[k])) * abs(inverse[k][j])
        for k in range(3)
    )) for j in range(3))
    assert math.prod(2 * bound + 1 for bound in bounds) < 100_000
    compatible = {}
    for shift in product(*(range(-bound, bound + 1) for bound in bounds)):
        translation = tuple(sum(shift[r] * rows[r][c] for r in range(3))
                            for c in range(3))
        if all(lo <= value <= hi for lo, value, hi in
               zip(lower, translation, upper)):
            compatible[shift] = translation
    return compatible


@pytest.mark.parametrize('lengths,axes,lower,upper,shift', [
    ((1, 1, 1), (True, True, True), (2, -3, 4), (2, -3, 4), (2, -3, 4)),
    ((-2, 3, 4), (True, True, True), (4, -3, 8), (4, -3, 8), (-2, -1, 2)),
    ((-2, 3), (True, False), (4, -1), (4, 1), (-2, 0)),
    ((2, 3, -4), (False, True, True), (-1, 3, 4), (1, 3, 4), (0, 1, -1)),
    ((2, 3), (False, False), (-1, -1), (1, 1), (0, 0)),
])
def test_signed_diagonal_and_partial_domains(lengths, axes, lower, upper, shift):
    native = _native()
    result = native.certify_native_translation(
        np.diag(lengths), native.CartesianCompatibilityBox(lower, upper),
        periodic_axes=axes,
    )
    assert result.shift == shift
    assert all(type(value) is int for value in result.shift)
    assert result.translation == tuple(a * b for a, b in zip(lengths, shift))
    assert result.certified
    assert result.diagnostics.examined_count == 1
    assert result.diagnostics.compatible_count == 1


def test_closed_boundaries_and_just_across_are_exact():
    native = _native()
    tiny = Fraction(1, 2**200)
    result = native.certify_native_translation(
        np.eye(3), native.CartesianCompatibilityBox((1, 0, 0), (1 + tiny, 0, 0)),
    )
    assert result.shift == (1, 0, 0)
    with pytest.raises(native.NativeTranslationInconsistencyError) as captured:
        native.certify_native_translation(
            np.eye(3), native.CartesianCompatibilityBox(
                (1 + tiny, 0, 0), (1 + 2 * tiny, 0, 0),
            ),
        )
    assert captured.value.diagnostics.compatible_count == 0
    assert captured.value.diagnostics.candidate_bound == 0


def test_envelope_constructor_exactifies_before_subtraction_and_uses_sign():
    native = _native()
    error = native.CartesianCompatibilityBox(
        (Fraction(-1, 4), 0, 0), (Fraction(1, 8), 0, 0),
    )
    box = native.translation_box_from_observation(
        (float(2**53), 3.0, 0.0), (-1.0, 2.0, 0.0), error,
    )
    assert box.lower == (Fraction(2**53 + 1) - Fraction(1, 4), 1, 0)
    assert box.upper == (Fraction(2**53 + 1) + Fraction(1, 8), 1, 0)
    result = native.certify_native_translation(np.eye(3), box)
    assert result.shift == (2**53 + 1, 1, 0)


@pytest.mark.parametrize('lower_x,expected_count', [(0, 1), (Fraction(1, 10), 0)])
def test_cartesian_filter_rejects_coefficient_box_false_positives(
    lower_x, expected_count,
):
    native = _native()
    lattice = np.array(((1, 0, 0), (.5, 1, 0), (0, 0, 1)))
    lower, upper = (lower_x, 0, 0), (Fraction(1, 4), 1, 0)
    oracle = _source_oracle(lattice, lower, upper)
    assert len(oracle) == expected_count
    box = native.CartesianCompatibilityBox(lower, upper)
    if expected_count:
        result = native.certify_native_translation(lattice, box)
        assert result.shift == (0, 0, 0)
        diagnostics = result.diagnostics
    else:
        with pytest.raises(native.NativeTranslationInconsistencyError) as captured:
            native.certify_native_translation(lattice, box)
        diagnostics = captured.value.diagnostics
    assert diagnostics.coefficient_widths == (1, 2, 1)
    assert diagnostics.candidate_bound == diagnostics.examined_count == 2
    assert diagnostics.compatible_count == expected_count


def test_ambiguity_finishes_enumeration_and_bounds_user_basis_witnesses():
    native = _native()
    limits = native.NativeTranslationLimits(max_witnesses=3)
    with pytest.raises(native.NativeTranslationAmbiguityError) as captured:
        native.certify_native_translation(
            np.eye(3), native.CartesianCompatibilityBox((-2, -2, -2), (2, 2, 2)),
            limits=limits,
        )
    diagnostics = captured.value.diagnostics
    assert diagnostics.candidate_bound == diagnostics.examined_count == 125
    assert diagnostics.compatible_count == 125
    assert len(diagnostics.witnesses) == 3
    assert len(set(diagnostics.witnesses)) == 3
    assert all(all(-2 <= x <= 2 for x in row) for row in diagnostics.witnesses)


@pytest.mark.parametrize('shear', [256, 2**63, 2**80])
def test_poor_shear_maps_unique_shift_without_int64_materialization(shear):
    native = _native()
    lattice = np.array(((1, 0, 0), (shear, 1, 0), (0, 0, 1)), dtype=float)
    assert Fraction(float(shear)) == shear
    result = native.certify_native_translation(
        lattice, native.CartesianCompatibilityBox(
            (Fraction(-1, 4), Fraction(3, 4), 0),
            (Fraction(1, 4), Fraction(5, 4), 0),
        ),
    )
    assert result.shift == (-shear, 1, 0)
    assert result.translation == (0, 1, 0)
    assert result.diagnostics.candidate_bound == 1
    assert result.diagnostics.max_mapped_coefficient_bits >= shear.bit_length()


def test_huge_reduced_coefficients_can_map_to_small_user_shift():
    native = _native()
    shear = 2**80
    point = (shear, 1, 0)
    result = native.certify_native_translation(
        ((1, 0, 0), (shear, 1, 0), (0, 0, 1)),
        native.CartesianCompatibilityBox(point, point),
    )
    assert result.shift == (0, 1, 0)
    assert result.translation == point
    assert result.diagnostics.max_reduced_coefficient_bits == 81


def test_ambiguity_keeps_shifts_outside_int64():
    native = _native()
    point = 2**80
    with pytest.raises(native.NativeTranslationAmbiguityError) as captured:
        native.certify_native_translation(np.eye(3), native.CartesianCompatibilityBox(
            (point, 0, 0), (point + 1, 0, 0),
        ))
    assert captured.value.diagnostics.compatible_count == 2
    assert set(captured.value.diagnostics.witnesses) == {
        (point, 0, 0), (point + 1, 0, 0),
    }


def test_reduced_rows_that_binary64_cannot_represent_remain_exact():
    native = _native()
    tiny = Fraction(1, 2**55)
    lattice = ((1.0, float(tiny), 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    point = (0, 1 - tiny, 0)
    result = native.certify_native_translation(
        lattice, native.CartesianCompatibilityBox(point, point),
    )
    assert result.shift == (-1, 1, 0)
    assert result.translation == point
    with pytest.raises(native.NativeTranslationInconsistencyError):
        native.certify_native_translation(
            lattice, native.CartesianCompatibilityBox((0, 1, 0), (0, 1, 0)),
        )


@pytest.mark.parametrize('left_handed', [False, True])
def test_determinant_cancellation_family(left_handed):
    native = _native()
    size = 2**27
    lattice = [(size, size - 1, 0), (size + 1, size, 0), (0, 0, 1)]
    expected = (size, 1 - size, 0)
    if left_handed:
        lattice[0], lattice[1] = lattice[1], lattice[0]
        expected = (1 - size, size, 0)
    point = (1, 0, 0)
    result = native.certify_native_translation(
        lattice, native.CartesianCompatibilityBox(point, point),
    )
    assert result.shift == expected
    assert result.translation == point


@pytest.mark.parametrize('lattice', [
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ((1, 0, 0), (.5, 1, 0), (.25, -.5, 1)),
    ((1, .5, 0), (0, 1, .5), (.25, 0, 1)),
])
def test_general_basis_counts_match_complete_independent_cofactor_oracle(lattice):
    native = _native()
    for lower, upper in [
        ((-1, -1, -1), (1, 1, 1)),
        ((Fraction(1, 3), 0, 0), (Fraction(2, 3), 1, 0)),
        ((0, 0, 0), (Fraction(1, 7), Fraction(1, 9), Fraction(1, 11))),
    ]:
        oracle = _source_oracle(lattice, lower, upper)
        box = native.CartesianCompatibilityBox(lower, upper)
        if len(oracle) == 1:
            result = native.certify_native_translation(lattice, box)
            assert oracle[result.shift] == result.translation
            diagnostics = result.diagnostics
        else:
            error = (native.NativeTranslationInconsistencyError if not oracle
                     else native.NativeTranslationAmbiguityError)
            with pytest.raises(error) as captured:
                native.certify_native_translation(lattice, box)
            diagnostics = captured.value.diagnostics
        assert diagnostics.compatible_count == len(oracle)
        assert diagnostics.examined_count == diagnostics.candidate_bound
        assert set(diagnostics.witnesses).issubset(oracle)


def test_nonperiodic_translation_component_must_admit_zero():
    native = _native()
    with pytest.raises(native.NativeTranslationInconsistencyError) as captured:
        native.certify_native_translation(
            np.eye(2), native.CartesianCompatibilityBox((0, 1), (0, 2)),
            periodic_axes=(True, False),
        )
    assert captured.value.diagnostics.compatible_count == 0
    assert captured.value.diagnostics.examined_count == 0
    assert captured.value.diagnostics.reason == 'nonperiodic-incompatibility'


def test_default_candidate_budget_preflights_complete_product():
    native = _native()
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            np.eye(3), native.CartesianCompatibilityBox((0, 0, 0), (100, 100, 100)),
        )
    error = captured.value
    assert error.resource == 'candidates'
    assert error.observed == 101**3
    assert error.configured_limit == 1_000_000
    assert error.diagnostics.coefficient_widths == (101, 101, 101)
    assert error.diagnostics.examined_count == 0
    assert error.diagnostics.compatible_count is None


def test_smaller_candidate_budget_cannot_be_bypassed_by_cached_preparation():
    native = _native()
    lattice = ((1, 0, 0), (.5, 1, 0), (0, 0, 1))
    box = native.CartesianCompatibilityBox((0, 0, 0), (Fraction(1, 4), 1, 0))
    assert native.certify_native_translation(lattice, box).shift == (0, 0, 0)
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            lattice, box, limits=native.NativeTranslationLimits(max_candidates=1),
        )
    assert captured.value.diagnostics.examined_count == 0
    assert captured.value.diagnostics.candidate_bound == 2


def test_integer_guard_observes_mapping_product_before_cancellation():
    native = _native()
    shear = 2**80
    point = (shear, 2, 0)
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            ((1, 0, 0), (shear, 1, 0), (0, 0, 1)),
            native.CartesianCompatibilityBox(point, point),
            limits=native.NativeTranslationLimits(max_integer_bits=81),
        )
    error = captured.value
    assert error.resource == 'integer_bits'
    assert error.observed == 82
    assert error.diagnostics.stage == 'shift-mapping'
    assert error.diagnostics.compatible_count == 1


def test_rational_guard_observes_inverse_bound_product():
    native = _native()
    point = (2**100, 0, 0)
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            np.diag((2.0**-100, 1, 1)),
            native.CartesianCompatibilityBox(point, point),
            limits=native.NativeTranslationLimits(max_rational_bits=200),
        )
    assert captured.value.resource == 'rational_bits'
    assert captured.value.observed == 201
    assert captured.value.diagnostics.examined_count == 0


def test_reduction_resources_keep_underlying_cause_and_never_fall_back():
    native = _native()
    lattice = ((1, 0, 0), (256, 1, 0), (0, 0, 1))
    box = native.CartesianCompatibilityBox((0, 0, 0), (0, 0, 0))
    assert native.certify_native_translation(lattice, box).shift == (0, 0, 0)
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            lattice, box,
            reduction_limits=replace(DEFAULT_REDUCTION_LIMITS, max_work=1),
        )
    assert captured.value.diagnostics.stage == 'basis-reduction'
    assert isinstance(captured.value.__cause__, ExactLatticeReductionResourceError)
    assert captured.value.diagnostics.examined_count == 0


def test_reduction_invariant_failure_stays_distinct(monkeypatch):
    native = _native()
    # Fault injection is necessary: a valid real reducer never deliberately
    # violates its certificate. No geometric expectation depends on this stub.

    def fail(*args, **kwargs):
        raise ExactLatticeReductionInvariantError('injected certificate failure')

    monkeypatch.setattr(native, '_prepare_basis', fail)
    with pytest.raises(native.NativeTranslationInvariantError) as captured:
        native.certify_native_translation(
            ((1, 0, 0), (.5, 1, 0), (0, 0, 1)),
            native.CartesianCompatibilityBox((0, 0, 0), (0, 0, 0)),
        )
    assert isinstance(captured.value.__cause__, ExactLatticeReductionInvariantError)
    assert captured.value.diagnostics.compatible_count is None


@pytest.mark.parametrize('lower,upper', [
    ((0,), (1,)), ((0, 0), (1, 1, 1)), ((1, 0), (0, 0)),
    ((float('nan'), 0), (1, 1)), ((0, 0), (float('inf'), 1)),
    ((0.0, 0), (1, 1)), ((False, 0), (1, 1)), ((0j, 0), (1, 1)),
    (None, (1, 1)), (('0', 0), (1, 1)),
])
def test_malformed_exact_boxes_are_value_errors(lower, upper):
    native = _native()
    with pytest.raises(ValueError):
        native.CartesianCompatibilityBox(lower, upper)


@pytest.mark.parametrize('lattice,axes', [
    ([[1, 0], [0]], None), ([[1, 0], [0, float('nan')]], None),
    ([[0, 0], [0, 1]], None), ([[1, 1], [0, 1]], None),
    ([[1, 0], [0, 1]], (1, True)), ([[1, 0], [0, 1]], (True,)),
    ([[1, 0, 0], [.5, 1, 0], [0, 0, 1]], (True, False, True)),
    ([[1, 0, 0], [2, 0, 0], [0, 0, 1]], None),
])
def test_malformed_or_unsupported_geometry_is_value_error(lattice, axes):
    native = _native()
    dimension = len(lattice)
    box = native.CartesianCompatibilityBox((0,) * dimension, (0,) * dimension)
    with pytest.raises(ValueError):
        native.certify_native_translation(lattice, box, periodic_axes=axes)


@pytest.mark.parametrize('name,value', [
    ('max_candidates', 0), ('max_candidates', 1.5), ('max_witnesses', False),
    ('max_candidates', 1_000_001),
    ('max_integer_bits', -1), ('max_rational_bits', '3'),
])
def test_malformed_resource_policies_are_value_errors(name, value):
    native = _native()
    with pytest.raises(ValueError):
        native.NativeTranslationLimits(**{name: value})


def test_native_cold_warm_work_metrics_include_exact_counts():
    native = _native()
    periodic_images._basis_cache_clear()
    lattice = ((1, 0, 0), (2**80, 1, 0), (0, 0, 1))
    box = native.CartesianCompatibilityBox(
        (Fraction(-1, 4), Fraction(3, 4), 0),
        (Fraction(1, 4), Fraction(5, 4), 0),
    )
    cold = native.certify_native_translation(lattice, box)
    before = periodic_images._basis_cache_info()
    warm = native.certify_native_translation(lattice, box)
    after = periodic_images._basis_cache_info()
    assert after.hits == before.hits + 1
    assert cold.shift == warm.shift == (-2**80, 1, 0)
    diagnostics = warm.diagnostics
    assert diagnostics.coefficient_widths == (1, 1, 1)
    assert diagnostics.candidate_bound == diagnostics.examined_count == 1
    assert diagnostics.compatible_count == 1
    assert diagnostics.max_integer_bits >= 81
    assert diagnostics.max_rational_bits >= 3
    assert diagnostics.reduction_diagnostics.work > 0


@pytest.mark.parametrize('seed', [0, 7, 18])
def test_random_unimodular_representation_uses_independently_composed_inverse(seed):
    native = _native()
    rng = random.Random(seed)
    rows = [[int(i == j) for j in range(3)] for i in range(3)]
    inverse = [[int(i == j) for j in range(3)] for i in range(3)]
    for _ in range(12):
        i, j = rng.sample(range(3), 2)
        multiplier = rng.choice((-3, -2, 2, 3))
        rows[i] = [left + multiplier * right
                   for left, right in zip(rows[i], rows[j])]
        # (E @ A)^-1 = A^-1 @ E^-1: update column j using column i.
        for k in range(3):
            inverse[k][j] -= multiplier * inverse[k][i]
    for i, j in product(range(3), repeat=2):
        assert sum(rows[i][k] * inverse[k][j] for k in range(3)) == int(i == j)
        assert Fraction(float(rows[i][j])) == rows[i][j]
    point = (2, -1, 3)
    expected = tuple(sum(point[k] * inverse[k][j] for k in range(3))
                     for j in range(3))
    box = native.CartesianCompatibilityBox(
        tuple(Fraction(value) - Fraction(1, 4) for value in point),
        tuple(Fraction(value) + Fraction(1, 4) for value in point),
    )
    result = native.certify_native_translation(rows, box)
    assert result.shift == expected
    assert result.translation == point
    assert result.diagnostics.candidate_bound == 1


def test_subnormal_near_dependent_basis_has_exact_huge_translation_coefficients():
    native = _native()
    tiny = float.fromhex('0x0.0000000000001p-1022')
    point = (0, 1, 0)
    result = native.certify_native_translation(
        ((1, 0, 0), (1, tiny, 0), (0, 0, 1)),
        native.CartesianCompatibilityBox(point, point),
    )
    assert result.shift == (-2**1074, 2**1074, 0)
    assert result.translation == point
    assert result.diagnostics.max_reduced_coefficient_bits == 1075


@pytest.mark.parametrize('point,resource,stage,observed', [
    (Fraction(1, 2**65_536), 'rational_bits', 'envelope', 65_537),
    (2**32_768, 'integer_bits', 'coefficient-bounds', 32_769),
], ids=('rational-limit', 'integer-limit'))
def test_default_semantic_bit_limits_are_structural(point, resource, stage, observed):
    native = _native()
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            np.eye(3), native.CartesianCompatibilityBox((point, 0, 0), (point, 0, 0)),
        )
    error = captured.value
    assert error.resource == resource
    assert error.observed == observed
    assert error.stage == stage
    assert error.diagnostics.examined_count == 0
    assert error.diagnostics.compatible_count is None


@pytest.mark.parametrize('observed,reference', [
    ((0, float('inf')), (0, 0)), ((float('nan'), 0), (0, 0)),
    ((0, 0), ('0', 0)), ((True, 0), (0, 0)), ((1j, 0), (0, 0)),
    ((0, 0, 0), (0, 0)), (None, (0, 0)), ((2**2000, 0), (0, 0)),
])
def test_malformed_binary64_observation_inputs_are_value_errors(observed, reference):
    native = _native()
    with pytest.raises(ValueError):
        native.translation_box_from_observation(
            observed, reference, native.CartesianCompatibilityBox((0, 0), (0, 0)),
        )


def test_envelope_construction_guard_observes_difference_before_cancellation():
    native = _native()
    magnitude = 2**80
    # Each operand and error endpoint fits 81 bits. The difference needs 82
    # bits, although adding the negative error yields an 81-bit endpoint.
    observed = float(magnitude)
    reference = float(-magnitude)
    errors = native.CartesianCompatibilityBox((-magnitude, 0), (-magnitude, 0))
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.translation_box_from_observation(
            (observed, 0), (reference, 0), errors,
            limits=native.NativeTranslationLimits(max_rational_bits=81),
        )
    assert captured.value.stage == 'envelope-construction'
    assert captured.value.resource == 'rational_bits'
    assert captured.value.observed == 82


@pytest.mark.parametrize('dimension', [2, 3])
def test_candidate_traversal_does_not_pool_large_integer_ranges(dimension):
    native = _native()
    lattice = np.eye(dimension)
    zero = (0,) * dimension
    native.certify_native_translation(
        lattice, native.CartesianCompatibilityBox(zero, zero),
    )
    width = 10_000
    start = 2**32_760
    lower = (start,) + (0,) * (dimension - 1)
    upper = (start + width - 1,) + (0,) * (dimension - 1)
    box = native.CartesianCompatibilityBox(lower, upper)
    tracemalloc.start()
    try:
        with pytest.raises(native.NativeTranslationAmbiguityError) as captured:
            native.certify_native_translation(lattice, box)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    diagnostics = captured.value.diagnostics
    assert diagnostics.examined_count == diagnostics.compatible_count == width
    assert len(diagnostics.witnesses) == 4
    # Pooling this range retains about 44 MB of 32,761-bit integers. A lazy
    # traversal retains only a few coefficients/witnesses; 2 MB leaves broad
    # interpreter/allocator margin while catching that multiplicative storage.
    assert peak_bytes < 2_000_000


@pytest.mark.parametrize('max_witnesses', [1, 2, 4])
def test_completed_ambiguity_survives_optional_witness_mapping_resource(max_witnesses):
    native = _native()
    limits = native.NativeTranslationLimits(
        max_integer_bits=4, max_witnesses=max_witnesses,
    )
    with pytest.raises(native.NativeTranslationAmbiguityError) as captured:
        native.certify_native_translation(
            ((1, 0, 0), (8, 1, 0), (0, 0, 1)),
            native.CartesianCompatibilityBox((0, 0, 0), (0, 2, 0)),
            limits=limits,
        )
    diagnostics = captured.value.diagnostics
    assert diagnostics.candidate_bound == diagnostics.examined_count == 3
    assert diagnostics.compatible_count == 3
    assert diagnostics.witnesses == ((0, 0, 0), (-8, 1, 0))[:max_witnesses]
    if max_witnesses < 3:
        assert diagnostics.witness_resource is None
    else:
        resource = diagnostics.witness_resource
        assert resource.stage == 'shift-mapping'
        assert resource.resource == 'integer_bits'
        assert resource.observed == 5
        assert resource.configured_limit == 4


def test_unique_shift_mapping_resource_still_prevents_success():
    native = _native()
    with pytest.raises(native.NativeTranslationResourceError) as captured:
        native.certify_native_translation(
            ((1, 0, 0), (8, 1, 0), (0, 0, 1)),
            native.CartesianCompatibilityBox((0, 2, 0), (0, 2, 0)),
            limits=native.NativeTranslationLimits(max_integer_bits=4),
        )
    assert captured.value.diagnostics.compatible_count == 1
    assert captured.value.stage == 'shift-mapping'
    assert captured.value.resource == 'integer_bits'
