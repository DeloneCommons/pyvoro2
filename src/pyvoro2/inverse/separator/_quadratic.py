"""Certified least-squares kernels for separator quadratics.

The public separator objective is expressed in source units as

``0.5 * confidence * residual**2``

plus optional L2 regularization.  The authoritative path solves the equivalent
augmented least-squares problem directly.  A guarded normal-equation fast path
is allowed only for conservatively conditioned systems, and every candidate is
still checked against the source objective.  Failure to certify a binary64
result is a numerical failure rather than an ``optimal`` solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Literal, Sequence

import numpy as np

from ._numerics import (
    _finite_product_exponents,
    _power_scaled_product,
    _stable_incidence_accumulate,
    _stable_norm,
    _stable_product,
    _stable_product_scalar,
    _stable_ratio_difference,
    _stable_ratio_product,
    _stable_ratio_product_scalar,
    _stable_scaled_difference,
    _stable_sum,
    _stable_sum_scalar,
    _stable_sum_products_sign,
    _stable_sum_squares,
    _two_product_error,
    _two_sum,
)
from ._objective import _l2_value, _mismatch_values_from_affine
from .model import SquaredLoss


_FLOAT_EPS = np.finfo(np.float64).eps
_MIN_SUBNORMAL = float.fromhex('0x0.0000000000001p-1022')
_TINY_EXACT_ORACLE_MAX_SITES = 16
_TINY_EXACT_ORACLE_MAX_ROWS = 256
_EXACT_GRADIENT_MAX_SITES = 8192
_EXACT_GRADIENT_MAX_ROWS = 65536
_ZERO_CANDIDATE_MAX_SITES = 256
_ZERO_CANDIDATE_MAX_ROWS = 4096
_CERTIFICATION_FACTOR = 512.0
_DENSE_REFINEMENT_CONDITION = 1.0 / math.sqrt(_FLOAT_EPS)
_DENSE_REFINEMENT_STEPS = 3
# Forming normal equations is a performance optimization only for systems
# whose conservative augmented-design condition bound remains modest.  Every
# candidate is certified against the source objective and falls back to the
# unsquared augmented system on any failure.
_NORMAL_FAST_CONDITION_LIMIT = 0.5 / math.sqrt(_FLOAT_EPS)


class QuadraticNumericalError(RuntimeError):
    """Raised when a quadratic component cannot be certified in binary64."""


def _require_scipy_sparse() -> None:
    """Validate the optional dependency for an explicitly sparse solve."""

    try:
        import scipy.sparse  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by API tests
        raise ImportError(
            "linear_backend='sparse' requires SciPy; install "
            'pyvoro2[sparse] or scipy'
        ) from exc


def _fraction(value: float) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def _fraction_frexp_exponent(value: Fraction) -> int:
    """Return the exponent used by ``frexp`` for a nonzero rational value."""

    numerator = abs(value.numerator)
    denominator = value.denominator
    floor_log2 = numerator.bit_length() - denominator.bit_length()
    if floor_log2 >= 0:
        if numerator < (denominator << floor_log2):
            floor_log2 -= 1
    elif (numerator << -floor_log2) < denominator:
        floor_log2 -= 1
    return floor_log2 + 1


def _component_mean(values: np.ndarray) -> float:
    """Return a stable finite component mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    if maximum <= np.finfo(np.float64).max / int(array.size):
        return float(math.fsum(array.tolist()) / int(array.size))
    exact = sum((_fraction(value) for value in array), Fraction(0))
    try:
        return float(exact / int(array.size))
    except OverflowError as exc:  # pragma: no cover - guarded by finite inputs
        raise QuadraticNumericalError(
            'component mean is outside the binary64 output range'
        ) from exc


def _zero_objective_candidate(
    n_sites: int,
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    reference: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray | None:
    """Return a representable zero-objective candidate when one is cheap.

    The ordinary rejection path is entirely vectorized.  Exact sign tests are
    reserved for a candidate whose rounded affine residuals are already all
    zero, so large inconsistent systems never enter per-row rational work.
    """

    if n_sites > _ZERO_CANDIDATE_MAX_SITES:
        return None

    active = np.asarray(confidence, dtype=np.float64) > 0.0
    I_active = np.asarray(I, dtype=np.int64)[active]
    J_active = np.asarray(J, dtype=np.int64)[active]
    alpha_active = np.asarray(alpha, dtype=np.float64)[active]
    beta_active = np.asarray(beta, dtype=np.float64)[active]
    target_active = np.asarray(target, dtype=np.float64)[active]
    if I_active.size > _ZERO_CANDIDATE_MAX_ROWS:
        return None

    if float(lambda_regularize) > 0.0:
        candidate = np.asarray(reference, dtype=np.float64).copy()
    else:
        differences = _stable_ratio_difference(
            target_active,
            beta_active,
            alpha_active,
        )
        if not np.all(np.isfinite(differences)):
            return None
        adjacency: list[list[tuple[int, float]]] = [
            [] for _ in range(n_sites)
        ]
        for site_i, site_j, difference in zip(
            I_active.tolist(),
            J_active.tolist(),
            differences.tolist(),
        ):
            adjacency[site_i].append((site_j, -difference))
            adjacency[site_j].append((site_i, difference))

        candidate = np.zeros(n_sites, dtype=np.float64)
        seen = np.zeros(n_sites, dtype=bool)
        for root in range(n_sites):
            if seen[root]:
                continue
            seen[root] = True
            stack = [root]
            while stack:
                site = stack.pop()
                value = float(candidate[site])
                for neighbor, increment in adjacency[site]:
                    if seen[neighbor]:
                        continue
                    proposed = _stable_sum_scalar(value, increment)
                    if not math.isfinite(proposed):
                        return None
                    candidate[neighbor] = proposed
                    seen[neighbor] = True
                    stack.append(neighbor)

    if I_active.size == 0:
        return candidate

    # A cheap rounded check rejects the overwhelmingly common nonzero case.
    # Only an apparently exact candidate reaches the authoritative exact-sign
    # classifier below.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        rounded_residual = (
            beta_active
            + alpha_active
            * (candidate[I_active] - candidate[J_active])
            - target_active
        )
    if not np.all(rounded_residual == 0.0):
        return None

    signs = _stable_sum_products_sign(
        (
            (beta_active,),
            (alpha_active, candidate[I_active]),
            (-1.0, alpha_active, candidate[J_active]),
            (-1.0, target_active),
        )
    )
    if np.any(signs != 0):
        return None
    return candidate


def _is_exact_zero_objective(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
) -> bool:
    """Prove that every active source-objective term is exactly zero."""

    candidate = np.asarray(weights, dtype=np.float64)
    if prepared.lambda_regularize > 0.0 and not np.array_equal(
        candidate,
        prepared.reference,
    ):
        return False
    if prepared.I.size == 0:
        return True

    # The shared sign helper rejects ordinary nonzero rows vectorially and
    # invokes exact finite-input arithmetic only for doubtful cancellations.
    signs = _stable_sum_products_sign(
        (
            (prepared.beta,),
            (prepared.alpha, candidate[prepared.I]),
            (-1.0, prepared.alpha, candidate[prepared.J]),
            (-1.0, prepared.target),
        )
    )
    return bool(np.all(signs == 0))


def _exact_scaled_exponent(
    factors: Sequence[float],
    *,
    difference: tuple[float, float] | None = None,
) -> int | None:
    value = Fraction(1)
    for factor in factors:
        value *= _fraction(factor)
    if difference is not None:
        value *= _fraction(difference[0]) - _fraction(difference[1])
    if value == 0:
        return None
    return _fraction_frexp_exponent(value)


def _scaled_difference_exponents(
    left: np.ndarray,
    right: np.ndarray | float,
    scale: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero masks and safe upper exponents for scaled differences.

    For finite binary64 operands, ``two_sum(left, -right)`` supplies a rounded
    high part and its exact low correction.  A nonzero finite difference has a
    nonzero high part, whose exponent is a safe upper exponent for the exact
    difference.  Only subtraction-overflow rows require rational fallback.
    """

    left_array, right_array, scale_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        difference_high, difference_low = _two_sum(
            left_array,
            -right_array,
        )
    valid, exponent = _finite_product_exponents(
        scale_array,
        difference_high,
    )
    finite_inputs = (
        np.isfinite(left_array)
        & np.isfinite(right_array)
        & np.isfinite(scale_array)
    )
    exceptional = (
        finite_inputs
        & (scale_array != 0.0)
        & (left_array != right_array)
        & (
            ~np.isfinite(difference_high)
            | ~np.isfinite(difference_low)
            | ~valid
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact = _exact_scaled_exponent(
            (float(scale_array.flat[flat_index]),),
            difference=(
                float(left_array.flat[flat_index]),
                float(right_array.flat[flat_index]),
            ),
        )
        if exact is not None:
            valid.flat[flat_index] = True
            exponent.flat[flat_index] = exact
    return valid, exponent


def _power_scaled_product_parts(
    power: int,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a compensated expansion of ``2**power * left * right``.

    Ordinary normal products use Dekker's error-free product on normalized
    mantissas.  Rational fallback is restricted to finite rows whose product
    lands outside that normal range.
    """

    left_array, right_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    high = _power_scaled_product(power, left_array, right_array)
    low = np.zeros(high.shape, dtype=np.float64)
    ordinary = (
        np.isfinite(left_array)
        & np.isfinite(right_array)
        & (left_array != 0.0)
        & (right_array != 0.0)
        & np.isfinite(high)
        & (np.abs(high) >= np.finfo(np.float64).tiny)
    )
    if np.any(ordinary):
        left_part, left_exponent = np.frexp(left_array[ordinary])
        right_part, right_exponent = np.frexp(right_array[ordinary])
        rounded_part = left_part * right_part
        error_part = _two_product_error(
            left_part,
            right_part,
            rounded_part,
        )
        exponent = (
            left_exponent.astype(np.int64)
            + right_exponent.astype(np.int64)
            + int(power)
        )
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            low[ordinary] = np.ldexp(error_part, exponent)

    exceptional = (
        np.isfinite(left_array)
        & np.isfinite(right_array)
        & (left_array != 0.0)
        & (right_array != 0.0)
        & ~ordinary
    )
    for flat_index in np.flatnonzero(exceptional):
        exact = (
            _fraction(left_array.flat[flat_index])
            * _fraction(right_array.flat[flat_index])
        )
        if power >= 0:
            exact *= 1 << power
        else:
            exact /= 1 << -power
        try:
            high_value = float(exact)
        except OverflowError:
            high_value = float('-inf') if exact < 0 else float('inf')
        high.flat[flat_index] = high_value
        if math.isfinite(high_value):
            remainder = exact - _fraction(high_value)
            try:
                low.flat[flat_index] = float(remainder)
            except OverflowError:
                low.flat[flat_index] = (
                    float('-inf') if remainder < 0 else float('inf')
                )
    return high, low


def _scaled_difference_parts(
    left: np.ndarray,
    right: np.ndarray | float,
    scale: np.ndarray | float,
    power: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return high/low parts of ``2**power * scale * (left-right)``."""

    left_array, right_array, scale_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        difference_high, difference_low = _two_sum(
            left_array,
            -right_array,
        )
    high, high_error = _power_scaled_product_parts(
        power,
        scale_array,
        difference_high,
    )
    low_high, low_error = _power_scaled_product_parts(
        power,
        scale_array,
        difference_low,
    )
    low = _stable_sum(high_error, low_high, low_error)

    finite_inputs = (
        np.isfinite(left_array)
        & np.isfinite(right_array)
        & np.isfinite(scale_array)
    )
    exceptional = (
        finite_inputs
        & (scale_array != 0.0)
        & (left_array != right_array)
        & (
            ~np.isfinite(difference_high)
            | ~np.isfinite(difference_low)
            | ~np.isfinite(high)
            | ~np.isfinite(low)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact = (
            _fraction(scale_array.flat[flat_index])
            * (
                _fraction(left_array.flat[flat_index])
                - _fraction(right_array.flat[flat_index])
            )
        )
        if power >= 0:
            exact *= 1 << power
        else:
            exact /= 1 << -power
        try:
            high_value = float(exact)
        except OverflowError:
            high_value = float('-inf') if exact < 0 else float('inf')
        high.flat[flat_index] = high_value
        if math.isfinite(high_value):
            remainder = exact - _fraction(high_value)
            try:
                low.flat[flat_index] = float(remainder)
            except OverflowError:
                low.flat[flat_index] = (
                    float('-inf') if remainder < 0 else float('inf')
                )
    return high, low


@dataclass(frozen=True, slots=True)
class _PreparedQuadratic:
    n_sites: int
    anchor: int
    I: np.ndarray
    J: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    target: np.ndarray
    strength: np.ndarray
    sqrt_strength: np.ndarray
    reference: np.ndarray
    lambda_regularize: float
    sqrt_lambda: float
    reference_mean: float
    variable_exponent: int
    design_exponent: int
    observation_coefficient: np.ndarray
    regularization_coefficient: float
    observation_rhs: np.ndarray
    observation_rhs_low: np.ndarray
    regularization_rhs: np.ndarray
    regularization_rhs_low: np.ndarray
    n_unknowns: int
    _singular_lower_cache: float | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def dense_design(self) -> np.ndarray:
        n_rows = int(self.I.size) + (
            self.n_sites if self.lambda_regularize > 0.0 else 0
        )
        design = np.zeros((n_rows, self.n_unknowns), dtype=np.float64)
        rows = np.arange(self.I.size, dtype=np.int64)
        if self.lambda_regularize > 0.0:
            np.add.at(
                design,
                (rows, self.I),
                self.observation_coefficient,
            )
            np.add.at(
                design,
                (rows, self.J),
                -self.observation_coefficient,
            )
            design[self.I.size :] = (
                self.regularization_coefficient
                * np.eye(self.n_sites, dtype=np.float64)
            )
            return design

        mask_i = self.I != self.anchor
        if np.any(mask_i):
            columns_i = self.I[mask_i] - (
                self.I[mask_i] > self.anchor
            )
            np.add.at(
                design,
                (rows[mask_i], columns_i),
                self.observation_coefficient[mask_i],
            )
        mask_j = self.J != self.anchor
        if np.any(mask_j):
            columns_j = self.J[mask_j] - (
                self.J[mask_j] > self.anchor
            )
            np.add.at(
                design,
                (rows[mask_j], columns_j),
                -self.observation_coefficient[mask_j],
            )
        return design

    def sparse_design(self):
        try:
            from scipy.sparse import coo_matrix, eye, vstack
        except ImportError as exc:  # pragma: no cover - exercised by caller
            raise ImportError(
                "linear_backend='sparse' requires SciPy; install "
                'pyvoro2[sparse] or scipy'
            ) from exc

        row_index = np.arange(self.I.size, dtype=np.int64)
        if self.lambda_regularize > 0.0:
            rows = np.concatenate((row_index, row_index))
            columns = np.concatenate((self.I, self.J))
            data = np.concatenate(
                (
                    self.observation_coefficient,
                    -self.observation_coefficient,
                )
            )
            observation = coo_matrix(
                (data, (rows, columns)),
                shape=(self.I.size, self.n_unknowns),
            ).tocsc()
            regularization = (
                self.regularization_coefficient
                * eye(self.n_sites, format='csc')
            )
            return vstack(
                (observation, regularization),
                format='csc',
            )

        rows_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        mask_i = self.I != self.anchor
        if np.any(mask_i):
            rows_parts.append(row_index[mask_i])
            column_parts.append(
                self.I[mask_i] - (self.I[mask_i] > self.anchor)
            )
            data_parts.append(self.observation_coefficient[mask_i])
        mask_j = self.J != self.anchor
        if np.any(mask_j):
            rows_parts.append(row_index[mask_j])
            column_parts.append(
                self.J[mask_j] - (self.J[mask_j] > self.anchor)
            )
            data_parts.append(-self.observation_coefficient[mask_j])
        if rows_parts:
            rows = np.concatenate(rows_parts)
            columns = np.concatenate(column_parts)
            data = np.concatenate(data_parts)
        else:
            rows = np.zeros(0, dtype=np.int64)
            columns = np.zeros(0, dtype=np.int64)
            data = np.zeros(0, dtype=np.float64)
        return coo_matrix(
            (data, (rows, columns)),
            shape=(self.I.size, self.n_unknowns),
        ).tocsc()

    def design_frobenius_square(self) -> float:
        """Return a scale-safe upper bound for the Hessian norm."""

        if self.lambda_regularize > 0.0:
            row_norm_factor = np.full(
                self.I.size,
                math.sqrt(2.0),
                dtype=np.float64,
            )
            observation_norms = _stable_product(
                self.observation_coefficient,
                row_norm_factor,
            )
            regularization_square = _stable_product_scalar(
                float(self.n_sites),
                self.regularization_coefficient,
                self.regularization_coefficient,
            )
        else:
            touches_anchor = (self.I == self.anchor) | (self.J == self.anchor)
            row_norm_factor = np.where(
                touches_anchor,
                1.0,
                math.sqrt(2.0),
            )
            observation_norms = _stable_product(
                self.observation_coefficient,
                row_norm_factor,
            )
            regularization_square = 0.0
        return _stable_sum_scalar(
            _stable_sum_squares(observation_norms),
            regularization_square,
        )

    def smallest_singular_lower_bound(self) -> float:
        """Return a proved lower bound for the gauge-fixed design."""

        if self._singular_lower_cache is not None:
            return self._singular_lower_cache
        if self.lambda_regularize > 0.0:
            # Stacking an incidence block above ``mu * I`` gives
            # ``||A x|| >= mu ||x||`` in every direction.
            lower = abs(self.regularization_coefficient)
            if lower > 0.0 and math.isfinite(lower):
                object.__setattr__(
                    self,
                    '_singular_lower_cache',
                    lower,
                )
                return lower
            raise QuadraticNumericalError(
                'scaled design has no certifiable singular-value bound'
            )
        coefficient = np.abs(self.observation_coefficient)
        if not np.any(coefficient > 0.0):
            raise QuadraticNumericalError(
                'quadratic component has no represented observation curvature'
            )
        # A maximum-bottleneck spanning tree avoids weakening the proof with
        # redundant low-confidence rows.  Root the tree at the fixed anchor.
        # The inverse of its reduced weighted incidence matrix has entry
        # ``1 / coefficient[e]`` exactly when edge ``e`` lies on a
        # root-to-site path, so ``sigma_min >= 1 / ||T^{-1}||_F``.
        parent = np.arange(self.n_sites, dtype=np.int64)
        rank = np.zeros(self.n_sites, dtype=np.int8)

        def find(site: int) -> int:
            root = site
            while int(parent[root]) != root:
                root = int(parent[root])
            while int(parent[site]) != site:
                next_site = int(parent[site])
                parent[site] = root
                site = next_site
            return root

        bottleneck = math.inf
        joined = 0
        tree_edges: list[tuple[int, int, float]] = []
        for row in np.argsort(-coefficient, kind='stable').tolist():
            if coefficient[row] == 0.0:
                break
            root_i = find(int(self.I[row]))
            root_j = find(int(self.J[row]))
            if root_i == root_j:
                continue
            if rank[root_i] < rank[root_j]:
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            if rank[root_i] == rank[root_j]:
                rank[root_i] += 1
            bottleneck = min(bottleneck, float(coefficient[row]))
            tree_edges.append(
                (
                    int(self.I[row]),
                    int(self.J[row]),
                    float(coefficient[row]),
                )
            )
            joined += 1
            if joined == self.n_sites - 1:
                break
        if joined != self.n_sites - 1:
            raise QuadraticNumericalError(
                'quadratic component is rank deficient after gauge fixing'
            )
        adjacency: list[list[tuple[int, float]]] = [
            [] for _ in range(self.n_sites)
        ]
        for site_i, site_j, edge_coefficient in tree_edges:
            adjacency[site_i].append((site_j, edge_coefficient))
            adjacency[site_j].append((site_i, edge_coefficient))
        rooted_parent = np.full(self.n_sites, -1, dtype=np.int64)
        parent_coefficient = np.ones(self.n_sites, dtype=np.float64)
        rooted_parent[self.anchor] = self.anchor
        order = [self.anchor]
        for site in order:
            for neighbor, edge_coefficient in adjacency[site]:
                if rooted_parent[neighbor] != -1:
                    continue
                rooted_parent[neighbor] = site
                parent_coefficient[neighbor] = edge_coefficient
                order.append(neighbor)
        if len(order) != self.n_sites:
            raise QuadraticNumericalError(
                'quadratic spanning-tree bound is disconnected'
            )
        subtree_size = np.ones(self.n_sites, dtype=np.int64)
        subtree_counts: list[int] = []
        subtree_coefficients: list[float] = []
        for site in reversed(order[1:]):
            parent_site = int(rooted_parent[site])
            subtree_counts.append(int(subtree_size[site]))
            subtree_coefficients.append(
                float(parent_coefficient[site])
            )
            subtree_size[parent_site] += subtree_size[site]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            ratio_upper = np.nextafter(
                bottleneck
                / np.asarray(subtree_coefficients, dtype=np.float64),
                math.inf,
            )
            square_upper = np.nextafter(
                ratio_upper * ratio_upper,
                math.inf,
            )
            normalized_terms = np.nextafter(
                np.asarray(subtree_counts, dtype=np.float64)
                * square_upper,
                math.inf,
            )
        if not np.all(np.isfinite(normalized_terms)):
            raise QuadraticNumericalError(
                'quadratic spanning-tree bound is not finite'
            )
        inverse_frobenius_square = math.nextafter(
            math.fsum(normalized_terms.tolist()),
            math.inf,
        )
        denominator = math.nextafter(
            math.sqrt(inverse_frobenius_square),
            math.inf,
        )
        lower = math.nextafter(
            _stable_ratio_product_scalar(
                (bottleneck,),
                (denominator,),
            ),
            0.0,
        )
        if lower == 0.0 or not math.isfinite(lower):
            raise QuadraticNumericalError(
                'scaled design has no certifiable singular-value bound'
            )
        object.__setattr__(
            self,
            '_singular_lower_cache',
            lower,
        )
        return lower

    def sparse_condition_upper_bound(self) -> float:
        """Conservatively bound the scaled design condition."""

        lower = self.smallest_singular_lower_bound()
        frobenius_square = self.design_frobenius_square()
        if frobenius_square <= 0.0 or not math.isfinite(frobenius_square):
            raise QuadraticNumericalError(
                'scaled sparse design norm cannot be certified'
            )
        return _stable_ratio_product_scalar(
            (math.sqrt(frobenius_square),),
            (lower,),
        )

    def rhs_parts_for_target(
        self,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_array = np.asarray(target, dtype=np.float64)
        if target_array.shape != self.target.shape:
            raise QuadraticNumericalError(
                'quadratic right-hand side has an unexpected shape'
            )
        observation_rhs, observation_rhs_low = _scaled_difference_parts(
            target_array,
            self.beta,
            self.sqrt_strength,
            -self.design_exponent,
        )
        if not (
            np.all(np.isfinite(observation_rhs))
            and np.all(np.isfinite(observation_rhs_low))
        ):
            raise QuadraticNumericalError(
                'scaled quadratic right-hand side is outside binary64 range'
            )
        represented_rhs = _stable_sum(
            observation_rhs,
            observation_rhs_low,
        )
        lost = (represented_rhs == 0.0) & (target_array != self.beta)
        if np.any(lost):
            raise QuadraticNumericalError(
                'quadratic right-hand-side range exceeds binary64 support'
            )
        if self.lambda_regularize <= 0.0:
            return observation_rhs, observation_rhs_low
        return (
            np.concatenate((observation_rhs, self.regularization_rhs)),
            np.concatenate(
                (observation_rhs_low, self.regularization_rhs_low)
            ),
        )

    def weights_from_unknown(self, unknown: np.ndarray) -> np.ndarray:
        scaled = np.asarray(unknown, dtype=np.float64)
        if not np.all(np.isfinite(scaled)):
            raise QuadraticNumericalError(
                'least-squares solve produced non-finite scaled weights'
            )
        contrast_values = _power_scaled_product(
            self.variable_exponent,
            scaled,
        )
        if not np.all(np.isfinite(contrast_values)):
            raise QuadraticNumericalError(
                'quadratic optimum is outside the binary64 weight range'
            )
        if self.lambda_regularize > 0.0:
            weights = _stable_sum(
                np.asarray(contrast_values, dtype=np.float64),
                self.reference_mean,
            )
        else:
            weights = np.zeros(self.n_sites, dtype=np.float64)
            free = np.arange(self.n_sites, dtype=np.int64) != self.anchor
            weights[free] = contrast_values
            # Preserve the historical public gauge (site zero equals zero)
            # even though the numerical solve anchors the strongest site.
            weights = _stable_scaled_difference(weights, weights[0], 1.0)
        if not np.all(np.isfinite(weights)):
            raise QuadraticNumericalError(
                'quadratic optimum is outside the binary64 weight range'
            )
        return weights

    def unknown_from_weights(self, weights: np.ndarray) -> np.ndarray:
        array = np.asarray(weights, dtype=np.float64)
        if self.lambda_regularize > 0.0:
            contrast = _stable_scaled_difference(
                array,
                self.reference_mean,
                1.0,
            )
        else:
            free = np.arange(self.n_sites, dtype=np.int64) != self.anchor
            contrast = _stable_scaled_difference(
                array[free],
                array[self.anchor],
                1.0,
            )
        return _power_scaled_product(-self.variable_exponent, contrast)

    @property
    def rhs_parts(self) -> tuple[np.ndarray, np.ndarray]:
        if self.lambda_regularize > 0.0:
            return (
                np.concatenate(
                    (self.observation_rhs, self.regularization_rhs)
                ),
                np.concatenate(
                    (
                        self.observation_rhs_low,
                        self.regularization_rhs_low,
                    )
                ),
            )
        return self.observation_rhs, self.observation_rhs_low

    @property
    def rhs(self) -> np.ndarray:
        high, low = self.rhs_parts
        return _stable_sum(high, low)


def _prepare_quadratic(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    strength: np.ndarray,
    reference: np.ndarray,
    lambda_regularize: float,
    *,
    rhs_hints: Sequence[np.ndarray] = (),
) -> _PreparedQuadratic:
    I_array = np.asarray(I, dtype=np.int64)
    J_array = np.asarray(J, dtype=np.int64)
    alpha_array = np.asarray(alpha, dtype=np.float64)
    beta_array = np.asarray(beta, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    strength_array = np.asarray(strength, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if not (
        I_array.shape
        == J_array.shape
        == alpha_array.shape
        == beta_array.shape
        == target_array.shape
        == strength_array.shape
    ):
        raise QuadraticNumericalError(
            'quadratic component arrays have inconsistent shapes'
        )
    n_sites = int(reference_array.size)
    if n_sites == 0:
        raise QuadraticNumericalError('quadratic component is empty')
    if np.any(strength_array < 0.0):
        raise QuadraticNumericalError(
            'quadratic row strengths must be nonnegative'
        )
    active = strength_array > 0.0
    I_active = I_array[active]
    J_active = J_array[active]
    alpha_active = alpha_array[active]
    beta_active = beta_array[active]
    target_active = target_array[active]
    strength_active = strength_array[active]
    if not np.all(
        np.isfinite(
            np.concatenate(
                (
                    alpha_active,
                    beta_active,
                    target_active,
                    strength_active,
                    reference_array,
                )
            )
        )
    ):
        raise QuadraticNumericalError(
            'quadratic component inputs must be finite'
        )
    if np.any(alpha_active == 0.0):
        raise QuadraticNumericalError(
            'quadratic component contains a zero affine coefficient'
        )

    lam = float(lambda_regularize)
    if not math.isfinite(lam) or lam < 0.0:
        raise QuadraticNumericalError(
            'quadratic L2 strength must be finite and nonnegative'
        )
    sqrt_strength = np.sqrt(strength_active)
    sqrt_lambda = math.sqrt(lam)
    reference_mean = _component_mean(reference_array) if lam > 0.0 else 0.0

    coefficient_valid, coefficient_exponents = _finite_product_exponents(
        sqrt_strength,
        alpha_active,
    )
    if np.any(coefficient_valid):
        max_coefficient_exp = int(
            np.max(coefficient_exponents[coefficient_valid])
        )
    elif lam > 0.0:
        max_coefficient_exp = math.frexp(sqrt_lambda)[1]
    else:
        raise QuadraticNumericalError(
            'quadratic component has no positive-curvature rows'
        )
    if lam > 0.0:
        max_coefficient_exp = max(
            max_coefficient_exp,
            math.frexp(sqrt_lambda)[1],
        )

    hint_arrays: list[np.ndarray] = []
    for hint in rhs_hints:
        hint_array = np.asarray(hint, dtype=np.float64)
        if hint_array.shape != target_array.shape:
            raise QuadraticNumericalError(
                'quadratic right-hand-side hint has an unexpected shape'
            )
        hint_arrays.append(hint_array[active])

    max_rhs_exp: int | None = None
    for hint in (target_active, *hint_arrays):
        rhs_valid, rhs_exponents = _scaled_difference_exponents(
            hint,
            beta_active,
            sqrt_strength,
        )
        if np.any(rhs_valid):
            candidate_exp = int(np.max(rhs_exponents[rhs_valid]))
            max_rhs_exp = (
                candidate_exp
                if max_rhs_exp is None
                else max(max_rhs_exp, candidate_exp)
            )
    if lam > 0.0:
        rhs_valid, rhs_exponents = _scaled_difference_exponents(
            reference_array,
            reference_mean,
            sqrt_lambda,
        )
        if np.any(rhs_valid):
            candidate_exp = int(np.max(rhs_exponents[rhs_valid]))
            max_rhs_exp = (
                candidate_exp
                if max_rhs_exp is None
                else max(max_rhs_exp, candidate_exp)
            )

    if max_rhs_exp is not None:
        variable_exponent = max_rhs_exp - max_coefficient_exp
        design_exponent = max_rhs_exp
    else:
        variable_exponent = 0
        design_exponent = max_coefficient_exp

    coefficient_power = variable_exponent - design_exponent
    observation_coefficient = _power_scaled_product(
        coefficient_power,
        sqrt_strength,
        alpha_active,
    )
    regularization_coefficient = (
        0.0
        if lam == 0.0
        else float(
            _power_scaled_product(
                coefficient_power,
                sqrt_lambda,
            )
        )
    )
    observation_rhs, observation_rhs_low = _scaled_difference_parts(
        target_active,
        beta_active,
        sqrt_strength,
        -design_exponent,
    )
    if lam == 0.0:
        regularization_rhs = np.zeros(0, dtype=np.float64)
        regularization_rhs_low = np.zeros(0, dtype=np.float64)
    else:
        regularization_rhs, regularization_rhs_low = (
            _scaled_difference_parts(
                reference_array,
                reference_mean,
                sqrt_lambda,
                -design_exponent,
            )
        )

    if not (
        np.all(np.isfinite(observation_coefficient))
        and math.isfinite(regularization_coefficient)
        and np.all(np.isfinite(observation_rhs))
        and np.all(np.isfinite(observation_rhs_low))
        and np.all(np.isfinite(regularization_rhs))
        and np.all(np.isfinite(regularization_rhs_low))
    ):
        raise QuadraticNumericalError(
            'scaled augmented least-squares design is outside binary64 range'
        )

    if np.any(observation_coefficient == 0.0):
        raise QuadraticNumericalError(
            'quadratic coefficient range exceeds binary64 support'
        )
    represented_observation_rhs = _stable_sum(
        observation_rhs,
        observation_rhs_low,
    )
    if np.any(
        (represented_observation_rhs == 0.0)
        & (target_active != beta_active)
    ):
        raise QuadraticNumericalError(
            'quadratic right-hand-side range exceeds binary64 support'
        )
    if lam > 0.0 and regularization_coefficient == 0.0:
        raise QuadraticNumericalError(
            'L2 curvature is below the supported binary64 design range'
        )
    if lam > 0.0:
        represented_regularization_rhs = _stable_sum(
            regularization_rhs,
            regularization_rhs_low,
        )
        if np.any(
            (represented_regularization_rhs == 0.0)
            & (reference_array != reference_mean)
        ):
            raise QuadraticNumericalError(
                'L2 right-hand-side range exceeds binary64 support'
            )

    if lam > 0.0:
        anchor = 0
    else:
        # Gauge fixing is mathematically arbitrary but numerically important.
        # Anchor the site with the largest scaled weighted degree rather than
        # whichever site happened to be numbered zero.  A final uniform shift
        # restores the historical site-zero output gauge.
        scaled_curvature = _stable_product(
            observation_coefficient,
            observation_coefficient,
        )
        weighted_degree = (
            np.bincount(
                I_active,
                weights=scaled_curvature,
                minlength=n_sites,
            )
            + np.bincount(
                J_active,
                weights=scaled_curvature,
                minlength=n_sites,
            )
        )
        if not np.all(np.isfinite(weighted_degree)):
            raise QuadraticNumericalError(
                'quadratic gauge anchor cannot be selected safely'
            )
        anchor = int(np.argmax(weighted_degree))

    return _PreparedQuadratic(
        n_sites=n_sites,
        anchor=anchor,
        I=I_active,
        J=J_active,
        alpha=alpha_active,
        beta=beta_active,
        target=target_active,
        strength=strength_active,
        sqrt_strength=sqrt_strength,
        reference=reference_array,
        lambda_regularize=lam,
        sqrt_lambda=sqrt_lambda,
        reference_mean=reference_mean,
        variable_exponent=variable_exponent,
        design_exponent=design_exponent,
        observation_coefficient=np.asarray(
            observation_coefficient, dtype=np.float64
        ),
        regularization_coefficient=regularization_coefficient,
        observation_rhs=np.asarray(observation_rhs, dtype=np.float64),
        observation_rhs_low=np.asarray(
            observation_rhs_low, dtype=np.float64
        ),
        regularization_rhs=np.asarray(
            regularization_rhs, dtype=np.float64
        ),
        regularization_rhs_low=np.asarray(
            regularization_rhs_low, dtype=np.float64
        ),
        n_unknowns=(n_sites if lam > 0.0 else max(0, n_sites - 1)),
    )


class _LeastSquaresFactor:
    condition: float | None
    largest_singular: float | None
    smallest_singular_lower_bound: float | None = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class _CertificationFactor(_LeastSquaresFactor):
    """Singular-value metadata for certifying an externally made candidate."""

    def __init__(self, prepared: _PreparedQuadratic) -> None:
        frobenius_square = prepared.design_frobenius_square()
        self.largest_singular = math.sqrt(frobenius_square)
        self.smallest_singular_lower_bound = (
            prepared.smallest_singular_lower_bound()
        )
        self.condition = _stable_ratio_product_scalar(
            (self.largest_singular,),
            (self.smallest_singular_lower_bound,),
        )


class _DenseLeastSquaresFactor(_LeastSquaresFactor):
    def __init__(self, design: np.ndarray) -> None:
        matrix = np.asarray(design, dtype=np.float64)
        if matrix.ndim != 2:
            raise QuadraticNumericalError(
                'dense least-squares design must be two-dimensional'
            )
        if matrix.shape[1] == 0:
            self.design = matrix
            self.q = np.zeros((matrix.shape[0], 0), dtype=np.float64)
            self.r = np.zeros((0, 0), dtype=np.float64)
            self.condition = 1.0
            self.largest_singular = 0.0
            return
        try:
            q, r = np.linalg.qr(matrix, mode='reduced')
            singular = np.linalg.svd(r, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise QuadraticNumericalError(
                'dense augmented least-squares factorization failed'
            ) from exc
        if singular.size != matrix.shape[1] or singular[-1] == 0.0:
            raise QuadraticNumericalError(
                'quadratic component is rank deficient after gauge fixing'
            )
        self.design = matrix
        self.q = q
        self.r = r
        self.largest_singular = float(singular[0])
        self.condition = _stable_ratio_product_scalar(
            (float(singular[0]),),
            (float(singular[-1]),),
        )
        self.extended_design = (
            matrix.astype(np.longdouble)
            if np.finfo(np.longdouble).eps < _FLOAT_EPS
            else None
        )

    def _accurate_residual(
        self,
        rhs: np.ndarray,
        solution: np.ndarray,
    ) -> tuple[np.ndarray, float | np.longdouble]:
        if self.extended_design is not None:
            residual_extended = (
                np.asarray(rhs, dtype=np.longdouble)
                - self.extended_design
                @ np.asarray(solution, dtype=np.longdouble)
            )
            score = np.sum(
                residual_extended * residual_extended,
                dtype=np.longdouble,
            )
            return np.asarray(residual_extended, dtype=np.float64), score

        fitted = np.asarray(
            [
                _stable_sum_scalar(
                    *_stable_product(row, solution).tolist()
                )
                for row in self.design
            ],
            dtype=np.float64,
        )
        residual = _stable_sum(rhs, -fitted)
        return residual, _stable_sum_squares(residual)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.r.size == 0:
            return np.zeros(0, dtype=np.float64)
        rhs_array = np.asarray(rhs, dtype=np.float64)
        try:
            projected = self.q.T @ rhs_array
            solution = np.linalg.solve(self.r, projected)
            if not np.all(np.isfinite(solution)):
                raise QuadraticNumericalError(
                    'dense augmented least-squares solve produced '
                    'non-finite unknowns'
                )
            if self.condition > _DENSE_REFINEMENT_CONDITION:
                best_solution = np.asarray(solution, dtype=np.float64)
                residual, best_score = self._accurate_residual(
                    rhs_array,
                    best_solution,
                )
                current = best_solution
                for _ in range(_DENSE_REFINEMENT_STEPS):
                    correction = np.linalg.solve(
                        self.r,
                        self.q.T @ residual,
                    )
                    if self.extended_design is not None:
                        candidate = np.asarray(
                            np.asarray(current, dtype=np.longdouble)
                            + np.asarray(correction, dtype=np.longdouble),
                            dtype=np.float64,
                        )
                    else:
                        candidate = _stable_sum(current, correction)
                    if np.array_equal(candidate, current):
                        break
                    residual, score = self._accurate_residual(
                        rhs_array,
                        candidate,
                    )
                    if score < best_score:
                        best_solution = candidate
                        best_score = score
                    current = candidate

                # A full-rank triangular solve can amplify a singular direction
                # that binary64 cannot resolve.  LAPACK's rank-revealing
                # least-squares candidate truncates only such directions.  It
                # is accepted solely when the accurately evaluated augmented
                # residual is smaller than every full-rank candidate.
                truncated = np.linalg.lstsq(
                    self.design,
                    rhs_array,
                    rcond=None,
                )[0]
                if np.all(np.isfinite(truncated)):
                    _truncated_residual, truncated_score = (
                        self._accurate_residual(rhs_array, truncated)
                    )
                    if truncated_score < best_score:
                        best_solution = np.asarray(
                            truncated, dtype=np.float64
                        )
                solution = best_solution
        except np.linalg.LinAlgError as exc:
            raise QuadraticNumericalError(
                'dense augmented least-squares solve failed'
            ) from exc
        return np.asarray(solution, dtype=np.float64)


class _SparseLeastSquaresFactor(_LeastSquaresFactor):
    def __init__(self, design) -> None:
        try:
            from scipy.sparse import bmat, eye
            from scipy.sparse.linalg import splu
        except ImportError as exc:  # pragma: no cover - exercised by caller
            raise ImportError(
                "linear_backend='sparse' requires SciPy; install "
                'pyvoro2[sparse] or scipy'
            ) from exc
        matrix = design.tocsc()
        n_rows, n_columns = matrix.shape
        if n_columns == 0:
            self.factor = None
            self.n_rows = n_rows
            self.n_columns = 0
            self.condition = 1.0
            self.largest_singular = 0.0
            return
        kkt = bmat(
            [
                [eye(n_rows, format='csc'), matrix],
                [matrix.T, None],
            ],
            format='csc',
        )
        try:
            self.factor = splu(kkt)
        except RuntimeError as exc:
            raise QuadraticNumericalError(
                'sparse augmented least-squares system is singular'
            ) from exc
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.condition = None
        self.largest_singular = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.n_columns == 0:
            return np.zeros(0, dtype=np.float64)
        assert self.factor is not None
        system_rhs = np.concatenate(
            (
                np.asarray(rhs, dtype=np.float64),
                np.zeros(self.n_columns, dtype=np.float64),
            )
        )
        try:
            solution = self.factor.solve(system_rhs)
        except RuntimeError as exc:
            raise QuadraticNumericalError(
                'sparse augmented least-squares solve failed'
            ) from exc
        return np.asarray(solution[self.n_rows:], dtype=np.float64)


class _NormalEquationFactor(_LeastSquaresFactor):
    """Fast, guarded normal-equation map for ordinary scaled systems."""

    def __init__(
        self,
        prepared: _PreparedQuadratic,
        backend: Literal['dense', 'sparse'],
        condition_bound: float,
    ) -> None:
        self.prepared = prepared
        self.backend = backend
        self.condition = float(condition_bound)
        self.smallest_singular_lower_bound = (
            prepared.smallest_singular_lower_bound()
        )
        self.largest_singular = math.sqrt(
            prepared.design_frobenius_square()
        )
        curvature = _stable_product(
            prepared.observation_coefficient,
            prepared.observation_coefficient,
        )
        if np.any(~np.isfinite(curvature)) or np.any(curvature == 0.0):
            raise QuadraticNumericalError(
                'guarded normal equations lost observation curvature'
            )
        regularization_curvature = 0.0
        if prepared.lambda_regularize > 0.0:
            regularization_curvature = _stable_product_scalar(
                prepared.regularization_coefficient,
                prepared.regularization_coefficient,
            )
            if (
                not math.isfinite(regularization_curvature)
                or regularization_curvature == 0.0
            ):
                raise QuadraticNumericalError(
                    'guarded normal equations lost L2 curvature'
                )

        if backend == 'dense':
            full = np.zeros(
                (prepared.n_sites, prepared.n_sites),
                dtype=np.float64,
            )
            np.add.at(full, (prepared.I, prepared.I), curvature)
            np.add.at(full, (prepared.J, prepared.J), curvature)
            np.add.at(full, (prepared.I, prepared.J), -curvature)
            np.add.at(full, (prepared.J, prepared.I), -curvature)
            if regularization_curvature != 0.0:
                full.flat[:: prepared.n_sites + 1] += (
                    regularization_curvature
                )
            if prepared.lambda_regularize > 0.0:
                self.matrix = full
            else:
                free = (
                    np.arange(prepared.n_sites, dtype=np.int64)
                    != prepared.anchor
                )
                self.matrix = full[np.ix_(free, free)]
            self.factor = None
            return

        if backend != 'sparse':
            raise ValueError(
                f'unsupported quadratic backend: {backend!r}'
            )
        try:
            from scipy.sparse import coo_matrix, eye
            from scipy.sparse.linalg import splu
        except ImportError as exc:  # pragma: no cover - caller validates
            raise ImportError(
                "linear_backend='sparse' requires SciPy; install "
                'pyvoro2[sparse] or scipy'
            ) from exc
        data = np.concatenate(
            (curvature, curvature, -curvature, -curvature)
        )
        rows = np.concatenate(
            (prepared.I, prepared.J, prepared.I, prepared.J)
        )
        columns = np.concatenate(
            (prepared.I, prepared.J, prepared.J, prepared.I)
        )
        matrix = coo_matrix(
            (data, (rows, columns)),
            shape=(prepared.n_sites, prepared.n_sites),
        ).tocsc()
        if regularization_curvature != 0.0:
            matrix = matrix + regularization_curvature * eye(
                prepared.n_sites,
                format='csc',
            )
        elif prepared.n_sites > 1:
            matrix = matrix[1:, :][:, 1:]
        else:
            matrix = matrix[:0, :0]
        try:
            self.factor = splu(matrix) if matrix.shape[0] else None
        except RuntimeError as exc:
            raise QuadraticNumericalError(
                'guarded sparse normal equations are singular'
            ) from exc
        self.matrix = None

    def _normal_rhs(self, rhs: np.ndarray) -> np.ndarray:
        prepared = self.prepared
        rhs_array = np.asarray(rhs, dtype=np.float64)
        expected_rows = int(prepared.I.size) + (
            prepared.n_sites
            if prepared.lambda_regularize > 0.0
            else 0
        )
        if rhs_array.shape != (expected_rows,):
            raise QuadraticNumericalError(
                'guarded normal-equation RHS has an unexpected shape'
            )
        row_rhs = _stable_product(
            prepared.observation_coefficient,
            rhs_array[: prepared.I.size],
        )
        # This is only the guarded ordinary-system candidate.  Direct
        # vectorized accumulation keeps the fast path competitive; the
        # authoritative source-gradient certificate rejects any material
        # cancellation and falls back to the augmented solve.
        full_rhs = (
            np.bincount(
                prepared.I,
                weights=row_rhs,
                minlength=prepared.n_sites,
            )
            - np.bincount(
                prepared.J,
                weights=row_rhs,
                minlength=prepared.n_sites,
            )
        )
        if not np.all(np.isfinite(full_rhs)):
            raise QuadraticNumericalError(
                'guarded normal-equation RHS accumulation failed'
            )
        if prepared.lambda_regularize > 0.0:
            regularization_rhs = _stable_product(
                prepared.regularization_coefficient,
                rhs_array[prepared.I.size :],
            )
            return _stable_sum(full_rhs, regularization_rhs)
        free = (
            np.arange(prepared.n_sites, dtype=np.int64)
            != prepared.anchor
        )
        return full_rhs[free]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        normal_rhs = self._normal_rhs(rhs)
        if normal_rhs.size == 0:
            return np.zeros(0, dtype=np.float64)
        try:
            if self.backend == 'dense':
                assert self.matrix is not None
                solution = np.linalg.solve(self.matrix, normal_rhs)
            else:
                assert self.factor is not None
                solution = self.factor.solve(normal_rhs)
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            raise QuadraticNumericalError(
                'guarded normal-equation solve failed'
            ) from exc
        solution = np.asarray(solution, dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            raise QuadraticNumericalError(
                'guarded normal-equation solve produced non-finite unknowns'
            )
        return solution


def _make_normal_factor(
    prepared: _PreparedQuadratic,
    backend: Literal['dense', 'sparse'],
) -> _NormalEquationFactor | None:
    """Return a guarded ordinary-system fast path, when conservative."""

    condition_bound = prepared.sparse_condition_upper_bound()
    if condition_bound > _NORMAL_FAST_CONDITION_LIMIT:
        return None
    return _NormalEquationFactor(prepared, backend, condition_bound)


def _solve_rhs_parts(
    factor: _LeastSquaresFactor,
    parts: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Solve a factored linear least-squares map for high/low RHS parts."""

    high, low = parts
    solution = factor.solve(high)
    if np.any(low != 0.0):
        solution = _stable_sum(solution, factor.solve(low))
    if not np.all(np.isfinite(solution)):
        raise QuadraticNumericalError(
            'augmented least-squares solve produced non-finite unknowns'
        )
    return np.asarray(solution, dtype=np.float64)


def _make_factor(
    prepared: _PreparedQuadratic,
    backend: Literal['dense', 'sparse'],
) -> _LeastSquaresFactor:
    lower_bound = prepared.smallest_singular_lower_bound()
    if backend == 'dense':
        factor = _DenseLeastSquaresFactor(prepared.dense_design())
        factor.smallest_singular_lower_bound = lower_bound
        return factor
    if backend == 'sparse':
        factor = _SparseLeastSquaresFactor(prepared.sparse_design())
        factor.condition = prepared.sparse_condition_upper_bound()
        factor.largest_singular = math.sqrt(
            prepared.design_frobenius_square()
        )
        factor.smallest_singular_lower_bound = lower_bound
        return factor
    raise ValueError(f'unsupported quadratic backend: {backend!r}')


def _quadratic_objective(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
) -> float:
    mismatch = _mismatch_values_from_affine(
        prepared.beta,
        prepared.alpha,
        np.asarray(weights, dtype=np.float64)[prepared.I],
        np.asarray(weights, dtype=np.float64)[prepared.J],
        prepared.target,
        prepared.strength,
        SquaredLoss(),
    )
    l2 = _l2_value(
        weights,
        prepared.reference,
        prepared.lambda_regularize,
    )
    return _stable_sum_scalar(*mismatch.tolist(), l2)


def _can_use_tiny_exact_oracle(prepared: _PreparedQuadratic) -> bool:
    """Return whether bounded rational certification is resource-safe."""

    return (
        prepared.n_sites <= _TINY_EXACT_ORACLE_MAX_SITES
        and prepared.I.size <= _TINY_EXACT_ORACLE_MAX_ROWS
    )


def _exact_fraction_solve(
    matrix: list[list[Fraction]],
    rhs: list[Fraction],
) -> list[Fraction]:
    size = len(rhs)
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(matrix[row][column]),
        )
        if matrix[pivot][column] == 0:
            raise QuadraticNumericalError(
                'tiny exact certification system is singular'
            )
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        pivot_value = matrix[column][column]
        for row in range(column + 1, size):
            if matrix[row][column] == 0:
                continue
            factor = matrix[row][column] / pivot_value
            matrix[row][column] = Fraction(0)
            for inner in range(column + 1, size):
                matrix[row][inner] -= factor * matrix[column][inner]
            rhs[row] -= factor * rhs[column]
    solution = [Fraction(0) for _ in range(size)]
    for row in range(size - 1, -1, -1):
        remainder = rhs[row] - sum(
            (
                matrix[row][column] * solution[column]
                for column in range(row + 1, size)
            ),
            Fraction(0),
        )
        solution[row] = remainder / matrix[row][row]
    return solution


def _tiny_exact_objective_gap(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
    *,
    required_mean: float | None,
) -> tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    np.ndarray | None,
    Fraction | None,
]:
    n_sites = prepared.n_sites
    lam = _fraction(prepared.lambda_regularize)
    matrix = [
        [Fraction(0) for _ in range(n_sites)]
        for _ in range(n_sites)
    ]
    rhs = [Fraction(0) for _ in range(n_sites)]
    for row in range(prepared.I.size):
        strength = _fraction(prepared.strength[row])
        alpha = _fraction(prepared.alpha[row])
        curvature = strength * alpha * alpha
        row_rhs = strength * alpha * (
            _fraction(prepared.target[row]) - _fraction(prepared.beta[row])
        )
        site_i = int(prepared.I[row])
        site_j = int(prepared.J[row])
        matrix[site_i][site_i] += curvature
        matrix[site_j][site_j] += curvature
        matrix[site_i][site_j] -= curvature
        matrix[site_j][site_i] -= curvature
        rhs[site_i] += row_rhs
        rhs[site_j] -= row_rhs
    if lam > 0:
        for site in range(n_sites):
            matrix[site][site] += lam
            rhs[site] += lam * _fraction(prepared.reference[site])
        optimum = _exact_fraction_solve(matrix, rhs)
    else:
        reduced = [row[1:] for row in matrix[1:]]
        optimum = [Fraction(0)] + _exact_fraction_solve(reduced, rhs[1:])
        if required_mean is not None:
            exact_mean = sum(optimum, Fraction(0)) / n_sites
            shift = _fraction(required_mean) - exact_mean
            optimum = [value + shift for value in optimum]

    def objective(candidate: Sequence[Fraction]) -> Fraction:
        value = Fraction(0)
        for row in range(prepared.I.size):
            residual = (
                _fraction(prepared.beta[row])
                + _fraction(prepared.alpha[row])
                * (
                    candidate[int(prepared.I[row])]
                    - candidate[int(prepared.J[row])]
                )
                - _fraction(prepared.target[row])
            )
            value += (
                Fraction(1, 2)
                * _fraction(prepared.strength[row])
                * residual
                * residual
            )
        if lam > 0:
            for site in range(n_sites):
                displacement = (
                    candidate[site] - _fraction(prepared.reference[site])
                )
                value += Fraction(1, 2) * lam * displacement * displacement
        return value

    candidate_exact = [_fraction(value) for value in weights]
    optimum_objective = objective(optimum)
    candidate_objective = objective(candidate_exact)
    rounded_weights: np.ndarray | None
    rounded_gap: Fraction | None
    try:
        rounded_weights = np.asarray(
            [float(value) for value in optimum],
            dtype=np.float64,
        )
    except OverflowError:
        rounded_weights = None
        rounded_gap = None
    else:
        if np.all(np.isfinite(rounded_weights)):
            rounded_objective = objective(
                [_fraction(value) for value in rounded_weights]
            )
            rounded_gap = rounded_objective - optimum_objective
        else:
            rounded_weights = None
            rounded_gap = None
    data_scale = Fraction(0)
    for row in range(prepared.I.size):
        offset = _fraction(prepared.target[row]) - _fraction(
            prepared.beta[row]
        )
        data_scale += (
            Fraction(1, 2)
            * _fraction(prepared.strength[row])
            * offset
            * offset
        )
    if lam > 0:
        for value in prepared.reference:
            data_scale += (
                Fraction(1, 2)
                * lam
                * _fraction(value)
                * _fraction(value)
            )
    return (
        candidate_objective - optimum_objective,
        candidate_objective,
        optimum_objective,
        data_scale,
        rounded_weights,
        rounded_gap,
    )


def _exact_certified_candidate(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
    *,
    factor_scale: float,
    allow_recovery: bool,
    required_mean: float | None,
) -> np.ndarray:
    """Select and certify a tiny component with an exact rational oracle.

    The oracle is deliberately bounded by ``_can_use_tiny_exact_oracle`` and
    is reached only after the ordinary floating-point certificates detect a
    doubtful candidate.  It may replace that candidate by the binary64
    rounding of the exact continuous optimum when the replacement has a
    smaller source-objective gap.
    """

    (
        exact_gap,
        exact_candidate_objective,
        exact_optimum,
        _exact_data,
        rounded_weights,
        rounded_gap,
    ) = _tiny_exact_objective_gap(
        prepared,
        weights,
        required_mean=required_mean,
    )

    selected = np.asarray(weights, dtype=np.float64)
    selected_gap = exact_gap
    selected_objective = exact_candidate_objective
    if (
        allow_recovery
        and rounded_weights is not None
        and rounded_gap is not None
        and rounded_gap < selected_gap
    ):
        selected = rounded_weights
        selected_gap = rounded_gap
        selected_objective = exact_optimum + rounded_gap

    exact_factor = Fraction.from_float(factor_scale)
    exact_eps = Fraction.from_float(_FLOAT_EPS)
    # The exact helper is a tighter implementation of the same universal
    # continuous-objective forward-gap rule used by the ordinary certificate.
    # It does not define a separate binary64-lattice objective and it cannot
    # turn coordinatewise rounding of the exact optimum into a proof of
    # optimality.  In particular, if the continuous optimum is exactly zero,
    # every positive-gap binary64 candidate fails because factor_scale * eps
    # is strictly smaller than one.
    exact_allowed = (
        exact_factor * exact_eps * abs(selected_objective)
    )
    if selected_gap > exact_allowed:
        raise QuadraticNumericalError(
            'quadratic optimum cannot be certified at binary64 output '
            'resolution'
        )
    return np.asarray(selected, dtype=np.float64)


def _exact_gradient_norm_upper(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
) -> float | None:
    """Return a bounded exact-dyadic gradient-norm upper bound."""

    if (
        prepared.n_sites > _EXACT_GRADIENT_MAX_SITES
        or prepared.I.size > _EXACT_GRADIENT_MAX_ROWS
    ):
        return None

    dyadic_cache: dict[float, tuple[int, int]] = {}

    def dyadic(value: float) -> tuple[int, int]:
        scalar = float(value)
        cached = dyadic_cache.get(scalar)
        if cached is not None:
            return cached
        numerator, denominator = scalar.as_integer_ratio()
        result = (
            int(numerator),
            -(int(denominator).bit_length() - 1),
        )
        dyadic_cache[scalar] = result
        return result

    def sum_dyadics(
        terms: Sequence[tuple[int, int]],
    ) -> tuple[int, int]:
        nonzero = tuple(term for term in terms if term[0] != 0)
        if not nonzero:
            return 0, 0
        exponent = min(term[1] for term in nonzero)
        numerator = sum(
            term[0] << (term[1] - exponent)
            for term in nonzero
        )
        return numerator, exponent

    gradient_numerator = [0 for _ in range(prepared.n_sites)]
    gradient_exponent = [0 for _ in range(prepared.n_sites)]

    def add_gradient(site: int, numerator: int, exponent: int) -> None:
        if numerator == 0:
            return
        current = gradient_numerator[site]
        if current == 0:
            gradient_numerator[site] = numerator
            gradient_exponent[site] = exponent
            return
        current_exponent = gradient_exponent[site]
        if exponent < current_exponent:
            current = (
                (current << (current_exponent - exponent))
                + numerator
            )
            current_exponent = exponent
        else:
            current += numerator << (exponent - current_exponent)
        gradient_numerator[site] = current
        gradient_exponent[site] = current_exponent

    candidate = np.asarray(weights, dtype=np.float64)
    for row in range(prepared.I.size):
        site_i = int(prepared.I[row])
        site_j = int(prepared.J[row])
        confidence_num, confidence_exp = dyadic(
            float(prepared.strength[row])
        )
        alpha_num, alpha_exp = dyadic(float(prepared.alpha[row]))
        beta_num, beta_exp = dyadic(float(prepared.beta[row]))
        target_num, target_exp = dyadic(float(prepared.target[row]))
        left_num, left_exp = dyadic(float(candidate[site_i]))
        right_num, right_exp = dyadic(float(candidate[site_j]))
        confidence_alpha_num = confidence_num * alpha_num
        confidence_alpha_exp = confidence_exp + alpha_exp
        alpha_square_num = confidence_alpha_num * alpha_num
        alpha_square_exp = confidence_alpha_exp + alpha_exp
        contribution_num, contribution_exp = sum_dyadics(
            (
                (
                    confidence_alpha_num * beta_num,
                    confidence_alpha_exp + beta_exp,
                ),
                (
                    alpha_square_num * left_num,
                    alpha_square_exp + left_exp,
                ),
                (
                    -alpha_square_num * right_num,
                    alpha_square_exp + right_exp,
                ),
                (
                    -confidence_alpha_num * target_num,
                    confidence_alpha_exp + target_exp,
                ),
            )
        )
        add_gradient(site_i, contribution_num, contribution_exp)
        add_gradient(site_j, -contribution_num, contribution_exp)
    if prepared.lambda_regularize > 0.0:
        lambda_num, lambda_exp = dyadic(
            prepared.lambda_regularize
        )
        for site in range(prepared.n_sites):
            candidate_num, candidate_exp = dyadic(
                float(candidate[site])
            )
            reference_num, reference_exp = dyadic(
                float(prepared.reference[site])
            )
            displacement_num, displacement_exp = sum_dyadics(
                (
                    (candidate_num, candidate_exp),
                    (-reference_num, reference_exp),
                )
            )
            add_gradient(
                site,
                lambda_num * displacement_num,
                lambda_exp + displacement_exp,
            )
        selected = range(prepared.n_sites)
    else:
        selected = (
            site
            for site in range(prepared.n_sites)
            if site != prepared.anchor
        )

    power = prepared.variable_exponent - 2 * prepared.design_exponent
    coordinate_upper: list[float] = []
    for site in selected:
        numerator = abs(gradient_numerator[site])
        if numerator == 0:
            continue
        exponent = gradient_exponent[site] + power
        bit_count = numerator.bit_length()
        shift = max(0, bit_count - 53)
        leading = numerator >> shift
        if shift and numerator != leading << shift:
            leading += 1
        try:
            upper = math.ldexp(float(leading), exponent + shift)
        except OverflowError:
            return math.inf
        if not math.isfinite(upper):
            return math.inf
        if upper == 0.0:
            upper = _MIN_SUBNORMAL
        else:
            upper = math.nextafter(upper, math.inf)
        coordinate_upper.append(upper)
    if not coordinate_upper:
        return 0.0
    scale = max(coordinate_upper)
    normalized_square_upper: list[float] = []
    for value in coordinate_upper:
        ratio = math.nextafter(value / scale, math.inf)
        normalized_square_upper.append(
            math.nextafter(ratio * ratio, math.inf)
        )
    square_sum_upper = math.nextafter(
        math.fsum(normalized_square_upper),
        math.inf,
    )
    root_upper = math.nextafter(
        math.sqrt(square_sum_upper),
        math.inf,
    )
    return math.nextafter(scale * root_upper, math.inf)


def _extended_gradient_norm_upper(
    prepared: _PreparedQuadratic,
    weights: np.ndarray,
) -> float | None:
    """Bound the exact source gradient with extended-precision arithmetic."""

    extended = np.longdouble
    extended_info = np.finfo(extended)
    if extended_info.eps >= _FLOAT_EPS:
        return None
    eps = extended(extended_info.eps)

    def gamma(count: np.ndarray | int) -> np.ndarray:
        scaled = np.asarray(count, dtype=extended) * eps
        return scaled / (extended(1.0) - scaled)

    candidate = np.asarray(weights, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        confidence = np.asarray(prepared.strength, dtype=extended)
        alpha = np.asarray(prepared.alpha, dtype=extended)
        beta = np.asarray(prepared.beta, dtype=extended)
        target = np.asarray(prepared.target, dtype=extended)
        left = np.asarray(candidate[prepared.I], dtype=extended)
        right = np.asarray(candidate[prepared.J], dtype=extended)
        term_beta = confidence * alpha * beta
        term_left = confidence * alpha * alpha * left
        term_right = -confidence * alpha * alpha * right
        term_target = -confidence * alpha * target
        row_gradient = (
            (term_beta + term_left)
            + term_right
            + term_target
        )
        row_absolute = (
            np.abs(term_beta)
            + np.abs(term_left)
            + np.abs(term_right)
            + np.abs(term_target)
        )
    if not (
        np.all(np.isfinite(row_gradient))
        and np.all(np.isfinite(row_absolute))
    ):
        return None

    full_gradient = np.zeros(prepared.n_sites, dtype=extended)
    absolute_accumulation = np.zeros(
        prepared.n_sites,
        dtype=extended,
    )
    np.add.at(full_gradient, prepared.I, row_gradient)
    np.add.at(full_gradient, prepared.J, -row_gradient)
    np.add.at(absolute_accumulation, prepared.I, row_absolute)
    np.add.at(absolute_accumulation, prepared.J, row_absolute)
    degree = np.bincount(
        np.concatenate((prepared.I, prepared.J)),
        minlength=prepared.n_sites,
    ).astype(np.int64, copy=False)

    if prepared.lambda_regularize > 0.0:
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            lam = extended(prepared.lambda_regularize)
            reference = np.asarray(
                prepared.reference,
                dtype=extended,
            )
            candidate_extended = np.asarray(candidate, dtype=extended)
            regularization_gradient = (
                lam * candidate_extended - lam * reference
            )
            regularization_absolute = (
                np.abs(lam * candidate_extended)
                + np.abs(lam * reference)
            )
        if not (
            np.all(np.isfinite(regularization_gradient))
            and np.all(np.isfinite(regularization_absolute))
        ):
            return None
        full_gradient += regularization_gradient
        absolute_accumulation += regularization_absolute
        degree += 1

    row_gamma = gamma(24)
    degree_gamma = gamma(np.maximum(degree, 1))
    if (
        not np.all(np.isfinite(degree_gamma))
        or not np.isfinite(row_gamma)
    ):
        return None
    absolute_upper = (
        absolute_accumulation
        / (extended(1.0) - degree_gamma)
        / (extended(1.0) - row_gamma)
    )
    coordinate_error = (
        row_gamma
        + degree_gamma
        + row_gamma * degree_gamma
    ) * absolute_upper
    coordinate_upper = np.abs(full_gradient) + coordinate_error
    if prepared.lambda_regularize == 0.0:
        coordinate_upper = np.delete(
            coordinate_upper,
            prepared.anchor,
        )
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        coordinate_upper = np.ldexp(
            coordinate_upper,
            prepared.variable_exponent - 2 * prepared.design_exponent,
        )
        square_sum = np.sum(
            coordinate_upper * coordinate_upper,
            dtype=extended,
        )
    if not np.isfinite(square_sum) or square_sum < 0.0:
        return None
    norm_gamma = gamma(2 * max(1, coordinate_upper.size) + 4)
    if not np.isfinite(norm_gamma):
        return None
    norm_upper = (
        np.sqrt(square_sum)
        / (extended(1.0) - norm_gamma)
    )
    if not np.isfinite(norm_upper):
        return math.inf
    try:
        rounded = float(norm_upper)
    except OverflowError:
        return math.inf
    if rounded == 0.0 and norm_upper > 0.0:
        return _MIN_SUBNORMAL
    conversion_allowance = _stable_product_scalar(
        64.0,
        _FLOAT_EPS,
        abs(rounded),
    )
    return math.nextafter(
        _stable_sum_scalar(rounded, conversion_allowance),
        math.inf,
    )


def _certify(
    prepared: _PreparedQuadratic,
    factor: _LeastSquaresFactor,
    weights: np.ndarray,
    *,
    allow_recovery: bool = True,
    required_mean: float | None = None,
) -> np.ndarray:
    """Return a source-objective-certified binary64 weight vector.

    Ordinary candidates must satisfy independent component-mean,
    stationarity/objective-gap, rank, and conditioning checks.  Guarded normal
    candidates additionally require a conservative forward-gap upper bound.
    A doubtful tiny component may invoke the bounded rational oracle; larger
    unsupported cases fail structurally instead of being reported as
    ``optimal``.
    """

    candidate = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(candidate)):
        raise QuadraticNumericalError(
            'quadratic solve produced non-finite weights'
        )
    objective = _quadratic_objective(prepared, candidate)
    if not math.isfinite(objective):
        raise QuadraticNumericalError(
            'quadratic solution has a non-finite authoritative objective'
        )
    if _is_exact_zero_objective(prepared, candidate):
        # Exact source residuals, rather than an underflowed floating
        # objective, provide the backend-independent global proof.
        return candidate

    factor_scale = _CERTIFICATION_FACTOR * max(
        1, prepared.n_unknowns
    )
    mean_failed = False
    if prepared.lambda_regularize > 0.0:
        total_difference = _stable_sum_scalar(
            *candidate.tolist(),
            *(-prepared.reference).tolist(),
        )
        mean_excess = _stable_ratio_product_scalar(
            (
                0.5,
                prepared.lambda_regularize,
                total_difference,
                total_difference,
            ),
            (float(prepared.n_sites),),
        )
        mean_allowed = _stable_product_scalar(
            factor_scale,
            _FLOAT_EPS,
            abs(objective),
        )
        mean_failed = (
            not math.isfinite(mean_excess)
            or mean_excess > mean_allowed
        )

    unknown = prepared.unknown_from_weights(candidate)
    if prepared.lambda_regularize > 0.0:
        site_unknown = unknown
    else:
        site_unknown = np.zeros(prepared.n_sites, dtype=np.float64)
        free = (
            np.arange(prepared.n_sites, dtype=np.int64)
            != prepared.anchor
        )
        site_unknown[free] = unknown
    scaled_difference = _stable_scaled_difference(
        site_unknown[prepared.I],
        site_unknown[prepared.J],
        1.0,
    )
    scaled_prediction = _stable_product(
        prepared.observation_coefficient,
        scaled_difference,
    )
    scaled_residual = _stable_sum(
        scaled_prediction,
        -prepared.observation_rhs,
        -prepared.observation_rhs_low,
    )
    confidence_over_root = _stable_ratio_product(
        (prepared.strength,),
        (prepared.sqrt_strength,),
    )
    row_gradient = _power_scaled_product(
        prepared.variable_exponent - prepared.design_exponent,
        confidence_over_root,
        prepared.alpha,
        scaled_residual,
    )
    full_gradient = _stable_incidence_accumulate(
        prepared.n_sites,
        prepared.I,
        prepared.J,
        row_gradient,
    )
    if prepared.lambda_regularize > 0.0:
        regularization_prediction = _stable_product(
            prepared.regularization_coefficient,
            site_unknown,
        )
        regularization_residual = _stable_sum(
            regularization_prediction,
            -prepared.regularization_rhs,
            -prepared.regularization_rhs_low,
        )
        lambda_over_root = _stable_ratio_product_scalar(
            (prepared.lambda_regularize,),
            (prepared.sqrt_lambda,),
        )
        regularization_gradient = _power_scaled_product(
            prepared.variable_exponent - prepared.design_exponent,
            lambda_over_root,
            regularization_residual,
        )
        full_gradient = _stable_sum(
            full_gradient,
            regularization_gradient,
        )
        gradient = full_gradient
    else:
        gradient = full_gradient[free]

    gradient_square = _stable_sum_squares(gradient)
    design_square = prepared.design_frobenius_square()
    if design_square <= 0.0 or not math.isfinite(design_square):
        raise QuadraticNumericalError(
            'quadratic design norm cannot be certified'
        )
    gap_lower_scaled = _stable_ratio_product_scalar(
        (0.5, gradient_square),
        (design_square,),
    )
    objective_scaled = float(
        _power_scaled_product(
            -2 * prepared.design_exponent,
            objective,
        )
    )
    allowed = _stable_product_scalar(
        factor_scale,
        _FLOAT_EPS,
        abs(objective_scaled),
    )

    gap_failed = (
        not math.isfinite(gap_lower_scaled)
        or gap_lower_scaled > allowed
    )

    smallest_singular = factor.smallest_singular_lower_bound
    factor_metadata_valid = (
        smallest_singular is not None
        and math.isfinite(smallest_singular)
        and smallest_singular > 0.0
    )
    if not factor_metadata_valid:
        raise QuadraticNumericalError(
            'quadratic forward objective gap cannot be certified because '
            'the factor path has no defensible smallest-singular-value bound'
        )

    # Bound evaluation error in A.T @ (A @ x - b) using Frobenius norms and
    # the standard gamma_k model for binary64 products and sums.  The large
    # operation count is deliberately conservative and covers the stable
    # affine, incidence, and regularization reductions above.
    design_norm = math.sqrt(design_square)
    unknown_norm = _stable_norm(unknown)
    rhs_high, rhs_low = prepared.rhs_parts
    rhs_norm_bound = _stable_sum_scalar(
        _stable_norm(rhs_high),
        _stable_norm(rhs_low),
    )
    residual_norm_bound = _stable_sum_scalar(
        _stable_product_scalar(design_norm, unknown_norm),
        rhs_norm_bound,
    )
    operation_count = 16 * (
        int(prepared.I.size) + prepared.n_unknowns + 1
    )
    gamma_numerator = operation_count * _FLOAT_EPS
    if gamma_numerator >= 0.5:
        gradient_roundoff_bound = math.inf
    else:
        gamma = gamma_numerator / (1.0 - gamma_numerator)
        gradient_roundoff_bound = _stable_product_scalar(
            gamma,
            design_norm,
            residual_norm_bound,
        )
        gradient_roundoff_bound = _stable_sum_scalar(
            gradient_roundoff_bound,
            _MIN_SUBNORMAL * operation_count,
        )
    gradient_norm_upper = _stable_sum_scalar(
        _stable_norm(gradient),
        gradient_roundoff_bound,
    )
    if math.isfinite(gradient_norm_upper):
        gradient_norm_upper = math.nextafter(
            gradient_norm_upper, math.inf
        )
    gap_upper_scaled = math.inf
    if factor_metadata_valid:
        assert smallest_singular is not None
        gap_upper_scaled = _stable_ratio_product_scalar(
            (0.5, gradient_norm_upper, gradient_norm_upper),
            (smallest_singular, smallest_singular),
        )
        if math.isfinite(gap_upper_scaled):
            gap_upper_scaled = math.nextafter(
                gap_upper_scaled, math.inf
            )
    forward_gap_failed = (
        not math.isfinite(gap_upper_scaled)
        or gap_upper_scaled > allowed
    )
    if forward_gap_failed:
        tighter_gradient_norm = _extended_gradient_norm_upper(
            prepared,
            candidate,
        )
        if tighter_gradient_norm is not None:
            assert smallest_singular is not None
            gap_upper_scaled = _stable_ratio_product_scalar(
                (
                    0.5,
                    tighter_gradient_norm,
                    tighter_gradient_norm,
                ),
                (smallest_singular, smallest_singular),
            )
            if math.isfinite(gap_upper_scaled):
                gap_upper_scaled = math.nextafter(
                    gap_upper_scaled, math.inf
                )
            forward_gap_failed = (
                not math.isfinite(gap_upper_scaled)
                or gap_upper_scaled > allowed
            )
    if forward_gap_failed:
        exact_gradient_norm = _exact_gradient_norm_upper(
            prepared,
            candidate,
        )
        if exact_gradient_norm is not None:
            assert smallest_singular is not None
            gap_upper_scaled = _stable_ratio_product_scalar(
                (
                    0.5,
                    exact_gradient_norm,
                    exact_gradient_norm,
                ),
                (smallest_singular, smallest_singular),
            )
            if math.isfinite(gap_upper_scaled):
                gap_upper_scaled = math.nextafter(
                    gap_upper_scaled, math.inf
                )
            forward_gap_failed = (
                not math.isfinite(gap_upper_scaled)
                or gap_upper_scaled > allowed
            )

    tiny_exact_trigger = (
        _can_use_tiny_exact_oracle(prepared)
        and (
            mean_failed
            or gap_failed
            or forward_gap_failed
        )
    )
    if tiny_exact_trigger:
        return _exact_certified_candidate(
            prepared,
            candidate,
            factor_scale=factor_scale,
            allow_recovery=allow_recovery,
            required_mean=required_mean,
        )

    if mean_failed:
        raise QuadraticNumericalError(
            'quadratic optimum cannot preserve its L2 component mean at '
            'binary64 output resolution'
        )
    if forward_gap_failed:
        raise QuadraticNumericalError(
            'quadratic candidate failed the forward objective-gap '
            'certificate'
        )
    if gap_failed:
        raise QuadraticNumericalError(
            'quadratic optimum failed the objective-gap certificate'
        )
    return candidate


@dataclass(slots=True)
class QuadraticWeightSystem:
    """Reusable centered/scaled least-squares system for ADMM weight steps."""

    prepared: _PreparedQuadratic
    factor: _LeastSquaresFactor

    @classmethod
    def build(
        cls,
        I: np.ndarray,
        J: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        target_hint: np.ndarray,
        row_strength: np.ndarray,
        reference: np.ndarray,
        lambda_regularize: float,
        *,
        backend: Literal['dense', 'sparse'] = 'dense',
        rhs_hints: Sequence[np.ndarray] = (),
    ) -> QuadraticWeightSystem:
        prepared = _prepare_quadratic(
            I,
            J,
            alpha,
            beta,
            target_hint,
            row_strength,
            reference,
            lambda_regularize,
            rhs_hints=rhs_hints,
        )
        factor = _make_factor(prepared, backend)
        return cls(prepared=prepared, factor=factor)

    def solve(self, target: np.ndarray) -> np.ndarray:
        rhs_parts = self.prepared.rhs_parts_for_target(target)
        unknown = _solve_rhs_parts(self.factor, rhs_parts)
        return self.prepared.weights_from_unknown(unknown)


def _solve_prepared_quadratic(
    prepared: _PreparedQuadratic,
    factor: _LeastSquaresFactor,
    *,
    required_mean: float | None = None,
) -> np.ndarray:
    """Solve one prepared system and apply all output certificates."""

    unknown = _solve_rhs_parts(factor, prepared.rhs_parts)
    weights = prepared.weights_from_unknown(unknown)
    if required_mean is not None:
        weights = _shift_to_required_mean(weights, required_mean)
    certified = _certify(
        prepared,
        factor,
        weights,
        required_mean=required_mean,
    )
    if (
        required_mean is not None
        and not np.array_equal(certified, weights)
    ):
        certified = _shift_to_required_mean(
            certified,
            required_mean,
        )
        certified = _certify(
            prepared,
            factor,
            certified,
            allow_recovery=False,
            required_mean=required_mean,
        )
    return certified


def _shift_to_required_mean(
    weights: np.ndarray,
    required_mean: float,
) -> np.ndarray:
    """Apply the public uniform mean gauge until its float map is stable."""

    shifted = np.asarray(weights, dtype=np.float64).copy()
    for _ in range(4):
        updated = shifted + (
            required_mean - float(np.mean(shifted))
        )
        if np.array_equal(updated, shifted):
            return shifted
        shifted = updated
    return shifted


def solve_quadratic_component(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    reference: np.ndarray,
    lambda_regularize: float,
    *,
    backend: Literal['dense', 'sparse'],
    required_mean: float | None = None,
) -> np.ndarray:
    """Solve and certify one connected squared-plus-L2 component."""

    if backend == 'sparse':
        _require_scipy_sparse()

    reference_array = np.asarray(reference, dtype=np.float64)
    n_sites = int(reference_array.size)
    # Exact zero objectives are complete optimality certificates.  The helper
    # is bounded before allocating row work, so it remains available for
    # moderately sized exact multiscale components without affecting the
    # ordinary large sparse path.
    zero_candidate = _zero_objective_candidate(
        n_sites,
        np.asarray(I, dtype=np.int64),
        np.asarray(J, dtype=np.int64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
        reference_array,
        float(lambda_regularize),
    )
    if zero_candidate is not None:
        if required_mean is None:
            return zero_candidate
        zero_candidate = _shift_to_required_mean(
            zero_candidate,
            required_mean,
        )
        prepared = _prepare_quadratic(
            I,
            J,
            alpha,
            beta,
            target,
            confidence,
            reference_array,
            lambda_regularize,
        )
        return _certify(
            prepared,
            _CertificationFactor(prepared),
            zero_candidate,
            required_mean=required_mean,
        )

    prepared = _prepare_quadratic(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        reference_array,
        lambda_regularize,
    )
    try:
        normal_prepared = prepared
        if (
            backend == 'sparse'
            and prepared.lambda_regularize == 0.0
            and prepared.anchor != 0
        ):
            # SuperLU's sparse normal solve is most reproducible in the
            # historical site-zero gauge.  The authoritative augmented
            # fallback retains the stronger weighted-degree anchor.
            normal_prepared = replace(prepared, anchor=0)
        normal_factor = _make_normal_factor(normal_prepared, backend)
        if normal_factor is not None:
            return _solve_prepared_quadratic(
                normal_prepared,
                normal_factor,
                required_mean=required_mean,
            )
    except QuadraticNumericalError:
        # Normal equations are only a guarded ordinary-system fast path.  The
        # unsquared augmented system remains authoritative whenever assembly,
        # solving, or source-objective certification is doubtful.
        pass

    factor = _make_factor(prepared, backend)
    return _solve_prepared_quadratic(
        prepared,
        factor,
        required_mean=required_mean,
    )


def certify_quadratic_component_candidate(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    reference: np.ndarray,
    lambda_regularize: float,
    weights: np.ndarray,
    *,
    backend: Literal['dense', 'sparse'],
    allow_recovery: bool = True,
    required_mean: float | None = None,
) -> np.ndarray:
    """Certify a final binary64 candidate against the source quadratic."""

    if backend == 'sparse':
        _require_scipy_sparse()
    prepared = _prepare_quadratic(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        np.asarray(reference, dtype=np.float64),
        lambda_regularize,
    )
    if prepared.lambda_regularize == 0.0 and prepared.anchor != 0:
        # Final public certification uses one factor-independent gauge-fixed
        # design.  Site zero is the documented connected-component output
        # gauge and is independent of the candidate-generation backend.
        prepared = replace(prepared, anchor=0)
    return _certify(
        prepared,
        _CertificationFactor(prepared),
        np.asarray(weights, dtype=np.float64),
        allow_recovery=allow_recovery,
        required_mean=required_mean,
    )


__all__ = [
    'QuadraticNumericalError',
    'QuadraticWeightSystem',
    'certify_quadratic_component_candidate',
    'solve_quadratic_component',
]
